# PHASE3_SUPERVISOR_IMPLEMENTATION_REPORT

> ⚠️ **历史快照 + 口径变更**：本文第 7/222 行"默认关闭增强模式"是 Phase 3 **当时**的口径。Phase 3.1 生产接线后已翻转为 **默认开启**：`ENABLE_SUPERVISOR_RUNTIME` 未设置 → True（`recommendation/__init__.py:38-45`），TEAM 路径走 `SupervisorAgent(enhanced=True, backend="langgraph")`；显式设 `0/false/no/off` 才回退 `AdaptiveTeamOrchestrator`。FSM 状态与确定性 REVIEW 规则本身未变。


> 项目：SmartImager — 多模态智能图库与对话推荐系统
> 阶段：Phase 3 — Supervisor Agent Implementation（受控监督层 / 有限状态机）
> 日期：2026-09-07
> 原则：不推翻 PlannerAgent / PlanValidator / DAGExecutor / SpecialistAgent / ToolRegistry；
>       只增加 Supervisor 控制层；默认关闭增强模式，保留既有行为作为 baseline。

---

## 1. 概述与目标架构

```text
User
 │
SupervisorAgent（新：PLAN → EXECUTE → REVIEW → REVISE/REPLAN → DONE/ABORT）
 │
PlannerAgent → PlanValidator
 │
DAGExecutor
 │
AgentRegistry → SpecialistAgent
 │
ToolRegistry（adapter 接入 Worker 注入位 + Agent 统一工具入口）
```

Supervisor 是一个**确定性先行的受控监督层**：第一阶段 REVIEW 不调用 LLM，
只用规则（task failed / retrieval empty / candidate below threshold /
synthesis input insufficient）产出 `SupervisorDecision`。

---

## 2. 需求覆盖矩阵

| # | 需求 | 状态 | 落地 |
|---|------|------|------|
| 1 | `supervisor.py` 有限状态机（PLAN/EXECUTE/REVIEW/REVISE/DONE/ABORT） | ✅ | `orchestration/supervisor.py` |
| 2 | Supervisor Decision（确定性规则 + `SupervisorDecision`） | ✅ | `review_execution()` / `SupervisorDecision` |
| 3 | RunState（request_id/plan/execution_results/agent_messages/tool_events/supervisor_trace） | ✅ | `RunState`（supervisor.py） |
| 4 | Specialist Communication（AgentMessage：task 下行 / report 上行） | ✅ | run_stream/_execute 组装 AgentMessage |
| 5 | ToolRegistry Integration（Supervisor/Agent 工具统一经 Registry，adapter 不改 Worker） | ✅ | `build_registry_backed_workers` / `RegistryAnalyzer` / `make_registry_retrieval_adapter` |
| 6 | 测试 `test_supervisor_state_machine.py`（normal success / empty retrieval retry / tool failure abort / max iteration） | ✅ | `tests/agent_team/test_supervisor_state_machine.py`（5 用例） |
| 7 | 报告 | ✅ | 本文件 |

---

## 3. 文件变更

**新增**
- `backend/ai_func/recommendation/orchestration/supervisor.py`（状态机 / 决策 / RunState / 适配器）
- `backend/tests/agent_team/test_supervisor_state_machine.py`

**修改（仅导出，无行为变更）**
- `orchestration/__init__.py`：导出 SupervisorAgent / SupervisorState / SupervisorDecision /
  RunState / review_execution / revise_plan / build_registry_backed_workers / 适配器与常量

**未修改（约束确认）**：`planner.py`、`models.py`(Validator/Task)、`executor.py`、
`workers.py`、`orchestrator.py`（AdaptiveTeamOrchestrator 基线保留）、`agent_cards.py`、
`specialist.py`、`agent_registry.py`、`tools/*`、DB / 前端 / SSE 契约。

---

## 4. 有限状态机设计

### 4.1 状态与转移

```text
         ┌──────────────────────────────────────────────┐
         ▼                                              │(decision=replan)
   PLAN ──► EXECUTE ──► REVIEW ──► (decision=revise) ──► REVISE ─┐(改参重跑)
    │             ▲                                       │
    │             └───────────────────────────────────────┘
    │ decision=continue → DONE
    └ decision=abort（planner/validator 失败、超限未解决） → ABORT
```

- **不允许无限循环**：`max_iterations`（默认 2）限制 EXECUTE 次数；达到后 REVIEW 一律 → ABORT；
- **默认关闭增强**：`enhanced=False` 时 REVIEW 恒为 continue，等价既有单次
  `Plan → Execute → Synthesize` 基线；
- **可配置**：`max_iterations` / `min_candidates` 均为构造参数。

### 4.2 确定性 REVIEW 规则（`review_execution`）

按优先级：
1. 存在 **FAILED 任务**（含工具层失败：RETRIEVAL_ERROR / WORKER_CRASH）→ `revise` / 超限 `abort`
2. **retrieve 成功但候选为空** → `revise` / 超限 `abort`
3. **synthesis 输入不足**（无有效候选/选择）→ `replan` / 超限 `abort`
4. **候选总数低于 min_candidates**（>0 且不足）→ `replan` / 超限 `abort`
5. 其余 → `continue`

`SupervisorDecision` 输出：`{decision: continue|revise|replan|abort, reason, affected_tasks}`。

### 4.3 REVISE 语义（第一阶段确定性）
- 对受影响 `retrieve_images` 任务放大 limit（+10，封顶 40）并标记 objective；
- 修订后整体重跑（DAGExecutor 新 ExecutionContext），保证依赖图完整；
- REPLAN 语义：携带上轮 reason 再次调用 PlannerAgent（受 max_iterations 约束）。

---

## 5. RunState（request 级状态容器）

| 字段 | 内容 |
|------|------|
| request_id | 一次 Supervisor 运行的 ID |
| plan | 当前 TaskPlan |
| execution_results | task_id → WorkerResult（每次 EXECUTE 后同步） |
| agent_messages | List[AgentMessage]（task 下行 / report 上行） |
| tool_events | List[ToolCallEvent]（经 ToolRegistry 调用产生的轨迹） |
| supervisor_trace | List[{state, attempt, decision, reason, affected_tasks}] |
| final_state / last_decision | 终止状态与最近决策 |

`RunState.to_dict()` 提供审计 / 测试 / 后续 Judge 消费的平面视图。

---

## 6. Specialist Communication（AgentMessage 接入）

Supervisor 在执行期按 Executor 事件构建消息，写入 `RunState.agent_messages`：

| 方向 | sender | receiver | message_type | payload |
|------|--------|----------|--------------|---------|
| Supervisor → Specialist | `supervisor` | `retrieval/organizer/vision`（按 task_type 路由） | `task` | task_id / task_type / objective / input / depends_on |
| Specialist → Supervisor | 对应 agent_id（synthesize 用 `synthesizer`） | `supervisor` | `report` | task 终态事件（status/output_summary/error） |

真实样例（normal success 场景，来自 run_state.agent_messages）：
```python
[('supervisor', 'retrieval', 'task'), ('retrieval', 'supervisor', 'report'),
 ('supervisor', 'synthesizer', 'task'), ('synthesizer', 'supervisor', 'report')]
```

---

## 7. ToolRegistry Integration（adapter 方案）

目标：Supervisor / Agent 调用工具统一经过 ToolRegistry，**不改 Worker 核心逻辑**。

实现：
- `build_registry_backed_workers(registry)`：以 ToolRegistry 为唯一数据面构造 Worker——
  - `RetrievalWorker(search_tool=make_registry_retrieval_adapter(registry))`
    → 检索调用经 `registry.call("search_images", ...)`（schema 校验 + ToolCallEvent）；
  - `VisionWorker(analyzer=RegistryAnalyzer(registry))` → 视觉调用经
    `registry.call("analyze_image", ...)`（RegistryAnalyzer 提供 `.client` 通过门禁，只转发
    schema 声明键）；
  - Metadata / Group Worker 保持纯数据操作不变（其工具 `filter_images / group_images` 仍
    注册于 registry，供 OrganizerAgent.call_tool 统一调用）。
- `SpecialistAgent(..., registry=reg)`：Agent 工具调用仍走既有 `call_tool` → registry；
- 因此 DAG 执行期间的工具调用（经 adapter）会自然产生 ToolCallEvent，并进入
  `RunState.tool_events`，供 REVIEW 与后续 LLM Judge / 调优消费。

> 说明：既有 `AdaptiveTeamOrchestrator` 默认路径（Worker 直连注入实现）保持不变；
> ToolRegistry 化是 Supervisor 增强路径的装配方式，可在 `SupervisorAgent(workers=...)`
> 传入，二者并存、行为隔离。

---

## 8. 行为与开关

| 项 | 默认 | 说明 |
|----|------|------|
| `enhanced` | False | 关闭增强 → 等价既有单次基线（空检索不重试等） |
| `max_iterations` | 2 | EXECUTE 次数上限；REVIEW 达上限即 abort |
| `min_candidates` | 1 | 候选数阈值（低于则 replan） |
| 事件契约 | — | `plan.created/plan.failed`、`task.*`、新增 `agent.review`、`assistant_delta`、`selection` |

新增 `agent.review` 事件（decision/reason/affected_tasks/attempt）遵循“未知事件忽略”的
前端兼容策略，可渐进接入。

---

## 9. 测试结果汇总

运行方式（repo 根目录）：`venv\Scripts\python.exe -X utf8 backend/tests/agent_team/<file>.py`

**Phase 3 新增：test_supervisor_state_machine.py（5 用例）**
| 用例 | 断言要点 | 结果 |
|------|----------|------|
| normal success | decisions=[continue]、DONE、单次 EXECUTE、task/report 消息存在 | ✅ PASS |
| empty retrieval retry | decisions=[revise, continue]、两次 EXECUTE、DONE、selection=[201,202] | ✅ PASS |
| tool failure abort | decisions=[revise, abort]、ABORT、ToolCallEvent 失败留痕 | ✅ PASS |
| max iteration limit | decisions=[replan, abort]、ABORT、无无限循环（EXECUTE=2） | ✅ PASS |
| baseline enhanced off | decisions=[continue]、空检索不重试（保留既有行为） | ✅ PASS |

**回归（Phase 1 / Phase 2 / 核心，全部通过）**
- Agent 层：agent_cards 6 + agent_registry 7 + specialist 10 = 23
- tools 层：tool_registry 11 + tool_schema 7 + tool_binding 7 = 25
- 核心：registration 19 / orchestrator 6 / executor 8 / workers 12 / sse 9 / dag 14 /
  plan_validation 20 / synthesizer 12 / planner 11 = 111

**合计：5（新增）+ 159（回归）= 164 passed / 0 failed。**

---

## 10. Runtime Trace 样例（真实组件输出）

normal success（SupervisorAgent 增强模式，工具经 ToolRegistry）：
```text
[Specialist:retrieval] RetrievalAgent executed task t1 (retrieve_images) via RetrievalWorker
[ToolRegistry:OK] agent=system tool=search_images latency_ms=0.0
DECISIONS= ['continue']
FINAL= done
TRACE= [
  {"state": "plan",    "attempt": 0, "decision": null,      "affected_tasks": []},
  {"state": "execute", "attempt": 1, "decision": null,      "affected_tasks": []},
  {"state": "review",  "attempt": 1, "decision": "continue","affected_tasks": [],
   "reason": "执行结果满足要求"},
  {"state": "done",    "attempt": 1, "decision": "continue","affected_tasks": []}]
MSGS= [('supervisor','retrieval','task'), ('retrieval','supervisor','report'),
       ('supervisor','synthesizer','task'), ('synthesizer','supervisor','report')]
TOOLS= [('search_images', True)]
SEL= [201, 202]
```

工具持续失败场景：REVIEW decisions=[revise, abort]，`ToolCallEvent(success=False)` 进入
`RunState.tool_events`；空检索场景 REVIEW decisions=[revise, continue]（第二次执行成功）；
候选低于阈值场景 REVIEW decisions=[replan, abort]。

---

## 11. 架构判断与 Phase 4 准备情况

**本阶段达成**
- 受控 Supervisor 层：有界、确定性优先、默认关闭、可配置；
- 不推翻既有执行链：Planner/Validator/DAGExecutor/AgentRegistry/Specialist/ToolRegistry 全保留；
- AgentMessage（task/report）接入 + RunState 全量运行审计数据；
- ToolRegistry 化装配（adapter）使工具调用统一可观测、可计量、可复盘。

**后续建议（Phase 4 +）**
1. **LLM 复盘（可选）**：REVIEW 确定性触发器之上叠加 LLM Judge 式复盘须在离线验证后以
   flag 开放（当前阶段按需求保持纯确定性）；
2. **Event Bus 落库**：RunState.agent_messages/tool_events/supervisor_trace 落库
   （request_sessions / agent_runs 扩展），支撑离线分析与 Phase 4 LLM-as-a-Judge；
3. **Supervisor 接入服务层**：`SupervisorAgent.run_stream` 事件契约已与
   AdaptiveTeamOrchestrator 兼容，可经 `_RecommendationService` 以 flag 切换（默认仍走基线）；
4. **并行执行 / 预算策略**：REVISE 全量重跑为简单正确实现，后续可收敛为“受影响子图重跑”。

---

## 12. 约束合规

| 约束 | 状态 |
|------|------|
| 不推翻 PlannerAgent | ✅ 未改 planner.py |
| 不推翻 PlanValidator / DAGExecutor | ✅ 未改 models.py / executor.py |
| 不推翻 SpecialistAgent / ToolRegistry | ✅ 未改其核心；新增 adapter/装配 |
| 不引入无限循环 | ✅ max_iterations=2 默认；测试证实无死循环 |
| 最大 review/replan 可配置 | ✅ max_iterations / min_candidates |
| 默认关闭增强 | ✅ enhanced=False 等价基线（单测覆盖） |
| 不引入 LangGraph / MCP SDK / 新 LLM Agent | ✅ 纯自研 pydantic 状态机 + 确定性规则 |


