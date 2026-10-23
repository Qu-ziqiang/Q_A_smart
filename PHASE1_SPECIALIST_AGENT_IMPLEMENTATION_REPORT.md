# PHASE1_SPECIALIST_AGENT_IMPLEMENTATION_REPORT

> 项目：SmartImager — 多模态智能图库与对话推荐系统
> 阶段：Phase 1 Upgrade —— Worker → Specialist Agent 角色化重构
> 日期：2026-09-07
> 性质：**只做“角色化封装”，不引入 Supervisor Loop / LLM Agent Loop，不改变执行逻辑**
> 代码基准：`Smartlmager-sqlite-vec/backend/ai_func/recommendation/orchestration/`

---

## 1. 概述

本阶段把既有执行链中的 **Worker 层**升级为 Multi-Agent 架构中的 **Specialist Agent**
（RetrievalAgent / OrganizerAgent / VisionAgent），使 SmartImager 从

```text
Planner → Typed Task DAG → WorkerRegistry → Worker.run() → Synthesizer
```

演进为 Multi-Agent Ready 形态：

```text
Planner → Typed Task DAG → AgentRegistry → SpecialistAgent.execute() → Worker.run() → Synthesizer
```

核心公式：**Specialist Agent = Agent Card + Adapter + Existing Worker**。
Worker 代码零改动、DAGExecutor 核心算法零改动、Planner / Task Model / Validator /
SSE 契约全部保持原样。

---

## 2. 需求覆盖矩阵

| Requirement | 内容 | 状态 | 落地位置 |
|-------------|------|------|----------|
| 1 | Agent Card 机制（agent_cards.py） | ✅ | `orchestration/agent_cards.py` |
| 2 | SpecialistAgent Adapter（specialist.py） | ✅ | `orchestration/specialist.py` |
| 3 | Agent Registry + 三个默认 Agent | ✅ | `orchestration/agent_registry.py` |
| 4 | Agent Message Protocol（messages.py） | ✅ | `orchestration/messages.py` |
| 5 | 接入执行链（orchestrator.py） | ✅ | `orchestrator._default_executor()` + `build_dispatched_workers()` |
| 6 | 新增测试 | ✅ | `test_agent_cards.py` / `test_agent_registry.py` / `test_specialist_agent.py` |
| 7 | Runtime Verification | ✅ | `demo_specialist_agents.py`（离线） |
| 8 | 实施报告 | ✅ | 本文件 |

---

## 3. 修改 / 新增文件列表

**新增模块（4 个）**
- `backend/ai_func/recommendation/orchestration/agent_cards.py`
- `backend/ai_func/recommendation/orchestration/specialist.py`
- `backend/ai_func/recommendation/orchestration/agent_registry.py`
- `backend/ai_func/recommendation/orchestration/messages.py`

**新增测试 / Demo（4 个）**
- `backend/tests/agent_team/test_agent_cards.py`
- `backend/tests/agent_team/test_agent_registry.py`
- `backend/tests/agent_team/test_specialist_agent.py`
- `backend/tests/agent_team/demo_specialist_agents.py`

**修改文件（2 个，行为不变）**
- `orchestration/orchestrator.py`：默认 Executor 改经 AgentRegistry 分发（新增可选构造参数 `agent_registry`，向后兼容）；
- `orchestration/__init__.py`：导出新 API 并更新包文档。

**未改动（约束确认）**：`models.py`、`planner.py`、`executor.py`、`workers.py`、
`gate.py`、`synthesizer.py`、`vision.py`、`memory.py`、Legacy `agent.py`、所有 repository/DB、
前端与 SSE 契约。

---

## 4. 新增模块说明

### 4.1 `agent_cards.py` — Agent Card（角色元数据，无业务逻辑）
- `AgentCard(BaseModel)`：`agent_id / name / description / supported_tasks(List[TaskType]) /
  available_tools / input_schema / output_schema / cost_level / enabled`（`extra="forbid"`）。
- `build_default_agent_cards(vision_ok=None)`：生成三角色卡片
  | agent_id | name | supported_tasks | available_tools | cost_level |
  |---|---|---|---|---|
  | retrieval | RetrievalAgent | retrieve_images | search_images, rerank | cheap |
  | organizer | OrganizerAgent | filter_metadata, group_images | filter_images, group_images | cheap |
  | vision | VisionAgent | analyze_visual | analyze_image | expensive |
- `VisionAgent.enabled` 跟随视觉能力门禁（`vision_ok` 显式传入或读 `vision_available()`），
  与 `default_workers()` / `registered_task_types()` 的既有口径一致。

### 4.2 `specialist.py` — SpecialistAgent（包装层）
- `SpecialistAgent(card: AgentCard, worker)`：`worker` 支持单 Worker 或 `Dict[TaskType, Worker]`；
- `execute(task, context)` → 按 `task.task_type` 解析 Worker → `Worker.run(task, context)`
  （Organizer 双任务由内部索引分发）；执行时打印角色日志：
  `[Specialist:{agent_id}] {name} executed task {task_id} ({type}) via {WorkerClass}`；
- `run(task, context)` 作为 Worker Protocol 兼容别名（DAGExecutor 以 `run()` 分派时直接可用）；
- 构造校验：card 必须是 AgentCard；worker 不能为 None；mapping 必须覆盖 `supported_tasks`；
  不支持的任务类型在 execute 时报 `TypeError`（由 Executor 收敛为 `WORKER_CRASH`）。

### 4.3 `agent_registry.py` — AgentRegistry + 分发辅助
- `register_agent(agent)` / `get_agent(agent_id)` / `list_agents()` / `get_agent_for_task(task_type)`；
- disabled Agent 可查询/列举，但不进入任务分发索引；
- `build_default_agent_registry(workers=None)`：按 worker 表实际存在的任务注册三角色
  （视觉关闭时自动不注册 VisionAgent —— 与能力门禁一致）；
- `build_dispatched_workers(base_workers, registry, synthesize_worker=None)`：执行链分发表——
  每个 task_type 优先 `registry.get_agent_for_task()`，未注册则回退原 Worker；
  SYNTHESIZE 允许外部覆盖（Orchestrator 注入 LLM SynthesizerAgent）。

### 4.4 `messages.py` — AgentMessage（通信信封雏形）
- `AgentMessage(message_id, sender, receiver, message_type, payload: dict)`；
- 本阶段不实现真实消息路由，仅为 Phase 3 Supervisor ↔ Specialist 通信冻结契约；
- 提供 `TYPE_TASK / TYPE_REPORT / TYPE_DECISION / TYPE_EVENT` 约定词。

---

## 5. 架构变化

### 5.1 分发包变化（Executor 内部不变）

```text
修改前：  Task ──► default_workers()[task.task_type] ──► worker.run(task, context)
修改后：  Task ──► AgentRegistry.get_agent_for_task(task.task_type)
                ──► SpecialistAgent.execute(task, context)
                ──► Worker.run(task, context)
```

### 5.2 职责边界

| 层 | 角色 | 说明 |
|----|------|------|
| Planner / Validator | 决策（规划） | 不变 |
| DAGExecutor | 调度（执行顺序/依赖/失败传播/预算） | 不变 |
| AgentRegistry | 角色注册与任务路由 | 新增 |
| SpecialistAgent | 角色化执行包装 | 新增 |
| Worker | 领域能力实现 | 不变 |

---

## 6. 测试结果汇总

运行方式（repo 根目录）：`venv\Scripts\python.exe -X utf8 backend/tests/agent_team/<file>.py`

**新增测试（Phase 1）**
| 测试文件 | 用例数 | 结果 |
|----------|-------:|------|
| test_agent_cards.py | 6 | 6 passed |
| test_agent_registry.py | 7 | 7 passed |
| test_specialist_agent.py | 10 | 10 passed |

**既有回归（无破坏）**
| 测试文件 | 用例数 | 结果 |
|----------|-------:|------|
| test_registration.py | 19 | 19 passed |
| test_orchestrator.py | 6 | 6 passed |
| test_executor.py | 8 | 8 passed |
| test_workers.py | 12 | 12 passed |
| test_dag_propagation.py | 14 | 14 passed |
| test_sse_contract.py | 9 | 9 passed |
| test_plan_validation.py | 20 | 20 passed |
| test_synthesizer.py | 12 | 12 passed |
| test_planner.py | 11 | 11 passed |
| test_integration_full_chain.py | 15 | 15 passed |
| test_gate.py | 13 | 13 passed |
| test_intent_rewrite.py | 23 | 23 passed |
| test_vision.py | 49 | 49 passed |

小计：新增 23 + 回归 211 = **234 passed / 0 failed**。

---

## 7. Runtime Verification 证据（Requirement 7）

运行：`venv\Scripts\python.exe -X utf8 backend/tests/agent_team/demo_specialist_agents.py`（离线，零 API / DB）

Demo 覆盖一次完整流程：
`User Query → ComplexityGate → PlannerAgent(Fake LLM) → PlanValidator(vision on)
 → DAGExecutor → AgentRegistry → SpecialistAgent.execute → Worker.run
 → DeterministicSynthesisWorker`

关键输出摘录（执行日志已体现三个 Specialist 角色）：

```text
== 1) Agent Card / AgentRegistry 角色视图（含视觉） ==
  agent_id=retrieval  name=RetrievalAgent   cost=cheap    tasks=['retrieve_images']
  agent_id=organizer  name=OrganizerAgent   cost=cheap    tasks=['filter_metadata', 'group_images']
  agent_id=vision     name=VisionAgent      cost=expensive tasks=['analyze_visual']

  Task → Specialist 映射：
    retrieve_images  → RetrievalAgent
    filter_metadata  → OrganizerAgent
    group_images     → OrganizerAgent
    analyze_visual   → VisionAgent

== 2) 完整流程：Planner → Validator → DAGExecutor → SpecialistAgent → Worker → Synthesis ==
  [Gate] route=team
  [Planner] plan_id=... tasks=[('t1','retrieve_images'),('t2','filter_metadata'),
             ('t3','group_images'),('t4','analyze_visual'),('t5','synthesize')]
  [Validator] validation passed
  [Executor] dispatch:
    retrieve_images → RetrievalAgent        filter_metadata → OrganizerAgent
    group_images    → OrganizerAgent        analyze_visual  → VisionAgent
    synthesize      → DeterministicSynthesisWorker

  [Specialist:retrieval] RetrievalAgent executed task t1 (retrieve_images) via _FW
  [Specialist:organizer] OrganizerAgent executed task t2 (filter_metadata) via _FW
  [Specialist:vision]    VisionAgent executed task t4 (analyze_visual) via _FW
  [Specialist:organizer] OrganizerAgent executed task t3 (group_images) via _FW
  task t1..t5: succeeded
  [Result] synthesis ids=[11, 12]
  [LogCheck] RetrievalAgent executed = True
  [LogCheck] OrganizerAgent executed = True
  [LogCheck] VisionAgent executed = True

== 3) Orchestrator 默认 Executor 已接入 AgentRegistry（静态核对） ==
  AgentRegistry agents: RetrievalAgent, OrganizerAgent          # 视觉未配置时的实际状态
  executor.workers[retrieve_images] → RetrievalAgent
  executor.workers[filter_metadata] → OrganizerAgent
  executor.workers[group_images]    → OrganizerAgent
  executor.workers[synthesize]      → SynthesizerAgent          # LLM 合成终点不变
```

说明：demo 第 3 节在本机（VISION_MODEL 未配置）显示 2 个 Agent；配置视觉模型后
VisionAgent 会自动随 `default_workers()`/`registered_task_types()` 一起出现
（单元测试 `test_default_registry_built_from_workers_including_vision` 已验证该路径）。

---

## 8. 约束合规性检查

| 约束 | 状态 |
|------|------|
| 不引入 LangGraph | ✅ 仅自研轻量模型（pydantic BaseModel + 简单 registry） |
| 不修改 Planner Prompt | ✅ `planner.py` / `prompts.py` 未动 |
| 不修改 DAG Executor 核心算法 | ✅ `executor.py` 未动，执行上限/依赖传播/失败隔离原样 |
| 不修改数据库 | ✅ 无 DDL / repository 改动 |
| 不修改前端 | ✅ 未触及 `frontend/` |
| 不引入 MCP SDK | ✅ 仅预留 MCP 风格字段，未加依赖 |
| 不创建新的 LLM Agent | ✅ SpecialistAgent 无 LLM 调用、无 Agent Loop |
| Worker 代码保持不变 | ✅ `workers.py` 未动；既有 Worker 测试 12/12 通过 |

---

## 9. 当前 Multi-Agent 能力分析

**本阶段已具备（Multi-Agent Ready）**
- Agent 身份 / 能力 / 工具 / IO 协议 / 成本 / 启用状态的统一描述（Agent Card）；
- Worker → Specialist 的角色化封装与执行入口（execute / run 协议兼容）；
- Specialist 注册、按任务路由、禁用不分发（AgentRegistry）；
- 执行链“Task → Agent → Worker”的可观测日志（每个角色 executed 行）；
- Supervisor ↔ Specialist 通信信封契约（AgentMessage，本阶段不投递）；
- 默认三角色（Retrieval / Organizer / Vision）与能力门禁一致。

**尚未具备（后续阶段）**
- Supervisor Agent（Phase 3）：规划后的复盘 / revise / replan / abort 决策回路；
- 真实消息路由与 Event Bus（Phase 3，基于 AgentMessage）；
- Tool Registry / MCP（Phase 2）：统一工具 schema 与多端消费；
- 记忆增强 / LLM-as-a-Judge（Phase 4）：质量与达成度评估。

---

## 10. 后续阶段建议

1. **Phase 2 — Tool Registry / MCP**：把 `search_images / filter_images / group_images /
   analyze_image` 收编为统一 ToolDef（单点 schema + 校验 + 计量），并让 Agent Card 的
   `available_tools` 与 registry 同源；条件成熟时再评估 MCP SDK 外化（见设计文档 §5.1）。
2. **Phase 3 — Supervisor Agent**：在 `supervisor.py` 内实现“plan → execute → review →
   revise/replan/abort”五态小状态机，复用 PlannerAgent / PlanValidator / DAGExecutor /
   AgentRegistry；默认关闭（flag 控制），并沿用 `MAX_EXECUTED_TASKS` 等既有预算上限。
3. **Phase 4 — Memory + LLM-as-a-Judge**：episodic memory 保留，新增 user-preference 层；
   Judge 只作离线评估第二层（诚实性 / 一致性 / 过程效率），不阻塞在线路径。
4. **能力扩展提示**：新增业务能力时，只需“新增 TaskType + 对应 Worker + 注册 Agent Card”，
   即可自动进入 Planner enabled 列表与 Specialist 分发——但注意 TaskType 是封闭枚举，
   扩展需走注册评审（见设计文档 §5.5 / Non-Goals）。

