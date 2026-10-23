# PHASE3_RUNTIME_CLOSURE_REPORT

> 项目：SmartImager — 多模态智能图库与对话推荐系统
> 阶段：Phase 3.2 — Runtime Contract Cleanup（收尾）
> 日期：2026-09-07
> 范围：不改架构、不加新 Agent、不改执行逻辑；仅修复审计发现的契约不一致并补齐可评估设施

---

## 0. 收尾内容清单

| 任务 | 内容 | 状态 |
|------|------|------|
| Task 1 | 修复 rerank Tool Schema 契约（JSON ↔ ImageCandidate，adapter 归一化） | ✅ |
| Task 2 | ToolCallEvent 持久化准备（EventCollector 抽象 + InMemory 默认 + Supervisor 接线） | ✅ |
| Task 3 | 全量运行验证 + Closure Report | ✅ 276 passed / 0 failed |

---

## 1. Architecture

```text
User / Service（ENABLE_SUPERVISOR_RUNTIME feature flag）
   │  （默认 False → AdaptiveTeamOrchestrator 基线；True → Supervisor 控制流）
   ▼
PlannerAgent
   ▼
PlanValidator
   ▼
SupervisorAgent（PLAN → EXECUTE → REVIEW → REVISE/REPLAN → DONE/ABORT）
   ▼
DAG Executor（Executor/Event）
   ▼
Specialist Agents（Retrieval / Organizer / Vision）
   ▼
Tool Registry（ToolDefinition + adapters，统一校验/计量/ToolCallEvent）
   ▼
Workers（既有实现，经注入位/适配器）
```

契约收口点：
- **ToolRegistry 不再认识领域对象**：外部只收 JSON（如 rerank.candidates=object[]），
  `tools/adapters.py::normalize_candidates` 归一化为 `ImageCandidate[]` 后再调用既有
  `rerank_candidates()`；
- **事件可评估**：Supervisor RunState 把 agent_messages / tool_events 双写到
  `EventCollector`（默认 InMemory），`snapshot()` 已按 Phase 4 Judge 需求暴露
  tool 名 / 成功率 / latency / 失败明细 / sender-receiver 计数。

---

## 2. Runtime Evidence（三个场景，离线真实组件复现）

### Case A — 普通检索：retrieve → synthesize
```text
[states]    plan → execute → review → done
[decisions] ['continue']
[final]     done
[plan]      [t1 retrieve_images, t2 synthesize]
[selection] [201, 202]
```
路径：Planner(fake) → Validator → Supervisor → Executor → RetrievalAgent →
ToolRegistry(search_images OK) → Worker → Review(continue) → Final。

### Case B — 复杂任务：retrieve → group → synthesize
```text
[states]      plan → execute → review → done
[plan]        [t1 retrieve_images, t2 group_images, t3 synthesize]
[agent_msgs]  supervisor→retrieval task / retrieval→supervisor report
              supervisor→organizer  task / organizer→supervisor  report
              supervisor→synthesizer task / synthesizer→supervisor report
[tool_events] [("search_images", true)]
[selection]   [301, 302, 303]
```
路径：RetrievalAgent + OrganizerAgent（group_images，按地点分组）→ 确定性合成。

### Case C — 失败恢复：retrieve(empty) → review → revise → retry → done
```text
[states]       plan → execute → review → revise → execute → review → done
[decisions]    ['revise', 'continue']
[search_calls] 2
[selection]    [301, 302, 303]
```
路径：首次检索为空 → REVIEW revise（扩大 limit 重跑）→ 重试成功 → continue → DONE。

> 补充（工具层失败，有界）：`tool failure → ['revise','abort']`，ToolCallEvent(success=False)
> 双留痕，无无限循环。基线一致性：SupervisorAgent(enhanced=False) 与 Phase2
> AdaptiveTeamOrchestrator 同数据 selection 完全一致，且基线事件流无 agent.review。

---

## 3. Task 1 证据：rerank JSON 契约

```text
ToolRegistry.call("rerank", {"query":"风景",
    "candidates":[{"id":101,"score":0.85,"metadata":{"location":"北京"}},
                  {"id":102,"score":0.76,"metadata":{"location":"上海"}}]})
=> [ImageCandidate(id=101), ImageCandidate(id=102)]   # 同池重排，无增删
```
- Schema 层 `candidates: object[]` 与 Runtime 输入一致（JSON）；
- 领域对象隔离在 `tools/adapters.py`（ImageCandidate 仅供内部兼容直传）；
- `test_tool_adapters.py` 5 用例覆盖（payload dict / list / 领域对象直传 / 非法输入 / registry 默认实现）。

---

## 4. Task 2 证据：EventCollector 抽象 + Supervisor 接线

```text
InMemoryEventCollector.record_tool_event(e) / record_agent_message(m)
snapshot() => {agent_message_count, tool_event_count, tool_success/failure_count,
               tool_names, tool_latency_ms, tool_failures[], agent_senders[]...}

SupervisorAgent(..., collector=col) 运行后：
  len(col.tool_events) == len(run_state.tool_events) == 1
  len(col.agent_messages) == len(run_state.agent_messages) >= 2   # 双写一致
```
- `orchestration/observability/event_collector.py`：`EventCollector`（抽象，无操作默认）
  + `InMemoryEventCollector`（进程内，`snapshot()` 面向 Judge）；
- Supervisor RunState 增加 `add_agent_message / add_tool_event`，内部列表与注入 Collector 双写；
- 未来 `SQLiteEventCollector → Judge Pipeline` 只需实现同一抽象，不改 Supervisor。

---

## 5. Regression（保持 0 failed）

复跑集（新增后）：
| 分组 | 套件与用例数 | 结果 |
|------|--------------|------|
| tools（4） | registry 11 / schema 7 / binding 7 / **adapters 5** | 30 passed |
| agent 层（3） | cards 6 / registry 7 / specialist 10 | 23 passed |
| supervisor / 接线（3） | state machine 5 / production integration 5 / **event collector 2** | 12 passed |
| 核心（13） | registration 19 / orchestrator 6 / executor 8 / workers 12 / sse 9 / dag 14 / plan_validation 20 / synthesizer 12 / planner 11 / gate 13 / intent 23 / vision 49 / integration_full_chain 15 | 211 passed |

**合计：276 passed / 0 failed。**

---

## 6. 本次收尾变更文件

**新增**
- `orchestration/tools/adapters.py`（normalize_candidates）
- `orchestration/observability/__init__.py`、`event_collector.py`
- `tests/agent_team/tools/test_tool_adapters.py`、`tests/agent_team/test_event_collector.py`

**修改（行为兼容）**
- `orchestration/tools/builtin_tools.py`：`_impl_rerank` 走 `normalize_candidates`
- `orchestration/supervisor.py`：SupervisorAgent 支持 `collector=` 注入；RunState 双写方法
- `orchestration/tools/__init__.py`、`orchestration/__init__.py`：导出 adapters / observability

**未修改**：Planner / Validator / DAGExecutor / Worker 核心 / Agent 核心 / DB / 前端 / SSE 契约。

---

## 7. 收尾判定

1. rerank 契约不一致已消除：Registry 只收 JSON，领域对象隔离在 adapter，接 MCP Client /
   LLM Tool Calling 不再有“schema 说 object、实现收 ImageCandidate”的断层；
2. 运行事件已具备可评估出口：Supervisor → RunState → Collector（InMemory），
   snapshot() 字段对齐 Phase 4 Judge 关注面（工具成功率/latency/失败、消息参与方）；
3. 无架构/执行逻辑变化，feature flag 默认关闭，Phase1/2 基线行为保持；
4. 全量回归 **276 passed / 0 failed**，Agent Runtime 达到可评估状态。
