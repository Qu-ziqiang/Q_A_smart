# PHASE 6 REPORT — LangGraph Supervisor Runtime Integration

> ⚠️ **历史快照（迁移前口径）**：文中 **454 passed** 为 LangGraph 引入时的门禁（448 原有用例 + 6 新用例）；当前门禁为 **463 passed**（2026-09-10 复测）。数据层已由 sqlite-vec 迁至 **PostgreSQL + pgvector**；LangGraph 仅承载 Supervisor 状态机这一层的结论不变。


> 目标：LangGraph 只替换 Supervisor 状态机执行；Planner / DAGExecutor / Specialist / Worker /
> ToolRegistry / MCP / Evaluation 一律不替换。
> 验证：既有 agent_team 全量 pytest **454 passed**（448 原有用例 + 6 个 LangGraph 新用例）。

---

## 1. Task 1 — 依赖评估（结论先行）

| 项 | 结论 |
|---|---|
| SDK | `langgraph 1.2.11`（2026-09 安装于 venv；随附 `langgraph-checkpoint`、`langchain-core` 等） |
| 传输/形态 | 进程内 `StateGraph` + 同步 `.stream()`；`MemorySaver` checkpointer（request 级 thread） |
| 兼容风险 | ① API 演进快（1.x），已锁定本机版本并在报告记录；② 状态对象需保持 request 级隔离（使用每请求独立 thread_id）；③ 循环边需自证有界（max_iterations 路由截断） |
| 迁移方式 | 叠加后端：`SupervisorAgent(backend="langgraph"|"legacy")`；enhanced=True 默认 LangGraph，enhanced=False 原样 legacy；SSE/API 不变 |

备选评估：自研 while-loop（现状，保留为 legacy 后端）；langchain agentic 框架（侵入运行时，违反约束）。LangGraph 的价值：状态图可编译、条件路由显式、checkpoint 免费。

### 1.1 与既有 FSM 的关系（Before/After）
```
Before:                         After:
SupervisorAgent                 SupervisorAgent (API 不变：run_stream/run_state)
  └ Python while-loop FSM         └ LangGraph StateGraph (compiled + MemorySaver)
      PLAN→EXECUTE→REVIEW             START → execute → review ──router──┐
      └REVISE/REPLAN→DONE/ABORT        revise ─────────────────────────┐   │
                                       finalize → END                  ▼   │
                                      （execute → review 循环由 router 截断）
```

---

## 2. 新包：`orchestration/supervisor_graph/`

| 文件 | 职责 |
|---|---|
| `state.py` | `AgentRuntimeState(TypedDict)`：request_id/goal/messages/plan/execution_results/agent_messages/tool_events/events/supervisor_decision/iteration_count/final_state/planner_failed；追加型字段用 `Annotated[List, _merge]` reducer |
| `graph.py` | 四个节点（execute/review/revise/finalize）+ `review_router` + `build_supervisor_graph()` |
| `runtime.py` | `LangGraphSupervisorRuntime`：初始规划 → 编译图（MemorySaver）→ 逐 superstep 取增量 events 并按 SSE 语义顺序 yield |

状态机等价映射（节点内全部复用既有函数，避免双实现漂移）：

| LangGraph 节点 | 复用/行为 |
|---|---|
| execute | `SupervisorAgent._execute`（DAGExecutor + AgentMessage + ToolCallEvent），attempt=iteration+1 |
| review | `review_execution()`（确定性规则）；decision=replan 时在节点内同步重规划（等价 legacy loop-head PLAN 分支：planner→validator→plan.created） |
| revise | `revise_plan()`（retrieve limit +=10/cap 40）→ 重新 execute |
| finalize | continue/abort → `finalize_output()` → assistant_delta + selection |
| review_router | continue→finalize；revise→revise；replan→execute；abort→finalize；`planner_failed`→END |

### 2.1 StateGraph 节点/边图

```
        ┌────────────────────────────────────────────────────────┐
 START →│ execute ──► review ──┬─ review_router ──► finalize ──► END
        │                       │   ├ continue ─────────┘
        │                       │   ├ abort    ─────────┘
        │                       │   ├ revise   ──► revise ─► execute（回环）
        │                       │   └ replan   ──► execute（回环；review 内已重规划）
        └───────────────────────┴ planner_failed → END（不产出 selection）
```

有界性保证：`execute` 每轮迭代 +1；review_execution 在 `attempt >= max_iterations`(默认 2) 时给出
abort；router 对 revise/replan 再做一次 `iteration_count >= max_iterations` 防御截断 → 无无限循环。


## 3. Runtime Logs（evidence，落盘 `backend/tests/agent_team/lg_runtime_traces.log`）

```
SCENARIO normal success:
  nodes visited : execute -> review -> finalize
  final_state   : done
  agent.review attempt=1 decision=continue reason="执行结果满足要求"
  selection ids=[201, 202]

SCENARIO empty -> revise -> retry:
  nodes visited : execute -> review -> revise -> execute -> review -> finalize
  agent.review attempt=1 decision=revise reason="检测到失败/空检索任务: failed=[] empty=['t1']"
  plan.created（objective 含 "Supervisor 重试：扩大候选"）
  agent.review attempt=2 decision=continue
  selection ids=[201, 202]   tool_events: search_images ok×2

SCENARIO tool failure -> revise -> abort:
  nodes visited : execute -> review -> revise -> execute -> review -> finalize
  agent.review attempt=2 decision=abort reason="达到最大执行次数(2)…"
  selection ids=[]            tool_events: search_images ok=false ×2（ToolCallEvent 失败留痕）

SCENARIO replan -> abort（候选 1 < 阈值 3）:
  nodes visited : execute -> review -> execute -> review -> finalize
  agent.review attempt=1 decision=replan；plan.created（重新规划）
  agent.review attempt=2 decision=abort reason="候选数 1 低于阈值 3 且达到上限"
```

## 4. Performance Comparison（本机，in-process Fake 全链路单请求）

| 后端 | 空检索→revise→retry 场景（5 次均值） |
|---|---|
| legacy（enhanced=True，原 while-loop） | 0.9 ms / run |
| LangGraph（StateGraph + MemorySaver） | 11.5 ms / run |
| 增量开销 | ≈10.6 ms / run（checkpoint + superstep 分帧） |

结论：纯 Fake 路径下 LangGraph 每请求增加约 10 ms；相对真实链路（sqlite-vec 检索 / 视觉 / MCP
stdio ~1.5 s）占比可忽略，换来显式状态图、条件路由与可恢复 checkpoint。

## 5. Tests（Task 7，全部本机通过）

新增 `backend/tests/agent_team/test_langgraph_supervisor.py`：6 passed
1. normal success（一次 execute → continue → done）
2. empty retrieval → revise → retry → done（execute attempts [1,2]）
3. tool failure → revise → abort（max_iterations 截断，ToolCallEvent 失败留痕）
4. baseline compatibility（enhanced=False 单次 legacy；langgraph 与 legacy(enhanced) 输出等价）
5. state persistence（MemorySaver checkpoint 保留 final_state/iteration_count/plan/tool_events/events）
6. replan 阶段 planner 失败 → plan.failed → END 直接终止（不产出 selection）

兼容回归（LangGraph 已作为 enhanced 默认后端跑通既有断言）：
- `test_supervisor_state_machine.py` 5 passed
- `test_supervisor_production_integration.py` 5 passed
- `test_evaluation_foundation.py` 6 passed（Evaluation 层复用 trace 正常）
- `backend/tests/agent_team/mcp/test_mcp_runtime.py` 3 passed（MCP 层经 LangGraph Supervisor 全链）
- 全量 `pytest backend/tests/agent_team -q` → **454 passed**

最终运行时（验证通过）：Planner → Validator → DAGExecutor → **LangGraph Supervisor** →
Specialist Agents → ToolRegistry（本地/MCP 远端）→（Evaluation 层离线复用同一 Runtime）。

## 6. Limitations

1. **默认后端变更**：enhanced=True 默认走 LangGraph（等价验证充分）；如需逐字保留原 FSM，
   显式 `backend="legacy"`。
2. **初始规划在图外**：plan.created 由 runtime 预生成后注入状态图；replan 仍可在 review 节点内
   完整重规划——语义等价但“初始 PLAN”不是独立节点。
3. **每请求独立 checkpoint thread**：MemorySaver 提供 request 级持久化；跨请求复用/中断恢复策略
   未实现（后续可做 thread 复用与 SQLite saver）。
4. **增量开销**：约 10 ms/请求（Fake），生产化需重新测量真实链路占比。
5. **LangGraph 版本锁定**：1.x API 演进快；报告记录 `langgraph 1.2.11`，升级需回归本套件。
6. **并行节点未使用**：四个节点保持串行（等价 legacy 语义）；LangGraph 的并行/条件并行能力未启用。

## 7. 交付物对照

| Task | 内容 | 状态 |
|---|---|---|
| 1 | 依赖评估（版本/风险/迁移） | ✅ §1 |
| 2 | `supervisor_graph/` AgentRuntimeState | ✅ state.py |
| 3 | execute/review/revise/finalize 节点 | ✅ graph.py |
| 4 | review_router + max_iterations 截断 | ✅ |
| 5 | SupervisorAgent 最小集成（backend 参数，API 不变） | ✅ |
| 6 | enhanced=False legacy / enhanced=True LangGraph；SSE 不变 | ✅ |
| 7 | test_langgraph_supervisor.py（6 项） | ✅ 6 passed |
| 8 | 本报告 | ✅ |

变更文件：新增 `orchestration/supervisor_graph/{__init__,state,graph,runtime}.py`、
`backend/tests/agent_team/test_langgraph_supervisor.py`、`lg_runtime_traces.log`；
修改 `orchestration/supervisor.py`（+`backend` 参数、run_stream 分发分支，其余逻辑未动）。
依赖：`langgraph==1.2.11`。
