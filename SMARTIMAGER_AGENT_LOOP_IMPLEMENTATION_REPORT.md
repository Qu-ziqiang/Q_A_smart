# SmartImager Agent 闭环升级实现报告

本次落地目标不是增加 Agent 数量，也不是让 Specialist 互聊或自带 LLM。  
真正完成的是：

> **角色分工 + 各自执行 + 结构化证据回传 + 集中式有界重规划**

即：

```text
Planner
  ↓ 决定做什么
Specialists
  ↓ 分别执行并交回专业证据
Supervisor
  ↓ 判断发生了什么，是否值得重新规划
Planner
  ↓ 根据 Observation 改变下一步策略
```

---

## 1. Modified Files

| 文件 | 修改内容 |
| --- | --- |
| `backend/ai_func/recommendation/orchestration/observation.py` | **新建**。`RoundObservation`、Retrieval/Organizer/Vision/Failure 切片、`build_observation()`、`classify_trigger()`、`render_for_planner()`、`assert_replan_changed()`。`LOW_RELEVANCE_THRESHOLD` 可配，默认 0.4，代码标明 exploratory。 |
| `backend/ai_func/recommendation/orchestration/supervisor.py` | REVIEW 改为 Observation 分类触发。空结果/低相关走 **replan**；执行失败走 **revise（同 query）**。`run_stream` 改为 `for _round in range(max_iterations)`，禁止 `while True`。新增 `invoke_planner` / `accept_plan`；第二轮 Observation 注入 Planner，Validator 全量再入；策略未变视为校验失败。SSE 增加 `agent.observe`、`REPLAN_REASON`、`REPLAN_ACTION`、`iteration`。 |
| `backend/ai_func/recommendation/orchestration/planner.py` | `create_plan(..., *, observation=None)`。`observation is None` 时第一轮 prompt 不变；第二轮把结构化 Observation 写入 user prompt。Planner 仍不执行任何业务工具。 |
| `backend/ai_func/recommendation/orchestration/supervisor_graph/graph.py` | `review_node` 构造 Observation 并入 review（不新增业务图节点）。`replan` 走 `Observation → Planner → Validator`，不再把 reason 字符串当唯一输入。 |
| `backend/ai_func/recommendation/orchestration/supervisor_graph/runtime.py` | 初始规划改用 `invoke_planner`，计入 Planner 调用次数。 |
| `backend/ai_func/recommendation/orchestration/__init__.py` | 导出 Observation 类型与 `build_observation`。 |
| `backend/ai_func/recommendation/evaluation/judge.py` | MockJudge 将 `replan` 与 `revise` 一并视为恢复信号（不改评分数据）。 |
| `backend/tests/agent_team/test_agent_loop.py` | **新建**。覆盖空结果 / 低相关 / 执行失败 / 非法 Replan / 两轮上限 / 正常成功 / Legacy。 |
| `backend/tests/agent_team/loop_support.py` | **新建**。测试用第二轮改 query 的 Fake Planner。 |
| `backend/tests/agent_team/run_agent_loop_benchmark.py` | **新建**。Before/After 对照 + Runtime Trace 样例。 |
| `backend/tests/agent_team/test_supervisor_state_machine.py` | 空检索期望由 `revise` 改为 `replan`。 |
| `backend/tests/agent_team/test_langgraph_supervisor.py` | 同上；LangGraph 与 legacy 后端仍等价。 |
| `backend/tests/agent_team/test_supervisor_production_integration.py` | 生产接线空检索恢复改为 replan。 |
| `backend/tests/agent_team/test_evaluation_foundation.py` | recovery case 期望决策改为 `replan, continue`。 |
| `backend/tests/agent_team/test_judge_contract.py` | recovery 判定兼容 replan。 |

未改 Worker / Specialist / DAGExecutor 核心架构。Specialist 仍只执行并回传 `WorkerResult`。`VISION_MODEL="null"` 关闭行为保持不变。Legacy / ComplexityGate 未取消。

---

## 2. Architecture

```text
Plan
 ↓  Validator
Execute          （Specialists → WorkerResult → RunState）
 ↓
Observe          （Supervisor 从 WorkerResult 构造 RoundObservation）
 ↓
Review           （分类 trigger；最多 2 轮）
 ├── continue → Done
 ├── revise   → 同 query 重试 Execute     （仅 EXECUTION_FAILURE）
 ├── replan   → Planner(observation)
 │                 ↓
 │               New TaskPlan
 │                 ↓
 │               PlanValidator          （全量再入；失败不部分执行）
 │                 ↓
 │               Execute
 └── abort    → Done/Abort
```

通信模型保持：

```text
Specialist → WorkerResult → RunState / ExecutionContext → Supervisor Observation → Planner
```

没有实现 `RetrievalAgent ↔ OrganizerAgent ↔ VisionAgent` 自主对话。

LangGraph 仍只承接 Supervisor 状态机：`execute → review → (revise|execute|finalize)`。  
Planner / DAGExecutor / Specialist **不是** LangGraph 业务节点。Observation 构造并入 `review`，未把三者改造成图节点。

---

## 3. Role Responsibilities

| Role | Decision | Execution | Observation |
| --- | --- | --- | --- |
| Planner | ✅ 第 1/2 次出 Typed DAG；可改 query / filter / group / 是否加 `analyze_visual` | ❌ 不执行检索、视觉、业务工具 | consume（仅第二轮） |
| Supervisor | ✅ 是否 Replan / Revise / Abort；控制迭代 | ❌ 不改 query、不重写 filter、不调用视觉模型、不代替 Judge | create |
| Validator | ❌ 不判断图片好坏、不改策略 | ❌ | validate（第 1、2 轮全量） |
| RetrievalAgent | ❌ 不改 query、不触发下一步 | ✅ | produce（候选数 / score / tags / ids） |
| OrganizerAgent | ❌ 不放宽 filter、不重规划 | ✅ | produce（input/output count / filter_empty / group_status） |
| VisionAgent | ❌ 不决定再检索 | ✅ | produce（inspected/passed / confidence / tags） |

---

## 4. New Observation Schema

`RoundObservation`（可序列化，不是日志字符串）：

```text
request_id
iteration
original_goal
trigger                  NONE | EMPTY_RETRIEVAL | LOW_RELEVANCE | EXECUTION_FAILURE | SYNTHESIS_FAILURE
suggested_action         NONE | RETRY_REWRITE | ADD_VISION | RETRY_SAME | REPLAN_STRUCT | ABORT

retrieval:
  status, empty, candidate_count, top1_score, topk_mean_score,
  tag_overlap, tag_hit, query, candidate_ids

organizer:
  status, input_count, output_count, filter_empty, group_status

vision:
  status, inspected_count, passed_count, confidence_summary, new_tags

failure:
  type, code, message

failed_task_ids
empty_retrieve_task_ids
vision_available
plan_had_vision
synthesis_has_selection
plan_task_types
current_plan_summary
top_titles
```

`render_for_planner()` 把它压成 bounded 文本，供第二轮 Planner 直接消费。  
空结果 / 低相关会明确要求：**第二轮 `retrieve.query` 必须与上一轮不同，禁止只增加 limit。**

---

## 5. Trigger Logic

Supervisor **不**用单一 `score < 0.4` 做决策。触发拆开：

### A. `EMPTY_RETRIEVAL`

```text
retrieve 成功且 candidate_count == 0
```

→ `replan` / `RETRY_REWRITE`

### B. `LOW_RELEVANCE`

必须同时满足：

```text
candidate_count > 0
AND top1_score < LOW_RELEVANCE_THRESHOLD   # 默认 0.4，exploratory
AND tag_hit == False                       # query 与 tags/title 关键词重叠 < 0.15
```

→ `replan` / `RETRY_REWRITE`  
若 `vision_available` 且本轮计划没有 `analyze_visual` → 建议 `ADD_VISION`。

### C. `EXECUTION_FAILURE`

工具异常 / Worker 失败 / 运行时错误。

→ `revise` / `RETRY_SAME`（允许同 query 重试）  
**不会**误判为低相关。

### D. `SYNTHESIS_FAILURE`

合成无 `image_ids`，或候选数低于 `min_candidates`。

→ `replan` / `REPLAN_STRUCT`

优先级：执行失败 > 空结果 > 低相关 > 合成失败 > 成功。

---

## 6. Replan Action Logic

| Action | 何时 | 第二轮必须发生的策略变化 |
| --- | --- | --- |
| `RETRY_REWRITE` | 空结果 / 低相关（无 Vision） | `retrieve.query` 必须改变；只加 `limit` 会被 `assert_replan_changed` 拒绝 |
| `ADD_VISION` | 低相关且 Vision 可用、首轮无视觉任务 | 新计划必须包含 `analyze_visual` |
| `RETRY_SAME` | 执行失败 | 允许同 query；走 `revise_plan`（可扩大 limit） |
| `REPLAN_STRUCT` | 合成失败 / 候选不足 | 回到 Planner 重新出 DAG |
| `ABORT` | 已达 2 轮仍失败 | 不再调用 Planner |

第一轮成功：**不会**第二次调用 Planner。

---

## 7. Validator Re-entry

最终执行链：

```text
Supervisor → Observation → Planner → New TaskPlan → PlanValidator → DAGExecutor → Specialists
```

严禁 Supervisor 直接改 Task 后执行。  
第二轮与第一轮使用**同一套** Validator 约束（task type / schema / whitelist / limit / vision limit / task count / dependency / cycle / depth / orphan / producer ancestry / retrieve requirement / 唯一 synthesize 终点）。

非法 Replan（例如未知依赖）→ `plan.failed` → **不进入第二次 Execute**。  
`RETRY_REWRITE` 但 query 未变 → `REPLAN_QUERY_UNCHANGED`，同样不部分执行。

---

## 8. Iteration Control

```text
for _round in range(max_iterations):   # 默认 2，不是 while True
    plan / replan / revise
    execute
    observe + review
    continue → DONE
    abort    → ABORT
    revise/replan → 下一轮
```

达到 2 轮仍失败 → ABORT，不会第三次 Planner。

---

## 9. LangGraph Changes

图拓扑保持：

```text
START → execute → review
                    ├── continue/abort → finalize → END
                    ├── revise → execute
                    └── replan → execute
```

变化仅在 `review_node` 内部：

- 调用 `review_execution(..., goal, request_id)` 生成 Observation
- 发出 `agent.observe`
- `replan` 时 `invoke_planner(observation=...)` + `accept_plan(...)`

没有把 Planner / Executor / Specialist 改成 LangGraph Node。

---

## 10. Tests

新文件 `backend/tests/agent_team/test_agent_loop.py`（11 passed）：

| 测试 | 验证 |
| --- | --- |
| `test_empty_retrieval_replans_with_new_query` | empty → observe → 第二次 Planner → query 改变 → Validator → execute → DONE |
| `test_low_relevance_replans` | 有候选 + 低 CLIP + tag mismatch → `LOW_RELEVANCE` → replan |
| `test_execution_failure_retries_same_query_not_low_relevance` | 工具失败 → revise；query 不变；不走低相关改写 |
| `test_replan_invalid_rejected_no_partial_execute` | 第二轮非法计划被 Validator 拒绝；`search.calls == 1` |
| `test_maximum_two_iterations_no_third_planner` | empty → replan → empty → abort；Planner 恰好 2 次 |
| `test_normal_success_single_planner` | 首轮成功，Planner 只调用 1 次 |
| `test_legacy_single_step_unaffected` | ComplexityGate 简单查询仍走 Legacy；enhanced=False 不重试 |
| `test_empty_and_low_relevance_are_distinct` | 两类 trigger 分开 |
| `test_langgraph_empty_replan_changes_query` | LangGraph 后端同样改 query |
| `test_planner_first_prompt_has_no_observation` | 第一轮 prompt 不含观察块 |
| `test_planner_second_prompt_consumes_observation` | 第二轮 prompt 含 Observation |

回归（均通过）：Supervisor FSM、LangGraph、生产接线、Evaluation、Judge、Memory、Planner、Registration、Orchestrator Legacy、MCP 工具失败仍为 `revise`。

---

## 11. Before / After Benchmark

脚本：`backend/tests/agent_team/run_agent_loop_benchmark.py`  
结果：`backend/tests/agent_team/outputs/agent_loop_benchmark.json`

5 个 Fake 场景，不改测试数据、不硬调阈值：

| 场景 | Before（旧 REVIEW） | After（Observe→Replan） |
| --- | --- | --- |
| success | continue，1 次 Planner | 相同 |
| empty_then_hit | **revise 同 query**，limit 思维 | **replan + query 改写** 后恢复 |
| low_relevance_then_hit | **当作成功**，返回低相关图 | **识别 LOW_RELEVANCE**，改 query 后恢复 |
| exec_fail | revise 同 query → abort | 相同（不误判低相关） |
| empty_then_empty | revise → abort | replan（query 已变）→ abort |

聚合指标：

| 指标 | Before | After |
| --- | --- | --- |
| recovery trigger rate | 0.6 | **0.8** |
| second-plan rate | 0.0 | **0.6** |
| action change rate | 0.0 | **0.6** |
| successful recovery rate | 0.2 | **0.4** |
| abort rate | 0.4 | 0.4 |
| Planner calls / request | 1.0 | 1.6 |
| latency ms mean（Fake） | 1.10 | 0.72 |
| invalid replan rate | 0.0 | 0.0 |

结论（刻意不追求 100%）：

> 第二轮**真的改变了策略**（query 从「城市建筑」变为「城市建筑 改写」），并且这种改变覆盖了旧路径漏掉的低相关失败。正常成功请求仍然只规划一次。

---

## 12. Runtime Trace Example

来源：`backend/tests/agent_team/outputs/agent_loop_trace_example.json`  
场景：第一轮空检索，第二轮改 query 后命中。

```text
iteration=1
  plan.created          retrieve_images → synthesize   query=城市建筑
  execute
  agent.observe         trigger=EMPTY_RETRIEVAL
                        REPLAN_REASON=EMPTY_RETRIEVAL
                        REPLAN_ACTION=RETRY_REWRITE
                        retrieval.empty=true candidate_count=0
  agent.review          decision=replan
  replan                Planner 第二次调用（observation 进入 prompt）

iteration=2
  plan.created          retrieve_images → synthesize   query=城市建筑 改写
  execute
  agent.observe         trigger=NONE  candidate_count=2 top1=0.91
  agent.review          decision=continue
  final                 selection=[201, 202]
```

`planner_calls=2`，检索 query：`城市建筑` → `城市建筑 改写`。  
这不是“代码里有一个 replan 函数”，而是一次可回放的 Observe → Replan 执行轨迹。

---

## 13. Known Limitations

- **LOW_RELEVANCE_THRESHOLD=0.4 仍为 exploratory**。CLIP score 定义为 `1 - cosine_distance`，尚未用真实图库 score distribution 标定；禁止把 0.4 写成生产结论。
- **Replan action space 有限**：当前主要是改 query / 可选加 Vision / 同 query 重试 / 结构重规划。没有自动改写复杂 filter 组合，也没有多假设并行。
- **Execution 仍串行**：DAGExecutor 未改为并行。
- **Specialist 不自主决策**：Retrieval / Organizer / Vision 没有 LLM，不互聊，不改下一步。这是设计约束，不是缺陷伪装。
- **MCP 仍非默认路径**：默认仍走 ToolRegistry adapter；MCP 测试仅覆盖可选路径。
- **Vision 默认关闭**：`VISION_MODEL="null"` 时 `ADD_VISION` 不会被建议。
- **Tag overlap 是关键词启发式**，不是语义匹配；中文整词切分可能偏粗。
- **第二轮策略变化依赖 Planner 真正消费 Observation**。测试用 Fake LLM 在见到观察块后改 query；真实 LLM 仍可能输出同 query，此时 Validator 会拒绝，不部分执行。
- **历史 200 条 evaluation JSON 里部分 `expected_decisions` 仍写着 `revise`**。本次未为刷分改那份数据；新闭环单测与 foundation 已改期望。
- **LangGraph 仍只做 Supervisor 编排**，不是 Multi-Agent 协商图。

---

## 验收对照

1. Specialist 只执行 + 回传证据  
2. 无 Agent-to-Agent 自由通信  
3. Supervisor 生成结构化 Observation  
4. Observation 进入第二轮 Planner  
5. 第二次 Planner 生成不同策略（query 改变）  
6. 新计划再次经过 Validator  
7. 空结果不再只是 limit+10  
8. 低相关与空结果分开  
9. execution failure 与 relevance failure 分开  
10. 最多两轮  
11. 正常请求仍一次 Planner  
12. Legacy / ComplexityGate 不受影响  
13. 新测试覆盖上述场景  
14. 有 Before/After 对照  
15. 有真实 Runtime Trace 证明 Observe → Replan  
16. 未新增 Specialist LLM，未实现 Agent 互聊  

本次升级到此停止，不再自行扩展 hybrid / 入库打标 / Specialist ReAct 等辅线。
