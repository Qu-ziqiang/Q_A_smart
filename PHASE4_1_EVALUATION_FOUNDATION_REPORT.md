# PHASE4_1_EVALUATION_FOUNDATION_REPORT

> 项目：SmartImager — 多模态智能图库与对话推荐系统
> 阶段：Phase 4.1 — Evaluation Foundation Implementation（离线评估基础设施）
> 日期：2026-09-07
> 约束：不改 planner/executor/workers/supervisor 核心逻辑；不引入 LLM Judge；
>       不修改数据库；不修改前端。

---

## 1. 目标

在已收敛的 Agent Runtime（Planner→Validator→Supervisor→DAG→Specialists→ToolRegistry→
Workers + RunState + EventCollector）之上，建立 **离线评估基础设施**：

- 输入：`EvaluationCase`（纯声明式用例规格 + 确定性预期）
- 执行：`EvaluationRunner`（case → runtime 执行）
- 输出：`EvaluationTrace`（源自 RunState / EventCollector，同源无二次实现）
- 汇总：`EvaluationSummary`（确定性预期检查，供 CI 门禁）

Phase 4.2 的 LLM-as-a-Judge 可直接消费 `EvaluationTrace`（结构字段已就绪），本阶段不引入评分。

---

## 2. 新增模块（不触碰任何核心模块）

```text
backend/ai_func/recommendation/evaluation/
├── __init__.py
├── models.py      EvaluationCase / EvaluationTrace / EvaluationSummary
└── runner.py      EvaluationRunner / build_evaluation_trace / expected_violations
```

**测试**：`backend/tests/agent_team/test_evaluation_foundation.py`（6 用例）

**复用（不改动）**：`SupervisorAgent.run_state`（plan / execution_results / agent_messages /
tool_events / supervisor_trace / final_state）、`EventCollector.snapshot()`、
`RunState.collector`（Phase 3.2 收口产物）。

---

## 3. 设计

### 3.1 EvaluationCase（数据模型）
```python
class EvaluationCase(BaseModel):
    case_id: str
    name: str = ""
    query: str
    goal: Optional[str] = None                 # 显式 goal（缺省用 query）
    messages: List[Dict[str, str]] = []        # 可选前置历史
    expected_agents: List[str] = []            # 应执行的 Specialist（确定性断言）
    expected_tools: List[str] = []             # 应调用的工具
    expected_decisions: Optional[List[str]]    # REVIEW decisions 序列（如 ['revise','continue']）
    expected_final_state: str = "done"
    expect_selection: bool = True
    tags: List[str] = []
```

### 3.2 EvaluationTrace（由 RunState + EventCollector 派生）
```python
class EvaluationTrace(BaseModel):
    case_id / request_id / duration_ms / plan_ok / final_state / error
    plan_tasks: List[dict]
    execution_status: Dict[str, str]           # task_id → status（metrics 同源）
    supervisor_states: List[str]               # plan/execute/review/revise/... 状态序列
    decisions: List[str]                       # 仅取 REVIEW 步骤决策（无重复终态）
    agent_messages: List[dict]                 # message_id/sender/receiver/message_type
    tool_events: List[dict]                    # tool_name/agent_id/success/latency_ms/error
    tool_summary: dict                         # total/success/failure/names/total_latency_ms
                                               #   + collector_snapshot（EventCollector 同源）
    selection_ids: List[int]
    metrics: dict                              # executed/succeeded/failed/skipped/selected_count
```

### 3.3 EvaluationRunner（离线执行）
```python
runner = EvaluationRunner(runtime_factory=lambda case: build_supervisor_for(case))
traces, summary = runner.run([case1, case2])   # runtime_factory 决定夹具（离线确定性）
```
- 单条失败不会中断整批：runtime 抛错写入 `trace.error`；
- `expected_violations(case, trace)` 返回违例列表（空 = 通过）；
- `EvaluationSummary {total, passed, failed, results[]}` 可直接做 CI 断言。

---

## 4. 用法示例（与既有确定性 harness 同风格）

```python
from backend.ai_func.recommendation.evaluation import EvaluationCase, EvaluationRunner

case = EvaluationCase(
    case_id="c_recover", query="找风景图片",
    tags=["recover"],
    expected_agents=["retrieval"],
    expected_tools=["search_images"],
    expected_decisions=["revise", "continue"],   # 空检索 → REVISE → retry → done
)
traces, summary = EvaluationRunner(runtime_factory=make_runtime).run([case])
assert summary.failed == 0
trace = traces[0]
print(trace.tool_summary["collector_snapshot"])  # EventCollector 快照（Judge 输入雏形）
```

---

## 5. 测试结果

**新增（Phase 4.1）：test_evaluation_foundation.py — 6 passed**
| 用例 | 验证 |
|------|------|
| case model defaults | 字段默认值 |
| runner success trace & summary | plan_ok/done/decisions/tool_summary/selection/metrics/通过 |
| recovery case | decisions=[revise,continue]、revise 状态、selection 非空、零违例 |
| expectation violations | 缺 vision/analyze_image/decisions 不符 → failed + mismatches |
| runtime error | 抛错捕获进 trace.error（不中断） |
| batch summary | total/passed/failed 计数 |

**回归（既有全部通过）**
- tools：registry 11 / schema 7 / binding 7 / adapters 5 = 30
- agent 层：cards 6 / registry 7 / specialist 10 = 23
- supervisor/observability：state machine 5 / production integration 5 / event collector 2 = 12
- 核心 13 套件：registration 19 / orchestrator 6 / executor 8 / workers 12 / sse 9 / dag 14 /
  plan_validation 20 / synthesizer 12 / planner 11 / gate 13 / intent 23 / vision 49 /
  integration_full_chain 15 = 211

**合计：282 passed / 0 failed（含本阶段新增 6）。**

---

## 6. 约束合规

| 约束 | 状态 |
|------|------|
| 不修改 planner/executor/workers/supervisor 核心逻辑 | ✅ evaluation 为独立包，仅读 RunState/Collector |
| 不引入 LLM Judge | ✅ 仅确定性预期检查 |
| 不修改数据库 | ✅ 无 DDL / repository 改动 |
| 不修改前端 | ✅ 未触及 frontend/ |
| 基于 RunState + EventCollector 构建 Trace | ✅ build_evaluation_trace 同源派生 |

---

## 7. 后续（Phase 4.2 预留点）

- `EvaluationTrace.tool_summary.collector_snapshot` 已携带工具级失败/延迟明细；
- `EvaluationTrace.decisions / supervisor_states / execution_status / selection_ids`
  已覆盖“Planner/Retrieval/Vision/Final”评估所需的确定性基座；
- 后续只需新增 `judge.py`（LLM Judge 合约 + runner 扩展），或在
  `expected_violations` 基础上增加评分维度，无需再改 Runtime。
