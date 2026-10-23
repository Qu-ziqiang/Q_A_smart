# PHASE4_2_JUDGE_CONTRACT_REPORT

> 项目：SmartImager — 多模态智能图库与对话推荐系统
> 阶段：Phase 4.2-A — LLM-as-a-Judge Contract Implementation（只建契约，不接真实 LLM）
> 日期：2026-09-07
> 约束：不修改 Runtime / Supervisor / EventCollector；不引入真实 LLM；不改数据库/前端。

---

## 1. 目标

在 Phase 4.1 EvaluationTrace 之上，为“LLM-as-a-Judge”冻结统一契约，并用 Mock 打通
`EvaluationTrace → JudgeInput → JudgeResult` 链路，使未来真实 LLM Judge（Phase 4.2-B）
只需实现同一 `Judge` 接口即可接入 `EvaluationRunner`。

---

## 2. 新增模块

```text
backend/ai_func/recommendation/evaluation/
├── __init__.py            （导出 + judge）
├── models.py              EvaluationCase / EvaluationTrace / EvaluationSummary（Phase 4.1）
├── runner.py              EvaluationRunner（扩展：可选 judge）
└── judge.py    ← 新增
        JudgeInput / JudgeResult
        Judge（ABC：evaluate / evaluate_trace）
        MockJudge（确定性实现）
        build_judge_input(trace)
```

测试：`backend/tests/agent_team/test_judge_contract.py`（6 用例）

---

## 3. Contract 设计

### 3.1 JudgeInput（结构化证据，无 LLM 生成物）
```python
class JudgeInput(BaseModel):
    case_id / request_id
    plan_ok: bool
    final_state: Optional[str]
    error: Optional[str]
    decisions: List[str]              # REVIEW decisions（不含终态重复）
    supervisor_states: List[str]
    execution_status: Dict[str, str]  # task_id → status
    metrics: Dict[str, Any]           # executed/succeeded/failed/skipped/selected_count
    tool_summary: Dict[str, Any]      # total/success/failure/names/latency + collector_snapshot
    selection_ids: List[int]
```
`build_judge_input(trace)`：对 `EvaluationTrace` 只读映射。

### 3.2 JudgeResult（统一输出）
```python
class JudgeResult(BaseModel):
    case_id / request_id
    verdict: Literal["pass", "flag", "fail"]
    criterion_scores: Dict[str, float]   # criterion → 0..5
    reasons: List[str]                   # 每条含证据来源
    error: Optional[str]
```

### 3.3 Judge 接口抽象
```python
class Judge(ABC):
    name: str
    criteria: List[str]
    @abstractmethod
    def evaluate(self, judge_input: JudgeInput) -> JudgeResult: ...
    def evaluate_trace(self, trace: EvaluationTrace) -> JudgeResult:  # 便捷入口
```

### 3.4 MockJudge（确定性，用于链路验证）
criteria：`execution` / `tool_health` / `selection_coverage`
- execution：计划执行成功度（成功任务占比；error/abort → 封顶 1.0）
- tool_health：工具成功率（失败 0 且调用>0 → 5.0；未调用 → 3.0 中性）
- selection_coverage：`3.0 + selected_count`，无推荐 → 0.0
- verdict：致命问题或任一 criterion < 阈值(4.0) → fail；经历 revise/replan 或工具失败
  或“无有效推荐” → flag；否则 pass。

---

## 4. EvaluationRunner 扩展（可选 judge）

```python
runner = EvaluationRunner(runtime_factory=..., judge=MockJudge())
traces, summary = runner.run(cases)
# summary.results[i]["judge"] = JudgeResult.model_dump()
# runner.judge_results = [JudgeResult, ...]
```
未注入 judge 时结果条目不含 `judge` 键（保持 Phase 4.1 行为，向后兼容）。

---

## 5. 测试结果

**新增：test_judge_contract.py — 6 passed**
| 用例 | 验证 |
|------|------|
| build_judge_input(trace) | JudgeInput 字段正确映射 |
| Judge 抽象不可实例化 | TypeError |
| MockJudge(success) | verdict=pass，三维 5.0 |
| MockJudge(recovery/revise) | verdict=flag，reason 含 revise |
| MockJudge(runtime error) | verdict=fail |
| Runner 可选 judge | results 含 judge + judge_results；无 judge 时不产出 |

**回归（全部通过）**
- evaluation：foundation 6 + judge 6 = 12
- tools：registry 11 / schema 7 / binding 7 / adapters 5 = 30
- agent 层：cards 6 / registry 7 / specialist 10 = 23
- supervisor/observability：state machine 5 / production integration 5 / event collector 2 = 12
- 核心 13 套件：registration 19 / orchestrator 6 / executor 8 / workers 12 / sse 9 / dag 14 /
  plan_validation 20 / synthesizer 12 / planner 11 / gate 13 / intent 23 / vision 49 /
  integration_full_chain 15 = 211

**合计：288 passed / 0 failed（含本阶段新增 6）。**

---

## 6. 约束合规

| 约束 | 状态 |
|------|------|
| 不修改 Runtime / Supervisor / EventCollector | ✅ 仅读 RunState/Collector 派生的 EvaluationTrace |
| 不引入真实 LLM | ✅ 仅 MockJudge（确定性） |
| 不修改数据库 / 前端 | ✅ |

---

## 7. Phase 4.2-B 接入点

真实 LLM Judge 只需：
1. `class LLMJudge(Judge)` 实现 `evaluate(JudgeInput) -> JudgeResult`
   （系统提示词 + rubric 同 `JudgeInput` 证据字段绑定）；
2. 小样本人工校准后接入 `EvaluationRunner(judge=LLMJudge(...))`；
3. `criterion_scores` / `reasons` 已与离线确定性层统一口径，可直接对比。
