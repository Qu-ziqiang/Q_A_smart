"""Phase 4.2-A — LLM-as-a-Judge Contract（仅契约与 Mock，不接真实 LLM）。

数据模型：
  - JudgeInput：从 EvaluationTrace 派生的“可判定输入”（结构化证据，无 LLM 生成物）
  - JudgeResult：{case_id, request_id, verdict: pass|flag|fail,
                 criterion_scores: {criteria: 0..5}, reasons[], error}

接口：
  - Judge（ABC）：evaluate(JudgeInput) -> JudgeResult
  - MockJudge：确定性实现，用于验证 EvaluationTrace → JudgeResult 链路

本阶段不调用任何 LLM；真实 LLM Judge 只需实现同一 Judge 接口即可接入
EvaluationRunner（Phase 4.2-B）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import EvaluationTrace


class JudgeInput(BaseModel):
    """喂给 Judge 的结构化证据（源自 EvaluationTrace）。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = ""
    request_id: str = ""
    query: str = ""
    goal: str = ""                        # Phase 4.3：用户目标（prompt Context）
    plan_tasks: List[Dict[str, Any]] = Field(default_factory=list)  # [{task_id,task_type}]
    agent_names: List[str] = Field(default_factory=list)
    plan_ok: bool = False
    final_state: Optional[str] = None
    error: Optional[str] = None
    decisions: List[str] = Field(default_factory=list)
    supervisor_states: List[str] = Field(default_factory=list)
    execution_status: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    tool_summary: Dict[str, Any] = Field(default_factory=dict)
    selection_ids: List[int] = Field(default_factory=list)


class JudgeResult(BaseModel):
    """Judge 输出：确定性/LLM 统一的评分结果。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = ""
    request_id: str = ""
    verdict: Literal["pass", "flag", "fail"] = "flag"
    criterion_scores: Dict[str, float] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class Judge(ABC):
    """Judge 接口抽象（Mock / 未来真实 LLM Judge 均实现本接口）。"""

    name: str = "judge"
    criteria: List[str] = []

    @abstractmethod
    def evaluate(self, judge_input: JudgeInput) -> JudgeResult:
        """输入结构化证据，输出 JudgeResult。"""

    def evaluate_trace(self, trace: EvaluationTrace) -> JudgeResult:
        """便捷入口：EvaluationTrace → JudgeInput → evaluate。"""
        return self.evaluate(build_judge_input(trace))


def build_judge_input(trace: EvaluationTrace) -> JudgeInput:
    """从 EvaluationTrace 构建 JudgeInput（只读映射，无副作用）。"""
    agent_names = sorted({
        m.get("sender") for m in trace.agent_messages
        if m.get("sender") and m.get("sender") not in ("supervisor", "synthesizer")
    })
    return JudgeInput(
        case_id=trace.case_id,
        request_id=trace.request_id,
        query=trace.query,
        goal=getattr(trace, "goal", None) or "",
        plan_tasks=[dict(t) for t in (getattr(trace, "plan_tasks", None) or [])],
        agent_names=agent_names,
        plan_ok=trace.plan_ok,
        final_state=trace.final_state,
        error=trace.error,
        decisions=list(trace.decisions),
        supervisor_states=list(trace.supervisor_states),
        execution_status=dict(trace.execution_status),
        metrics=dict(trace.metrics),
        tool_summary=dict(trace.tool_summary),
        selection_ids=list(trace.selection_ids),
    )


# ---------------------------------------------------------------------------
# MockJudge：确定性启发式评分（用于验证链路，不代表真实质量判定）
# ---------------------------------------------------------------------------
class MockJudge(Judge):
    """确定性 Mock：按 trace 信号打分，产出 pass/flag/fail。

    criteria：
      - execution         计划执行是否成功（plan_ok / final_state / 任务失败率）
      - tool_health       工具调用成功率 / 失败数
      - selection_coverage最终是否有有效推荐
    """

    name = "mock"
    criteria = ["execution", "tool_health", "selection_coverage"]

    def __init__(self, threshold: float = 4.0):
        self.threshold = float(threshold)

    def evaluate(self, judge_input: JudgeInput) -> JudgeResult:
        reasons: List[str] = []
        fatal = judge_input.error is not None or not judge_input.plan_ok
        if judge_input.final_state == "abort":
            fatal = True
            reasons.append("final_state=abort")

        # -- execution ---------------------------------------------------
        succeeded = int(judge_input.metrics.get("succeeded", 0))
        failed = int(judge_input.metrics.get("failed", 0))
        executed = succeeded + failed
        ratio = (succeeded / executed) if executed else 0.0
        exec_score = 5.0 * ratio
        if fatal:
            exec_score = min(exec_score, 1.0)
            reasons.append("执行异常（error/plan 失败/abort）")
        if judge_input.decisions and any(d in ("revise", "replan") for d in judge_input.decisions):
            reasons.append("经历 revise/replan 恢复")

        # -- tool_health -------------------------------------------------
        tool_total = int(judge_input.tool_summary.get("total", 0) or 0)
        tool_fail = int(judge_input.tool_summary.get("failure", 0) or 0)
        if tool_total:
            tool_score = 5.0 * (1.0 - tool_fail / tool_total)
            if tool_fail:
                reasons.append(f"工具失败 {tool_fail}/{tool_total}")
        else:
            tool_score = 3.0  # 未调用工具视为中性
            reasons.append("无工具调用（中性 3.0）")

        # -- selection_coverage ------------------------------------------
        selected = len(judge_input.selection_ids)
        sel_score = min(5.0, 3.0 + float(selected)) if selected else 0.0
        if not selected:
            reasons.append("无有效推荐")

        scores = {
            "execution": round(max(0.0, min(5.0, exec_score)), 2),
            "tool_health": round(max(0.0, min(5.0, tool_score)), 2),
            "selection_coverage": round(max(0.0, min(5.0, sel_score)), 2),
        }

        if fatal or min(scores.values()) < self.threshold:
            verdict: Literal["pass", "flag", "fail"] = "fail"
        elif any(d in ("revise", "replan") for d in judge_input.decisions) \
                or tool_fail or "无有效推荐" in reasons:
            verdict = "flag"
        else:
            verdict = "pass"
        if not reasons:
            reasons.append("通过确定性启发式检查")

        return JudgeResult(
            case_id=judge_input.case_id,
            request_id=judge_input.request_id,
            verdict=verdict,
            criterion_scores=scores,
            reasons=reasons,
        )
