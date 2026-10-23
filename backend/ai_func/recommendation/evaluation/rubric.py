"""Phase 4.2-B — Judge Rubric（评分标准定义）。

JudgeCriterion：一个评分维度（0–5）。
JUDGE_CRITERIA：LLMJudge / MockJudge / 未来评分方统一使用的能力清单。
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, ConfigDict


class JudgeCriterion(BaseModel):
    """评分维度定义。"""

    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    description: str
    evaluation_focus: str          # Judge 应关注什么（注入 prompt）
    min_score: int = 0
    max_score: int = 5


JUDGE_CRITERIA: Dict[str, JudgeCriterion] = {
    "planning_quality": JudgeCriterion(
        key="planning_quality",
        name="Planning Quality",
        description="计划质量：计划是否符合用户目标、任务顺序是否合理。",
        evaluation_focus="核对计划任务类型与顺序能否达成 query 目标；"
                         "是否存在明显多余或缺失的任务。",
    ),
    "agent_routing": JudgeCriterion(
        key="agent_routing",
        name="Agent Routing",
        description="Agent 分工：检索/整理/视觉 Specialist 是否被正确调用。",
        evaluation_focus="检查任务类型是否路由到正确的 Specialist；分工是否合理。",
    ),
    "tool_usage": JudgeCriterion(
        key="tool_usage",
        name="Tool Usage",
        description="工具使用：调用是否成功、是否存在明显错误调用。",
        evaluation_focus="评估工具成功/失败比例与异常调用（如空检索/工具失败是否被正确触发）。",
    ),
    "recovery_behavior": JudgeCriterion(
        key="recovery_behavior",
        name="Recovery Behavior",
        description="恢复行为：empty retrieval / tool failure 后 retry/revise/replan 是否合理。",
        evaluation_focus="评估 REVIEW 决策序列（revise/replan/abort）是否匹配实际失败，"
                         "是否在预算内恢复。",
    ),
    "final_result_quality": JudgeCriterion(
        key="final_result_quality",
        name="Final Result Quality",
        description="最终结果质量：终态、selection 是否满足用户目标。",
        evaluation_focus="评估 final_state、推荐数量与 query 目标的匹配度。",
    ),
}


def criterion_keys() -> List[str]:
    return list(JUDGE_CRITERIA.keys())


def rubric_for_prompt() -> str:
    """把 rubric 渲染为 prompt 段（bounded，避免 token 膨胀）。"""
    lines = []
    for key, c in JUDGE_CRITERIA.items():
        lines.append(
            f"- {key}（{c.name}）: {c.description} 关注点: {c.evaluation_focus} "
            f"评分 {c.min_score}-{c.max_score}"
        )
    return "\n".join(lines)
