"""Judge Rubric v2（开发期回归信号，不覆盖冻结 200 的旧 rubric）。

跨版本不可比。产品决策对齐 Gate FN/FP、TEAM 率、简单 query 均 Task 数、延迟。
"""

from __future__ import annotations

from .rubric import JUDGE_CRITERIA, JudgeCriterion, criterion_keys

JUDGE_CRITERIA_V2 = dict(JUDGE_CRITERIA)
JUDGE_CRITERIA_V2["planning_quality"] = JudgeCriterion(
    key="planning_quality",
    name="Planning Quality",
    description="最小充分计划=满分；能达成目标的最短 Typed DAG 优于更长套餐。",
    evaluation_focus=(
        "最小充分=满分；每多一个用户未要求的可选任务（rerank/dedup/select/verify）扣分；"
        "VISION_MODEL 不可用时缺席 analyze/verify 不扣分。"
    ),
)


def rubric_v2_for_prompt() -> str:
    lines = [
        "评分锚点：最小充分计划满分；冗余任务扣分；能力不可用不惩罚。",
        "效率比 = overall / max(平均任务数, 1)，仅报告不单独当 pass 门槛。",
    ]
    for key, c in JUDGE_CRITERIA_V2.items():
        lines.append(
            f"- {key}（{c.name}）: {c.description} 关注点: {c.evaluation_focus}"
        )
    return "\n".join(lines)


def efficiency_ratio(overall: float, avg_tasks: float) -> float:
    return round(float(overall) / max(float(avg_tasks), 1.0), 4)
