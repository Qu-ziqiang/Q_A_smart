"""空计划硬底线：重试一次后 fallback 为 Legacy 单步检索。Validator 仍拒绝空计划。"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from .models import (
    PlanValidationResult,
    PlannerFailure,
    TaskDraft,
    TaskPlanDraft,
    TaskType,
)
from .opt_flags import STATS

_EXECUTABLE = {
    TaskType.REWRITE_QUERY,
    TaskType.EXPAND_QUERY,
    TaskType.SCOPE_DATASET,
    TaskType.FILTER_BY_LABELS,
    TaskType.RETRIEVE_IMAGES,
    TaskType.RETRIEVE_SIMILAR,
    TaskType.RERANK_CANDIDATES,
    TaskType.FILTER_METADATA,
    TaskType.DEDUPLICATE_IMAGES,
    TaskType.GROUP_IMAGES,
    TaskType.SORT_IMAGES,
    TaskType.SELECT_IMAGES,
    TaskType.ANALYZE_VISUAL,
    TaskType.VERIFY_VISUAL_CONDITION,
    TaskType.READ_VISIBLE_TEXT,
    TaskType.LIST_OBJECTS,
    TaskType.COUNT_NAMED_OBJECT,
    TaskType.JUDGE_MAIN_SUBJECT,
    TaskType.VERIFY_SPATIAL,
}


def _tt(task: Any) -> Optional[TaskType]:
    t = getattr(task, "task_type", task)
    if isinstance(t, TaskType):
        return t
    try:
        return TaskType(str(t))
    except Exception:
        return None


def has_executable_tasks(plan: Any) -> bool:
    tasks = list(getattr(plan, "tasks", None) or [])
    return any(_tt(t) in _EXECUTABLE for t in tasks)


def fallback_retrieve_synth(
    goal: str, previous_queries: Optional[List[str]] = None
) -> TaskPlanDraft:
    q = (goal or "").strip() or "找相关图片"
    old = {x.strip().lower() for x in (previous_queries or []) if x and str(x).strip()}
    if q.strip().lower() in old:
        q = f"{q} 相关候选"
    return TaskPlanDraft(
        goal=q[:120],
        synthesis_task_id="synth_1",
        tasks=[
            TaskDraft(
                task_id="retrieve_1",
                task_type=TaskType.RETRIEVE_IMAGES,
                objective="单步检索",
                input={
                    "query": q,
                    "vector_targets": ["title", "description", "image"],
                    "limit": 20,
                },
                depends_on=[],
            ),
            TaskDraft(
                task_id="synth_1",
                task_type=TaskType.SYNTHESIZE,
                objective="汇总推荐",
                input={"recommendation_limit": 10, "presentation": "flat"},
                depends_on=["retrieve_1"],
            ),
        ],
    )


def ensure_executable_plan(
    *,
    create_draft: Callable[[Optional[List[str]]], Any],
    validate: Callable[[Any], PlanValidationResult],
    goal: str = "",
    previous_queries: Optional[List[str]] = None,
) -> PlanValidationResult:
    """空计划或 Validator 拒绝：带错误重生成一次；仍失败 → retrieve→synthesize。

    Validator 本身仍拒绝空/非法草稿；本函数保证调用方拿到可执行计划。
    """
    errors: List[str] = []
    for attempt in range(2):
        repair = errors if attempt else None
        try:
            draft = create_draft(repair)
        except PlannerFailure as e:
            errors = [str(e)]
            if attempt == 0:
                STATS.plan_floor_retry += 1
            continue
        if not has_executable_tasks(draft):
            errors = ["计划不含可执行 Task"]
            if attempt == 0:
                STATS.plan_floor_retry += 1
            continue
        vres = validate(draft)
        if vres.valid and vres.plan is not None and has_executable_tasks(vres.plan):
            return vres
        errors = list(vres.errors or ["validation_failed"])
        if not has_executable_tasks(draft) and not errors:
            errors = ["计划不含可执行 Task"]
        if attempt == 0:
            STATS.plan_floor_retry += 1
            continue
    STATS.plan_floor_fallback += 1
    return validate(fallback_retrieve_synth(goal, previous_queries))
