"""软 lint：Gate 判简单（Legacy）时，剥掉可选 Task，只留 retrieve + synthesize。

非整单拒绝。SM_PLAN_LINT=1 才生效。剥离记 STATS.plan_lint_stripped。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .gate import ComplexityGate
from .models import RouteType, TaskType
from .opt_flags import STATS, plan_lint_enabled

_OPTIONAL = {
    TaskType.RERANK_CANDIDATES,
    TaskType.DEDUPLICATE_IMAGES,
    TaskType.SELECT_IMAGES,
    TaskType.SORT_IMAGES,
    TaskType.VERIFY_VISUAL_CONDITION,
    TaskType.ANALYZE_VISUAL,
    TaskType.FILTER_METADATA,
    TaskType.GROUP_IMAGES,
}

_KEEP = {TaskType.RETRIEVE_IMAGES, TaskType.SYNTHESIZE}


def _tt(task) -> TaskType:
    t = getattr(task, "task_type", task)
    if isinstance(t, TaskType):
        return t
    return TaskType(str(t))


def should_lint_query(query: str, gate: Optional[ComplexityGate] = None) -> bool:
    g = gate or ComplexityGate()
    return g.decide(query=query).route == RouteType.LEGACY


def strip_optional_tasks(draft: Any) -> Tuple[Any, List[str]]:
    """返回 (可能新的 draft, 被剥掉的 task_type 列表)。"""
    tasks = list(getattr(draft, "tasks", None) or [])
    if not tasks:
        return draft, []
    removed = []
    kept = []
    for t in tasks:
        if _tt(t) in _OPTIONAL:
            removed.append(_tt(t).value)
        else:
            kept.append(t)
    if not removed:
        return draft, []
    retrieves = [t for t in kept if _tt(t) == TaskType.RETRIEVE_IMAGES]
    synths = [t for t in kept if _tt(t) == TaskType.SYNTHESIZE]
    if not retrieves or not synths:
        return draft, []
    r0, s0 = retrieves[0], synths[0]
    try:
        s0 = s0.model_copy(update={"depends_on": [r0.task_id]})
    except Exception:
        s0.depends_on = [r0.task_id]
    new_tasks = [r0, s0]
    try:
        new_draft = draft.model_copy(update={
            "tasks": new_tasks,
            "synthesis_task_id": s0.task_id,
        })
    except Exception:
        draft.tasks = new_tasks
        draft.synthesis_task_id = s0.task_id
        new_draft = draft
    return new_draft, removed


def maybe_lint_plan(query: str, draft: Any) -> Any:
    if not plan_lint_enabled() or draft is None:
        return draft
    if not should_lint_query(query):
        return draft
    new_draft, removed = strip_optional_tasks(draft)
    if removed:
        STATS.plan_lint_stripped = getattr(STATS, "plan_lint_stripped", 0) + 1
    return new_draft
