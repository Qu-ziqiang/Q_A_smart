"""P1 软 lint：Gate 判简单时剥离可选 Task。默认关，不硬拒。"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.models import (
    TaskDraft,
    TaskPlanDraft,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.opt_flags import (
    STATS,
    reset_opt_stats,
)
from backend.ai_func.recommendation.orchestration.plan_lint import (
    maybe_lint_plan,
    should_lint_query,
    strip_optional_tasks,
)


def _draft_bloat() -> TaskPlanDraft:
    r = TaskDraft(
        task_id="retrieve_1",
        task_type=TaskType.RETRIEVE_IMAGES,
        objective="找图",
        input={"query": "找猫"},
    )
    d = TaskDraft(
        task_id="dedup_1",
        task_type=TaskType.DEDUPLICATE_IMAGES,
        objective="去重",
        input={},
        depends_on=["retrieve_1"],
    )
    s = TaskDraft(
        task_id="select_1",
        task_type=TaskType.SELECT_IMAGES,
        objective="选",
        input={"strategy": "top_n", "limit": 5},
        depends_on=["dedup_1"],
    )
    y = TaskDraft(
        task_id="synth_1",
        task_type=TaskType.SYNTHESIZE,
        objective="答",
        input={},
        depends_on=["select_1"],
    )
    return TaskPlanDraft(goal="找猫", tasks=[r, d, s, y], synthesis_task_id="synth_1")


def test_should_lint_simple_not_visual():
    assert should_lint_query("找红色花朵照片") is True
    assert should_lint_query("找派对照片，再确认画面中是否有生日蛋糕") is False


def test_strip_keeps_retrieve_and_synth():
    new, removed = strip_optional_tasks(_draft_bloat())
    types = [t.task_type for t in new.tasks]
    assert types == [TaskType.RETRIEVE_IMAGES, TaskType.SYNTHESIZE]
    assert "deduplicate_images" in removed
    assert "select_images" in removed
    assert new.tasks[1].depends_on == ["retrieve_1"]


def test_maybe_lint_off_by_default():
    os.environ.pop("SM_PLAN_LINT", None)
    reset_opt_stats()
    draft = _draft_bloat()
    out = maybe_lint_plan("找猫", draft)
    assert len(out.tasks) == 4
    assert STATS.plan_lint_stripped == 0


def test_maybe_lint_strips_when_flag_on():
    os.environ["SM_PLAN_LINT"] = "1"
    reset_opt_stats()
    try:
        out = maybe_lint_plan("找猫", _draft_bloat())
        assert len(out.tasks) == 2
        assert STATS.plan_lint_stripped == 1
    finally:
        os.environ.pop("SM_PLAN_LINT", None)
        reset_opt_stats()


def test_maybe_lint_skips_team_queries():
    os.environ["SM_PLAN_LINT"] = "1"
    reset_opt_stats()
    try:
        q = "找派对照片，再确认画面中是否有生日蛋糕"
        out = maybe_lint_plan(q, _draft_bloat())
        assert len(out.tasks) == 4
        assert STATS.plan_lint_stripped == 0
    finally:
        os.environ.pop("SM_PLAN_LINT", None)
        reset_opt_stats()
