"""空计划硬底线 + Gate 政策分级。零真实 API。"""

from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.models import (
    PlanValidator,
    PlannerFailure,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.opt_flags import STATS, reset_opt_stats
from backend.ai_func.recommendation.orchestration.plan_floor import (
    ensure_executable_plan,
    fallback_retrieve_synth,
    has_executable_tasks,
)
from backend.ai_func.recommendation.orchestration.planner import (
    PlannerAgent,
    routing_mode_for_query,
    routing_policy_for_mode,
)


def test_simple_query_gets_min_policy():
    assert routing_mode_for_query("找红色花朵照片") == "simple"
    text = routing_policy_for_mode("simple")
    assert "禁止默认套餐" in text
    assert "禁止空计划" not in text


def test_complex_query_gets_full_catalog_policy():
    q = "找派对照片，再确认画面中是否有生日蛋糕"
    assert routing_mode_for_query(q) == "complex"
    text = routing_policy_for_mode("complex")
    assert "禁止空计划" in text
    assert "禁止默认套餐" not in text


def test_create_plan_injects_policy_by_gate():
    captured = []

    def llm(messages):
        captured.append(messages[0]["content"])
        return json.dumps({
            "goal": "找花",
            "synthesis_task_id": "t2",
            "tasks": [
                {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
                 "input": {"query": "花"}, "depends_on": []},
                {"task_id": "t2", "task_type": "synthesize", "objective": "汇总",
                 "input": {}, "depends_on": ["t1"]},
            ],
        }, ensure_ascii=False)

    PlannerAgent(llm=llm).create_plan(goal="找红色花朵照片")
    PlannerAgent(llm=llm).create_plan(goal="找派对照片，再确认画面中是否有生日蛋糕")
    assert "禁止默认套餐" in captured[0]
    assert "禁止空计划" in captured[1]
    assert "禁止默认套餐" not in captured[1]


def test_empty_plan_retries_then_fallback():
    reset_opt_stats()
    n = {"c": 0}

    def create(repair):
        n["c"] += 1
        raise PlannerFailure("boom")

    vres = ensure_executable_plan(
        create_draft=create,
        validate=lambda d: PlanValidator().validate(d),
        goal="找猫",
    )
    assert n["c"] == 2
    assert STATS.plan_floor_retry == 1
    assert STATS.plan_floor_fallback == 1
    assert vres.valid, vres.errors
    assert has_executable_tasks(vres.plan)
    types = [t.task_type for t in vres.plan.tasks]
    assert types == [TaskType.RETRIEVE_IMAGES, TaskType.SYNTHESIZE]


def test_fallback_never_empty():
    draft = fallback_retrieve_synth("找狗")
    assert has_executable_tasks(draft)
    r = PlanValidator().validate(draft)
    assert r.valid, r.errors


def test_invalid_plan_retries_then_fallback():
    reset_opt_stats()
    n = {"c": 0}

    def create(repair):
        n["c"] += 1
        return fallback_retrieve_synth("找猫").model_copy(update={
            "tasks": fallback_retrieve_synth("找猫").tasks[:1],
        })

    vres = ensure_executable_plan(
        create_draft=create,
        validate=lambda d: PlanValidator().validate(d),
        goal="找猫",
    )
    assert n["c"] == 2
    assert STATS.plan_floor_retry == 1
    assert STATS.plan_floor_fallback == 1
    assert vres.valid, vres.errors
    assert has_executable_tasks(vres.plan)
