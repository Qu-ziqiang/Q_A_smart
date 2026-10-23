"""Task-catalog A/B 装配测试：不消耗真实 API。"""

import json
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

import importlib.util

from backend.ai_func.recommendation.evaluation.benchmark import load_benchmark_cases
from backend.ai_func.recommendation.evaluation.benchmark_harness import (
    build_real_planner_runtime,
    _plan_json,
)
from backend.ai_func.recommendation.evaluation.catalog_ab import (
    capabilities_for,
    enabled_task_types,
    pin_registered_catalog,
)
from backend.ai_func.recommendation.evaluation.models import EvaluationCase
from backend.ai_func.recommendation.orchestration.models import (
    TaskType,
    registered_task_types,
)

_SCRIPT = os.path.join(
    os.path.abspath(_REPO_ROOT), "scripts", "run_task_catalog_ab_benchmark.py"
)
_spec = importlib.util.spec_from_file_location("task_catalog_ab_script", _SCRIPT)
_ab = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ab)


def test_before_catalog_excludes_new_types():
    before = enabled_task_types("before", vision=True)
    assert "retrieve_images" in before
    assert "analyze_visual" in before
    assert "synthesize" in before
    assert "select_images" not in before
    assert "sort_images" not in before
    assert "verify_visual_condition" not in before
    after = enabled_task_types("after", vision=True)
    assert "select_images" in after
    assert "verify_visual_condition" in after
    assert len(after) == 10


def test_pin_registered_catalog_restores():
    original = set(registered_task_types(False))
    with pin_registered_catalog("before", vision=False) as enabled:
        assert "sort_images" not in enabled
        pinned = registered_task_types(True)
        assert TaskType.SORT_IMAGES not in pinned
        assert TaskType.RETRIEVE_IMAGES in pinned
    assert set(registered_task_types(False)) == original


def test_before_prompt_does_not_leak_new_tasks():
    caps = capabilities_for("before", vision=True)
    from backend.ai_func.recommendation.evaluation.catalog_ab import CatalogPinnedPlanner

    prompt = CatalogPinnedPlanner._build_system_prompt(caps)
    assert "select_images" not in prompt
    assert "verify_visual_condition" not in prompt
    assert "analyze_visual" in prompt


def test_real_planner_runtime_accepts_injected_llm():
    case = EvaluationCase(
        case_id="ab_unit_1",
        query="找旅行照片，按地点分组后再每组选2张",
        goal="分组选择",
        tags=["organization", "success"],
    )
    runtime = build_real_planner_runtime(
        case, "after",
        planner_llm=lambda messages: json.dumps(_plan_json(case), ensure_ascii=False),
    )
    events = list(runtime.run_stream([{"role": "user", "content": case.query}]))
    assert events
    assert getattr(runtime, "last_plan_ok", False) or getattr(runtime, "run_state", None)


def test_load_ab_cases_has_more_than_200():
    extra = load_benchmark_cases(os.path.join(
        os.path.abspath(_REPO_ROOT), "backend", "ai_func", "recommendation",
        "evaluation", "datasets", "smartimager_task_catalog_ab_extra.json",
    ))
    cases = _ab.load_ab_cases(include_extra=True)
    assert len(extra) == 40
    assert len(cases) >= 240
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))


def test_compare_delta_math():
    before = {
        "overall": {"overall_score": 4.0, "pass_rate": 0.3, "plan_ok_rate": 0.8},
        "scores": {k: 4.0 for k in (
            "planning_quality", "agent_routing", "tool_usage",
            "recovery_behavior", "final_result_quality",
        )},
    }
    after = {
        "overall": {"overall_score": 4.2, "pass_rate": 0.4, "plan_ok_rate": 0.9},
        "scores": {k: 4.2 for k in before["scores"]},
    }
    cmp = _ab._compare(before, after)
    assert cmp["overall_score"]["delta"] == 0.2
    assert cmp["scores"]["agent_routing"]["delta"] == 0.2


def test_resolve_credentials_requires_key():
    class Args:
        provider = "openai"
        api_key = ""
        base_url = ""
        model = ""

    old = {k: os.environ.pop(k, None) for k in (
        "LLM_JUDGE_API_KEY", "OPENAI_API_KEY",
    )}
    try:
        try:
            _ab.resolve_credentials(Args())
            assert False, "should require key"
        except Exception as e:
            assert "API Key" in str(e)
    finally:
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


def test_persist_env_keeps_key():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, ".env")
        _ab.persist_env("sk-test-keep", "https://api.deepseek.com", "deepseek-chat", path)
        text = open(path, encoding="utf-8").read()
        assert "LLM_JUDGE_API_KEY=sk-test-keep" in text
        _ab.persist_env("sk-test-keep-2", "https://api.deepseek.com", "deepseek-chat", path)
        assert "sk-test-keep-2" in open(path, encoding="utf-8").read()
