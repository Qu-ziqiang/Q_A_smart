"""Routing Benchmark — 数据集完整性 + 离线评分器测试（审计 §10）。

普通 regression 不调用真实 API；真实 probe 需显式授权（见 routing_benchmark_28.py）。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_routing_benchmark.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.tests.agent_team.routing_benchmark_28 import (
    GOLD_QUERIES,
    VISION_NEGATIVE_CATEGORIES,
    VISION_POSITIVE_CATEGORIES,
    aggregate_scores,
    score_plan,
    vision_negative_set,
    vision_positive_set,
)


def test_dataset_28_queries_7_categories():
    assert len(GOLD_QUERIES) == 28
    from collections import Counter
    cats = Counter(q["category"] for q in GOLD_QUERIES)
    assert set(cats) == {"A", "B", "C", "D", "E", "F", "G"}
    assert all(v == 4 for v in cats.values())
    ids = [q["id"] for q in GOLD_QUERIES]
    assert ids == list(range(1, 29))          # 连续唯一 id
    assert len(set(ids)) == 28


def test_vision_positive_and_negative_sets():
    pos = vision_positive_set()
    neg = vision_negative_set()
    assert len(pos) == 12 and len(neg) == 16
    assert set(VISION_POSITIVE_CATEGORIES) == {"D", "E", "G"}
    assert set(VISION_NEGATIVE_CATEGORIES) == {"A", "B", "C", "F"}
    assert all(q["vision_required"] for q in pos)
    assert all(not q["vision_required"] for q in neg)


def test_every_query_has_expected_fields():
    for q in GOLD_QUERIES:
        assert q["expected_route"] in ("legacy", "team")
        assert "retrieve_images" in q["expected_task_types"]
        assert "synthesize" in q["expected_task_types"]
        assert isinstance(q["vision_required"], bool)
        assert q["note"]


def test_scorer_exact_match():
    q = {"id": 13, "expected_task_types": ["retrieve_images", "analyze_visual", "synthesize"],
         "vision_required": True}
    r = score_plan(q, ["retrieve_images", "analyze_visual", "synthesize"])
    assert r["exact_task_types_match"] is True
    assert r["vision_false_negative"] is False and r["vision_false_positive"] is False
    assert r["valid"] is True


def test_scorer_vision_false_negative():
    q = {"id": 13, "expected_task_types": ["retrieve_images", "analyze_visual", "synthesize"],
         "vision_required": True}
    r = score_plan(q, ["retrieve_images", "synthesize"])     # 漏掉 vision
    assert r["exact_task_types_match"] is False
    assert r["vision_false_negative"] is True
    assert r["missing"] == ["analyze_visual"]


def test_scorer_vision_false_positive():
    q = {"id": 9, "expected_task_types": ["retrieve_images", "group_images", "synthesize"],
         "vision_required": False}
    r = score_plan(q, ["retrieve_images", "group_images", "analyze_visual", "synthesize"])
    assert r["vision_false_positive"] is True
    assert r["extra"] == ["analyze_visual"]


def test_scorer_invalid_marked():
    q = {"id": 5, "expected_task_types": ["retrieve_images", "filter_metadata", "synthesize"],
         "vision_required": False}
    r = score_plan(q, ["retrieve_images", "synthesize"], validation_ok=False)
    assert r["valid"] is False


def test_aggregate_scores():
    qa = {"id": 1, "expected_task_types": ["retrieve_images", "synthesize"], "vision_required": False}
    results = [
        score_plan(qa, ["retrieve_images", "synthesize"]),
        score_plan(qa, ["retrieve_images", "analyze_visual", "synthesize"], validation_ok=False),
    ]
    agg = aggregate_scores(results)
    assert agg["n"] == 2
    assert agg["exact_task_types_match"] == 1
    assert agg["vision_false_positive"] == 1
    assert agg["invalid"] == 1


# ---------------------------------------------------------------------------
# 纯 python 运行器
# ---------------------------------------------------------------------------
def _main():
    failed = 0
    passed = 0
    mod_globals = globals()
    for name in sorted(mod_globals):
        if name.startswith("test_"):
            fn = mod_globals[name]
            try:
                fn()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print(f"\n== {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
