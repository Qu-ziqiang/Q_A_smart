"""Phase 1-A — ComplexityGate 单元测试。

运行方式（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_gate.py
若已安装 pytest，也可直接 pytest 收集同名 test_* 函数。
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import ComplexityGate
from backend.ai_func.recommendation.orchestration.models import RouteType


def _gate():
    return ComplexityGate()


# ---------------------------------------------------------------------------
# 简单请求 → legacy
# ---------------------------------------------------------------------------
def test_gate_simple_cat_photos():
    d = _gate().decide(query="找猫的照片")
    assert d.route == RouteType.LEGACY


def test_gate_simple_greeting():
    d = _gate().decide(query="你好")
    assert d.route == RouteType.LEGACY


def test_gate_simple_ambiguous():
    d = _gate().decide(query="推荐点好看的图")
    assert d.route == RouteType.LEGACY


def test_gate_simple_retrieval_with_filter():
    d = _gate().decide(query="找 2025 年的海边照片")
    assert d.route == RouteType.LEGACY


def test_gate_simple_single_topic():
    d = _gate().decide(query="我想看一些好看的壁纸图片")
    assert d.route == RouteType.LEGACY


def test_gate_empty_query():
    d = _gate().decide(query="")
    assert d.route == RouteType.LEGACY


def test_gate_messages_uses_last_user():
    msgs = [
        {"role": "user", "content": "找猫的照片"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "分别找猫和狗的照片"},
    ]
    d = _gate().decide(messages=msgs)
    assert d.route == RouteType.TEAM


# ---------------------------------------------------------------------------
# 复杂请求 → team
# ---------------------------------------------------------------------------
def test_gate_complex_group_by_location():
    d = _gate().decide(query="找出旅行照片，按地点分类，然后推荐最合适的几张")
    assert d.route == RouteType.TEAM
    assert d.matched_features, "应记录命中的特征"


def test_gate_complex_multi_stage():
    d = _gate().decide(query="先找出旅行照片，再按时间归组，每组推荐两张")
    assert d.route == RouteType.TEAM


def test_gate_complex_multi_set():
    d = _gate().decide(query="分别找猫和狗的照片")
    assert d.route == RouteType.TEAM


def test_gate_complex_e2e_demo_query():
    d = _gate().decide(query="从旅行照片中找出横向图片，按地点分类，每组推荐最多 2 张")
    assert d.route == RouteType.TEAM


# ---------------------------------------------------------------------------
# 保守性：不认识的请求一律 legacy
# ---------------------------------------------------------------------------
def test_gate_visual_verify_goes_team():
    d = _gate().decide(query="找派对照片，再确认画面中是否有生日蛋糕")
    assert d.route == RouteType.TEAM
    assert any("visual_verify" in f for f in d.matched_features)


def test_gate_style_retrieval_stays_legacy():
    """颜色/风格当检索词，不是像素核验。"""
    assert _gate().decide(query="找红色花朵照片").route == RouteType.LEGACY
    assert _gate().decide(query="找水墨画风格的山景图").route == RouteType.LEGACY


def test_gate_routing_gold_28_layer_a():
    """Layer A：28 条 routing gold 的 expected_route。视觉核验应进 TEAM，纯检索仍 Legacy。"""
    from backend.tests.agent_team.routing_benchmark_28 import GOLD_QUERIES

    gate = _gate()
    fn, fp = [], []
    for q in GOLD_QUERIES:
        got = gate.decide(query=q["query"]).route.value
        exp = q["expected_route"]
        if exp == "team" and got != "team":
            fn.append(q["id"])
        if exp == "legacy" and got != "legacy":
            fp.append(q["id"])
    assert fn == [], f"Gate FN（该 TEAM 却 LEGACY）: {fn}"
    assert fp == [], f"Gate FP（该 LEGACY 却 TEAM）: {fp}"


def test_gate_conservative_unknown_phrase():
    d = _gate().decide(query="给我来点灵感")
    assert d.route == RouteType.LEGACY


def test_gate_null_messages():
    d = _gate().decide(messages=None)
    assert d.route == RouteType.LEGACY


# ---------------------------------------------------------------------------
# 纯 python 运行器（无 pytest 时使用）
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
