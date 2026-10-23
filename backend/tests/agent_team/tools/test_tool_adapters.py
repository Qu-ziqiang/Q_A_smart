"""Phase 3.2 — Tool rerank JSON 契约 + adapters 单元测试。

覆盖：
  - normalize_candidates：JSON dict → ImageCandidate[]（领域对象隔离在 adapter）
  - normalize_candidates 容错：dict{'candidates':[...]} / list / ImageCandidate 直传
  - 非法项 / 非法 payload 类型报 TypeError
  - ToolRegistry.call('rerank', {candidates:[JSON...]}) 走默认实现成功，返回 ImageCandidate 列表

零真实 DB / LLM。运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/tools/test_tool_adapters.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    ToolRegistry,
    normalize_candidates,
    register_builtin_tools,
)
from backend.ai_func.recommendation.orchestration.workers import ImageCandidate


def _json_rows():
    return [
        {"id": 101, "score": 0.85, "metadata": {"location": "北京"}, "tags": ["风景"],
         "public_url": "/static/101.jpg"},
        {"id": 102, "score": 0.76, "metadata": {"location": "上海"}, "tags": ["风景"],
         "public_url": "/static/102.jpg"},
    ]


def test_normalize_candidates_from_payload_dict():
    rows = _json_rows()
    cands = normalize_candidates({"candidates": rows})
    assert len(cands) == 2
    assert cands[0].id == 101
    assert abs(float(cands[0].score) - 0.85) < 1e-6
    assert cands[0].metadata == {"location": "北京"}
    assert all(isinstance(c, ImageCandidate) for c in cands)


def test_normalize_candidates_from_list():
    rows = _json_rows()
    cands = normalize_candidates(rows)
    assert [c.id for c in cands] == [101, 102]


def test_normalize_candidates_passes_through_domain_objects():
    obj = ImageCandidate(id=7, score=0.5)
    cands = normalize_candidates([obj])
    assert cands[0] is obj  # 内部调用方兼容：领域对象直传


def test_normalize_candidates_rejects_bad_inputs():
    try:
        normalize_candidates("nope")
    except TypeError:
        pass
    else:
        raise AssertionError("非法 payload 类型应抛 TypeError")
    try:
        normalize_candidates({"candidates": [{"id": "not-int"}]})  # ImageCandidate 校验失败
    except TypeError:
        pass
    else:
        raise AssertionError("非法 candidate 应抛 TypeError")


def test_registry_rerank_json_contract_with_default_impl():
    reg = ToolRegistry()
    register_builtin_tools(reg)  # 默认实现（不经 override）
    result = reg.call("rerank", {"query": "风景", "candidates": _json_rows()})
    # 返回 ImageCandidate 列表，且候选集合与顺序一致（同池重排，无增删）
    assert len(result) == 2
    assert all(isinstance(c, ImageCandidate) for c in result)
    assert [c.id for c in result] == [101, 102]
    # ToolRegistry 层面不再出现领域对象参数
    event = reg.trace()[-1]
    assert event.tool_name == "rerank"
    assert event.success is True


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
