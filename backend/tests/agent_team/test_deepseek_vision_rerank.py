"""看图重排只重排已有 id，不调用网络。"""

from types import SimpleNamespace

from backend.ai_func.recommendation.orchestration.llm_rerank import (
    _order_from_text,
    rerank_with_deepseek_vision,
)


def _cand(i):
    return SimpleNamespace(id=i, title="", filepath="", public_url="")


def test_order_keeps_the_same_ids():
    cands = [_cand(3), _cand(1), _cand(2)]
    ranked = _order_from_text('{"order": [1, 3]}', cands)
    assert [c.id for c in ranked] == [1, 3, 2]


def test_vision_rerank_uses_injected_response_without_api():
    cands = [_cand(8), _cand(5)]
    ordered, used = rerank_with_deepseek_vision(
        "有鸟的风景",
        cands,
        llm=lambda query, items: '{"order": [5, 8]}',
    )
    assert used == "deepseek_vision"
    assert [c.id for c in ordered] == [5, 8]
