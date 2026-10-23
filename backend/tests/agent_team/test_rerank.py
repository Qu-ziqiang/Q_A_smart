"""Rerank R1 — 单元 + RetrievalWorker 集成测试（test_rerank.py）。

覆盖：vector 单独排序 / keyword 改变排序 / metadata 改变排序 / 组合 / 同分稳定 /
空候选 / 单候选 / 缺字段 / ID 集合不变 / 数量不变 / 多次一致 / 权重全零与原序一致 /
RetrievalWorker 集成（开启可重排、关闭保持原序、reranker 异常回退原序）。

运行（repo 根）：
    venv\\Scripts\\python.exe -X utf8 -m pytest backend/tests/agent_team/test_rerank.py -q
"""

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.models import (
    MetadataPredicate,
    RetrieveImagesInput,
    Task,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.rerank import DEFAULT_WEIGHTS, rerank_candidates
from backend.ai_func.recommendation.orchestration.workers import (
    ExecutionContext,
    ImageCandidate,
    RetrievalWorker,
)

_WV = {"vector": 1.0, "keyword": 0.0, "metadata": 0.0}
_WK = {"vector": 1.0, "keyword": 1.0, "metadata": 0.0}
_WM = {"vector": 1.0, "keyword": 0.0, "metadata": 2.0}
_W0 = {"vector": 0.0, "keyword": 0.0, "metadata": 0.0}


def _cand(iid, score=0.5, title="", desc="", tags=None, metadata=None):
    return ImageCandidate(id=iid, score=score, title=title, description=desc,
                          tags=list(tags or []), metadata=dict(metadata or {}))


def _ids(cands):
    return [c.id for c in cands]


def _assert_pool_preserved(before, after):
    assert set(_ids(before)) == set(_ids(after))
    assert len(after) == len(before)


# ---------------------------------------------------------------------------
# 1) vector score 单独排序
# ---------------------------------------------------------------------------
def test_vector_only_ordering():
    cands = [_cand(1, 0.9, "a"), _cand(2, 0.5, "b"), _cand(3, 0.7, "c")]
    out = rerank_candidates("任意查询文本", cands, weights=_WV)
    assert _ids(out) == [1, 3, 2]
    _assert_pool_preserved(cands, out)


# ---------------------------------------------------------------------------
# 2) keyword overlap 能改变排序
# ---------------------------------------------------------------------------
def test_keyword_can_change_order():
    cands = [
        _cand(1, 0.8, "普通风景"),
        _cand(2, 0.7, "长城日出", "山顶看长城日出", ["长城"]),
        _cand(3, 0.6, "另一张"),
    ]
    out = rerank_candidates("长城日出", cands, weights=_WK)
    assert _ids(out) == [2, 1, 3]
    _assert_pool_preserved(cands, out)


def test_keyword_overlap_reuses_existing_tokenizer():
    # query 中英混合：latin 词与 CJK bigram 都能参与 overlap
    cands = [
        _cand(1, 0.9, "普通", tags=[]),                    # 无重叠 → 纯 vector 占优
        _cand(2, 0.8, "海边日落", tags=["sunset"]),         # latin+bigram 双命中
    ]
    out = rerank_candidates("海边 sunset", cands,
                            weights={"vector": 1.0, "keyword": 1.5, "metadata": 0.0})
    assert _ids(out) == [2, 1]
    _assert_pool_preserved(cands, out)


# ---------------------------------------------------------------------------
# 3) metadata match 能改变排序
# ---------------------------------------------------------------------------
def test_metadata_can_change_order():
    intent = SimpleNamespace(metadata_constraints=[
        MetadataPredicate(field="location", operator="eq", value="北京")])
    cands = [
        _cand(1, 0.8, "a", metadata={}),
        _cand(2, 0.7, "b", metadata={"location": "北京"}),
        _cand(3, 0.6, "c", metadata={"location": "北京"}),
    ]
    out = rerank_candidates("旅行", cands, intent=intent, weights=_WM)
    assert _ids(out) == [2, 3, 1]
    _assert_pool_preserved(cands, out)


def test_no_intent_metadata_is_neutral():
    cands = [_cand(1, 0.9, "a"), _cand(2, 0.6, "b")]
    out = rerank_candidates("q", cands)
    assert _ids(out) == [1, 2]


# ---------------------------------------------------------------------------
# 4) 多信号组合
# ---------------------------------------------------------------------------
def test_combined_signals():
    intent = SimpleNamespace(metadata_constraints=[
        MetadataPredicate(field="location", operator="eq", value="北京")])
    cands = [
        _cand(1, 0.9, "长城", metadata={"location": "北京"}),   # kw + meta 双命中
        _cand(2, 0.95, "无关", metadata={}),                    # vector 最高但无信号
        _cand(3, 0.7, "长城", metadata={}),                     # 仅 kw
    ]
    out = rerank_candidates("长城", cands, intent=intent,
                            weights={"vector": 1.0, "keyword": 1.0, "metadata": 1.0})
    assert _ids(out) == [1, 2, 3]
    _assert_pool_preserved(cands, out)


# ---------------------------------------------------------------------------
# 5) 同分稳定排序
# ---------------------------------------------------------------------------
def test_stable_tie_keeps_original_order():
    cands = [_cand(1, 0.9, "sun cat"), _cand(2, 0.9, "sun cat")]
    out = rerank_candidates("sun cat", cands)
    assert _ids(out) == [1, 2]


# ---------------------------------------------------------------------------
# 6) 空候选 / 单候选
# ---------------------------------------------------------------------------
def test_empty_candidates():
    assert rerank_candidates("q", []) == []


def test_single_candidate():
    c = _cand(1, 0.9, "x")
    out = rerank_candidates("q", [c])
    assert len(out) == 1 and out[0].id == 1


# ---------------------------------------------------------------------------
# 8) 缺字段 / 缺分数不崩溃且保持原序
# ---------------------------------------------------------------------------
def test_missing_fields_no_crash():
    cands = [
        ImageCandidate(id=1, score=0.9),
        ImageCandidate(id=2, score=0.8),
        ImageCandidate(id=3, score=0.7),
    ]
    out = rerank_candidates("q", cands)
    assert _ids(out) == [1, 2, 3]
    _assert_pool_preserved(cands, out)


def test_missing_vector_score_keeps_original_order():
    cands = [
        ImageCandidate(id=1, score=None, title="命中"),
        ImageCandidate(id=2, score=None, title="其他"),
    ]
    out = rerank_candidates("命中", cands)
    assert _ids(out) == [1, 2]
    _assert_pool_preserved(cands, out)


# ---------------------------------------------------------------------------
# 9/10) ID 集合与数量不变
# ---------------------------------------------------------------------------
def test_pool_preserved_across_reorders():
    cands = [_cand(1, 0.8, "x"), _cand(2, 0.7, "长城日出", tags=["长城"]), _cand(3, 0.6, "y")]
    out = rerank_candidates("长城日出", cands, weights=_WK)
    assert set(_ids(out)) == {1, 2, 3}
    assert len(out) == 3
    assert all(isinstance(c, ImageCandidate) for c in out)


# ---------------------------------------------------------------------------
# 11) 输入相同 → 多次一致（确定性）
# ---------------------------------------------------------------------------
def test_deterministic_multiple_runs():
    cands = [_cand(1, 0.8, "a"), _cand(2, 0.7, "长城日出", tags=["长城"]), _cand(3, 0.6, "b")]
    a = _ids(rerank_candidates("长城日出", cands, weights=_WK))
    b = _ids(rerank_candidates("长城日出", cands, weights=_WK))
    assert a == b


def test_rerank_does_not_mutate_input():
    cands = [_cand(1, 0.8, "长城日出"), _cand(2, 0.7, "其他")]
    before = [(c.id, c.score, c.title) for c in cands]
    rerank_candidates("长城日出", cands, weights=_WK)
    after = [(c.id, c.score, c.title) for c in cands]
    assert before == after


# ---------------------------------------------------------------------------
# 12) 权重全零 → 原序；RetrievalWorker 关闭/异常回退 → 原序
# ---------------------------------------------------------------------------
def test_zero_weights_fallback_original_order():
    cands = [_cand(1, 0.5, "a"), _cand(2, 0.9, "长城日出"), _cand(3, 0.7, "c")]
    out = rerank_candidates("长城日出", cands, weights=_W0)
    assert _ids(out) == [1, 2, 3]
    _assert_pool_preserved(cands, out)


def _task(q="长城日出"):
    return Task(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, objective="o",
                input=RetrieveImagesInput(query=q, limit=10))


def _rows():
    return [
        {"id": 1, "score": 0.9, "title": "山", "tags": '["山"]',
         "metadata": '{}', "public_url": "/s/1.jpg", "description": "d", "file_type": "jpg"},
        {"id": 2, "score": 0.9, "title": "长城日出", "tags": '["长城"]',
         "metadata": '{}', "public_url": "/s/2.jpg", "description": "长城日出", "file_type": "jpg"},
    ]


def test_retrieval_worker_integration_rerank_reorders_same_pool():
    worker = RetrievalWorker(search_tool=lambda **kw: list(_rows()))
    res = worker.run(_task(), ExecutionContext(None))
    assert res.status == TaskStatus.SUCCEEDED
    ids = [c.id for c in res.output.candidates]
    assert set(ids) == {1, 2}
    assert len(ids) == 2
    assert ids == [2, 1]


def test_retrieval_worker_rerank_disabled_keeps_row_order():
    worker = RetrievalWorker(search_tool=lambda **kw: list(_rows()), rerank_enabled=False)
    res = worker.run(_task(), ExecutionContext(None))
    assert [c.id for c in res.output.candidates] == [1, 2]


def test_retrieval_worker_reranker_exception_falls_back_original():
    def boom(query, cands):
        raise RuntimeError("rerank boom")

    worker = RetrievalWorker(search_tool=lambda **kw: list(_rows()), reranker=boom)
    res = worker.run(_task(), ExecutionContext(None))
    assert res.status == TaskStatus.SUCCEEDED
    assert [c.id for c in res.output.candidates] == [1, 2]


# ---------------------------------------------------------------------------
def _main():
    failed = passed = 0
    g = globals()
    for name in sorted(g):
        if name.startswith("test_"):
            try:
                g[name]()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n== rerank: {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())


