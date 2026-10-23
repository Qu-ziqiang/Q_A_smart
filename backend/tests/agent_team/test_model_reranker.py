"""R2 MultimodalReranker — 可注入模型型重排接口 + adapter + fallback 测试。

覆盖：score 单位化/clamp/非有限报错 / 同池稳定排序 / 不动输入对象与数量 / ID 集合不变 /
provider 失败(RelevanceError)与未知异常 → fallback R1（保留 R1 结果）/ 注入自定义 fallback /
空/单候选 / env 未配置 → build 返回 None、default_model_reranker 等价 R1 /
env 配置未知 provider 或 openai_chat 未配 Key → dispatcher 自动回退 R1 /
RetrievalWorker 集成：默认路径零回归 + R2 dispatcher 失败时 SUCCEEDED 且=R1 结果。

运行（repo 根）：
    venv\\Scripts\\python.exe -X utf8 -m pytest backend/tests/agent_team/test_model_reranker.py -q
"""

import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.model_reranker import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PROVIDER,
    RelevanceError,
    build_multimodal_reranker,
    default_model_reranker,
    MultimodalReranker,
    OpenAICompatChatRerankerProvider,
)
from backend.ai_func.recommendation.orchestration.rerank import rerank_candidates
from backend.ai_func.recommendation.orchestration.models import (
    RetrieveImagesInput,
    Task,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.workers import (
    ExecutionContext,
    ImageCandidate,
    RetrievalWorker,
)


def _cand(iid, score=0.5, title="", desc="", tags=None):
    return ImageCandidate(id=iid, score=score, title=title, description=desc,
                          tags=list(tags or []), metadata={})


def _ids(cands):
    return [c.id for c in cands]


class _FakeProvider:
    """确定性 provider：func(image) -> raw score；可配置 fail_ids 触发 RelevanceError。"""

    name = "fake"

    def __init__(self, func=None, fail_ids=()):
        self._func = func or (lambda img: img.score)
        self._fail_ids = set(fail_ids)
        self.calls = []

    def relevance(self, query, image):
        self.calls.append(image.id)
        if image.id in self._fail_ids:
            raise RelevanceError(f"provider boom for image {image.id}")
        return self._func(image)


def _fake_reranker(scores, fail_ids=()):
    return MultimodalReranker(_FakeProvider(lambda img: scores.get(img.id, img.score),
                                            fail_ids=fail_ids))


def _restore(keys, saved):
    for k in keys:
        if saved.get(k) is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = saved[k]


# ---------------------------------------------------------------------------
# 1) score：0~1 单位化与非法输入
# ---------------------------------------------------------------------------
def test_score_clamps_to_unit_and_is_finite():
    rk = _fake_reranker({1: 1.7, 2: -0.2, 3: 0.64})
    assert rk.score("q", _cand(1)) == 1.0
    assert rk.score("q", _cand(2)) == 0.0
    assert abs(rk.score("q", _cand(3)) - 0.64) < 1e-9


def test_score_rejects_nonfinite():
    def bad(img):
        if img.id == 1:
            return float("nan")
        return 0.5

    rk = MultimodalReranker(_FakeProvider(bad))
    try:
        rk.score("q", _cand(1))
        raise AssertionError("expected RelevanceError")
    except RelevanceError:
        pass
    assert rk.score("q", _cand(2)) == 0.5  # 其它候选不受影响


def test_score_rejects_non_numeric_provider_output():
    def bad(img):
        return None

    rk = MultimodalReranker(_FakeProvider(bad))
    try:
        rk.score("q", _cand(1))
        raise AssertionError("expected RelevanceError")
    except RelevanceError:
        pass


# ---------------------------------------------------------------------------
# 2) rerank：contract（数量不变 / ID 集合不变 / 只改排序 / 确定性）
# ---------------------------------------------------------------------------
def test_rerank_orders_by_score_desc_and_keeps_pool():
    cands = [_cand(1, 0.5, "a"), _cand(2, 0.9, "长城日出", tags=["长城"]), _cand(3, 0.7, "b")]
    rk = _fake_reranker({1: 0.1, 2: 0.99, 3: 0.5})
    out = rk.rerank("长城日出", cands)
    assert _ids(out) == [2, 3, 1]
    assert set(_ids(out)) == {1, 2, 3}
    assert len(out) == len(cands)


def test_rerank_stable_ties_keep_input_order():
    cands = [_cand(1), _cand(2), _cand(3)]
    rk = _fake_reranker({1: 0.5, 2: 0.5, 3: 0.5})  # 同分 → 原序
    assert _ids(rk.rerank("q", cands)) == [1, 2, 3]
    cands2 = [_cand(1), _cand(2), _cand(3)]
    assert _ids(rk.rerank("q", cands2)) == [1, 2, 3]  # 多次一致


def test_rerank_returns_same_objects_and_no_mutation():
    cands = [_cand(1, 0.9, "a"), _cand(2, 0.6, "长城日出")]
    before = [(c.id, c.score, c.title) for c in cands]
    out = _fake_reranker({1: 0.2, 2: 0.8}).rerank("长城日出", cands)
    after = [(c.id, c.score, c.title) for c in cands]
    assert before == after                       # 输入未被改写
    assert all(any(c is o for o in out) for c in cands)  # 同一批对象


# ---------------------------------------------------------------------------
# 3) fallback：provider 失败 → R1（默认），且绝不抛出
# ---------------------------------------------------------------------------
def test_rerank_provider_error_falls_back_to_r1_default():
    cands = [_cand(1, 0.9, "山"), _cand(2, 0.9, "长城日出", tags=["长城"])]
    rk = _fake_reranker({}, fail_ids={1, 2})  # 全部失败
    out = rk.rerank("长城日出", cands)
    expected = _ids(rerank_candidates("长城日出", cands))
    assert _ids(out) == expected              # 回退 = R1 结果（保留 R1）
    assert set(_ids(out)) == {1, 2} and len(out) == 2


def test_rerank_provider_partial_failure_discards_scores_and_falls_back():
    cands = [_cand(1, 0.9, "a"), _cand(2, 0.9, "长城日出", tags=["长城"]), _cand(3, 0.5, "b")]
    rk = _fake_reranker({1: 0.99, 3: 0.1}, fail_ids={2})  # 第 2 张失败
    out = rk.rerank("长城日出", cands)
    assert _ids(out) == _ids(rerank_candidates("长城日出", cands))  # 不采用部分结果


def test_rerank_unexpected_exception_also_falls_back():
    def explode(img):
        raise RuntimeError("unexpected crash")

    rk = MultimodalReranker(_FakeProvider(explode))
    cands = [_cand(1), _cand(2)]
    out = rk.rerank("q", cands)
    assert _ids(out) == [1, 2] or _ids(out) == _ids(rerank_candidates("q", cands))


def test_rerank_injected_custom_fallback_used():
    cands = [_cand(1, 0.5, "a"), _cand(2, 0.9, "长城日出")]
    fallback_called = []

    def my_fallback(q, cs):
        fallback_called.append(q)
        return list(reversed(cs))

    rk = MultimodalReranker(_FakeProvider(lambda img: img.score, fail_ids={1, 2}),
                            fallback=my_fallback)
    out = rk.rerank("长城日出", cands)
    assert fallback_called == ["长城日出"]
    assert _ids(out) == [2, 1]


# ---------------------------------------------------------------------------
# 4) env 工厂与 default_model_reranker（默认不配置 → 等价 R1，零付费调用）
# ---------------------------------------------------------------------------
_ENV_KEYS = (ENV_PROVIDER, ENV_API_KEY, ENV_BASE_URL, ENV_MODEL)


def test_build_unset_env_returns_none():
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        assert build_multimodal_reranker() is None
    finally:
        _restore(_ENV_KEYS, saved)


def test_default_model_reranker_unset_env_equals_r1():
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        cands = [_cand(1, 0.9, "山"), _cand(2, 0.9, "长城日出", tags=["长城"])]
        out = default_model_reranker()("长城日出", cands)
        assert _ids(out) == _ids(rerank_candidates("长城日出", cands))
    finally:
        _restore(_ENV_KEYS, saved)


def test_build_unknown_provider_falls_back_to_r1():
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        os.environ[ENV_PROVIDER] = "not_a_provider"
        for k in _ENV_KEYS:
            if k != ENV_PROVIDER:
                os.environ.pop(k, None)
        cands = [_cand(1, 0.9, "山"), _cand(2, 0.9, "长城日出", tags=["长城"])]
        out = default_model_reranker()("长城日出", cands)
        assert _ids(out) == _ids(rerank_candidates("长城日出", cands))
    finally:
        _restore(_ENV_KEYS, saved)


def test_openai_chat_provider_without_key_unavailable_and_falls_back():
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        os.environ[ENV_PROVIDER] = "openai_chat"
        for k in _ENV_KEYS:
            if k != ENV_PROVIDER:
                os.environ.pop(k, None)   # 无 model/key/base_url → 不可用
        prov = OpenAICompatChatRerankerProvider()
        try:
            prov.relevance("q", _cand(1))
            raise AssertionError("expected RelevanceError")
        except RelevanceError:
            pass
        cands = [_cand(1, 0.9, "山"), _cand(2, 0.9, "长城日出", tags=["长城"])]
        out = default_model_reranker()("长城日出", cands)
        assert _ids(out) == _ids(rerank_candidates("长城日出", cands))  # 回退 R1
    finally:
        _restore(_ENV_KEYS, saved)


def test_explicit_provider_argument_used():
    rk = build_multimodal_reranker(provider_name="")
    assert rk is None


# ---------------------------------------------------------------------------
# 5) RetrievalWorker 集成：默认零回归 + R2 dispatcher 失败仍 SUCCEEDED
# ---------------------------------------------------------------------------
def _rows():
    return [
        {"id": 1, "score": 0.9, "title": "山", "tags": '["山"]',
         "metadata": '{}', "public_url": "/s/1.jpg", "description": "d", "file_type": "jpg"},
        {"id": 2, "score": 0.9, "title": "长城日出", "tags": '["长城"]',
         "metadata": '{}', "public_url": "/s/2.jpg", "description": "长城日出", "file_type": "jpg"},
    ]


def _task(q="长城日出"):
    return Task(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, objective="o",
                input=RetrieveImagesInput(query=q, limit=10))


def test_worker_default_env_unset_still_r1_reorder():
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        worker = RetrievalWorker(search_tool=lambda **kw: list(_rows()))
        res = worker.run(_task(), ExecutionContext(None))
        assert res.status == TaskStatus.SUCCEEDED
        assert _ids(res.output.candidates) == [2, 1]  # 与历史 R1 行为一致
    finally:
        _restore(_ENV_KEYS, saved)


def test_worker_with_failing_model_dispatcher_succeeds_via_r1():
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        os.environ[ENV_PROVIDER] = "openai_chat"  # 无 Key/Model → provider 不可用
        for k in _ENV_KEYS:
            if k != ENV_PROVIDER:
                os.environ.pop(k, None)
        worker = RetrievalWorker(search_tool=lambda **kw: list(_rows()))
        res = worker.run(_task(), ExecutionContext(None))
        assert res.status == TaskStatus.SUCCEEDED          # 不导致请求失败
        assert set(_ids(res.output.candidates)) == {1, 2}   # 候选数量/集合不变
        assert _ids(res.output.candidates) == _ids(rerank_candidates("长城日出", res.output.candidates))
    finally:
        _restore(_ENV_KEYS, saved)


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
    print(f"\n== model_reranker: {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())


