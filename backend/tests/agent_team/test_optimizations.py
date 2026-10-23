"""A2/A1/B1/C2 优化单测：全部 mock LLM/VLM，不打真实 API。

运行（repo 根）：
    venv\\Scripts\\python.exe -X utf8 -m pytest backend/tests/agent_team/test_optimizations.py -q
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.model_reranker import (
    OpenAICompatChatRerankerProvider,
    RelevanceError,
    MultimodalReranker,
    default_model_reranker,
    parse_rank_json,
)
from backend.ai_func.recommendation.orchestration.models import (
    AnalyzeVisualInput,
    GroupImagesInput,
    RetrieveImagesInput,
    SynthesizeInput,
    Task,
    TaskPlan,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.opt_flags import (
    STATS,
    clip_top_gap,
    reset_opt_stats,
    should_trigger_mm_rerank,
)
from backend.ai_func.recommendation.orchestration.plan_cache import (
    PlanCache,
    get_plan_cache,
    normalize_query,
    registry_fingerprint,
)
from backend.ai_func.recommendation.orchestration.rerank import rerank_candidates
from backend.ai_func.recommendation.orchestration.supervisor import SupervisorAgent
from backend.ai_func.recommendation.orchestration.workers import (
    GroupOutput,
    ImageCandidate,
    RetrievalOutput,
    SynthesisOutput,
    WorkerResult,
)

_FLAG_KEYS = (
    "SM_RERANK_GAP_TRIGGER",
    "SM_RERANK_GAP_TAU",
    "SM_RERANK_BATCH",
    "SM_RERANK_BATCH_MAX",
    "SM_DAG_PARALLEL",
    "SM_PLAN_CACHE",
    "SM_PLAN_LINT",
    "MM_RERANK_PROVIDER",
)


def _save_env():
    return {k: os.environ.get(k) for k in _FLAG_KEYS}


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _cand(iid, score=0.5, title=""):
    return ImageCandidate(id=iid, score=score, title=title, description="", tags=[], metadata={})


def _ids(cands):
    return [c.id for c in cands]


class _FakeProvider:
    name = "fake"

    def __init__(self, func=None, fail_ids=()):
        self._func = func or (lambda img: img.score)
        self._fail_ids = set(fail_ids)
        self.calls = []

    def relevance(self, query, image):
        self.calls.append(image.id)
        if image.id in self._fail_ids:
            raise RelevanceError("boom")
        return self._func(image)


class _BatchProvider:
    name = "batch_fake"

    def __init__(self, orders=None, fail_times=0, raws=None):
        self.orders = list(orders or [])
        self.fail_times = fail_times
        self.raws = list(raws or [])
        self.calls = 0

    def relevance(self, query, image):
        return image.score

    def rank_batch(self, query, candidates):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RelevanceError("batch timeout")
        if self.raws:
            return parse_rank_json(self.raws.pop(0), len(candidates))
        if self.orders:
            return list(self.orders.pop(0))
        return list(range(len(candidates) - 1, -1, -1))


# ---------------------------------------------------------------------------
# A2 分差触发
# ---------------------------------------------------------------------------
def test_clip_top_gap_and_trigger_off_always_true():
    saved = _save_env()
    try:
        os.environ.pop("SM_RERANK_GAP_TRIGGER", None)
        cands = [_cand(1, 0.9), _cand(2, 0.89), _cand(3, 0.88), _cand(4, 0.87), _cand(5, 0.86)]
        assert abs(clip_top_gap(cands) - 0.04) < 1e-9
        assert should_trigger_mm_rerank(cands) is True
    finally:
        _restore(saved)


def test_gap_trigger_skips_confident_clip():
    saved = _save_env()
    try:
        os.environ["SM_RERANK_GAP_TRIGGER"] = "1"
        os.environ["SM_RERANK_GAP_TAU"] = "0.08"
        os.environ["MM_RERANK_PROVIDER"] = "openai_chat"
        reset_opt_stats()
        confident = [_cand(1, 0.95), _cand(2, 0.7), _cand(3, 0.6), _cand(4, 0.5), _cand(5, 0.4)]
        assert should_trigger_mm_rerank(confident) is False
        out = default_model_reranker()("日落", confident)
        assert _ids(out) == [1, 2, 3, 4, 5]
        assert STATS.rerank_skipped == 1
        assert STATS.rerank_triggered == 0
    finally:
        _restore(saved)


def test_gap_trigger_fires_when_scores_close():
    saved = _save_env()
    try:
        os.environ["SM_RERANK_GAP_TRIGGER"] = "1"
        os.environ["SM_RERANK_GAP_TAU"] = "0.08"
        close = [_cand(1, 0.51), _cand(2, 0.50), _cand(3, 0.49), _cand(4, 0.48), _cand(5, 0.47)]
        assert should_trigger_mm_rerank(close) is True
    finally:
        _restore(saved)


# ---------------------------------------------------------------------------
# A1 批处理 + JSON 重试 + 超时回退
# ---------------------------------------------------------------------------
def test_parse_rank_json_one_based_and_retry_raw():
    assert parse_rank_json('{"order": [3, 1, 2], "reason": "x"}', 3) == [2, 0, 1]
    assert parse_rank_json('prefix {"order": [0, 2, 1]}', 3) == [0, 2, 1]
    try:
        parse_rank_json("not json", 3)
        raise AssertionError("expected RelevanceError")
    except RelevanceError:
        pass


def test_batch_rerank_one_call_reorders():
    saved = _save_env()
    try:
        os.environ["SM_RERANK_BATCH"] = "1"
        cands = [_cand(1, 0.9), _cand(2, 0.8), _cand(3, 0.7)]
        prov = _BatchProvider(orders=[[2, 0, 1]])
        out = MultimodalReranker(prov).rerank("q", cands)
        assert _ids(out) == [3, 1, 2]
        assert prov.calls == 1
        assert STATS.batch_ok >= 1
    finally:
        _restore(saved)


def test_batch_k_over_cap_two_rounds():
    saved = _save_env()
    try:
        os.environ["SM_RERANK_BATCH"] = "1"
        os.environ["SM_RERANK_BATCH_MAX"] = "3"
        cands = [_cand(i, 0.5) for i in range(1, 6)]
        # 第一轮选出 5→3，第二轮精排 3
        first = list(range(5))
        second = [2, 0, 1]
        prov = _BatchProvider(orders=[first, second])
        out = MultimodalReranker(prov).rerank("q", cands)
        assert prov.calls == 2
        assert _ids(out)[:3] == [3, 1, 2]
        assert set(_ids(out)) == {1, 2, 3, 4, 5}
    finally:
        _restore(saved)


def test_batch_timeout_falls_back_to_r1_not_sequential():
    saved = _save_env()
    try:
        os.environ["SM_RERANK_BATCH"] = "1"
        cands = [_cand(1, 0.9, "山"), _cand(2, 0.9, "长城日出")]
        seq_calls = []
        p = _BatchProvider(fail_times=3)
        p.relevance = lambda q, img: seq_calls.append(img.id) or 0.1
        out = MultimodalReranker(p).rerank("长城日出", cands)
        assert seq_calls == []
        assert _ids(out) == _ids(rerank_candidates("长城日出", cands))
    finally:
        _restore(saved)


def test_batch_json_fail_falls_back_without_sequential():
    saved = _save_env()
    try:
        os.environ["SM_RERANK_BATCH"] = "1"
        cands = [_cand(1), _cand(2)]

        class _OnceBad:
            name = "once"

            def __init__(self):
                self.calls = 0

            def relevance(self, q, img):
                raise AssertionError("should not sequential")

            def rank_batch(self, q, cs):
                self.calls += 1
                raise RelevanceError("bad json")

        prov = _OnceBad()
        out = MultimodalReranker(prov).rerank("长城日出", cands)
        assert prov.calls == 1
        assert set(_ids(out)) == {1, 2}
    finally:
        _restore(saved)


def test_batch_flag_off_uses_sequential():
    saved = _save_env()
    try:
        os.environ.pop("SM_RERANK_BATCH", None)
        cands = [_cand(1, 0.2), _cand(2, 0.9)]
        prov = _BatchProvider(orders=[[0, 1]])
        out = MultimodalReranker(prov).rerank("q", cands)
        assert prov.calls == 0
        assert _ids(out) == [2, 1]
    finally:
        _restore(saved)


# ---------------------------------------------------------------------------
# B1 DAG 并行
# ---------------------------------------------------------------------------
def _task(tid, ttype, inp, depends=None):
    return Task(task_id=tid, task_type=ttype, objective="o", input=inp, depends_on=depends or [])


def _plan(synth, tasks):
    return TaskPlan(plan_version="1.0", plan_id="p", goal="g", tasks=tasks, synthesis_task_id=synth)


def _ok(tid, ttype, output=None):
    return WorkerResult(task_id=tid, task_type=ttype, status=TaskStatus.SUCCEEDED, output=output)


class _SleepWorker:
    def __init__(self, ttype, seconds, stamp):
        self.task_type = ttype
        self.seconds = seconds
        self.stamp = stamp

    def run(self, task, context):
        self.stamp.append((task.task_id, "start", time.time()))
        time.sleep(self.seconds)
        self.stamp.append((task.task_id, "end", time.time()))
        if task.task_type == TaskType.RETRIEVE_IMAGES:
            return _ok(task.task_id, task.task_type,
                       RetrievalOutput(candidates=[_cand(1)], query="q", total=1))
        if task.task_type == TaskType.ANALYZE_VISUAL:
            return _ok(task.task_id, task.task_type)
        if task.task_type == TaskType.GROUP_IMAGES:
            return _ok(task.task_id, task.task_type, GroupOutput(group_by="x", groups=[]))
        return _ok(task.task_id, task.task_type, SynthesisOutput(image_ids=[1], candidates=[]))


def test_dag_parallel_flag_off_stays_serial():
    saved = _save_env()
    try:
        os.environ.pop("SM_DAG_PARALLEL", None)
        stamp = []
        w = _SleepWorker(TaskType.RETRIEVE_IMAGES, 0.05, stamp)
        plan = _plan("t3", [
            _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="a")),
            _task("t2", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="b")),
            _task("t3", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t1", "t2"]),
        ])
        t0 = time.time()
        list(DAGExecutor(workers={
            TaskType.RETRIEVE_IMAGES: w,
            TaskType.SYNTHESIZE: _SleepWorker(TaskType.SYNTHESIZE, 0.0, stamp),
        }).execute_stream(plan))
        assert time.time() - t0 >= 0.09
    finally:
        _restore(saved)


def test_dag_parallel_two_retrieve_faster_than_sum():
    saved = _save_env()
    try:
        os.environ["SM_DAG_PARALLEL"] = "1"
        stamp = []
        w = _SleepWorker(TaskType.RETRIEVE_IMAGES, 0.12, stamp)
        plan = _plan("t3", [
            _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="a")),
            _task("t2", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="b")),
            _task("t3", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t1", "t2"]),
        ])
        t0 = time.time()
        ex = DAGExecutor(workers={
            TaskType.RETRIEVE_IMAGES: w,
            TaskType.SYNTHESIZE: _SleepWorker(TaskType.SYNTHESIZE, 0.0, stamp),
        })
        list(ex.execute_stream(plan))
        elapsed = time.time() - t0
        assert elapsed < 0.28
        assert ex.result.succeeded_count == 3
    finally:
        _restore(saved)


def test_dag_parallel_vision_stays_serial():
    saved = _save_env()
    try:
        os.environ["SM_DAG_PARALLEL"] = "1"
        stamp = []
        plan = _plan("t4", [
            _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="a")),
            _task("t2", TaskType.ANALYZE_VISUAL, AnalyzeVisualInput(), depends=["t1"]),
            _task("t3", TaskType.ANALYZE_VISUAL, AnalyzeVisualInput(), depends=["t1"]),
            _task("t4", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t2", "t3"]),
        ])
        workers = {
            TaskType.RETRIEVE_IMAGES: _SleepWorker(TaskType.RETRIEVE_IMAGES, 0.0, stamp),
            TaskType.ANALYZE_VISUAL: _SleepWorker(TaskType.ANALYZE_VISUAL, 0.08, stamp),
            TaskType.SYNTHESIZE: _SleepWorker(TaskType.SYNTHESIZE, 0.0, stamp),
        }
        t0 = time.time()
        list(DAGExecutor(workers=workers).execute_stream(plan))
        assert time.time() - t0 >= 0.15
        ends = {tid: t for tid, kind, t in stamp if kind == "end"}
        starts = {tid: t for tid, kind, t in stamp if kind == "start"}
        # 两个 vision 不重叠
        a, b = "t2", "t3"
        assert ends[a] <= starts[b] + 1e-3 or ends[b] <= starts[a] + 1e-3
    finally:
        _restore(saved)


# ---------------------------------------------------------------------------
# C2 计划缓存
# ---------------------------------------------------------------------------
def test_normalize_and_exact_hit():
    c = PlanCache(ttl_seconds=60)
    c.put("找 日落，", {"plan_id": "p1"}, "regA")
    assert c.get("找 日落!", "regA")["plan_id"] == "p1"
    assert normalize_query("找 日落，") == normalize_query("找 日落!")


def test_semantic_hit_with_embed_fn():
    def emb(q):
        return [1.0, 0.0] if "日落" in q or "夕阳" in q else [0.0, 1.0]

    c = PlanCache(ttl_seconds=60, similarity=0.95, embed_fn=emb)
    c.put("海边日落", {"plan_id": "p2"}, "regA")
    assert c.get("海边夕阳", "regA")["plan_id"] == "p2"
    assert c.get("一只猫", "regA") is None


def test_ttl_and_registry_invalidation():
    clock = {"t": 1000.0}
    c = PlanCache(ttl_seconds=10, now_fn=lambda: clock["t"])
    c.put("q", {"plan_id": "old"}, "regA")
    clock["t"] = 1011.0
    assert c.get("q", "regA") is None
    c.put("q", {"plan_id": "new"}, "regA")
    assert c.get("q", "regB") is None
    assert c.get("q", "regA")["plan_id"] == "new"


def test_supervisor_cache_skips_planner_llm():
    saved = _save_env()
    try:
        os.environ["SM_PLAN_CACHE"] = "1"
        get_plan_cache().clear()
        reset_opt_stats()
        calls = []

        class _P:
            def create_plan(self, **kwargs):
                calls.append(kwargs.get("goal"))
                return {"plan_id": "draft", "goal": kwargs.get("goal")}

        sup = SupervisorAgent(planner=_P())
        a = sup.invoke_planner("找日落照片")
        b = sup.invoke_planner("找日落照片")
        assert calls == ["找日落照片"]
        assert a["plan_id"] == b["plan_id"] == "draft"
        assert STATS.cache_hit == 1
        assert sup.planner_call_count == 1
    finally:
        get_plan_cache().clear()
        _restore(saved)


def test_supervisor_replan_bypasses_cache():
    saved = _save_env()
    try:
        os.environ["SM_PLAN_CACHE"] = "1"
        get_plan_cache().clear()
        calls = []

        class _P:
            def create_plan(self, **kwargs):
                calls.append(kwargs.get("observation"))
                return {"n": len(calls)}

        sup = SupervisorAgent(planner=_P())
        sup.invoke_planner("同一句", observation=None)
        sup.invoke_planner("同一句", observation={"empty": True})
        assert len(calls) == 2
        assert calls[1] == {"empty": True}
    finally:
        get_plan_cache().clear()
        _restore(saved)


def test_registry_fingerprint_stable():
    fp = registry_fingerprint(vision_available_flag=False)
    assert "retrieve_images" in fp
    assert "analyze_visual" not in fp
    fp2 = registry_fingerprint(vision_available_flag=True)
    assert "analyze_visual" in fp2


def test_openai_batch_json_retry_then_ok():
    prov = OpenAICompatChatRerankerProvider()
    prov._reason = ""
    n = {"i": 0}

    def _fake_chat(query, candidates):
        n["i"] += 1
        if n["i"] == 1:
            return "not-json"
        return '{"order": [2, 1], "reason": "ok"}'

    prov._chat_batch = _fake_chat
    assert prov.rank_batch("q", [_cand(1), _cand(2)]) == [1, 0]
    assert n["i"] == 2
