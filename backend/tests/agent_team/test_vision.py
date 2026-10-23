"""Phase 2 Stage 1/3 — Vision Contract / Parser / VisionWorker 单元测试。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_vision.py
"""

import json
import os
import sys
import tempfile
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from pydantic import ValidationError

from backend.ai_func.image_analysis import ImageAnalysisError, ImageAnalysisErrorCode
from backend.ai_func.recommendation.orchestration.models import (
    AnalyzeVisualInput,
    DEFAULT_VISION_CANDIDATES,
    MAX_VISUAL_CANDIDATES,
    Task,
    TaskStatus,
    TaskType,
    VISION_ITEM_TIMEOUT_SECONDS,
)
from backend.ai_func.recommendation.orchestration.vision import (
    MAX_VISION_TOTAL_EVIDENCE_CHARS,
    VisionErrorCode,
    VisionItemResult,
    VisionStatus,
    VisionTaskOutput,
    build_vision_output,
    cap_evidence_total,
    parse_vision_response,
    vision_item_failed,
    vision_item_success,
    vision_systemic_failure,
)
from backend.ai_func.recommendation.orchestration.workers import (
    ExecutionContext,
    ImageCandidate,
    RetrievalOutput,
    VisionWorker,
    WorkerResult,
    build_vision_prompt,
    status_summary,
)


def _valid_raw():
    return json.dumps({
        "title": "Red circle and blue triangle",
        "description": "A white background with a red circle on the left, a blue triangle on the right, and text VISION-PROBE-2026 at the bottom.",
        "tags": ["red circle", "blue triangle", "text"],
        "evidence": "visible shapes",
        "confidence": 0.95,
    })


# ---------------------------------------------------------------------------
# 成功路径
# ---------------------------------------------------------------------------
def test_parse_success():
    r = parse_vision_response(_valid_raw(), image_id=42)
    assert r.status == VisionStatus.SUCCESS
    assert r.image_id == 42
    assert r.title and r.description and r.tags
    assert r.error_code is None
    assert r.confidence == 0.95


def test_parse_success_fenced_json():
    raw = "```json\n" + _valid_raw() + "\n```"
    r = parse_vision_response(raw, image_id=1)
    assert r.status == VisionStatus.SUCCESS


def test_parse_missing_optional_fields_defaults():
    raw = json.dumps({"title": "t", "description": "d"})
    r = parse_vision_response(raw, image_id=1)
    assert r.status == VisionStatus.SUCCESS
    assert r.tags == []
    assert r.confidence == 0.0
    assert r.evidence == ""


def test_extra_forbid_on_model():
    try:
        VisionItemResult(image_id=1, title="t", description="d", evil="x")
        assert False, "应拒绝未知字段"
    except ValidationError:
        pass


def test_vision_task_output_extra_forbid():
    try:
        VisionTaskOutput(items=[], total=0, success_count=0, failed_count=0, junk=True)
        assert False, "应拒绝未知字段"
    except ValidationError:
        pass


# ---------------------------------------------------------------------------
# 失败路径（parser 绝不抛异常，一律收敛为 ITEM_FAILED）
# ---------------------------------------------------------------------------
def test_parse_empty_output():
    r = parse_vision_response("", image_id=1)
    assert r.status == VisionStatus.ITEM_FAILED
    assert r.error_code == VisionErrorCode.PARSE_ERROR


def test_parse_malformed_json():
    r = parse_vision_response("this is not json", image_id=1)
    assert r.status == VisionStatus.ITEM_FAILED
    assert r.error_code == VisionErrorCode.PARSE_ERROR


def test_parse_non_dict():
    r = parse_vision_response("[1,2,3]", image_id=1)
    assert r.status == VisionStatus.ITEM_FAILED


def test_parse_unknown_fields():
    raw = json.dumps({"title": "t", "description": "d", "location": "Beijing"})
    r = parse_vision_response(raw, image_id=1)
    assert r.status == VisionStatus.ITEM_FAILED
    assert r.error_code == VisionErrorCode.PARSE_ERROR
    assert "location" in r.failure_reason


def test_parse_missing_required_fields():
    raw = json.dumps({"title": "", "description": "  "})
    r = parse_vision_response(raw, image_id=1)
    assert r.status == VisionStatus.ITEM_FAILED
    assert r.error_code == VisionErrorCode.PARSE_ERROR


def test_parse_invalid_image_id():
    r = parse_vision_response(_valid_raw(), image_id=-5)
    assert r.status == VisionStatus.ITEM_FAILED
    assert r.error_code == VisionErrorCode.INVALID_IMAGE_ID
    assert r.image_id == 0  # 保留哨兵值


def test_parse_invalid_confidence_type():
    raw = json.dumps({"title": "t", "description": "d", "confidence": ["x"]})
    r = parse_vision_response(raw, image_id=1)
    assert r.status == VisionStatus.ITEM_FAILED
    assert r.error_code == VisionErrorCode.PARSE_ERROR


def test_parse_invalid_confidence_string():
    raw = json.dumps({"title": "t", "description": "d", "confidence": "abc"})
    r = parse_vision_response(raw, image_id=1)
    assert r.status == VisionStatus.ITEM_FAILED


def test_parse_confidence_clamped():
    raw = json.dumps({"title": "t", "description": "d", "confidence": 3.7})
    r = parse_vision_response(raw, image_id=1)
    assert r.status == VisionStatus.SUCCESS
    assert r.confidence == 1.0


def test_parse_overlong_fields_truncated():
    raw = json.dumps({"title": "x" * 100, "description": "y" * 500, "tags": ["z" * 50] * 20})
    r = parse_vision_response(raw, image_id=1)
    assert r.status == VisionStatus.SUCCESS
    assert len(r.title) <= 60
    assert len(r.description) <= 200
    assert len(r.tags) <= 8
    assert all(len(t) <= 30 for t in r.tags)


def test_parse_tags_non_list():
    raw = json.dumps({"title": "t", "description": "d", "tags": "notalist"})
    r = parse_vision_response(raw, image_id=1)
    assert r.status == VisionStatus.ITEM_FAILED


# ---------------------------------------------------------------------------
# 构造辅助与聚合
# ---------------------------------------------------------------------------
def test_vision_systemic_failure():
    out = vision_systemic_failure(VisionErrorCode.PROVIDER_ERROR, "provider down")
    assert out.systemic_failed is True
    assert out.systemic_error_code == VisionErrorCode.PROVIDER_ERROR
    assert out.total == 0


def test_build_vision_output_counts():
    items = [
        vision_item_success(1, title="a", description="desc a"),
        vision_item_success(2, title="b", description="desc b"),
        vision_item_failed(3, VisionErrorCode.TIMEOUT, "timeout"),
        vision_item_failed(4, VisionErrorCode.PARSE_ERROR, "parse"),
    ]
    out = build_vision_output(items, criteria=["red"])
    assert out.total == 4
    assert out.success_count == 2
    assert out.failed_count == 2
    assert out.systemic_failed is False
    assert out.criteria == ["red"]


def test_vision_success_helper_truncates():
    r = vision_item_success(7, title="x" * 100, description="y" * 500, tags=["z" * 40] * 10)
    assert r.status == VisionStatus.SUCCESS
    assert len(r.title) <= 60
    assert len(r.description) <= 200
    assert len(r.tags) <= 8


def test_vision_status_enum_values():
    assert VisionStatus.SUCCESS.value == "success"
    assert VisionStatus.ITEM_FAILED.value == "item_failed"
    assert VisionStatus.SYSTEMIC_FAILED.value == "systemic_failed"
    assert VisionErrorCode.TIMEOUT.value == "timeout"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 2 Stage 3 — VisionWorker 测试（Fake ImageAnalysis，零真实 API）
# ---------------------------------------------------------------------------
class _FakeAnalyzer:
    """Fake ImageAnalysis：暴露 .client 属性 + analyze_image(**kwargs)。

    outcomes: 每次调用的结果序列。元素为 dict（成功，经 Stage 1 parser）或
    Exception（失败，走 _analyze_one 异常映射）。序列耗尽后返回默认成功 dict。
    """

    def __init__(self, outcomes=(), client_ok=True):
        self.client = object() if client_ok else None
        self._outcomes = list(outcomes)
        self.calls = []

    def analyze_image(self, **kwargs):
        self.calls.append(kwargs)
        if self._outcomes:
            outcome = self._outcomes.pop(0)
        else:
            outcome = {"title": "ok", "description": "desc", "tags": ["t"],
                       "evidence": "e", "confidence": 0.9}
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _ok_dict(iid):
    return {"title": f"标题{iid}", "description": f"描述{iid}", "tags": ["t"],
            "evidence": f"证据{iid}", "confidence": 0.8}


def _vtask(depends, criteria=None, max_candidates=None, detail="low"):
    kwargs = dict(criteria=criteria or [], detail=detail)
    if max_candidates is not None:
        kwargs["max_candidates"] = max_candidates
    inp = AnalyzeVisualInput(**kwargs)
    return Task(task_id="tv", task_type=TaskType.ANALYZE_VISUAL, objective="o",
                input=inp, depends_on=depends or [])


def _vctx(*deps):
    ctx = ExecutionContext(None)
    for tid, res in deps:
        ctx.results[tid] = res
    return ctx


def _vcand(iid, filepath=None, public_url=None):
    url = public_url if public_url is not None else f"/static/images/{iid}.jpg"
    return ImageCandidate(id=iid, filepath=filepath, public_url=url)


def _vretr(*cands):
    return WorkerResult(
        task_id="t1", task_type=TaskType.RETRIEVE_IMAGES,
        status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=list(cands), query="q", total=len(cands)),
    )


# -- Candidate -----------------------------------------------------------
def test_vision_worker_normal_4_candidates_success():
    fake = _FakeAnalyzer([_ok_dict(1), _ok_dict(2), _ok_dict(3), _ok_dict(4)])
    worker = VisionWorker(analyzer=fake)
    task = _vtask(["t1"], criteria=["旅行"], max_candidates=4)
    res = worker.run(task, _vctx(("t1", _vretr(*[_vcand(i) for i in range(1, 5)]))))
    assert res.status == TaskStatus.SUCCEEDED
    out = res.output
    assert out.total == 4 and out.success_count == 4 and out.failed_count == 0
    assert out.systemic_failed is False
    assert len(fake.calls) == 4
    assert [it.image_id for it in out.items] == [1, 2, 3, 4]


def test_vision_worker_limit_cap_8():
    fake = _FakeAnalyzer()
    worker = VisionWorker(analyzer=fake)
    task = _vtask(["t1"], max_candidates=8)
    cands = [_vcand(i) for i in range(1, 11)]
    res = worker.run(task, _vctx(("t1", _vretr(*cands))))
    assert res.output.total == 8
    assert len(fake.calls) == 8
    assert [it.image_id for it in res.output.items] == list(range(1, 9))


def test_vision_worker_default_limit_4():
    fake = _FakeAnalyzer()
    worker = VisionWorker(analyzer=fake)
    task = _vtask(["t1"])
    assert task.input.max_candidates == DEFAULT_VISION_CANDIDATES
    cands = [_vcand(i) for i in range(1, 7)]
    res = worker.run(task, _vctx(("t1", _vretr(*cands))))
    assert res.output.total == 4
    assert len(fake.calls) == 4


def test_vision_worker_dedup_preserves_order():
    fake = _FakeAnalyzer()
    worker = VisionWorker(analyzer=fake)
    task = _vtask(["t1", "t2"], max_candidates=8)
    dep_a = _vretr(_vcand(5), _vcand(2), _vcand(5), _vcand(8))
    dep_b = WorkerResult(
        task_id="t2", task_type=TaskType.RETRIEVE_IMAGES,
        status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=[_vcand(2), _vcand(8)], query="q", total=2),
    )
    res = worker.run(task, _vctx(("t1", dep_a), ("t2", dep_b)))
    assert [it.image_id for it in res.output.items] == [5, 2, 8]
    assert len(fake.calls) == 3


def test_vision_worker_invalid_ids_skipped():
    fake = _FakeAnalyzer()
    worker = VisionWorker(analyzer=fake)
    task = _vtask(["t1"], max_candidates=8)
    res = worker.run(task, _vctx(("t1", _vretr(_vcand(0), _vcand(-3), _vcand(7)))))
    assert res.status == TaskStatus.SUCCEEDED
    assert [it.image_id for it in res.output.items] == [7]
    assert len(fake.calls) == 1


def test_vision_worker_empty_candidates_no_provider_call():
    fake = _FakeAnalyzer()
    worker = VisionWorker(analyzer=fake)
    task = _vtask(["t1"], max_candidates=8)
    res = worker.run(task, _vctx(("t1", _vretr())))
    assert res.status == TaskStatus.SUCCEEDED
    assert res.output.total == 0 and res.output.success_count == 0
    assert len(fake.calls) == 0

# 纯 python 运行器
# ---------------------------------------------------------------------------
# -- Success -------------------------------------------------------------
def test_vision_worker_single_success_fields():
    fake = _FakeAnalyzer([_ok_dict(3)])
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"]), _vctx(("t1", _vretr(_vcand(3)))))
    it = res.output.items[0]
    assert it.status == VisionStatus.SUCCESS
    assert it.image_id == 3
    assert it.title == "标题3" and it.description == "描述3"
    assert it.tags == ["t"] and it.evidence == "证据3"
    assert it.confidence == 0.8
    assert it.error_code is None


# -- Partial Failure -----------------------------------------------------
def test_vision_worker_partial_timeout():
    fake = _FakeAnalyzer([
        _ok_dict(1),
        ImageAnalysisError(ImageAnalysisErrorCode.TIMEOUT, "timeout"),
        _ok_dict(3),
    ])
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"], max_candidates=8),
                     _vctx(("t1", _vretr(_vcand(1), _vcand(2), _vcand(3)))))
    out = res.output
    assert out.success_count == 2 and out.failed_count == 1
    assert out.systemic_failed is False
    failed = [it for it in out.items if it.status == VisionStatus.ITEM_FAILED]
    assert len(failed) == 1 and failed[0].image_id == 2
    assert failed[0].error_code == VisionErrorCode.TIMEOUT


def test_vision_worker_partial_provider_error():
    fake = _FakeAnalyzer([
        ImageAnalysisError(ImageAnalysisErrorCode.PROVIDER_ERROR, "provider down"),
        _ok_dict(2),
    ])
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"], max_candidates=8),
                     _vctx(("t1", _vretr(_vcand(1), _vcand(2)))))
    failed = [it for it in res.output.items if it.status == VisionStatus.ITEM_FAILED]
    assert failed[0].error_code == VisionErrorCode.PROVIDER_ERROR


def test_vision_worker_partial_parse_error():
    fake = _FakeAnalyzer([
        _ok_dict(1),
        ImageAnalysisError(ImageAnalysisErrorCode.PARSE_ERROR, "parse fail"),
    ])
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"], max_candidates=8),
                     _vctx(("t1", _vretr(_vcand(1), _vcand(2)))))
    failed = [it for it in res.output.items if it.status == VisionStatus.ITEM_FAILED]
    assert failed[0].error_code == VisionErrorCode.PARSE_ERROR


# -- All Failed ----------------------------------------------------------
def test_vision_worker_all_failed_not_systemic():
    fake = _FakeAnalyzer([ImageAnalysisError(ImageAnalysisErrorCode.TIMEOUT, "t")] * 4)
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"], max_candidates=8),
                     _vctx(("t1", _vretr(*[_vcand(i) for i in range(1, 5)]))))
    out = res.output
    assert res.status == TaskStatus.SUCCEEDED
    assert out.success_count == 0 and out.failed_count == 4
    assert out.systemic_failed is False


# -- Systemic ------------------------------------------------------------
def test_vision_worker_client_unavailable_systemic():
    fake = _FakeAnalyzer(client_ok=False)
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"]), _vctx(("t1", _vretr(_vcand(1)))))
    assert res.status == TaskStatus.SUCCEEDED
    out = res.output
    assert out.systemic_failed is True
    assert out.systemic_error_code == VisionErrorCode.CLIENT_UNAVAILABLE
    assert len(fake.calls) == 0

# -- Timeout / Strict Boundary -------------------------------------------
def test_vision_worker_explicit_timeout_passed():
    fake = _FakeAnalyzer([_ok_dict(1)])
    worker = VisionWorker(analyzer=fake, item_timeout=12.5)
    res = worker.run(_vtask(["t1"]), _vctx(("t1", _vretr(_vcand(1)))))
    assert fake.calls[0]["timeout"] == 12.5


def test_vision_worker_default_timeout_is_constant():
    fake = _FakeAnalyzer([_ok_dict(1)])
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"]), _vctx(("t1", _vretr(_vcand(1)))))
    assert fake.calls[0]["timeout"] == VISION_ITEM_TIMEOUT_SECONDS


def test_vision_worker_strict_boundary_enabled():
    fake = _FakeAnalyzer([_ok_dict(1)])
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"]), _vctx(("t1", _vretr(_vcand(1)))))
    assert fake.calls[0]["raise_on_error"] is True


# -- Prompt --------------------------------------------------------------
def test_vision_worker_prompt_criteria_aware():
    fake = _FakeAnalyzer([_ok_dict(1)])
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"], criteria=["旅行记录", "风景"]),
                     _vctx(("t1", _vretr(_vcand(1)))))
    prompt = fake.calls[0]["prompt"]
    assert "旅行记录" in prompt and "风景" in prompt
    assert "tool" not in prompt.lower()
    assert "history" not in prompt.lower()
    assert "registry" not in prompt.lower()


def test_vision_worker_prompt_image_grounded_no_history():
    fake = _FakeAnalyzer([_ok_dict(1)])
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"], criteria=["人物"]),
                     _vctx(("t1", _vretr(_vcand(1)))))
    prompt = fake.calls[0]["prompt"]
    assert "只能描述图片中实际可观察到的内容" in prompt
    assert "unknown" in prompt
    assert "JSON" in prompt
    assert "history" not in prompt.lower()


def test_build_vision_prompt_empty_criteria():
    p = build_vision_prompt()
    assert "无特定关注点" in p

# -- Safety --------------------------------------------------------------
def test_vision_worker_no_retry():
    fake = _FakeAnalyzer([ImageAnalysisError(ImageAnalysisErrorCode.TIMEOUT, "t")] * 3)
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"], max_candidates=8),
                     _vctx(("t1", _vretr(*[_vcand(i) for i in range(1, 4)]))))
    assert len(fake.calls) == 3  # 每张恰好一次，无 retry


def test_vision_worker_no_sensitive_leak():
    path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
    try:
        with open(path, "wb") as f:
            f.write(b"fake-image")
        fake = _FakeAnalyzer([_ok_dict(1)])
        worker = VisionWorker(analyzer=fake)
        res = worker.run(_vtask(["t1"]),
                         _vctx(("t1", _vretr(_vcand(1, filepath=path, public_url=None)))))
        text = json.dumps(res.output.model_dump(), ensure_ascii=False)
        assert path not in text
        assert "base64" not in text
        assert "filepath" not in text
        assert "public_url" not in text
    finally:
        os.unlink(path)


def test_vision_worker_single_model_no_switch():
    fake = _FakeAnalyzer([_ok_dict(1), _ok_dict(2)])
    worker = VisionWorker(analyzer=fake, model="m-vision-1")
    res = worker.run(_vtask(["t1"], max_candidates=8),
                     _vctx(("t1", _vretr(_vcand(1), _vcand(2)))))
    assert len({c["model"] for c in fake.calls}) == 1
    assert all(c["model"] == "m-vision-1" for c in fake.calls)


# -- Image Source --------------------------------------------------------
def test_vision_worker_prefers_local_filepath():
    path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
    try:
        with open(path, "wb") as f:
            f.write(b"x")
        fake = _FakeAnalyzer([_ok_dict(1)])
        worker = VisionWorker(analyzer=fake)
        res = worker.run(_vtask(["t1"]), _vctx(("t1", _vretr(_vcand(1, filepath=path)))))
        assert fake.calls[0]["local_image_path"] == path
        assert "image_url" not in fake.calls[0]
    finally:
        os.unlink(path)


def test_vision_worker_falls_back_to_url_when_filepath_missing():
    fake = _FakeAnalyzer([_ok_dict(1)])
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"]),
                     _vctx(("t1", _vretr(_vcand(1, filepath="Z:/nope.jpg")))))
    assert "image_url" in fake.calls[0]
    assert "local_image_path" not in fake.calls[0]


def test_vision_worker_no_source_item_failed():
    fake = _FakeAnalyzer()
    worker = VisionWorker(analyzer=fake)
    cand = ImageCandidate(id=9, filepath=None, public_url=None)
    res = worker.run(_vtask(["t1"]), _vctx(("t1", _vretr(cand))))
    it = res.output.items[0]
    assert it.status == VisionStatus.ITEM_FAILED
    assert it.error_code == VisionErrorCode.INVALID_IMAGE_ID
    assert len(fake.calls) == 0


# -- Error Mapping -------------------------------------------------------
def test_vision_worker_unsupported_mime_mapping():
    fake = _FakeAnalyzer(
        [ImageAnalysisError(ImageAnalysisErrorCode.UNSUPPORTED_MIME, "unsupported mime")]
    )
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"]), _vctx(("t1", _vretr(_vcand(1)))))
    it = res.output.items[0]
    assert it.status == VisionStatus.ITEM_FAILED
    assert it.error_code == VisionErrorCode.UNSUPPORTED_MIME


def test_vision_worker_unknown_exception_mapped():
    fake = _FakeAnalyzer([RuntimeError("boom")])
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"]), _vctx(("t1", _vretr(_vcand(1)))))
    it = res.output.items[0]
    assert it.status == VisionStatus.ITEM_FAILED
    assert it.error_code == VisionErrorCode.UNKNOWN_ERROR


# -- Input Type（Task 模型在构造时即强制 input 类型匹配，该防御路径不可达；见 models.Task）


# -- Evidence Budget -----------------------------------------------------
def test_cap_evidence_total_budget():
    items = [vision_item_success(i, title=f"t{i}", description="d",
                                 evidence="x" * 300) for i in range(1, 9)]
    capped = cap_evidence_total(items, budget=100)
    total = sum(len(it.evidence) for it in capped if it.status == VisionStatus.SUCCESS)
    assert total <= 100


def test_vision_worker_evidence_budget_constant_respected():
    outcomes = [
        {"title": f"t{i}", "description": "d", "tags": ["t"],
         "evidence": "e" * 300, "confidence": 0.9} for i in range(1, 9)
    ]
    fake = _FakeAnalyzer(outcomes)
    worker = VisionWorker(analyzer=fake)
    res = worker.run(_vtask(["t1"], max_candidates=8),
                     _vctx(("t1", _vretr(*[_vcand(i) for i in range(1, 9)]))))
    total = sum(len(it.evidence) for it in res.output.items if it.status == VisionStatus.SUCCESS)
    assert total <= MAX_VISION_TOTAL_EVIDENCE_CHARS


# -- status_summary ------------------------------------------------------
def test_status_summary_vision():
    out = build_vision_output([vision_item_success(1, title="a", description="d")])
    res = WorkerResult(task_id="tv", task_type=TaskType.ANALYZE_VISUAL,
                       status=TaskStatus.SUCCEEDED, output=out)
    assert "视觉分析 1/1" in status_summary(res)
    sys_res = WorkerResult(
        task_id="tv", task_type=TaskType.ANALYZE_VISUAL, status=TaskStatus.SUCCEEDED,
        output=vision_systemic_failure(VisionErrorCode.CLIENT_UNAVAILABLE, "no client"),
    )
    assert "systemic" in status_summary(sys_res)

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
