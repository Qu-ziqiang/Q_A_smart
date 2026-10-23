"""窄视觉任务：读字、列物体、计数、主体、空间关系。Fake analyzer，不调模型。"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.models import (
    CountNamedObjectInput,
    JudgeMainSubjectInput,
    ListObjectsInput,
    ReadVisibleTextInput,
    RetrieveImagesInput,
    Task,
    TaskStatus,
    TaskType,
    VerifySpatialInput,
)
from backend.ai_func.recommendation.orchestration.workers import (
    ExecutionContext,
    FocusedVisionWorker,
    ImageCandidate,
    RetrievalOutput,
    WorkerResult,
)


class _FakeAnalyzer:
    client = object()

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def analyze_image(self, **kwargs):
        self.prompts.append(kwargs.get("prompt") or "")
        return dict(self.payload)


def _cand(cid, url="/a.jpg"):
    return ImageCandidate(id=cid, title="t", public_url=url, score=0.5)


def _ctx(cands):
    ctx = ExecutionContext(None)
    ctx.results["r1"] = WorkerResult(
        task_id="r1",
        task_type=TaskType.RETRIEVE_IMAGES,
        status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=cands, query="q", total=len(cands)),
    )
    return ctx


def _task(tid, ttype, inp):
    return Task(task_id=tid, task_type=ttype, objective="o", input=inp, depends_on=["r1"])


def test_read_visible_text_keeps_candidate_and_ocr():
    analyzer = _FakeAnalyzer({"text": "咖啡店", "confidence": 0.8, "evidence": "招牌"})
    worker = FocusedVisionWorker(TaskType.READ_VISIBLE_TEXT, analyzer=analyzer)
    res = worker.run(
        _task("v1", TaskType.READ_VISIBLE_TEXT, ReadVisibleTextInput(max_candidates=4)),
        _ctx([_cand(3)]),
    )
    assert res.status == TaskStatus.SUCCEEDED
    assert res.output.candidates[0].id == 3
    assert res.output.candidates[0].vision.ocr_text == "咖啡店"
    assert "文字" in analyzer.prompts[0]


def test_list_objects_and_count():
    analyzer = _FakeAnalyzer({"objects": ["bear", "chair"], "confidence": 0.7, "evidence": "可见"})
    res = FocusedVisionWorker(TaskType.LIST_OBJECTS, analyzer=analyzer).run(
        _task("v2", TaskType.LIST_OBJECTS, ListObjectsInput()),
        _ctx([_cand(1)]),
    )
    assert res.output.items[0].objects == ["bear", "chair"]

    counter = _FakeAnalyzer({"count": 2, "confidence": 0.6, "evidence": "两只"})
    res = FocusedVisionWorker(TaskType.COUNT_NAMED_OBJECT, analyzer=counter).run(
        _task("v3", TaskType.COUNT_NAMED_OBJECT, CountNamedObjectInput(object_name="鸟")),
        _ctx([_cand(1)]),
    )
    assert res.output.items[0].count == 2
    assert len(res.output.candidates) == 1


def test_main_subject_and_spatial_drop_failures():
    analyzer = _FakeAnalyzer({"passed": False, "subject": "chair", "confidence": 0.9, "evidence": "主体是椅子"})
    res = FocusedVisionWorker(TaskType.JUDGE_MAIN_SUBJECT, analyzer=analyzer).run(
        _task("v4", TaskType.JUDGE_MAIN_SUBJECT, JudgeMainSubjectInput(subject="bear")),
        _ctx([_cand(1), _cand(2, "/b.jpg")]),
    )
    assert res.output.inspected_count == 2
    assert res.output.candidates == []

    spatial = _FakeAnalyzer({"passed": True, "confidence": 0.85, "evidence": "熊在桌面上"})
    res = FocusedVisionWorker(TaskType.VERIFY_SPATIAL, analyzer=spatial).run(
        _task("v5", TaskType.VERIFY_SPATIAL, VerifySpatialInput(relation="熊在桌子上")),
        _ctx([_cand(8)]),
    )
    assert [c.id for c in res.output.candidates] == [8]
    assert res.output.items[0].passed is True


def test_empty_focus_fails():
    res = FocusedVisionWorker(TaskType.JUDGE_MAIN_SUBJECT).run(
        _task("v6", TaskType.JUDGE_MAIN_SUBJECT, JudgeMainSubjectInput(subject="  ")),
        _ctx([_cand(1)]),
    )
    assert res.status == TaskStatus.FAILED
