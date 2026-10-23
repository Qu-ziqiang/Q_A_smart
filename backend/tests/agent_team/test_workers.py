"""Phase 1-B — Workers 单元测试（Fake tool / Fake repo，不调真实 API）。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_workers.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.models import (
    FilterMetadataInput,
    GroupImagesInput,
    MetadataPredicate,
    RetrieveImagesInput,
    RewriteQueryInput,
    Task,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.workers import (
    ExecutionContext,
    GroupImagesWorker,
    GroupOutput,
    ImageCandidate,
    MetadataWorker,
    MetadataOutput,
    RetrievalOutput,
    RetrievalWorker,
    RewriteOutput,
    RewriteWorker,
    WorkerResult,
)


def _task(tid, ttype, input_model, depends=None):
    return Task(
        task_id=tid, task_type=ttype, objective="o",
        input=input_model, depends_on=depends or [],
    )


def _ctx(*deps):
    ctx = ExecutionContext(None)
    for tid, res in deps:
        ctx.results[tid] = res
    return ctx


def _cand(id, score=0.5, tags=(), width=None, height=None, metadata=None, file_type="jpg", created_at=None):
    return ImageCandidate(
        id=id, score=score, title=f"t{id}", tags=list(tags),
        width=width, height=height, metadata=metadata or {},
        file_type=file_type, created_at=created_at,
        public_url=f"/static/images/{id}.jpg",
    )


def _retr_result(*cands):
    return WorkerResult(
        task_id="t1", task_type=TaskType.RETRIEVE_IMAGES,
        status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=list(cands), query="q", total=len(cands)),
    )


# ---------------------------------------------------------------------------
# RetrievalWorker
# ---------------------------------------------------------------------------
def test_retrieval_worker_uses_typed_input_and_normalizes():
    calls = {}

    def fake_search(query, targets, filters, limit):
        calls["args"] = (query, targets, filters, limit)
        return [
            {"id": 1, "score": 0.9, "title": "A", "tags": '["cat"]', "metadata": '{"location": "北京"}',
             "width": 100, "height": 200, "file_type": "jpg", "public_url": "/static/images/a.jpg",
             "filepath": "data/images/a.jpg", "description": "desc", "created_at": "2025-01-01T00:00:00"},
        ]

    worker = RetrievalWorker(search_tool=fake_search)
    task = _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="猫", limit=10))
    res = worker.run(task, _ctx())
    assert res.status == TaskStatus.SUCCEEDED
    assert calls["args"] == ("猫", ["title", "description", "image"], {}, 10)
    out = res.output
    assert isinstance(out, RetrievalOutput)
    assert out.total == 1
    assert out.candidates[0].id == 1
    assert out.candidates[0].tags == ["cat"]
    assert out.candidates[0].metadata == {"location": "北京"}


def test_rewrite_task_then_retrieve_uses_it():
    calls = {}

    def fake_search(query, targets, filters, limit):
        calls["args"] = (query, filters, limit)
        return [
            {"id": 1, "score": 0.9, "title": "猫", "tags": [], "metadata": {},
             "width": 400, "height": 200, "file_type": "jpg", "public_url": "/a.jpg",
             "filepath": "", "description": "", "created_at": "2024-01-01"},
            {"id": 2, "score": 0.8, "title": "竖猫", "tags": [], "metadata": {},
             "width": 200, "height": 400, "file_type": "jpg", "public_url": "/b.jpg",
             "filepath": "", "description": "", "created_at": "2024-01-01"},
        ]

    rewrite = RewriteWorker().run(
        _task("t0", TaskType.REWRITE_QUERY, RewriteQueryInput(query="请帮我找横向 JPG 猫的照片谢谢")),
        _ctx(),
    )
    assert rewrite.status == TaskStatus.SUCCEEDED
    assert isinstance(rewrite.output, RewriteOutput)
    assert "谢谢" not in rewrite.output.query
    assert rewrite.output.filters.get("file_type") == "jpg"
    assert rewrite.output.orientation == "landscape"

    bare = RetrievalWorker(search_tool=fake_search, rerank_enabled=False).run(
        _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="请帮我找横向 JPG 猫的照片谢谢", limit=10)),
        _ctx(),
    )
    assert bare.output.query == "请帮我找横向 JPG 猫的照片谢谢"

    worker = RetrievalWorker(search_tool=fake_search, rerank_enabled=False)
    task = _task(
        "t1", TaskType.RETRIEVE_IMAGES,
        RetrieveImagesInput(query="原句", limit=10),
        depends=["t0"],
    )
    res = worker.run(task, _ctx(("t0", rewrite)))
    assert res.status == TaskStatus.SUCCEEDED
    query, filters, limit = calls["args"]
    assert query == rewrite.output.query
    assert "猫" in query
    assert filters.get("file_type") == "jpg"
    assert limit == 10
    assert [c.id for c in res.output.candidates] == [1]


def test_retrieval_worker_zero_results_is_succeeded():
    worker = RetrievalWorker(search_tool=lambda **kw: [])
    task = _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="x"))
    res = worker.run(task, _ctx())
    assert res.status == TaskStatus.SUCCEEDED
    assert res.output.total == 0


def test_retrieval_worker_error_returns_failed():
    def boom(**kw):
        raise RuntimeError("search down")

    worker = RetrievalWorker(search_tool=boom)
    task = _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="x"))
    res = worker.run(task, _ctx())
    assert res.status == TaskStatus.FAILED
    assert res.error.code == "RETRIEVAL_ERROR"


# ---------------------------------------------------------------------------
# MetadataWorker
# ---------------------------------------------------------------------------
def test_metadata_worker_filters_by_orientation():
    cands = [
        _cand(1, width=100, height=50),   # landscape
        _cand(2, width=50, height=100),   # portrait
        _cand(3, width=60, height=60),    # square
    ]
    task = _task(
        "t2", TaskType.FILTER_METADATA,
        FilterMetadataInput(predicates=[MetadataPredicate(field="orientation", operator="eq", value="landscape")]),
        depends=["t1"],
    )
    res = MetadataWorker().run(task, _ctx(("t1", _retr_result(*cands))))
    assert res.status == TaskStatus.SUCCEEDED
    out = res.output
    assert [c.id for c in out.candidates] == [1]
    assert out.rejected_ids == [2, 3]


def test_metadata_worker_tags_contains():
    cands = [_cand(1, tags=["sunset"]), _cand(2, tags=["cat"])]
    task = _task(
        "t2", TaskType.FILTER_METADATA,
        FilterMetadataInput(predicates=[MetadataPredicate(field="tags", operator="contains", value="sunset")]),
        depends=["t1"],
    )
    res = MetadataWorker().run(task, _ctx(("t1", _retr_result(*cands))))
    assert [c.id for c in res.output.candidates] == [1]


def test_metadata_worker_gte_width():
    cands = [_cand(1, width=800), _cand(2, width=400)]
    task = _task(
        "t2", TaskType.FILTER_METADATA,
        FilterMetadataInput(predicates=[MetadataPredicate(field="width", operator="gte", value=500)]),
        depends=["t1"],
    )
    res = MetadataWorker().run(task, _ctx(("t1", _retr_result(*cands))))
    assert [c.id for c in res.output.candidates] == [1]


def test_metadata_worker_missing_field_excluded():
    cands = [_cand(1, width=None, height=None), _cand(2, width=100, height=50)]
    task = _task(
        "t2", TaskType.FILTER_METADATA,
        FilterMetadataInput(predicates=[MetadataPredicate(field="orientation", operator="eq", value="landscape")]),
        depends=["t1"],
    )
    res = MetadataWorker().run(task, _ctx(("t1", _retr_result(*cands))))
    assert [c.id for c in res.output.candidates] == [2]


def test_metadata_worker_dedups_across_deps():
    dep_a = _retr_result(_cand(1), _cand(2))
    dep_b = WorkerResult(
        task_id="t1b", task_type=TaskType.RETRIEVE_IMAGES,
        status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=[_cand(2), _cand(3)], query="q", total=2),
    )
    task = _task("t2", TaskType.FILTER_METADATA, FilterMetadataInput(), depends=["t1", "t1b"])
    res = MetadataWorker().run(task, _ctx(("t1", dep_a), ("t1b", dep_b)))
    assert [c.id for c in res.output.candidates] == [1, 2, 3]


# ---------------------------------------------------------------------------
# GroupImagesWorker
# ---------------------------------------------------------------------------
def test_group_by_location():
    cands = [
        _cand(1, metadata={"location": "北京"}),
        _cand(2, metadata={"location": "上海"}),
        _cand(3, metadata={"location": "北京"}),
    ]
    task = _task("t3", TaskType.GROUP_IMAGES, GroupImagesInput(group_by="location"), depends=["t1"])
    res = GroupImagesWorker().run(task, _ctx(("t1", _retr_result(*cands))))
    assert res.status == TaskStatus.SUCCEEDED
    out = res.output
    assert isinstance(out, GroupOutput)
    keys = [g.key for g in out.groups]
    assert keys == ["北京", "上海"]
    assert [c.id for c in out.groups[0].candidates] == [1, 3]
    assert out.groups[0].image_ids == [1, 3]


def test_group_by_orientation_derived():
    cands = [_cand(1, width=100, height=50), _cand(2, width=50, height=100)]
    task = _task("t3", TaskType.GROUP_IMAGES, GroupImagesInput(group_by="orientation"), depends=["t1"])
    res = GroupImagesWorker().run(task, _ctx(("t1", _retr_result(*cands))))
    keys = [g.key for g in res.output.groups]
    assert keys == ["landscape", "portrait"]


def test_group_unknown_when_field_missing():
    cands = [_cand(1), _cand(2)]
    task = _task(
        "t3", TaskType.GROUP_IMAGES,
        GroupImagesInput(group_by="location", unknown_group_name="未分类"),
        depends=["t1"],
    )
    res = GroupImagesWorker().run(task, _ctx(("t1", _retr_result(*cands))))
    assert [g.key for g in res.output.groups] == ["未分类"]
    assert len(res.output.groups[0].image_ids) == 2


def test_group_by_year_from_created_at():
    cands = [_cand(1, created_at="2024-05-01T10:00:00"), _cand(2, created_at="2025-01-02T08:00:00")]
    task = _task("t3", TaskType.GROUP_IMAGES, GroupImagesInput(group_by="year"), depends=["t1"])
    res = GroupImagesWorker().run(task, _ctx(("t1", _retr_result(*cands))))
    assert [g.key for g in res.output.groups] == ["2024", "2025"]


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
