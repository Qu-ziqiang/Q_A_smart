"""检索 Agent 新任务：expand / scope / labels / similar（Fake，不碰真实库）。"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.models import (
    ExpandQueryInput,
    FilterByLabelsInput,
    RetrieveImagesInput,
    RetrieveSimilarInput,
    RewriteQueryInput,
    ScopeDatasetInput,
    Task,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.observation import (
    ReplanAction,
    RoundObservation,
    RetrievalSlice,
    assert_replan_changed,
    plan_has_expand,
)
from backend.ai_func.recommendation.orchestration.workers import (
    ExecutionContext,
    ExpandQueryWorker,
    FilterByLabelsWorker,
    RetrievalWorker,
    RetrieveSimilarWorker,
    RewriteWorker,
    ScopeDatasetOutput,
    ScopeDatasetWorker,
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


def test_rewrite_extracts_dataset_hint():
    res = RewriteWorker().run(
        _task("t0", TaskType.REWRITE_QUERY, RewriteQueryInput(query="在 VOC 里找猫")),
        _ctx(),
    )
    assert res.status == TaskStatus.SUCCEEDED
    assert "VOC" in (res.output.dataset_hint or "").upper() or res.output.dataset_hint.lower() == "voc"


def test_expand_query_uses_injected_llm():
    worker = ExpandQueryWorker(llm_fn=lambda q, vocab: "person child")
    res = worker.run(
        _task("e1", TaskType.EXPAND_QUERY, ExpandQueryInput(query="男孩")),
        _ctx(),
    )
    assert res.status == TaskStatus.SUCCEEDED
    assert res.output.query == "person child"
    assert res.output.source_query == "男孩"


def test_retrieve_prefers_expand_over_rewrite():
    calls = {}

    def fake_search(query, targets, filters, limit):
        calls["query"] = query
        calls["filters"] = filters
        return []

    rewrite = RewriteWorker().run(
        _task("r0", TaskType.REWRITE_QUERY, RewriteQueryInput(query="请找猫谢谢")),
        _ctx(),
    )
    expand = ExpandQueryWorker(llm_fn=lambda q, vocab: "cat feline").run(
        _task("e0", TaskType.EXPAND_QUERY, ExpandQueryInput(query="猫")),
        _ctx(),
    )
    scope = WorkerResult(
        task_id="s0",
        task_type=TaskType.SCOPE_DATASET,
        status=TaskStatus.SUCCEEDED,
        output=ScopeDatasetOutput(dataset_id=7, dataset_name="voc2010", label_vocab=["cat"]),
    )
    RetrievalWorker(search_tool=fake_search, rerank_enabled=False).run(
        _task(
            "t1", TaskType.RETRIEVE_IMAGES,
            RetrieveImagesInput(query="原句", limit=5),
            depends=["r0", "e0", "s0"],
        ),
        _ctx(("r0", rewrite), ("e0", expand), ("s0", scope)),
    )
    assert calls["query"] == "cat feline"
    assert calls["filters"].get("dataset_id") == 7


def test_filter_by_labels_and_match():
    seen = {}

    def fake_labels(labels, dataset_id=None, match="all", limit=40):
        seen["args"] = (list(labels), dataset_id, match, limit)
        return [
            {"id": 1, "score": 1.0, "title": "a", "tags": ["bear", "chair"],
             "metadata": {}, "width": 10, "height": 10, "file_type": "jpg",
             "public_url": "/a.jpg", "filepath": "", "description": "", "created_at": ""},
        ]

    scope = WorkerResult(
        task_id="s0",
        task_type=TaskType.SCOPE_DATASET,
        status=TaskStatus.SUCCEEDED,
        output=ScopeDatasetOutput(
            dataset_id=3, dataset_name="voc2010",
            label_vocab=["bear", "chair"],
        ),
    )
    res = FilterByLabelsWorker(search_tool=fake_labels).run(
        _task(
            "f1", TaskType.FILTER_BY_LABELS,
            FilterByLabelsInput(labels=["bear", "chair"], match="all", limit=10),
            depends=["s0"],
        ),
        _ctx(("s0", scope)),
    )
    assert res.status == TaskStatus.SUCCEEDED
    assert seen["args"] == (["bear", "chair"], 3, "all", 10)
    assert [c.id for c in res.output.candidates] == [1]


def test_retrieve_similar_passes_dataset():
    seen = {}

    def fake_sim(image_id, dataset_id=None, limit=20):
        seen["args"] = (image_id, dataset_id, limit)
        return [
            {"id": 9, "score": 0.8, "title": "x", "tags": [], "metadata": {},
             "width": 1, "height": 1, "file_type": "jpg", "public_url": "",
             "filepath": "", "description": "", "created_at": ""},
        ]

    scope = WorkerResult(
        task_id="s0",
        task_type=TaskType.SCOPE_DATASET,
        status=TaskStatus.SUCCEEDED,
        output=ScopeDatasetOutput(dataset_id=2, dataset_name="voc2010"),
    )
    res = RetrieveSimilarWorker(search_tool=fake_sim).run(
        _task(
            "sim", TaskType.RETRIEVE_SIMILAR,
            RetrieveSimilarInput(image_id=5, limit=8),
            depends=["s0"],
        ),
        _ctx(("s0", scope)),
    )
    assert res.status == TaskStatus.SUCCEEDED
    assert seen["args"] == (5, 2, 8)
    assert res.output.candidates[0].id == 9


def test_replan_accepts_expand_query_task():
    from backend.ai_func.recommendation.orchestration.models import (
        TaskPlan,
        SynthesizeInput,
    )

    obs = RoundObservation(
        iteration=1,
        original_goal="找男孩",
        suggested_action=ReplanAction.RETRY_REWRITE,
        retrieval=RetrievalSlice(query="男孩", candidate_count=0, empty=True),
    )
    plan = TaskPlan(
        plan_id="p1",
        goal="找男孩",
        synthesis_task_id="s1",
        tasks=[
            _task("e1", TaskType.EXPAND_QUERY, ExpandQueryInput(query="男孩")),
            _task(
                "r1", TaskType.RETRIEVE_IMAGES,
                RetrieveImagesInput(query="男孩", limit=20),
                depends=["e1"],
            ),
            _task("s1", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["r1"]),
        ],
    )
    assert plan_has_expand(plan)
    assert assert_replan_changed(obs, plan) is None


def test_scope_worker_resolves_voc_name(monkeypatch):
    class _FakeRepo:
        def ensure_schema_seed(self):
            return None

        def get_by_id(self, dataset_id):
            return None

        def get_by_name(self, name):
            if name.lower() in ("voc", "voc2010"):
                return {"id": 11, "name": "voc2010", "label_vocab": ["cat"]}
            return None

        def list_datasets(self):
            return [{"name": "voc2010"}]

    import backend.ai_func.recommendation.orchestration.workers as wmod

    monkeypatch.setattr(
        "backend.db_func.repositories.datasets.DatasetRepository",
        _FakeRepo,
    )
    # ScopeDatasetWorker imports DatasetRepository inside run from relative path
    monkeypatch.setattr(
        wmod,
        "ScopeDatasetWorker",
        ScopeDatasetWorker,
    )

    # Patch the import target used inside run()
    import backend.db_func.repositories.datasets as dsmod
    monkeypatch.setattr(dsmod, "DatasetRepository", _FakeRepo)

    res = ScopeDatasetWorker().run(
        _task("s0", TaskType.SCOPE_DATASET, ScopeDatasetInput(dataset_name="VOC")),
        _ctx(),
    )
    assert res.status == TaskStatus.SUCCEEDED
    assert res.output.dataset_id == 11
    assert res.output.dataset_name == "voc2010"
