"""Task 能力扩充：新 TaskType 的 Validator / Executor / Specialist 组合测试。

覆盖用户要求的 6 类 DAG，以及 filter≠select、id 去重边界、旧路径回归。
全部 Fake search / Fake analyzer，零真实 API / DB。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_task_expansion.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    DAGExecutor,
    PlanValidator,
    TaskDraft,
    TaskPlanDraft,
    TaskStatus,
    TaskType,
    build_default_agent_registry,
    default_workers,
)
from backend.ai_func.recommendation.orchestration.models import (
    REGISTERED_TASK_TYPES,
    registered_task_types,
)
from backend.ai_func.recommendation.orchestration.workers import (
    DedupOutput,
    DedupWorker,
    DeterministicSynthesisWorker,
    GroupImagesWorker,
    ImageCandidate,
    MetadataWorker,
    RerankOutput,
    RerankWorker,
    RetrievalOutput,
    RetrievalWorker,
    SelectOutput,
    SelectWorker,
    SortOutput,
    SortWorker,
    VerifyTaskOutput,
    VerifyVisualWorker,
    WorkerResult,
)


def _task(tid, ttype, objective="o", input=None, depends=None):
    return TaskDraft(
        task_id=tid,
        task_type=ttype,
        objective=objective,
        input=input or {},
        depends_on=depends or [],
    )


def _draft(synth_id, tasks, goal="g"):
    return TaskPlanDraft(goal=goal, synthesis_task_id=synth_id, tasks=tasks)


def _rows():
    return [
        {"id": 1, "score": 0.91, "title": "北京街景", "tags": ["city"],
         "metadata": {"location": "北京"}, "width": 200, "height": 100,
         "file_type": "jpg", "public_url": "http://example.test/1.jpg",
         "filepath": "", "description": "街", "created_at": "2024-01-01T00:00:00"},
        {"id": 2, "score": 0.82, "title": "上海夜景", "tags": ["city"],
         "metadata": {"location": "上海"}, "width": 80, "height": 160,
         "file_type": "jpg", "public_url": "http://example.test/2.jpg",
         "filepath": "", "description": "夜", "created_at": "2023-06-01T00:00:00"},
        {"id": 3, "score": 0.71, "title": "北京公园", "tags": ["park"],
         "metadata": {"location": "北京"}, "width": 120, "height": 120,
         "file_type": "png", "public_url": "http://example.test/3.jpg",
         "filepath": "", "description": "园", "created_at": "2025-03-01T00:00:00"},
        {"id": 4, "score": 0.60, "title": "杭州西湖", "tags": ["lake"],
         "metadata": {"location": "杭州"}, "width": 300, "height": 150,
         "file_type": "jpg", "public_url": "http://example.test/4.jpg",
         "filepath": "", "description": "湖", "created_at": "2022-08-01T00:00:00"},
    ]


def _search(rows=None):
    payload = list(rows if rows is not None else _rows())

    def fn(query, targets, filters, limit):
        return payload[:limit]

    return fn


class _DupRetrieve:
    """故意返回重复 id，用于证明 deduplicate_images 按 id 去重。"""

    task_type = TaskType.RETRIEVE_IMAGES

    def run(self, task, context):
        base = [
            ImageCandidate(id=1, score=0.9, title="a", public_url="http://example.test/1.jpg",
                           file_type="jpg", metadata={"location": "北京"}, width=100, height=50),
            ImageCandidate(id=1, score=0.8, title="a-dup", public_url="http://example.test/1.jpg",
                           file_type="jpg", metadata={"location": "北京"}, width=100, height=50),
            ImageCandidate(id=2, score=0.7, title="b", public_url="http://example.test/2.jpg",
                           file_type="jpg", metadata={"location": "上海"}, width=50, height=100),
        ]
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED,
            output=RetrievalOutput(candidates=base, query="q", total=3),
        )


class _FakeVerifier:
    """按 URL 中的 id 决定是否通过。1/3 通过，其余不通过。"""

    def analyze_image(self, **kwargs):
        url = kwargs.get("image_url") or kwargs.get("local_image_path") or ""
        cid = 0
        for n in (1, 2, 3, 4):
            if f"/{n}.jpg" in url:
                cid = n
                break
        passed = cid in (1, 3)
        return {
            "passed": passed,
            "evidence": "可见自行车" if passed else "未见目标",
            "confidence": 0.88 if passed else 0.21,
            "tags": ["bicycle"] if passed else ["other"],
            "title": "核验",
            "description": "条件判断",
        }


def _workers(*, search=None, verifier=None, retrieve=None, rerank_enabled=False):
    w = {
        TaskType.RETRIEVE_IMAGES: retrieve or RetrievalWorker(
            search_tool=search or _search(), rerank_enabled=rerank_enabled,
        ),
        TaskType.RERANK_CANDIDATES: RerankWorker(),
        TaskType.FILTER_METADATA: MetadataWorker(),
        TaskType.DEDUPLICATE_IMAGES: DedupWorker(),
        TaskType.GROUP_IMAGES: GroupImagesWorker(),
        TaskType.SORT_IMAGES: SortWorker(),
        TaskType.SELECT_IMAGES: SelectWorker(),
        TaskType.SYNTHESIZE: DeterministicSynthesisWorker(),
    }
    if verifier is not None:
        w[TaskType.VERIFY_VISUAL_CONDITION] = VerifyVisualWorker(analyzer=verifier)
    return w


def _run(draft, workers, vision=False):
    vres = PlanValidator().validate(draft, vision_available_flag=vision)
    assert vres.valid, vres.errors
    ex = DAGExecutor(workers=workers)
    events = list(ex.execute_stream(vres.plan))
    assert all(
        r.status == TaskStatus.SUCCEEDED
        for r in ex.context.results.values()
    ), {k: (r.status, getattr(r.error, "code", None)) for k, r in ex.context.results.items()}
    return vres.plan, ex.context.results, events


def _ids(result):
    out = result.output
    if hasattr(out, "selected_ids") and out.selected_ids:
        return list(out.selected_ids)
    if hasattr(out, "image_ids") and out.image_ids:
        return list(out.image_ids)
    if hasattr(out, "ordered_ids") and out.ordered_ids:
        return list(out.ordered_ids)
    return [c.id for c in getattr(out, "candidates", [])]


# ---------------------------------------------------------------------------
# Case 1: retrieve → rerank → synthesize
# ---------------------------------------------------------------------------
def test_case1_retrieve_rerank_synthesize():
    def reverse_rerank(query, cands):
        return list(reversed(cands))

    workers = _workers()
    workers[TaskType.RERANK_CANDIDATES] = RerankWorker(reranker=reverse_rerank)
    draft = _draft("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "街景", "limit": 4}),
        _task("t2", TaskType.RERANK_CANDIDATES, input={"provider": "r1"}, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t2"]),
    ])
    plan, results, _ = _run(draft, workers)
    assert [t.task_type for t in plan.tasks] == [
        TaskType.RETRIEVE_IMAGES, TaskType.RERANK_CANDIDATES, TaskType.SYNTHESIZE,
    ]
    retr_ids = _ids(results["t1"])
    rank_ids = _ids(results["t2"])
    assert set(rank_ids) == set(retr_ids)
    assert rank_ids == list(reversed(retr_ids))
    assert isinstance(results["t2"].output, RerankOutput)
    assert results["t2"].output.total == len(retr_ids)
    assert results["t3"].output.image_ids


# ---------------------------------------------------------------------------
# Case 2: retrieve → filter → dedup → synthesize
# ---------------------------------------------------------------------------
def test_case2_retrieve_filter_dedup_synthesize():
    # filter 内部已按 id 去重，因此本链的 dedup 是幂等收口；重复 id 的真实删除见 case 6。
    workers = _workers()
    draft = _draft("t4", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q", "limit": 4}),
        _task("t2", TaskType.FILTER_METADATA, input={
            "predicates": [{"field": "file_type", "operator": "eq", "value": "jpg"}],
        }, depends=["t1"]),
        _task("t3", TaskType.DEDUPLICATE_IMAGES, input={}, depends=["t2"]),
        _task("t4", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t3"]),
    ])
    plan, results, _ = _run(draft, workers)
    assert [t.task_type.value for t in plan.tasks] == [
        "retrieve_images", "filter_metadata", "deduplicate_images", "synthesize",
    ]
    assert [c.id for c in results["t2"].output.candidates] == [1, 2, 4]
    dedup = results["t3"].output
    assert isinstance(dedup, DedupOutput)
    assert dedup.removed_count == 0
    assert [c.id for c in dedup.candidates] == [1, 2, 4]


# ---------------------------------------------------------------------------
# Case 3: retrieve → group → select → synthesize
# ---------------------------------------------------------------------------
def test_case3_retrieve_group_select_synthesize():
    workers = _workers()
    draft = _draft("t4", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "旅行", "limit": 4}),
        _task("t2", TaskType.GROUP_IMAGES, input={"group_by": "location"}, depends=["t1"]),
        _task("t3", TaskType.SELECT_IMAGES, input={"strategy": "per_group", "limit": 1}, depends=["t2"]),
        _task("t4", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t3"]),
    ])
    plan, results, _ = _run(draft, workers)
    groups = results["t2"].output.groups
    assert {g.key for g in groups} == {"北京", "上海", "杭州"}
    selected = results["t3"].output
    assert isinstance(selected, SelectOutput)
    assert selected.strategy == "per_group"
    assert len(selected.selected_ids) == 3
    assert set(selected.selected_ids) <= {1, 2, 3, 4}
    assert "per_group" in selected.selection_reason


# ---------------------------------------------------------------------------
# Case 4: retrieve → filter → group → select → synthesize
# ---------------------------------------------------------------------------
def test_case4_retrieve_filter_group_select_synthesize():
    workers = _workers()
    draft = _draft("t5", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "城市", "limit": 4}),
        _task("t2", TaskType.FILTER_METADATA, input={
            "predicates": [{"field": "file_type", "operator": "eq", "value": "jpg"}],
        }, depends=["t1"]),
        _task("t3", TaskType.GROUP_IMAGES, input={"group_by": "location"}, depends=["t2"]),
        _task("t4", TaskType.SELECT_IMAGES, input={"strategy": "per_group", "limit": 1}, depends=["t3"]),
        _task("t5", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t4"]),
    ])
    _, results, _ = _run(draft, workers)
    kept = [c.id for c in results["t2"].output.candidates]
    assert 3 not in kept  # png 被 filter 删除
    assert set(kept) == {1, 2, 4}
    assert {g.key for g in results["t3"].output.groups} == {"北京", "上海", "杭州"}
    assert len(results["t4"].output.selected_ids) == 3


# ---------------------------------------------------------------------------
# Case 5: retrieve → verify → synthesize
# ---------------------------------------------------------------------------
def test_case5_retrieve_verify_synthesize():
    workers = _workers(verifier=_FakeVerifier())
    draft = _draft("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "自行车", "limit": 4}),
        _task("t2", TaskType.VERIFY_VISUAL_CONDITION, input={
            "condition": "是否包含自行车", "max_candidates": 4,
        }, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t2"]),
    ])
    plan, results, _ = _run(draft, workers, vision=True)
    assert plan.tasks[1].task_type == TaskType.VERIFY_VISUAL_CONDITION
    vo = results["t2"].output
    assert isinstance(vo, VerifyTaskOutput)
    assert vo.inspected_count == 4
    assert vo.passed_ids == [1, 3]
    assert [c.id for c in vo.candidates] == [1, 3]
    assert results["t3"].output.image_ids == [1, 3]


# ---------------------------------------------------------------------------
# Case 6: retrieve → dedup → group → verify → select → synthesize
# ---------------------------------------------------------------------------
def test_case6_retrieve_dedup_group_verify_select_synthesize():
    workers = _workers(retrieve=_DupRetrieve(), verifier=_FakeVerifier())
    draft = _draft("t6", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _task("t2", TaskType.DEDUPLICATE_IMAGES, input={}, depends=["t1"]),
        _task("t3", TaskType.GROUP_IMAGES, input={"group_by": "location"}, depends=["t2"]),
        _task("t4", TaskType.VERIFY_VISUAL_CONDITION, input={
            "condition": "是否包含自行车", "max_candidates": 4,
        }, depends=["t3"]),
        _task("t5", TaskType.SELECT_IMAGES, input={"strategy": "top_n", "limit": 1}, depends=["t4"]),
        _task("t6", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t5"]),
    ])
    plan, results, _ = _run(draft, workers, vision=True)
    assert [t.task_type.value for t in plan.tasks] == [
        "retrieve_images", "deduplicate_images", "group_images",
        "verify_visual_condition", "select_images", "synthesize",
    ]
    assert results["t2"].output.removed_count == 1
    assert results["t4"].output.passed_ids == [1]
    assert results["t5"].output.selected_ids == [1]
    assert results["t6"].output.image_ids == [1]


# ---------------------------------------------------------------------------
# 边界：filter ≠ select；sort 不召回；rerank 不增召回
# ---------------------------------------------------------------------------
def test_filter_is_not_select():
    workers = _workers()
    filter_draft = _draft("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q", "limit": 4}),
        _task("t2", TaskType.FILTER_METADATA, input={
            "predicates": [{"field": "file_type", "operator": "eq", "value": "jpg"}],
        }, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, depends=["t2"]),
    ])
    select_draft = _draft("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q", "limit": 4}),
        _task("t2", TaskType.SELECT_IMAGES, input={"strategy": "top_n", "limit": 2}, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, depends=["t2"]),
    ])
    _, f_res, _ = _run(filter_draft, workers)
    _, s_res, _ = _run(select_draft, workers)
    filtered = [c.id for c in f_res["t2"].output.candidates]
    selected = s_res["t2"].output.selected_ids
    assert filtered == [1, 2, 4]          # 谓词删除 png
    assert selected == [1, 2]             # 数量截取，不看 file_type
    assert 3 not in filtered and 3 not in selected
    assert type(f_res["t2"].output).__name__ != type(s_res["t2"].output).__name__


def test_sort_changes_order_not_membership():
    workers = _workers()
    draft = _draft("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q", "limit": 4}),
        _task("t2", TaskType.SORT_IMAGES, input={"sort_by": "width", "order": "asc"}, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, depends=["t2"]),
    ])
    _, results, _ = _run(draft, workers)
    out = results["t2"].output
    assert isinstance(out, SortOutput)
    widths = [c.width for c in out.candidates]
    assert widths == sorted(widths)
    assert {c.id for c in out.candidates} == {1, 2, 3, 4}


def test_rerank_does_not_add_recall():
    workers = _workers()
    draft = _draft("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q", "limit": 3}),
        _task("t2", TaskType.RERANK_CANDIDATES, input={"provider": "r1"}, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, depends=["t2"]),
    ])
    _, results, _ = _run(draft, workers)
    assert set(_ids(results["t2"])) == set(_ids(results["t1"]))
    assert results["t2"].output.total == results["t1"].output.total


# ---------------------------------------------------------------------------
# Validator / 注册 / Specialist
# ---------------------------------------------------------------------------
def test_validator_rejects_verify_when_vision_off():
    d = _draft("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _task("t2", TaskType.VERIFY_VISUAL_CONDITION, input={"condition": "有车"}, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, depends=["t2"]),
    ])
    r = PlanValidator().validate(d, vision_available_flag=False)
    assert not r.valid
    assert any("VISION_UNAVAILABLE" in e or "WORKER_UNAVAILABLE" in e for e in r.errors)


def test_validator_rejects_consumer_without_producer():
    d = _draft("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _task("t2", TaskType.RERANK_CANDIDATES, input={}, depends=[]),
        _task("t3", TaskType.SYNTHESIZE, depends=["t1", "t2"]),
    ])
    r = PlanValidator().validate(d)
    assert not r.valid
    assert any("NO_PRODUCER" in e for e in r.errors)


def test_new_types_registered_without_vision():
    reg = registered_task_types(False)
    for t in (
        TaskType.RERANK_CANDIDATES, TaskType.DEDUPLICATE_IMAGES,
        TaskType.SORT_IMAGES, TaskType.SELECT_IMAGES,
    ):
        assert t in reg
        assert t in REGISTERED_TASK_TYPES
        assert t in default_workers()
    assert TaskType.VERIFY_VISUAL_CONDITION not in reg
    assert TaskType.ANALYZE_VISUAL not in registered_task_types(False)


def test_specialists_own_new_tasks():
    workers = _workers(verifier=_FakeVerifier())
    registry = build_default_agent_registry(workers=workers)
    assert registry.get_agent_for_task(TaskType.RERANK_CANDIDATES).card.agent_id == "retrieval"
    assert registry.get_agent_for_task(TaskType.DEDUPLICATE_IMAGES).card.agent_id == "organizer"
    assert registry.get_agent_for_task(TaskType.SORT_IMAGES).card.agent_id == "organizer"
    assert registry.get_agent_for_task(TaskType.SELECT_IMAGES).card.agent_id == "organizer"
    assert registry.get_agent_for_task(TaskType.VERIFY_VISUAL_CONDITION).card.agent_id == "vision"
    assert len(registry.list_agents()) == 3


def test_legacy_retrieve_synthesize_still_valid():
    workers = _workers()
    draft = _draft("t2", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "旅行", "limit": 4}),
        _task("t2", TaskType.SYNTHESIZE, input={"recommendation_limit": 3}, depends=["t1"]),
    ])
    plan, results, _ = _run(draft, workers)
    assert [t.task_type for t in plan.tasks] == [TaskType.RETRIEVE_IMAGES, TaskType.SYNTHESIZE]
    assert results["t2"].output.image_ids == [1, 2, 3]


def test_result_shape_is_worker_result():
    workers = _workers()
    draft = _draft("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q", "limit": 2}),
        _task("t2", TaskType.SELECT_IMAGES, input={"strategy": "top_n", "limit": 1}, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, depends=["t2"]),
    ])
    _, results, _ = _run(draft, workers)
    for tid, res in results.items():
        assert isinstance(res, WorkerResult)
        assert res.task_id == tid
        assert res.status == TaskStatus.SUCCEEDED
        assert res.output is not None
        assert res.error is None


# ---------------------------------------------------------------------------
def _main():
    failed = 0
    passed = 0
    for name in sorted(globals()):
        if name.startswith("test_"):
            fn = globals()[name]
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
