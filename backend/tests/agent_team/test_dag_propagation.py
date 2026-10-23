"""Phase 2 Stage 5 — DAG & Candidate Propagation 单元测试。

覆盖：
  - collect_candidates 消费 VisionOutput（按 image_id 合并证据）
  - Vision 失败 item / systemic 不注入证据；vision-only 不创造候选
  - DAG 执行 Case A-D：全成功 / 部分失败 / 全失败 / systemic → synth 均 SUCCEEDED
  - Synthesizer 收到 bounded vision evidence（无 filepath/public_url/base64）
  - candidate identity 保持 / 不产生重复
  - Validator DAG 规则（推荐结构合法 / producer / orphan）

全部 Fake/Mock（Fake search / Fake analyzer / Fake LLM），零真实 API。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_dag_propagation.py
"""

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.image_analysis import ImageAnalysisError, ImageAnalysisErrorCode
from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.models import (
    AnalyzeVisualInput,
    GroupImagesInput,
    PlanValidator,
    RetrieveImagesInput,
    SynthesizeInput,
    Task,
    TaskDraft,
    TaskPlan,
    TaskPlanDraft,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.synthesizer import SynthesizerAgent
from backend.ai_func.recommendation.orchestration.vision import (
    VisionErrorCode,
    VisionEvidence,
    VisionStatus,
    build_vision_output,
    vision_item_failed,
    vision_item_success,
    vision_systemic_failure,
)
from backend.ai_func.recommendation.orchestration.workers import (
    ExecutionContext,
    GroupImagesWorker,
    ImageCandidate,
    RetrievalOutput,
    RetrievalWorker,
    VisionWorker,
    WorkerResult,
    collect_candidates,
    dedup_candidates,
)


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------
def _vision_plan():
    return TaskPlan(
        plan_version="1.0", plan_id="plan_v", goal="找旅行照片",
        synthesis_task_id="t4",
        tasks=[
            Task(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, objective="r",
                 input=RetrieveImagesInput(query="旅行"), depends_on=[]),
            Task(task_id="t2", task_type=TaskType.GROUP_IMAGES, objective="g",
                 input=GroupImagesInput(group_by="location"), depends_on=["t1"]),
            Task(task_id="t3", task_type=TaskType.ANALYZE_VISUAL, objective="v",
                 input=AnalyzeVisualInput(criteria=["旅行"], max_candidates=8), depends_on=["t1"]),
            Task(task_id="t4", task_type=TaskType.SYNTHESIZE, objective="s",
                 input=SynthesizeInput(recommendation_limit=10), depends_on=["t2", "t3"]),
        ],
    )


def _rows():
    return [
        {"id": 1, "score": 0.9, "title": "a", "tags": [], "metadata": {},
         "public_url": "/s/1.jpg", "filepath": None},
        {"id": 2, "score": 0.8, "title": "b", "tags": [], "metadata": {},
         "public_url": "/s/2.jpg", "filepath": None},
        {"id": 3, "score": 0.7, "title": "c", "tags": [], "metadata": {},
         "public_url": "/s/3.jpg", "filepath": None},
    ]


class _FakeAnalyzer:
    def __init__(self, outcomes=(), client_ok=True):
        self.client = object() if client_ok else None
        self._outcomes = list(outcomes)
        self.calls = []

    def analyze_image(self, **kwargs):
        self.calls.append(kwargs)
        if self._outcomes:
            out = self._outcomes.pop(0)
        else:
            out = {"title": "ok", "description": "desc", "tags": ["t"],
                   "evidence": "e", "confidence": 0.9}
        if isinstance(out, Exception):
            raise out
        return out


class _SpyLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.response


def _ok_dict(iid):
    return {"title": f"v{iid}", "description": f"d{iid}", "tags": ["t"],
            "evidence": f"e{iid}", "confidence": 0.8}


def _exec(rows, outcomes=(), client_ok=True, synth_ids=None):
    retr = RetrievalWorker(search_tool=lambda **kw: list(rows))
    group = GroupImagesWorker()
    vision = VisionWorker(analyzer=_FakeAnalyzer(outcomes, client_ok=client_ok))
    spy = _SpyLLM(json.dumps({"answer": "推荐以下", "image_ids": synth_ids or [1, 2, 3]}))
    synth = SynthesizerAgent(llm=spy)
    ex = DAGExecutor(workers={
        TaskType.RETRIEVE_IMAGES: retr,
        TaskType.GROUP_IMAGES: group,
        TaskType.ANALYZE_VISUAL: vision,
        TaskType.SYNTHESIZE: synth,
    })
    return ex, spy


def _run_plan(ex):
    return list(ex.execute_stream(_vision_plan()))

# ---------------------------------------------------------------------------
# collect_candidates 消费 VisionOutput
# ---------------------------------------------------------------------------
def test_collect_candidates_consumes_vision_output():
    ctx = ExecutionContext(None)
    ctx.results["t1"] = WorkerResult(
        task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(
            candidates=[ImageCandidate(id=i, public_url=f"/s/{i}.jpg") for i in (1, 2, 3)],
            query="q", total=3),
    )
    vo = build_vision_output([
        vision_item_success(1, title="v1", description="d1", tags=["t"], evidence="e1", confidence=0.8),
        vision_item_failed(2, VisionErrorCode.TIMEOUT, "timeout"),
        vision_item_success(3, title="v3", description="d3", tags=["t"], evidence="e3", confidence=0.7),
        vision_item_success(9, title="v9", description="d9"),  # vision-only，无既有候选
    ])
    ctx.results["t3"] = WorkerResult(
        task_id="t3", task_type=TaskType.ANALYZE_VISUAL, status=TaskStatus.SUCCEEDED, output=vo,
    )
    task = Task(task_id="t4", task_type=TaskType.SYNTHESIZE, objective="s",
                input=SynthesizeInput(), depends_on=["t1", "t3"])
    cands = dedup_candidates(collect_candidates(task, ctx.results))
    assert [c.id for c in cands] == [1, 2, 3]          # 不引入 vision-only id 9
    by_id = {c.id: c for c in cands}
    assert by_id[1].vision is not None
    assert by_id[1].vision.evidence == "e1"
    assert by_id[1].vision.title == "v1"
    assert by_id[2].vision is None                     # failed 不注入
    assert by_id[3].vision is not None
    assert len(cands) == len({c.id for c in cands})    # 无重复候选


def test_collect_candidates_vision_systemic_no_injection():
    ctx = ExecutionContext(None)
    ctx.results["t1"] = WorkerResult(
        task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=[ImageCandidate(id=1, public_url="/s/1.jpg")], query="q", total=1),
    )
    ctx.results["t3"] = WorkerResult(
        task_id="t3", task_type=TaskType.ANALYZE_VISUAL, status=TaskStatus.SUCCEEDED,
        output=vision_systemic_failure(VisionErrorCode.CLIENT_UNAVAILABLE, "no client"),
    )
    task = Task(task_id="t4", task_type=TaskType.SYNTHESIZE, objective="s",
                input=SynthesizeInput(), depends_on=["t1", "t3"])
    cands = collect_candidates(task, ctx.results)
    assert len(cands) == 1
    assert cands[0].vision is None


def test_collect_candidates_without_vision_unaffected():
    ctx = ExecutionContext(None)
    cands_in = [ImageCandidate(id=1, public_url="/s/1.jpg"), ImageCandidate(id=2, public_url="/s/2.jpg")]
    ctx.results["t1"] = WorkerResult(
        task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=cands_in, query="q", total=2),
    )
    task = Task(task_id="t2", task_type=TaskType.GROUP_IMAGES, objective="g",
                input=GroupImagesInput(group_by="location"), depends_on=["t1"])
    cands = collect_candidates(task, ctx.results)
    assert [c.id for c in cands] == [1, 2]
    assert all(c.vision is None for c in cands)


def test_vision_evidence_whitelist_fields():
    ev = VisionEvidence(image_id=1, title="t", description="d", tags=["a"],
                        evidence="e", confidence=0.5)
    assert set(ev.model_dump().keys()) == {
        "image_id", "title", "description", "tags", "evidence", "confidence",
    }
    try:
        VisionEvidence(image_id=1, title="t", description="d", junk=1)
        assert False, "extra=forbid 应拒绝未知字段"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DAG 执行：Case A — 全部成功，vision evidence 到达 Synthesizer
# ---------------------------------------------------------------------------
def test_dag_case_a_all_success_with_evidence():
    ex, spy = _exec(_rows(), outcomes=[_ok_dict(1), _ok_dict(2), _ok_dict(3)])
    _run_plan(ex)
    res = ex.context.results
    for t in ("t1", "t2", "t3", "t4"):
        assert res[t].status == TaskStatus.SUCCEEDED, t
    vo = res["t3"].output
    assert vo.success_count == 3 and vo.failed_count == 0
    out = res["t4"].output
    assert out.synthesis_mode == "llm"
    by_id = {c.id: c for c in out.candidates}
    assert by_id[1].vision is not None and by_id[1].vision.evidence == "e1"
    # Synthesizer payload 包含 bounded vision evidence
    payload = json.loads(spy.calls[0][1]["content"])
    c1 = [c for c in payload["candidates"] if c["id"] == 1][0]
    assert c1["vision"]["title"] == "v1"
    assert c1["vision"]["evidence"] == "e1"
    assert c1["vision"]["confidence"] == 0.8


# ---------------------------------------------------------------------------
# DAG 执行：Case B — Vision 部分失败（2 成功 1 失败）→ synth 仍执行
# ---------------------------------------------------------------------------
def test_dag_case_b_vision_partial_failure():
    ex, spy = _exec(_rows(),
                    outcomes=[_ok_dict(1),
                              ImageAnalysisError(ImageAnalysisErrorCode.TIMEOUT, "timeout"),
                              _ok_dict(3)])
    _run_plan(ex)
    res = ex.context.results
    assert res["t3"].status == TaskStatus.SUCCEEDED
    vo = res["t3"].output
    assert vo.success_count == 2 and vo.failed_count == 1 and vo.systemic_failed is False
    assert res["t4"].status == TaskStatus.SUCCEEDED
    by_id = {c.id: c for c in res["t4"].output.candidates}
    assert by_id[1].vision is not None
    assert by_id[2].vision is None    # 失败 item 不注入伪证据
    assert by_id[3].vision is not None


# ---------------------------------------------------------------------------
# DAG 执行：Case C — Vision 全部 item 失败 → synth 仍执行（不升级 systemic）
# ---------------------------------------------------------------------------
def test_dag_case_c_vision_all_failed():
    ex, spy = _exec(_rows(),
                    outcomes=[ImageAnalysisError(ImageAnalysisErrorCode.TIMEOUT, "t")] * 3)
    _run_plan(ex)
    res = ex.context.results
    assert res["t3"].status == TaskStatus.SUCCEEDED
    vo = res["t3"].output
    assert vo.success_count == 0 and vo.failed_count == 3 and vo.systemic_failed is False
    assert res["t4"].status == TaskStatus.SUCCEEDED
    assert all(c.vision is None for c in res["t4"].output.candidates)


# ---------------------------------------------------------------------------
# DAG 执行：Case D — Vision systemic failure → synth 仍执行（contract 保持）
# ---------------------------------------------------------------------------
def test_dag_case_d_vision_systemic_failure():
    ex, spy = _exec(_rows(), outcomes=(), client_ok=False)
    _run_plan(ex)
    res = ex.context.results
    assert res["t3"].status == TaskStatus.SUCCEEDED
    vo = res["t3"].output
    assert vo.systemic_failed is True
    assert vo.systemic_error_code == VisionErrorCode.CLIENT_UNAVAILABLE
    assert res["t4"].status == TaskStatus.SUCCEEDED
    assert all(c.vision is None for c in res["t4"].output.candidates)


# ---------------------------------------------------------------------------
# Candidate identity / duplicate / 泄露
# ---------------------------------------------------------------------------
def test_dag_vision_preserves_candidate_identity():
    ex, spy = _exec(_rows(), outcomes=[_ok_dict(1), _ok_dict(2), _ok_dict(3)])
    _run_plan(ex)
    by_id = {c.id: c for c in ex.context.results["t4"].output.candidates}
    assert by_id[1].title == "a"            # 原始 title 保留
    assert by_id[1].score == 0.9            # 原始 score 保留
    assert by_id[1].vision.title == "v1"    # vision 是独立字段


def test_dag_no_duplicate_candidates_from_vision():
    ex, spy = _exec(_rows(), outcomes=[_ok_dict(1), _ok_dict(2), _ok_dict(3)])
    _run_plan(ex)
    out = ex.context.results["t4"].output
    ids = [c.id for c in out.candidates]
    assert len(ids) == len(set(ids)) == 3


def test_dag_vision_no_sensitive_leak_in_payload():
    ex, spy = _exec(_rows(), outcomes=[_ok_dict(1), _ok_dict(2), _ok_dict(3)])
    _run_plan(ex)
    payload = json.loads(spy.calls[0][1]["content"])
    for c in payload["candidates"]:
        v = c.get("vision")
        if v is not None:
            assert "filepath" not in v
            assert "public_url" not in v
            assert "base64" not in json.dumps(v)
            assert "raw" not in json.dumps(v).lower()
    # VisionEvidence 序列化不含内部字段
    out = ex.context.results["t4"].output
    for c in out.candidates:
        if c.vision is not None:
            keys = set(c.vision.model_dump().keys())
            assert keys <= {"image_id", "title", "description", "tags", "evidence", "confidence"}


# ---------------------------------------------------------------------------
# Validator DAG 规则
# ---------------------------------------------------------------------------
def _vdraft(tasks, synth_id):
    return TaskPlanDraft(goal="g", synthesis_task_id=synth_id, tasks=tasks)


def _vtask(tid, ttype, input=None, depends=None):
    return TaskDraft(task_id=tid, task_type=ttype, objective="o",
                     input=input or {}, depends_on=depends or [])


def test_dag_validation_recommended_structure_legal():
    d = _vdraft([
        _vtask("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _vtask("t2", TaskType.GROUP_IMAGES, input={"group_by": "location"}, depends=["t1"]),
        _vtask("t3", TaskType.ANALYZE_VISUAL, input={"criteria": ["c"]}, depends=["t1"]),
        _vtask("t4", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t2", "t3"]),
    ], synth_id="t4")
    r = PlanValidator().validate(d, vision_available_flag=True)
    assert r.valid, r.errors


def test_dag_vision_no_producer_rejected():
    d = _vdraft([
        _vtask("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _vtask("t2", TaskType.ANALYZE_VISUAL, input={"criteria": ["c"]}, depends=[]),
        _vtask("t3", TaskType.SYNTHESIZE, depends=["t2"]),
    ], synth_id="t3")
    r = PlanValidator().validate(d, vision_available_flag=True)
    assert not r.valid
    assert any("NO_PRODUCER" in e for e in r.errors)


def test_dag_vision_not_reaching_synth_rejected():
    d = _vdraft([
        _vtask("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _vtask("t2", TaskType.ANALYZE_VISUAL, input={"criteria": ["c"]}, depends=["t1"]),
        _vtask("t3", TaskType.SYNTHESIZE, depends=["t1"]),
    ], synth_id="t3")
    r = PlanValidator().validate(d, vision_available_flag=True)
    assert not r.valid
    assert any("ORPHAN_TASK" in e for e in r.errors)


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

