"""Phase 2 Stage 8 — Integration and Regression 集成测试。

保留真实内部 orchestration（Gate → Planner → Validator → Executor → Workers →
Synthesizer → SSE），Fake 外部依赖（search / LLM / vision provider）。

覆盖：
  A1 legacy_boundary_e2e / B team_without_vision / C full_team_fake_vision
  vision partial / all-failed / systemic
  planner_vision_enabled / disabled
  selection validation in full chain
  legacy fallback boundary（planner 失败回退；vision 失败不回退）
  full team SSE / leakage full chain / contract serialization
  real service + fake provider（HTTP 集成）

零真实 API；不修改配置；不触碰生产 DB。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_integration_full_chain.py
"""

import json
import os
import sys
import traceback
from contextlib import contextmanager

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.ai_func.recommendation.orchestration.models as models_mod
import backend.ai_func.recommendation.orchestration.planner as planner_mod
import backend.routers.ai.recommendation as router_mod
from backend.ai_func.image_analysis import ImageAnalysisError, ImageAnalysisErrorCode
from backend.ai_func.recommendation import ENABLE_SUPERVISOR_RUNTIME, _RecommendationService
from backend.ai_func.recommendation.orchestration import (
    AdaptiveTeamOrchestrator,
    ComplexityGate,
    PlanValidator,
    PlannerAgent,
    PlannerFailure,
    TaskPlan,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.models import (
    AnalyzeVisualInput,
    RetrieveImagesInput,
    SynthesizeInput,
    Task,
)
from backend.ai_func.recommendation.orchestration.synthesizer import (
    SynthesisResult,
    SynthesizerAgent,
)
from backend.ai_func.recommendation.orchestration.vision import (
    VisionErrorCode,
    VisionEvidence,
    VisionTaskOutput,
    build_vision_output,
    vision_item_failed,
    vision_item_success,
)
from backend.ai_func.recommendation.orchestration.workers import (
    GroupImagesWorker,
    ImageCandidate,
    RetrievalWorker,
    VisionWorker,
    WorkerResult,
    default_workers,
)

# ---------------------------------------------------------------------------
# Fakes（外部依赖）
# ---------------------------------------------------------------------------
class _FakeAgent:
    """Fake Legacy agent：产出 assistant_delta + selection。"""

    def __init__(self, delta="找到猫的照片。", ids=(1,)):
        self.delta = delta
        self.ids = list(ids)
        self.called = False

    def run_stream(self, **kwargs):
        self.called = True
        yield json.dumps({"event": "assistant_delta", "delta": self.delta}, ensure_ascii=False)
        yield json.dumps({"event": "selection", "ids": self.ids,
                          "images": [{"id": 1, "score": 0.9, "title": "猫", "tags": [],
                                      "public_url": "/s/1.jpg"}]}, ensure_ascii=False)


class _FakeLLM:
    def __init__(self, response="", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


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


class _FakeMsgRepo:
    def __init__(self, *a, **k):
        pass

    def list_messages(self, conv_id, limit=40):
        return []

    def add_message(self, *a, **k):
        return None


class _FakeSessionRepo:
    def __init__(self, *a, **k):
        pass

    def update_by_request_id(self, *a, **k):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rows():
    return [
        {"id": 101, "score": 0.95, "title": "bike", "tags": [], "metadata": {},
         "public_url": "/s/101.jpg", "filepath": None},
        {"id": 102, "score": 0.90, "title": "mountain", "tags": [], "metadata": {},
         "public_url": "/s/102.jpg", "filepath": None},
        {"id": 103, "score": 0.85, "title": "lake", "tags": [], "metadata": {},
         "public_url": "/s/103.jpg", "filepath": None},
    ]


def _bike_outcome():
    return {"title": "red bicycle",
            "description": "A person riding a red bicycle",
            "tags": ["bicycle", "outdoor"],
            "evidence": "A red bicycle is visibly present",
            "confidence": 0.95}


def _ok_outcome(iid):
    return {"title": f"v{iid}", "description": f"d{iid}", "tags": ["t"],
            "evidence": f"e{iid}", "confidence": 0.8}


def _search(rows):
    return lambda **kw: list(rows)


def _plan_json(with_vision=True):
    tasks = [
        {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
         "input": {"query": "旅行"}, "depends_on": []},
    ]
    if with_vision:
        tasks += [
            {"task_id": "t2", "task_type": "group_images", "objective": "分组",
             "input": {"group_by": "location"}, "depends_on": ["t1"]},
            {"task_id": "t3", "task_type": "analyze_visual", "objective": "视觉",
             "input": {"criteria": ["旅行记录"], "max_candidates": 8}, "depends_on": ["t1"]},
            {"task_id": "t4", "task_type": "synthesize", "objective": "合成",
             "input": {"recommendation_limit": 10}, "depends_on": ["t2", "t3"]},
        ]
        synth_id = "t4"
    else:
        tasks += [
            {"task_id": "t2", "task_type": "group_images", "objective": "分组",
             "input": {"group_by": "location"}, "depends_on": ["t1"]},
            {"task_id": "t3", "task_type": "synthesize", "objective": "合成",
             "input": {"recommendation_limit": 10}, "depends_on": ["t2"]},
        ]
        synth_id = "t3"
    return json.dumps({"goal": "找旅行照片", "synthesis_task_id": synth_id,
                       "tasks": tasks}, ensure_ascii=False)


class _VisionOn:
    def __enter__(self):
        self._orig = (models_mod.vision_available, planner_mod.vision_available)
        models_mod.vision_available = lambda: True
        planner_mod.vision_available = lambda: True
        return self

    def __exit__(self, *a):
        models_mod.vision_available, planner_mod.vision_available = self._orig


def _build_orchestrator(search_rows, analyzer_outcomes=(), client_ok=True,
                        synth_llm=None, planner_llm=None, include_vision=True):
    planner = PlannerAgent(llm=planner_llm or _FakeLLM(_plan_json(with_vision=include_vision)))
    workers = {
        TaskType.RETRIEVE_IMAGES: RetrievalWorker(search_tool=_search(search_rows)),
        TaskType.GROUP_IMAGES: GroupImagesWorker(),
    }
    if include_vision:
        workers[TaskType.ANALYZE_VISUAL] = VisionWorker(
            analyzer=_FakeAnalyzer(analyzer_outcomes, client_ok=client_ok))
    synth = SynthesizerAgent(llm=synth_llm or _FakeLLM(
        json.dumps({"answer": "推荐 101", "image_ids": [101, 999, 101], "rationale": "r"})))
    workers[TaskType.SYNTHESIZE] = synth
    executor = DAGExecutor(workers=workers)
    return AdaptiveTeamOrchestrator(planner=planner, executor=executor)


def _make_service(agent=None, orchestrator=None):
    svc = _RecommendationService()
    svc._agent = agent if agent is not None else _FakeAgent()
    svc._gate = ComplexityGate()
    svc._orchestrator = orchestrator
    return svc


@contextmanager
def _disable_supervisor():
    """本文件测 AdaptiveTeamOrchestrator 注入路径，显式关掉默认 Supervisor。"""
    old = os.environ.get(ENABLE_SUPERVISOR_RUNTIME)
    os.environ[ENABLE_SUPERVISOR_RUNTIME] = "false"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ENABLE_SUPERVISOR_RUNTIME, None)
        else:
            os.environ[ENABLE_SUPERVISOR_RUNTIME] = old


def _run_service(svc, query):
    events = []
    with _disable_supervisor():
        for raw in svc.stream_core(messages=[{"role": "user", "content": query}]):
            try:
                events.append(json.loads(raw))
            except Exception:
                continue
    return events


_COMPLEX_QUERY = "找适合做旅行记录的自行车照片，按地点分组再推荐最合适的"
_SIMPLE_QUERY = "找猫的照片"


# ---------------------------------------------------------------------------
# Test 1 — Legacy Boundary E2E
# ---------------------------------------------------------------------------
def test_legacy_boundary_e2e():
    agent = _FakeAgent()
    svc = _make_service(agent=agent)
    events = _run_service(svc, _SIMPLE_QUERY)
    names = [e["event"] for e in events]
    assert names == ["assistant_delta", "selection"]
    assert agent.called
    assert not any(n.startswith(("plan.", "task.")) for n in names)  # 无 Team 事件


# ---------------------------------------------------------------------------
# Test 2 — Team without Vision（vision_available=False，真实配置）
# ---------------------------------------------------------------------------
def test_team_without_vision():
    planner_llm = _FakeLLM(_plan_json(with_vision=False))
    synth_llm = _FakeLLM(json.dumps({"answer": "推荐", "image_ids": [101, 102], "rationale": "r"}))
    svc = _make_service(orchestrator=_build_orchestrator(
        _rows(), synth_llm=synth_llm, planner_llm=planner_llm, include_vision=False))
    events = _run_service(svc, _COMPLEX_QUERY)
    names = [e["event"] for e in events]
    assert "plan.created" in names
    completed_types = {e["task_type"] for e in events if e["event"] == "task.completed"}
    assert "retrieve_images" in completed_types and "group_images" in completed_types
    assert "synthesize" in completed_types
    assert not any(e.get("task_type") == "analyze_visual" for e in events if e["event"].startswith("task."))
    assert names[-2] == "assistant_delta" and names[-1] == "selection"


# ---------------------------------------------------------------------------
# Test 3 — Full Team + Fake Vision（核心集成）
# ---------------------------------------------------------------------------
def test_full_team_fake_vision():
    captured = {}

    def spy_llm(messages):
        captured["payload"] = json.loads(messages[1]["content"])
        return json.dumps({"answer": "推荐自行车照片", "image_ids": [101, 102], "rationale": "视觉证据"})

    with _VisionOn():
        svc = _make_service(orchestrator=_build_orchestrator(
            _rows(), analyzer_outcomes=[_bike_outcome(), _ok_outcome(102), _ok_outcome(103)],
            synth_llm=spy_llm))
        events = _run_service(svc, _COMPLEX_QUERY)
    names = [e["event"] for e in events]
    assert "plan.created" in names
    assert not any(e["event"] == "plan.failed" for e in events)
    # 任务顺序（串行确定）：retrieve → group → vision → synthesize
    order = [e["task_type"] for e in events if e["event"] == "task.started"]
    assert order[0] == "retrieve_images"
    assert order.index("group_images") < order.index("synthesize")
    assert order.index("analyze_visual") < order.index("synthesize")
    # Vision evidence 到达 Synthesizer payload（candidate 101）
    c101 = [c for c in captured["payload"]["candidates"] if c["id"] == 101][0]
    assert c101["vision"]["title"] == "red bicycle"
    assert c101["vision"]["evidence"] == "A red bicycle is visibly present"
    assert c101["vision"]["confidence"] == 0.95
    # Final selection
    sel = [e for e in events if e["event"] == "selection"][-1]
    assert sel["ids"] == [101, 102]
    assert svc._orchestrator.last_plan_ok is True


# ---------------------------------------------------------------------------
# Test 4 — Vision Partial Failure Integration
# ---------------------------------------------------------------------------
def test_vision_partial_failure():
    captured = {}

    def spy_llm(messages):
        captured["payload"] = json.loads(messages[1]["content"])
        return json.dumps({"answer": "ok", "image_ids": [101, 102, 103], "rationale": "r"})

    outcomes = [_ok_outcome(101),
                ImageAnalysisError(ImageAnalysisErrorCode.TIMEOUT, "timeout"),
                _ok_outcome(103)]
    with _VisionOn():
        svc = _make_service(orchestrator=_build_orchestrator(
            _rows(), analyzer_outcomes=outcomes, synth_llm=spy_llm))
        events = _run_service(svc, _COMPLEX_QUERY)
    vis = [e for e in events if e["event"] == "task.completed"
           and e["task_type"] == "analyze_visual"][0]
    assert vis["output_summary"]["vision"]["success_count"] == 2
    assert vis["output_summary"]["vision"]["failed_count"] == 1
    assert vis["output_summary"]["vision"]["systemic_failed"] is False
    assert any(e["event"] == "task.completed" and e["task_type"] == "synthesize" for e in events)
    assert events[-1]["event"] == "selection"
    by_id = {c["id"]: c for c in captured["payload"]["candidates"]}
    assert "vision" not in by_id[102]
    assert by_id[101]["vision"]["title"] == "v101"
    assert by_id[103]["vision"]["title"] == "v103"


# ---------------------------------------------------------------------------
# Test 5 — Vision All Items Failed
# ---------------------------------------------------------------------------
def test_vision_all_failed():
    captured = {}

    def spy_llm(messages):
        captured["payload"] = json.loads(messages[1]["content"])
        return json.dumps({"answer": "ok", "image_ids": [101, 102, 103], "rationale": "r"})

    outcomes = [ImageAnalysisError(ImageAnalysisErrorCode.TIMEOUT, "t")] * 3
    with _VisionOn():
        svc = _make_service(orchestrator=_build_orchestrator(
            _rows(), analyzer_outcomes=outcomes, synth_llm=spy_llm))
        events = _run_service(svc, _COMPLEX_QUERY)
    vis = [e for e in events if e["event"] == "task.completed"
           and e["task_type"] == "analyze_visual"][0]
    assert vis["output_summary"]["vision"] == {
        "candidate_count": 3, "success_count": 0, "failed_count": 3, "systemic_failed": False,
    }
    assert any(e["event"] == "task.completed" and e["task_type"] == "synthesize" for e in events)
    for c in captured["payload"]["candidates"]:
        assert "vision" not in c


# ---------------------------------------------------------------------------
# Test 6 — Vision Systemic Failure
# ---------------------------------------------------------------------------
def test_vision_systemic_failure():
    captured = {}

    def spy_llm(messages):
        captured["payload"] = json.loads(messages[1]["content"])
        return json.dumps({"answer": "ok", "image_ids": [101, 102, 103], "rationale": "r"})

    with _VisionOn():
        svc = _make_service(orchestrator=_build_orchestrator(
            _rows(), analyzer_outcomes=(), client_ok=False, synth_llm=spy_llm))
        events = _run_service(svc, _COMPLEX_QUERY)
    vis = [e for e in events if e["event"] == "task.completed"
           and e["task_type"] == "analyze_visual"][0]
    assert vis["output_summary"]["vision"]["systemic_failed"] is True
    assert vis["output_summary"]["vision"]["candidate_count"] == 0
    assert any(e["event"] == "task.completed" and e["task_type"] == "synthesize" for e in events)
    for c in captured["payload"]["candidates"]:
        assert "vision" not in c


# ---------------------------------------------------------------------------
# Test 7 — Planner Vision Enabled（Validator 接受 + Executor 执行 VisionWorker）
# ---------------------------------------------------------------------------
def test_planner_vision_enabled():
    planner_llm = _FakeLLM(_plan_json(with_vision=True))
    synth_llm = _FakeLLM(json.dumps({"answer": "ok", "image_ids": [101], "rationale": "r"}))
    with _VisionOn():
        svc = _make_service(orchestrator=_build_orchestrator(
            _rows(), analyzer_outcomes=[_bike_outcome(), _ok_outcome(102), _ok_outcome(103)],
            synth_llm=synth_llm, planner_llm=planner_llm))
        events = _run_service(svc, _COMPLEX_QUERY)
    assert not any(e["event"] == "plan.failed" for e in events)
    assert any(e["event"] == "task.started" and e["task_type"] == "analyze_visual" for e in events)
    assert any(e["event"] == "task.completed" and e["task_type"] == "analyze_visual" for e in events)


# ---------------------------------------------------------------------------
# Test 8 — Planner Vision Disabled（Validator 拒绝 + 不执行 Vision + Legacy fallback）
# ---------------------------------------------------------------------------
def test_planner_vision_disabled():
    # vision off（真实配置，不 patch）
    planner_llm = _FakeLLM(_plan_json(with_vision=True))   # planner 仍产出 Vision 计划
    agent = _FakeAgent()
    svc = _make_service(agent=agent, orchestrator=_build_orchestrator(
        _rows(), include_vision=False, planner_llm=planner_llm))
    events = _run_service(svc, _COMPLEX_QUERY)
    # Validator 拒绝 Vision 计划 → 硬底线 fallback retrieve→synth，不交空计划
    assert not any(e["event"] == "plan.failed" for e in events)
    assert not agent.called
    assert not any(e.get("task_type") == "analyze_visual" for e in events if e["event"].startswith("task."))


# ---------------------------------------------------------------------------
# Test 9 — Selection Validation in Full Chain（[101,999,101] → [101]）
# ---------------------------------------------------------------------------
def test_selection_validation_in_full_chain():
    captured = {}

    def spy_llm(messages):
        captured["payload"] = json.loads(messages[1]["content"])
        return json.dumps({"answer": "a", "image_ids": [101, 999, 101], "rationale": "r"})

    with _VisionOn():
        svc = _make_service(orchestrator=_build_orchestrator(
            _rows(), analyzer_outcomes=[_bike_outcome(), _ok_outcome(102), _ok_outcome(103)],
            synth_llm=spy_llm))
        events = _run_service(svc, _COMPLEX_QUERY)
    sel = [e for e in events if e["event"] == "selection"][-1]
    assert sel["ids"] == [101]           # 999 被 validate_selection 过滤，重复去重
    assert 999 not in sel["ids"]
    # payload 候选仍包含 3 个（selection 只挑合法 id）
    assert len(captured["payload"]["candidates"]) == 3


# ---------------------------------------------------------------------------
# Test 10 — Legacy Fallback Boundary（planner 失败→回退；vision 失败→不回退）
# ---------------------------------------------------------------------------
def test_legacy_fallback_boundary():
    # (a) Planner 失败 → 硬底线 retrieve→synth，不交空计划、不回 Legacy
    planner_llm = _FakeLLM(error=PlannerFailure("planner boom"))
    agent_a = _FakeAgent()
    svc_a = _make_service(agent=agent_a, orchestrator=_build_orchestrator(
        _rows(), include_vision=False, planner_llm=planner_llm))
    events_a = _run_service(svc_a, _COMPLEX_QUERY)
    assert not any(e["event"] == "plan.failed" for e in events_a)
    assert agent_a.called is False
    assert svc_a._orchestrator.last_plan_ok is True

    # (b) Vision 失败（partial）→ 不触发 Legacy fallback
    agent_b = _FakeAgent()
    with _VisionOn():
        svc_b = _make_service(agent=agent_b, orchestrator=_build_orchestrator(
            _rows(),
            analyzer_outcomes=[_ok_outcome(101),
                               ImageAnalysisError(ImageAnalysisErrorCode.TIMEOUT, "t"),
                               _ok_outcome(103)]))
        events_b = _run_service(svc_b, _COMPLEX_QUERY)
    assert not any(e["event"] == "plan.failed" for e in events_b)
    assert agent_b.called is False
    assert events_b[-1]["event"] == "selection"
    assert svc_b._orchestrator.last_plan_ok is True

    # (c) Synthesizer 失败（LLM 非法输出 → deterministic fallback）→ 不触发 Legacy
    agent_c = _FakeAgent()
    bad_synth = _FakeLLM(response="这不是 JSON")
    with _VisionOn():
        svc_c = _make_service(agent=agent_c, orchestrator=_build_orchestrator(
            _rows(), analyzer_outcomes=[_bike_outcome(), _ok_outcome(102), _ok_outcome(103)],
            synth_llm=bad_synth))
        events_c = _run_service(svc_c, _COMPLEX_QUERY)
    assert not any(e["event"] == "plan.failed" for e in events_c)
    assert agent_c.called is False
    synth_task = [e for e in events_c if e["event"] == "task.completed"
                  and e["task_type"] == "synthesize"][0]
    assert synth_task["output_summary"]["synthesis"]["degraded"] is True


# ---------------------------------------------------------------------------
# Test 11 — Full Team SSE（HTTP 帧级）
# ---------------------------------------------------------------------------
def _parse_sse(text):
    frames = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        ev = ""
        data_lines = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                ev = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
            else:
                data_lines.append(line.strip())
        data_text = "".join(data_lines)
        frames.append({"event": ev, "data": json.loads(data_text) if data_text else {}})
    return frames


def _run_http(svc, query):
    router_mod.recommendation_service = svc
    router_mod.ConversationMessageRepository = _FakeMsgRepo
    router_mod.RequestSessionRepository = _FakeSessionRepo
    app = FastAPI()
    app.include_router(router_mod.router, prefix="/api/v1/ai")
    client = TestClient(app)
    with _disable_supervisor():
        resp = client.post("/api/v1/ai/recommend/chat/stream", json={"query": query})
    assert resp.status_code == 200
    return _parse_sse(resp.text)


def test_full_team_sse():
    with _VisionOn():
        svc = _make_service(orchestrator=_build_orchestrator(
            _rows(), analyzer_outcomes=[_bike_outcome(), _ok_outcome(102), _ok_outcome(103)]))
        frames = _run_http(svc, _COMPLEX_QUERY)
    names = [f["event"] for f in frames]
    assert names[0] == "rewrite_start"
    assert "plan.created" in names
    assert any(n.startswith("task.") for n in names)
    assert names.count("complete") == 1
    assert names[-1] == "complete"
    for f in frames:
        if f["event"] == "task.completed" and f["data"].get("task_type") == "analyze_visual":
            vs = f["data"]["output_summary"]["vision"]
            assert set(vs.keys()) == {"candidate_count", "success_count", "failed_count", "systemic_failed"}


def test_legacy_sse_contract():
    svc = _make_service(agent=_FakeAgent())
    frames = _run_http(svc, _SIMPLE_QUERY)
    names = [f["event"] for f in frames]
    assert names == ["rewrite_start", "assistant_delta", "complete"]   # 不新增事件类型


# ---------------------------------------------------------------------------
# Test 12 — Leakage Full Chain
# ---------------------------------------------------------------------------
def test_leakage_full_chain():
    with _VisionOn():
        svc = _make_service(orchestrator=_build_orchestrator(
            _rows(), analyzer_outcomes=[_bike_outcome(), _ok_outcome(102), _ok_outcome(103)]))
        events = _run_service(svc, _COMPLEX_QUERY)
    text = json.dumps(events, ensure_ascii=False).lower()
    for bad in ("filepath", "base64", "raw_response", "authorization", "api_key",
                "sk-", "data:image/", ":\\", "/secret"):
        assert bad not in text, f"事件流不应包含 {bad}"
    for e in events:
        if e["event"] == "task.completed" and e.get("task_type") == "analyze_visual":
            os_ = e["output_summary"]
            assert "title" not in json.dumps(os_, ensure_ascii=False)
            assert "evidence" not in json.dumps(os_, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Test 13 — Contract Serialization / Deserialization
# ---------------------------------------------------------------------------
def test_contract_serialization():
    # Task（含 AnalyzeVisualInput）
    t = Task(task_id="t3", task_type=TaskType.ANALYZE_VISUAL, objective="o",
             input=AnalyzeVisualInput(criteria=["c"], max_candidates=4), depends_on=["t1"])
    t2 = Task.model_validate(t.model_dump())
    assert t2 == t
    assert isinstance(t2.input, AnalyzeVisualInput)
    # TaskPlan
    plan = TaskPlan(plan_version="1.0", plan_id="p", goal="g", synthesis_task_id="t4",
                    tasks=[Task(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, objective="o",
                                input=RetrieveImagesInput(query="q"), depends_on=[]),
                           Task(task_id="t4", task_type=TaskType.SYNTHESIZE, objective="o",
                                input=SynthesizeInput(), depends_on=["t1"])])
    plan2 = TaskPlan.model_validate(plan.model_dump())
    assert plan2 == plan and plan2.synthesis_task_id == "t4"
    # VisionTaskOutput / VisionEvidence（image_id identity）
    vo = build_vision_output([vision_item_success(1, title="a", description="d", tags=["t"],
                                                  evidence="e", confidence=0.9)])
    vo2 = VisionTaskOutput.model_validate(vo.model_dump())
    assert vo2.success_count == 1 and vo2.total == 1
    ev = vo2.items[0]
    assert ev.image_id == 1
    cand = ImageCandidate(id=1, public_url="/s/1.jpg")
    assert ev.image_id == cand.id          # 与候选 identity 一致
    # WorkerResult / SynthesisResult
    wr = WorkerResult(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES,
                      status=TaskStatus.SUCCEEDED, output=None)
    assert WorkerResult.model_validate(wr.model_dump()).status == TaskStatus.SUCCEEDED
    sr = SynthesisResult(answer="a", image_ids=[1, 2], rationale="r")
    sr2 = SynthesisResult.model_validate(sr.model_dump())
    assert sr2.image_ids == [1, 2] and sr2.answer == "a"


# ---------------------------------------------------------------------------
# Test 14 — Real Service + Fake Provider（HTTP real / orchestration real / provider fake）
# ---------------------------------------------------------------------------
def test_real_service_fake_provider():
    # 真实 _RecommendationService（真实 gate + 真实 stream_core 路由逻辑）
    svc = _RecommendationService()
    svc._agent = _FakeAgent()
    svc._gate = ComplexityGate()
    with _VisionOn():
        svc._orchestrator = _build_orchestrator(
            _rows(), analyzer_outcomes=[_bike_outcome(), _ok_outcome(102), _ok_outcome(103)])
        frames_legacy = _run_http(svc, _SIMPLE_QUERY)
        assert [f["event"] for f in frames_legacy] == ["rewrite_start", "assistant_delta", "complete"]
        frames_team = _run_http(svc, _COMPLEX_QUERY)
    names = [f["event"] for f in frames_team]
    assert names[0] == "rewrite_start"
    assert "plan.created" in names
    assert any(f["event"] == "task.started" and f["data"]["task_type"] == "analyze_visual"
               for f in frames_team)
    assert names[-1] == "complete"


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

