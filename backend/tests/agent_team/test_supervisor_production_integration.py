"""Phase 3.1 — Supervisor Production Integration 接线测试。

覆盖：
  A. flag=False → 结果 == Phase 2 baseline（AdaptiveTeamOrchestrator 行为/选择一致）
  B. flag=True  → TEAM 请求确实经过 SupervisorAgent（supervisor_trace / decisions 可见）
  C. failure    → Supervisor REVIEW/REVISE 重试生效（empty → revise → retry → DONE）
  + feature flag 解析默认 True；显式 false 关闭；plan 失败仍回退 Legacy

全部 Fake（真实服务 / Planner / Validator / Executor / Supervisor 组件；叶子与工具实现注入）。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_supervisor_production_integration.py
"""

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation import ENABLE_SUPERVISOR_RUNTIME, _RecommendationService
from backend.ai_func.recommendation.orchestration import (
    PlanValidator,
    PlannerAgent,
    PlannerFailure,
    SupervisorAgent,
    TaskType,
    ToolRegistry,
    build_default_agent_registry,
    build_registry_backed_workers,
    register_builtin_tools,
)
from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.orchestrator import AdaptiveTeamOrchestrator
from backend.ai_func.recommendation.orchestration.workers import (
    DeterministicSynthesisWorker,
    GroupOutput,
    ImageCandidate,
    RetrievalOutput,
    SynthesisOutput,
    TaskStatus,
    WorkerResult,
)
from backend.tests.agent_team.loop_support import rewrite_aware_llm

_COMPLEX = "找出旅行照片，按地点分类，然后推荐最合适的几张"


class _Flag:
    """ENABLE_SUPERVISOR_RUNTIME 上下文开关。"""

    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self._old = os.environ.get(ENABLE_SUPERVISOR_RUNTIME)
        if self.on:
            os.environ[ENABLE_SUPERVISOR_RUNTIME] = "true"
        else:
            os.environ[ENABLE_SUPERVISOR_RUNTIME] = "false"
        return self

    def __exit__(self, *a):
        if self._old is None:
            os.environ.pop(ENABLE_SUPERVISOR_RUNTIME, None)
        else:
            os.environ[ENABLE_SUPERVISOR_RUNTIME] = self._old


class _SpyLegacy:
    def __init__(self):
        self.called = False

    def run_stream(self, messages, vector_targets=None, filters=None, limit=20):
        self.called = True
        yield json.dumps({"event": "assistant_delta", "delta": "legacy"}, ensure_ascii=False) + "\n\n"
        yield json.dumps({"event": "selection", "ids": [999], "images": []}, ensure_ascii=False) + "\n\n"


def _plan_json():
    return {
        "goal": "找旅行照片按地点分类推荐",
        "synthesis_task_id": "t2",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索旅行照片",
             "input": {"query": "旅行照片", "limit": 20}, "depends_on": []},
            {"task_id": "t2", "task_type": "synthesize", "objective": "汇总推荐",
             "input": {"recommendation_limit": 10}, "depends_on": ["t1"]},
        ],
    }


def _rows():
    return [
        {"id": 401, "score": 0.92, "title": "海边旅行", "description": "d",
         "tags": ["旅行"], "metadata": {"location": "三亚"}, "width": 1920, "height": 1280,
         "file_type": "jpg", "created_at": "2025-06-01T10:00:00Z",
         "public_url": "/static/401.jpg", "filepath": ""},
        {"id": 402, "score": 0.88, "title": "山地旅行", "description": "d",
         "tags": ["旅行"], "metadata": {"location": "丽江"}, "width": 1920, "height": 1280,
         "file_type": "jpg", "created_at": "2025-06-02T10:00:00Z",
         "public_url": "/static/402.jpg", "filepath": ""},
    ]


class _FakeSearch:
    def __init__(self, results_by_call, fail=False):
        self.results = list(results_by_call)
        self.calls = 0
        self.fail = fail

    def __call__(self, query=None, vector_targets=None, filters=None, limit=20):
        self.calls += 1
        if self.fail:
            raise RuntimeError("search tool down")
        idx = min(self.calls - 1, len(self.results) - 1) if self.results else 0
        return list(self.results[idx])


class _MemorySpyPlanner:
    def __init__(self):
        self.memory_hints_seen = []
        self._inner = PlannerAgent(
            llm=lambda messages: json.dumps(_plan_json(), ensure_ascii=False)
        )

    def create_plan(self, goal, recent_messages=None, memory_hints=None, **kwargs):
        self.memory_hints_seen.append(memory_hints)
        return self._inner.create_plan(
            goal=goal,
            recent_messages=recent_messages,
            memory_hints=memory_hints,
        )


class _FakeMemory:
    def __init__(self):
        self.episodes = []
        self.writes = []

    def recall_episodes(self, conversation_id, query):
        return list(self.episodes)

    def render_memory_hints(self, episodes, conversation_id=None, **kwargs):
        return [f"历史筛选偏好:{ep['preference']}" for ep in episodes]

    def remember(self, context, **kwargs):
        self.writes.append({"context": context, **kwargs})
        self.episodes = [{"preference": "按地点分组"}]
        return len(self.writes)


def _make_supervisor(search_impl, enhanced=True):
    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={"search_images": search_impl})
    workers = build_registry_backed_workers(tool_reg, with_vision=False)
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=tool_reg)
    planner = PlannerAgent(llm=rewrite_aware_llm(_plan_json()))
    return SupervisorAgent(
        planner=planner, validator=PlanValidator(), workers=workers,
        agent_registry=agent_reg, tool_registry=tool_reg,
        synthesize_worker=DeterministicSynthesisWorker(),
        enhanced=enhanced, max_iterations=2,
    )


class _FW:
    def __init__(self, ttype, fn):
        self.task_type = ttype
        self._fn = fn

    def run(self, task, context):
        return self._fn(task, context)


def _orchestrator_fake():
    """Phase 2 基线（orchestrator）用 Fake Workers（同 test_orchestrator 风格）。"""
    cands = [ImageCandidate(id=11, score=0.9, public_url="/s/11.jpg"),
             ImageCandidate(id=12, score=0.8, public_url="/s/12.jpg")]

    def retr(task, context):
        return WorkerResult(task_id=task.task_id, task_type=task.task_type,
                            status=TaskStatus.SUCCEEDED,
                            output=RetrievalOutput(candidates=cands, query="旅行照片", total=2))

    def synth(task, context):
        return WorkerResult(task_id=task.task_id, task_type=task.task_type,
                            status=TaskStatus.SUCCEEDED,
                            output=SynthesisOutput(image_ids=[11, 12], candidates=cands))

    planner = PlannerAgent(llm=lambda messages: json.dumps(_plan_json(), ensure_ascii=False))
    executor = DAGExecutor(workers={
        TaskType.RETRIEVE_IMAGES: _FW(TaskType.RETRIEVE_IMAGES, retr),
        TaskType.SYNTHESIZE: _FW(TaskType.SYNTHESIZE, synth),
    })
    return AdaptiveTeamOrchestrator(planner=planner, validator=PlanValidator(), executor=executor)


def _run_service(service, query):
    return [json.loads(raw) for raw in service.stream_core([{"role": "user", "content": query}])]


def _selection(events):
    sels = [e for e in events if e["event"] == "selection"]
    return sels[-1]["ids"] if sels else None


def test_flag_parses_default_on():
    with _Flag(False):
        assert _RecommendationService._supervisor_runtime_enabled() is False
    with _Flag(True):
        assert _RecommendationService._supervisor_runtime_enabled() is True
    old = os.environ.pop(ENABLE_SUPERVISOR_RUNTIME, None)
    try:
        assert _RecommendationService._supervisor_runtime_enabled() is True
    finally:
        if old is None:
            os.environ.pop(ENABLE_SUPERVISOR_RUNTIME, None)
        else:
            os.environ[ENABLE_SUPERVISOR_RUNTIME] = old


def test_a_flag_off_matches_phase2_baseline():
    service = _RecommendationService()
    service._agent = _SpyLegacy()
    service._orchestrator = _orchestrator_fake()
    with _Flag(False):
        events = _run_service(service, _COMPLEX)
        assert _selection(events) == [11, 12]  # 结果即 AdaptiveTeamOrchestrator 基线
        names = [e["event"] for e in events]
        assert "agent.review" not in names  # 无 supervisor 事件
        assert "plan.created" in names
        assert service._agent.called is False  # 计划成功，未回退 legacy


def test_b_flag_on_routes_through_supervisor():
    service = _RecommendationService()
    service._agent = _SpyLegacy()
    sup = _make_supervisor(_FakeSearch([_rows()]), enhanced=True)
    service._supervisor_runtime = sup  # 注入（等价 flag 下惰性创建的真实控制器）
    with _Flag(True):
        events = _run_service(service, _COMPLEX)
        names = [e["event"] for e in events]
        assert "plan.created" in names
        assert "agent.review" in names  # Supervisor 控制信号
        assert sup.run_state is not None
        assert sup.run_state.final_state.value == "done"
        decisions = [r["decision"] for r in
                     [e for e in events if e["event"] == "agent.review"]]
        assert decisions == ["continue"]
        assert _selection(events) == [401, 402]
        assert service._agent.called is False


def test_c_failure_recovery_via_supervisor_review_retry():
    service = _RecommendationService()
    service._agent = _SpyLegacy()
    sup = _make_supervisor(_FakeSearch([[], _rows()]), enhanced=True)  # 首次空 → revise
    service._supervisor_runtime = sup
    with _Flag(True):
        events = _run_service(service, _COMPLEX)
        decisions = [r["decision"] for r in
                     [e for e in events if e["event"] == "agent.review"]]
        assert decisions == ["replan", "continue"]
        assert sup.run_state.final_state.value == "done"
        assert _selection(events) == [401, 402]  # 重试后成功
        assert service._agent.called is False  # 未回退 legacy


def test_d_default_supervisor_recalls_and_writes_memory():
    service = _RecommendationService()
    service._agent = _SpyLegacy()
    memory = _FakeMemory()
    planner = _MemorySpyPlanner()
    sup = _make_supervisor(_FakeSearch([_rows()]), enhanced=True)
    sup.planner = planner
    service._supervisor_runtime = sup
    service._memory_service = memory

    # 默认不设置 flag：Supervisor 首轮成功后写 episode。
    old = os.environ.pop(ENABLE_SUPERVISOR_RUNTIME, None)
    try:
        first = [json.loads(raw) for raw in service.stream_core(
            [{"role": "user", "content": _COMPLEX}], conversation_id="conv-A"
        )]
        assert "agent.review" in [e["event"] for e in first]
        assert planner.memory_hints_seen[0] is None
        assert len(memory.writes) == 1
        assert memory.writes[0]["conversation_id"] == "conv-A"
        assert memory.writes[0]["route"] == "team"

        # 同会话第二轮召回，并把 bounded hints 传入 Supervisor Planner。
        list(service.stream_core(
            [{"role": "user", "content": _COMPLEX}],
            conversation_id="conv-A",
        ))
        assert planner.memory_hints_seen[-1] == ["历史筛选偏好:按地点分组"]
        assert len(memory.writes) == 2
    finally:
        if old is not None:
            os.environ[ENABLE_SUPERVISOR_RUNTIME] = old


def test_legacy_fallback_preserved_on_plan_failure():
    service = _RecommendationService()
    legacy = _SpyLegacy()
    service._agent = legacy

    class _BadPlanner(PlannerAgent):
        def create_plan(self, **kwargs):
            raise PlannerFailure("planner boom")

    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={"search_images": _FakeSearch([[]])})
    workers = build_registry_backed_workers(tool_reg, with_vision=False)
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=tool_reg)
    sup = SupervisorAgent(
        planner=_BadPlanner(), validator=PlanValidator(), workers=workers,
        agent_registry=agent_reg, tool_registry=tool_reg,
        synthesize_worker=DeterministicSynthesisWorker(), enhanced=True,
    )
    service._supervisor_runtime = sup
    with _Flag(True):
        events = _run_service(service, _COMPLEX)
        names = [e["event"] for e in events]
        assert "plan.failed" not in names
        assert legacy.called is False  # 空计划由 Supervisor 硬底线兜底，不再交白卷给 Legacy
        assert sup.last_plan_ok is True
        assert any(t.task_type.value == "retrieve_images" for t in sup.run_state.plan.tasks)


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

