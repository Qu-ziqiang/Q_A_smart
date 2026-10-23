"""Phase 1-B — 集成测试：Gate → Planner(Fake) → Validator → Executor → Worker。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_orchestrator.py
"""

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation import ENABLE_SUPERVISOR_RUNTIME
from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.models import TaskType
from backend.ai_func.recommendation.orchestration.orchestrator import AdaptiveTeamOrchestrator
from backend.ai_func.recommendation.orchestration.planner import PlannerAgent
from backend.ai_func.recommendation.orchestration.workers import (
    GroupOutput,
    ImageCandidate,
    RetrievalOutput,
    SynthesisOutput,
    TaskStatus,
    WorkerResult,
)

_COMPLEX_QUERY = "找出旅行照片，按地点分类，然后推荐最合适的几张"
_SIMPLE_QUERY = "找猫的照片"


def _valid_plan_json():
    return json.dumps(
        {
            "goal": "找旅行照片按地点分类推荐",
            "synthesis_task_id": "t3",
            "tasks": [
                {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索旅行照片",
                 "input": {"query": "旅行照片", "limit": 20}, "depends_on": []},
                {"task_id": "t2", "task_type": "group_images", "objective": "按地点分组",
                 "input": {"group_by": "location"}, "depends_on": ["t1"]},
                {"task_id": "t3", "task_type": "synthesize", "objective": "汇总推荐",
                 "input": {"recommendation_limit": 10}, "depends_on": ["t2"]},
            ],
        },
        ensure_ascii=False,
    )


def _cycle_plan_json():
    return json.dumps(
        {
            "goal": "x",
            "synthesis_task_id": "t2",
            "tasks": [
                {"task_id": "t1", "task_type": "retrieve_images", "objective": "r",
                 "input": {"query": "x"}, "depends_on": ["t2"]},
                {"task_id": "t2", "task_type": "synthesize", "objective": "s",
                 "input": {}, "depends_on": ["t1"]},
            ],
        }
    )


class _FW:
    """Fake Worker：把普通函数 fn(task, context) 包装成 Worker。"""

    def __init__(self, ttype, fn):
        self.task_type = ttype
        self._fn = fn

    def run(self, task, context):
        return self._fn(task, context)


def _fake_workers():
    def retr(task, context):
        return WorkerResult(
            task_id=task.task_id, task_type=TaskType.RETRIEVE_IMAGES,
            status=TaskStatus.SUCCEEDED,
            output=RetrievalOutput(
                candidates=[ImageCandidate(id=1, score=0.9, public_url="/s/1.jpg"),
                            ImageCandidate(id=2, score=0.8, public_url="/s/2.jpg",
                                           metadata={"location": "北京"})],
                query="旅行照片", total=2,
            ),
        )

    def group(task, context):
        return WorkerResult(
            task_id=task.task_id, task_type=TaskType.GROUP_IMAGES,
            status=TaskStatus.SUCCEEDED,
            output=GroupOutput(group_by="location", groups=[]),
        )

    def synth(task, context):
        return WorkerResult(
            task_id=task.task_id, task_type=TaskType.SYNTHESIZE,
            status=TaskStatus.SUCCEEDED,
            output=SynthesisOutput(image_ids=[1, 2], candidates=[
                ImageCandidate(id=1, score=0.9, public_url="/s/1.jpg"),
                ImageCandidate(id=2, score=0.8, public_url="/s/2.jpg"),
            ]),
        )

    return {
        TaskType.RETRIEVE_IMAGES: _FW(TaskType.RETRIEVE_IMAGES, retr),
        TaskType.GROUP_IMAGES: _FW(TaskType.GROUP_IMAGES, group),
        TaskType.SYNTHESIZE: _FW(TaskType.SYNTHESIZE, synth),
    }


def _orchestrator(response):
    planner = PlannerAgent(llm=lambda messages: response)
    executor = DAGExecutor(workers=_fake_workers())
    return AdaptiveTeamOrchestrator(planner=planner, executor=executor)


def _events(msgs):
    return [json.loads(raw) for raw in _orchestrator(_valid_plan_json()).run_stream(msgs)]


def _event_names(evs):
    return [e.get("event") for e in evs]


# ---------------------------------------------------------------------------
def test_orchestrator_full_flow_events():
    evs = _events([{"role": "user", "content": _COMPLEX_QUERY}])
    names = _event_names(evs)
    assert "plan.created" in names
    assert "task.started" in names
    assert "task.completed" in names
    assert "assistant_delta" in names
    assert "selection" in names
    select = evs[-1]
    assert select["event"] == "selection"
    assert select["ids"] == [1, 2]


def test_orchestrator_planner_failure_falls_back_retrieve():
    orch = _orchestrator("这不是 JSON")
    evs = [json.loads(raw) for raw in orch.run_stream([{"role": "user", "content": _COMPLEX_QUERY}])]
    names = _event_names(evs)
    assert names[0] == "plan.created"
    assert orch.last_plan_ok is True
    assert any(t.task_type.value == "retrieve_images" for t in orch.last_plan.tasks)


def test_orchestrator_validation_failure_falls_back_retrieve():
    orch = _orchestrator(_cycle_plan_json())
    evs = [json.loads(raw) for raw in orch.run_stream([{"role": "user", "content": _COMPLEX_QUERY}])]
    names = _event_names(evs)
    assert names[0] == "plan.created"
    assert orch.last_plan_ok is True
    assert any(t.task_type.value == "retrieve_images" for t in orch.last_plan.tasks)


def test_service_integration_simple_goes_legacy():
    from backend.ai_func.recommendation import _RecommendationService

    class FakeAgent:
        def run_stream(self, messages, vector_targets=None, filters=None, limit=20):
            yield json.dumps({"event": "assistant_delta", "delta": "legacy"}, ensure_ascii=False) + "\n\n"
            yield json.dumps({"event": "selection", "ids": [7], "images": []}, ensure_ascii=False) + "\n\n"

    service = _RecommendationService()
    service._agent = FakeAgent()
    evs = [json.loads(raw) for raw in service.stream_core([{"role": "user", "content": _SIMPLE_QUERY}])]
    names = _event_names(evs)
    assert "plan.created" not in names
    assert names[-1] == "selection"


def test_service_integration_complex_goes_team_no_legacy_fallback():
    from backend.ai_func.recommendation import _RecommendationService

    class SpyAgent:
        def __init__(self):
            self.called = False

        def run_stream(self, messages, vector_targets=None, filters=None, limit=20):
            self.called = True
            yield json.dumps({"event": "selection", "ids": [], "images": []}, ensure_ascii=False) + "\n\n"

    agent = SpyAgent()
    service = _RecommendationService()
    service._agent = agent
    service._orchestrator = _orchestrator(_valid_plan_json())

    old = os.environ.get(ENABLE_SUPERVISOR_RUNTIME)
    os.environ[ENABLE_SUPERVISOR_RUNTIME] = "false"
    try:
        evs = [json.loads(raw) for raw in service.stream_core([{"role": "user", "content": _COMPLEX_QUERY}])]
    finally:
        if old is None:
            os.environ.pop(ENABLE_SUPERVISOR_RUNTIME, None)
        else:
            os.environ[ENABLE_SUPERVISOR_RUNTIME] = old
    names = _event_names(evs)
    assert "plan.created" in names
    assert "selection" in names
    assert agent.called is False  # 复杂路径不立即回退 Legacy


def test_service_integration_plan_failed_falls_back_legacy():
    from backend.ai_func.recommendation import _RecommendationService

    class SpyAgent:
        def __init__(self):
            self.called = False

        def run_stream(self, messages, vector_targets=None, filters=None, limit=20):
            self.called = True
            yield json.dumps({"event": "selection", "ids": [7], "images": []}, ensure_ascii=False) + "\n\n"

    agent = SpyAgent()
    service = _RecommendationService()
    service._agent = agent
    service._orchestrator = _orchestrator("bad json")

    old = os.environ.get(ENABLE_SUPERVISOR_RUNTIME)
    os.environ[ENABLE_SUPERVISOR_RUNTIME] = "false"
    try:
        evs = [json.loads(raw) for raw in service.stream_core([{"role": "user", "content": _COMPLEX_QUERY}])]
    finally:
        if old is None:
            os.environ.pop(ENABLE_SUPERVISOR_RUNTIME, None)
        else:
            os.environ[ENABLE_SUPERVISOR_RUNTIME] = old
    names = _event_names(evs)
    assert names[0] == "plan.created"
    assert names[-1] == "selection"
    assert agent.called is False
    assert service._orchestrator.last_plan_ok is True


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
