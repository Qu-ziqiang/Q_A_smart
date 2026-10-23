"""Lightweight Agent Memory — 集成测试（test_memory_integration.py）。

覆盖（§21）：
  Case 1  第一轮 Team 成功 → agent_episodes = 1
  Case 2  第二轮同会话相关 query → recall 1 episode → memory hints 进入 Planner prompt
  Case 3  冲突：历史 grouped，当前"不要分组" → prompt 含 current-query precedence，Memory 不强制 TaskType
  Case 4  不同 conversation 隔离
  Case 5  DELETE conversation → episodes(A) = 0
  附加    _RecommendationService.stream_core 线程 conversation_id（§16）

运行（repo 根）：
    venv\\Scripts\\python.exe -X utf8 -m pytest backend/tests/agent_team/test_memory_integration.py -q
"""

import json
import os
import sqlite3
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation import ENABLE_SUPERVISOR_RUNTIME
from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.memory import EpisodicMemoryService
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
from backend.db_func.repositories.agent_episodes import (
    AGENT_EPISODES_INDEXES_SQL,
    AGENT_EPISODES_TABLE_SQL,
    AgentEpisodeRepository,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(AGENT_EPISODES_TABLE_SQL)
    for s in AGENT_EPISODES_INDEXES_SQL:
        conn.execute(s)
    conn.commit()
    return conn


def _svc():
    return EpisodicMemoryService(repo=AgentEpisodeRepository(conn=_make_db()))


def _group_plan_json(goal="找旅行照片按地点整理"):
    return json.dumps({
        "goal": goal, "synthesis_task_id": "t3",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
             "input": {"query": goal, "limit": 20}, "depends_on": []},
            {"task_id": "t2", "task_type": "group_images", "objective": "分组",
             "input": {"group_by": "location"}, "depends_on": ["t1"]},
            {"task_id": "t3", "task_type": "synthesize", "objective": "汇总",
             "input": {"recommendation_limit": 10}, "depends_on": ["t2"]},
        ],
    }, ensure_ascii=False)


def _retrieve_plan_json(goal="夜景照片"):
    return json.dumps({
        "goal": goal, "synthesis_task_id": "t2",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
             "input": {"query": goal, "limit": 20}, "depends_on": []},
            {"task_id": "t2", "task_type": "synthesize", "objective": "汇总",
             "input": {"recommendation_limit": 10}, "depends_on": ["t1"]},
        ],
    }, ensure_ascii=False)


class _SpyPlanner:
    """Fake Planner：记录收到的 memory_hints 与渲染后的 user prompt，并按当前 goal 返回计划。

    分组规则（模拟"当前 query wins"）：goal 含"不要分组"或"夜景"→ 不分组计划；否则分组计划。
    """

    def __init__(self):
        self.last_memory_hints = None
        self.last_user_prompt = None
        self._inner = PlannerAgent(llm=lambda msgs: self._plan_for(msgs))

    @staticmethod
    def _plan_for(messages):
        goal = messages[-1].get("content") or ""
        return _retrieve_plan_json() if ("不要分组" in goal or "夜景" in goal) else _group_plan_json()

    def create_plan(self, goal, recent_messages=None, memory_hints=None, capabilities=None):
        self.last_memory_hints = memory_hints
        self.last_user_prompt = PlannerAgent._build_user_prompt(goal, recent_messages, memory_hints)
        return self._inner.create_plan(goal, recent_messages, memory_hints, capabilities)


class _FW:
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
                            ImageCandidate(id=2, score=0.8, public_url="/s/2.jpg")],
                query="x", total=2,
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


def _orch(planner):
    return AdaptiveTeamOrchestrator(
        planner=planner, executor=DAGExecutor(workers=_fake_workers())
    )


def _run(orch, messages, conversation_id, svc):
    return [json.loads(raw) for raw in orch.run_stream(
        messages, conversation_id=conversation_id, memory_service=svc)]


def _plan_types(evs):
    for e in evs:
        if e.get("event") == "plan.created":
            return [t["task_type"] for t in e["tasks"]]
    return []

# ---------------------------------------------------------------------------
# Case 1: 第一轮 Team 成功 → 1 episode
# ---------------------------------------------------------------------------
def test_case1_first_turn_writes_episode():
    svc = _svc()
    evs = _run(_orch(_SpyPlanner()),
               [{"role": "user", "content": "找旅行照片并按地点整理"}], "A", svc)
    assert any(e["event"] == "selection" for e in evs)
    rows = svc.list_recent("A")
    assert len(rows) == 1
    assert rows[0]["success"] == 1
    assert rows[0]["query_summary"] == "找旅行照片按地点整理"


# ---------------------------------------------------------------------------
# Case 2: 第二轮同会话相关 query → recall + hints 进入 Planner prompt
# ---------------------------------------------------------------------------
def test_case2_second_turn_recalls_into_planner_prompt():
    svc = _svc()
    _run(_orch(_SpyPlanner()),
         [{"role": "user", "content": "找旅行照片并按地点整理"}], "A", svc)
    assert len(svc.list_recent("A")) == 1
    p2 = _SpyPlanner()
    _run(_orch(p2),
         [{"role": "user", "content": "再帮我找类似的旅行照片"}], "A", svc)
    assert p2.last_memory_hints, "第二轮应收到 memory hints"
    assert "找旅行照片按地点整理" in p2.last_user_prompt


# ---------------------------------------------------------------------------
# Case 3: 冲突 — Memory 不覆盖当前 query
# ---------------------------------------------------------------------------
def test_case3_conflict_memory_does_not_override():
    svc = _svc()
    _run(_orch(_SpyPlanner()),
         [{"role": "user", "content": "找旅行照片并按地点整理"}], "A", svc)
    assert len(svc.list_recent("A")) == 1
    p2 = _SpyPlanner()
    evs = _run(_orch(p2),
               [{"role": "user", "content": "不要分组，只要夜景照片"}], "A", svc)
    assert p2.last_memory_hints
    assert "以当前目标为准" in p2.last_user_prompt
    assert "当前用户请求优先级最高" in p2.last_user_prompt
    types = _plan_types(evs)
    assert "group_images" not in types
    assert types == ["retrieve_images", "synthesize"]


# ---------------------------------------------------------------------------
# Case 4: conversation 隔离
# ---------------------------------------------------------------------------
def test_case4_conversation_isolation():
    svc = _svc()
    _run(_orch(_SpyPlanner()),
         [{"role": "user", "content": "找旅行照片并按地点整理"}], "A", svc)
    assert len(svc.list_recent("A")) == 1
    pB = _SpyPlanner()
    _run(_orch(pB),
         [{"role": "user", "content": "找旅行照片并按地点整理"}], "B", svc)
    assert pB.last_memory_hints is None, "B 不应看到 A 的 memory"
    rows_b = svc.list_recent("B")
    assert len(rows_b) == 1  # B 自身成功运行写了 1 条（仅 B 的）
    assert all(r["conversation_id"] == "B" for r in rows_b)


# ---------------------------------------------------------------------------
# Case 5: DELETE conversation → episodes 级联删除
# ---------------------------------------------------------------------------
def test_case5_delete_conversation_cascades():
    svc = _svc()
    _run(_orch(_SpyPlanner()),
         [{"role": "user", "content": "找旅行照片并按地点整理"}], "A", svc)
    assert len(svc.list_recent("A")) == 1
    deleted = svc._repo.delete_by_conversation("A")
    assert deleted >= 1
    assert svc.list_recent("A") == []
    assert svc._repo.count("A") == 0


# ---------------------------------------------------------------------------
# 附加：_RecommendationService.stream_core 线程 conversation_id（§16）
# ---------------------------------------------------------------------------
def test_service_stream_core_threads_conversation_id():
    from backend.ai_func.recommendation import _RecommendationService

    class FakeAgent:
        def run_stream(self, messages, vector_targets=None, filters=None, limit=20):
            yield json.dumps({"event": "assistant_delta", "delta": "legacy"}, ensure_ascii=False) + "\n\n"
            yield json.dumps({"event": "selection", "ids": [7], "images": []}, ensure_ascii=False) + "\n\n"

    svc = _svc()
    service = _RecommendationService()
    service._agent = FakeAgent()
    service._orchestrator = _orch(_SpyPlanner())
    service._memory_service = svc

    old = os.environ.get(ENABLE_SUPERVISOR_RUNTIME)
    os.environ[ENABLE_SUPERVISOR_RUNTIME] = "false"
    try:
        evs = [json.loads(raw) for raw in service.stream_core(
            [{"role": "user", "content": "找旅行照片并按地点整理"}], conversation_id="A")]
        assert any(e["event"] == "selection" for e in evs)
        assert len(svc.list_recent("A")) == 1  # conversation_id 已穿透到 orchestrator/memory
        assert "plan.created" in [e["event"] for e in evs]

        svc2 = _svc()
        service._memory_service = svc2
        list(service.stream_core([{"role": "user", "content": "找旅行照片并按地点整理"}]))
        assert svc2.list_recent("A") == []
    finally:
        if old is None:
            os.environ.pop(ENABLE_SUPERVISOR_RUNTIME, None)
        else:
            os.environ[ENABLE_SUPERVISOR_RUNTIME] = old


# ---------------------------------------------------------------------------
def _main():
    failed = passed = 0
    g = globals()
    for name in sorted(g):
        if name.startswith("test_"):
            try:
                g[name]()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n== memory integration: {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

