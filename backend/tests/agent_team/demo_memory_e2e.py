"""Lightweight Agent Memory — 多轮 E2E Demo（§23）。

Conversation A：
  Turn 1: 找旅行照片并按地点整理            → Team 成功 → episode 写入
  Turn 2: 再帮我找类似的旅行照片，并优先做视觉核验
                                          → Memory recall → Top-1 hint
                                          → Planner 收到 bounded hint
                                          → Planner 仍遵循当前 query（视觉核验）

说明：本 demo 使用 **Fake Planner + 真实 Memory 路径**（真实 SQLite agent_episodes 表，
真实 EpisodicMemoryService / 写入门控 / recall 排序 / TTL / 渲染）。未调用任何真实 LLM。

运行（repo 根）：
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/demo_memory_e2e.py
"""

import json
import os
import sqlite3
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.memory import EpisodicMemoryService
from backend.ai_func.recommendation.orchestration.models import TaskType
from backend.ai_func.recommendation.orchestration.orchestrator import AdaptiveTeamOrchestrator
from backend.ai_func.recommendation.orchestration.planner import PlannerAgent
from backend.ai_func.recommendation.orchestration.vision import VisionTaskOutput
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


class _SpyPlanner:
    """Fake Planner（不调用 LLM）：按当前 goal 返回计划并记录 user prompt / hints。"""

    def __init__(self):
        self.last_user_prompt = None
        self.last_memory_hints = None
        self._current_goal = ""
        self._inner = PlannerAgent(llm=lambda msgs: self._plan_for_goal(self._current_goal))

    @staticmethod
    def _plan_for_goal(goal):
        if "视觉核验" in goal or "核验" in goal:
            return json.dumps({
                "goal": goal, "synthesis_task_id": "t3",
                "tasks": [
                    {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
                     "input": {"query": goal, "limit": 20}, "depends_on": []},
                    {"task_id": "t2", "task_type": "analyze_visual", "objective": "视觉核验",
                     "input": {"criteria": ["确认是类似旅行场景"], "max_candidates": 4}, "depends_on": ["t1"]},
                    {"task_id": "t3", "task_type": "synthesize", "objective": "汇总",
                     "input": {"recommendation_limit": 10}, "depends_on": ["t1", "t2"]},
                ],
            }, ensure_ascii=False)
        return json.dumps({
            "goal": goal, "synthesis_task_id": "t3",
            "tasks": [
                {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
                 "input": {"query": goal, "limit": 20}, "depends_on": []},
                {"task_id": "t2", "task_type": "group_images", "objective": "按地点分组",
                 "input": {"group_by": "location"}, "depends_on": ["t1"]},
                {"task_id": "t3", "task_type": "synthesize", "objective": "汇总",
                 "input": {"recommendation_limit": 10}, "depends_on": ["t2"]},
            ],
        }, ensure_ascii=False)

    def create_plan(self, goal, recent_messages=None, memory_hints=None, capabilities=None):
        self._current_goal = goal
        self.last_memory_hints = memory_hints
        self.last_user_prompt = PlannerAgent._build_user_prompt(goal, recent_messages, memory_hints)
        return self._inner.create_plan(goal, recent_messages, memory_hints, capabilities)


class _FW:
    def __init__(self, ttype, fn):
        self.task_type = ttype
        self._fn = fn

    def run(self, task, context):
        return self._fn(task, context)


def _workers():
    def retr(task, context):
        return WorkerResult(
            task_id=task.task_id, task_type=TaskType.RETRIEVE_IMAGES,
            status=TaskStatus.SUCCEEDED,
            output=RetrievalOutput(
                candidates=[ImageCandidate(id=1, score=0.9, public_url="/s/1.jpg"),
                            ImageCandidate(id=2, score=0.8, public_url="/s/2.jpg"),
                            ImageCandidate(id=3, score=0.7, public_url="/s/3.jpg")],
                query="x", total=3,
            ),
        )

    def group(task, context):
        return WorkerResult(
            task_id=task.task_id, task_type=TaskType.GROUP_IMAGES,
            status=TaskStatus.SUCCEEDED,
            output=GroupOutput(group_by="location", groups=[]),
        )

    def vision(task, context):
        return WorkerResult(
            task_id=task.task_id, task_type=TaskType.ANALYZE_VISUAL,
            status=TaskStatus.SUCCEEDED,
            output=VisionTaskOutput(
                items=[], total=3, success_count=3, failed_count=0, systemic_failed=False,
            ),
        )

    def synth(task, context):
        return WorkerResult(
            task_id=task.task_id, task_type=TaskType.SYNTHESIZE,
            status=TaskStatus.SUCCEEDED,
            output=SynthesisOutput(image_ids=[1, 2, 3], candidates=[
                ImageCandidate(id=1, score=0.9, public_url="/s/1.jpg"),
                ImageCandidate(id=2, score=0.8, public_url="/s/2.jpg"),
                ImageCandidate(id=3, score=0.7, public_url="/s/3.jpg"),
            ]),
        )

    return {
        TaskType.RETRIEVE_IMAGES: _FW(TaskType.RETRIEVE_IMAGES, retr),
        TaskType.GROUP_IMAGES: _FW(TaskType.GROUP_IMAGES, group),
        TaskType.ANALYZE_VISUAL: _FW(TaskType.ANALYZE_VISUAL, vision),
        TaskType.SYNTHESIZE: _FW(TaskType.SYNTHESIZE, synth),
    }


def _run(orch, messages, conversation_id, svc, tag):
    print(f"\n=== {tag} ===")
    print(f"query: {messages[-1]['content']}")
    events = [json.loads(raw) for raw in orch.run_stream(
        messages, conversation_id=conversation_id, memory_service=svc)]
    for e in events:
        if e["event"] == "plan.created":
            print("plan:", [t["task_type"] for t in e["tasks"]])
        elif e["event"] == "selection":
            print("selection ids:", e["ids"])
    return events


def _orch(planner):
    return AdaptiveTeamOrchestrator(planner=planner, executor=DAGExecutor(workers=_workers()))



def main():
    print("SmartImager — Lightweight Agent Memory 多轮 Demo（Fake Planner + 真实 Memory 路径）")
    print("NOTE: 未调用真实 LLM；Planner 为 Fake（确定性返回计划），Memory 链为真实实现。")
    print("NOTE: runtime patch vision_available()=True（仅本进程，不改 config.yaml），"
          "以演示含 analyze_visual 的第二轮计划可通过 Validator。")

    # 进程内 capability snapshot：vision 可用（不改 config；与 Stage 9 E2E 同模式）
    import backend.ai_func.recommendation.orchestration.models as _models_mod
    import backend.ai_func.recommendation.orchestration.planner as _planner_mod
    _orig = (_models_mod.vision_available, _planner_mod.vision_available)
    _models_mod.vision_available = lambda: True
    _planner_mod.vision_available = lambda: True

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(AGENT_EPISODES_TABLE_SQL)
    for s in AGENT_EPISODES_INDEXES_SQL:
        conn.execute(s)
    conn.commit()
    svc = EpisodicMemoryService(repo=AgentEpisodeRepository(conn=conn))
    conv = "conversation-A"

    # Turn 1 —— 分组任务成功 → episode 写入
    p1 = _SpyPlanner()
    _run(_orch(p1), [{"role": "user", "content": "找旅行照片并按地点整理"}], conv, svc, "Turn 1")
    eps1 = svc.list_recent(conv)
    print(f"\n[Memory] agent_episodes after Turn 1 = {len(eps1)}")
    ep = eps1[0]
    print(f"  query_summary     = {ep['query_summary']}")
    print(f"  strategy_summary  = {ep['strategy_summary']}")
    print(f"  result_summary    = {ep['result_summary']}")
    print(f"  preference_hints  = {ep['preference_hints']}")
    print(f"  success           = {ep['success']}")

    # Turn 2 —— 相关 query + 视觉核验：recall → hint → Planner 收到并仍跟随当前 query
    p2 = _SpyPlanner()
    _run(_orch(p2), [{"role": "user", "content": "再帮我找类似的旅行照片，并优先做视觉核验"}], conv, svc, "Turn 2")

    recalled = svc.recall_episodes(conv, "再帮我找类似的旅行照片，并优先做视觉核验")
    hints = svc.render_memory_hints(recalled)
    print(f"\n[Memory] recall: {len(recalled)} episode(s), total rendered chars = "
          f"{sum(len(h) for h in hints)}")
    for h in hints:
        print(f"  hint: {h[:120]}...")
    print(f"\n[Planner] memory_hints received  = {'YES' if p2.last_memory_hints else 'NO'}")
    print("[Planner] user prompt (excerpt):")
    print("  " + p2.last_user_prompt.replace("\n", "\n  ")[:500])

    assert p2.last_memory_hints, "Turn 2 应收到 memory hint"
    assert "找旅行照片并按地点整理" in p2.last_user_prompt
    assert "以当前目标为准" in p2.last_user_prompt
    eps2 = svc.list_recent(conv)
    print(f"\n[Memory] agent_episodes after Turn 2 = {len(eps2)}")
    print(f"  Turn2 strategy_summary = {eps2[0]['strategy_summary']}")

    print("\n[Demo PASS] 第二轮 Planner 收到 bounded hint，且仍按当前 query 选择 analyze_visual。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

