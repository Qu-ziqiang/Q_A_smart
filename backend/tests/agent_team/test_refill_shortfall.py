"""筛选后张数不够时，Supervisor 要再检索一轮，而不是直接交卷。"""

import json
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    PlanValidator,
    PlannerAgent,
    SupervisorAgent,
    SupervisorState,
    ToolRegistry,
    build_default_agent_registry,
    build_registry_backed_workers,
    register_builtin_tools,
)
from backend.ai_func.recommendation.orchestration.workers import DeterministicSynthesisWorker
from backend.tests.agent_team.loop_support import rewrite_aware_llm


def _row(i, bird):
    tags = ["风景", "鸟"] if bird else ["风景"]
    return {
        "id": i, "score": 0.9, "title": f"风景{i}", "description": "风景",
        "tags": tags, "metadata": {}, "width": 800, "height": 600,
        "file_type": "jpg", "created_at": "2025-06-01T10:00:00Z",
        "public_url": f"/static/{i}.jpg", "filepath": "",
    }


def _first_page():
    rows = [_row(i, bird=False) for i in range(1, 9)]
    rows.extend([_row(9, bird=True), _row(10, bird=True)])
    return rows


def _second_page():
    return [_row(i, bird=True) for i in range(11, 21)]


def _plan_json():
    return {
        "goal": "找十张有鸟的风景图",
        "synthesis_task_id": "t3",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索风景",
             "input": {"query": "风景", "limit": 10}, "depends_on": []},
            {"task_id": "t2", "task_type": "filter_metadata", "objective": "只留有鸟的",
             "input": {"predicates": [{"field": "tags", "operator": "contains", "value": "鸟"}]},
             "depends_on": ["t1"]},
            {"task_id": "t3", "task_type": "synthesize", "objective": "汇总",
             "input": {"recommendation_limit": 10}, "depends_on": ["t2"]},
        ],
    }


class _FakeSearch:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = 0

    def __call__(self, query=None, vector_targets=None, filters=None, limit=20):
        idx = min(self.calls, len(self.pages) - 1)
        self.calls += 1
        return list(self.pages[idx])[: int(limit)]


def test_filter_shortfall_retrieves_again_then_fills():
    search = _FakeSearch([_first_page(), _second_page()])
    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={"search_images": search})
    workers = build_registry_backed_workers(tool_reg, with_vision=False)
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=tool_reg)
    supervisor = SupervisorAgent(
        planner=PlannerAgent(llm=rewrite_aware_llm(_plan_json())),
        validator=PlanValidator(),
        workers=workers,
        agent_registry=agent_reg,
        tool_registry=tool_reg,
        synthesize_worker=DeterministicSynthesisWorker(),
        enhanced=True,
        max_iterations=2,
    )
    events = [json.loads(raw) for raw in supervisor.run_stream(
        [{"role": "user", "content": "找十张有鸟的风景图"}]
    )]
    decisions = [e["decision"] for e in events if e["event"] == "agent.review"]
    selections = [e for e in events if e["event"] == "selection"]
    assert decisions == ["replan", "continue"]
    assert supervisor.run_state.final_state == SupervisorState.DONE
    assert search.calls == 2
    assert selections and selections[-1]["ids"] == list(range(11, 21))
