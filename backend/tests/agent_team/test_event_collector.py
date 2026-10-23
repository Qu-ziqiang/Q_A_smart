"""Phase 3.2 — Event Collector 抽象 + Supervisor 接线单元测试。

覆盖：
  - EventCollector / InMemoryEventCollector：record_* 与 snapshot()
  - SupervisorAgent 注入 collector：RunState 与 Collector 双写一致
  - Supervisor 运行期间 tool_events / agent_messages 进入 Collector

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_event_collector.py
"""

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    InMemoryEventCollector,
    PlanValidator,
    PlannerAgent,
    SupervisorAgent,
    ToolRegistry,
    build_default_agent_registry,
    build_registry_backed_workers,
    register_builtin_tools,
)
from backend.ai_func.recommendation.orchestration.workers import DeterministicSynthesisWorker


class _FakeSearch:
    def __init__(self, results):
        self.results = list(results)

    def __call__(self, query=None, vector_targets=None, filters=None, limit=20):
        return [dict(r) for r in self.results]


def _rows():
    return [
        {"id": 501, "score": 0.9, "title": "风景A", "description": "d", "tags": ["风景"],
         "metadata": {"location": "北京"}, "width": 1600, "height": 1200, "file_type": "jpg",
         "created_at": "2025-06-01T10:00:00Z", "public_url": "/static/501.jpg", "filepath": ""},
    ]


def _plan_json():
    return {
        "goal": "找风景图片",
        "synthesis_task_id": "t2",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
             "input": {"query": "风景", "limit": 20}, "depends_on": []},
            {"task_id": "t2", "task_type": "synthesize", "objective": "汇总",
             "input": {"recommendation_limit": 5}, "depends_on": ["t1"]},
        ],
    }


def test_in_memory_collector_records_and_snapshots():
    col = InMemoryEventCollector()

    class _E:
        tool_name = "search_images"
        success = True
        latency_ms = 12.0
        agent_id = "retrieval"
        error = None

    class _M:
        sender = "supervisor"
        receiver = "retrieval"

    col.record_tool_event(_E())
    col.record_tool_event(_E())
    col.record_agent_message(_M())
    col.record_agent_message(_M())
    snap = col.snapshot()
    assert snap["tool_event_count"] == 2
    assert snap["agent_message_count"] == 2
    assert snap["tool_success_count"] == 2 and snap["tool_failure_count"] == 0
    assert len(col.tool_events) == 2 and len(col.agent_messages) == 2
    col.reset()
    assert col.snapshot()["tool_event_count"] == 0


def test_supervisor_routes_events_to_injected_collector():
    collector = InMemoryEventCollector()
    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={"search_images": _FakeSearch(_rows())})
    workers = build_registry_backed_workers(tool_reg, with_vision=False)
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=tool_reg)
    planner = PlannerAgent(llm=lambda messages: json.dumps(_plan_json(), ensure_ascii=False))
    sup = SupervisorAgent(
        planner=planner, validator=PlanValidator(), workers=workers,
        agent_registry=agent_reg, tool_registry=tool_reg,
        synthesize_worker=DeterministicSynthesisWorker(),
        enhanced=True, collector=collector,
    )
    events = [json.loads(raw) for raw in sup.run_stream(
        [{"role": "user", "content": "找风景图片"}])]
    names = [e["event"] for e in events]
    assert "selection" in names and "agent.review" in names
    # Collector 与 RunState 双写一致
    assert len(collector.tool_events) == len(sup.run_state.tool_events) == 1
    assert len(collector.agent_messages) == len(sup.run_state.agent_messages) >= 2
    snap = collector.snapshot()
    assert snap["tool_event_count"] == 1
    assert snap["tool_names"] == ["search_images"]


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
