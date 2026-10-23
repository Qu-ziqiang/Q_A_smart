"""Phase 6 — LangGraph Supervisor Runtime 测试。

覆盖：
  1. normal success（LangGraph：一次 EXECUTE → continue → finalize → done）
  2. empty retrieval → review revise → retry → done
  3. tool failure → revise → max_iterations → abort（无无限循环）
  4. baseline compatibility：enhanced=False legacy 单次行为不变；
     langgraph 后端输出与 legacy(enhanced) 等价
  5. state persistence：MemorySaver checkpoint 保留 AgentRuntimeState

全部 Fake / Mock（经 ToolRegistry + adapter 注入），零真实 DB / LLM / 视觉 API。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_langgraph_supervisor.py
"""

import json
import os
import sys
import traceback

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
from backend.ai_func.recommendation.orchestration.workers import (
    DeterministicSynthesisWorker,
)
from backend.tests.agent_team.loop_support import rewrite_aware_llm


def _rows():
    return [
        {"id": 201, "score": 0.91, "title": "城市建筑A", "description": "a", "tags": ["建筑"],
         "metadata": {"location": "北京"}, "width": 1920, "height": 1280,
         "file_type": "jpg", "created_at": "2025-06-01T10:00:00Z",
         "public_url": "/static/201.jpg", "filepath": ""},
        {"id": 202, "score": 0.84, "title": "城市建筑B", "description": "b", "tags": ["建筑"],
         "metadata": {"location": "上海"}, "width": 1920, "height": 1080,
         "file_type": "jpg", "created_at": "2025-06-02T10:00:00Z",
         "public_url": "/static/202.jpg", "filepath": ""},
    ]


def _plan_json():
    return {
        "goal": "找城市建筑图片",
        "synthesis_task_id": "t2",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索城市建筑图片",
             "input": {"query": "城市建筑", "limit": 20}, "depends_on": []},
            {"task_id": "t2", "task_type": "synthesize", "objective": "汇总推荐",
             "input": {"recommendation_limit": 10}, "depends_on": ["t1"]},
        ],
    }


class _FakeSearch:
    """按调用次序返回固定结果的 search_images 实现（经 ToolRegistry 注入）。"""

    def __init__(self, results_by_call, fail=False):
        self.results = list(results_by_call)
        self.calls = 0
        self.fail = fail

    def __call__(self, query=None, vector_targets=None, filters=None, limit=20):
        if self.fail:
            self.calls += 1
            raise RuntimeError("search tool down")
        idx = min(self.calls, len(self.results) - 1) if self.results else 0
        self.calls += 1
        return list(self.results[idx])


def _make_supervisor(search_impl, *, enhanced=True, max_iterations=2, min_candidates=1,
                     backend="langgraph", planner=None):
    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={"search_images": search_impl})
    workers = build_registry_backed_workers(tool_reg, with_vision=False)
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=tool_reg)
    if planner is None:
        planner = PlannerAgent(llm=rewrite_aware_llm(_plan_json()))
    return SupervisorAgent(
        planner=planner, validator=PlanValidator(), workers=workers,
        agent_registry=agent_reg, tool_registry=tool_reg,
        synthesize_worker=DeterministicSynthesisWorker(),
        enhanced=enhanced, max_iterations=max_iterations,
        min_candidates=min_candidates, backend=backend,
    )


def _run(supervisor):
    query = "找城市建筑图片"
    events = [json.loads(raw) for raw in supervisor.run_stream(
        [{"role": "user", "content": query}])]
    reviews = [e for e in events if e["event"] == "agent.review"]
    decisions = [r["decision"] for r in reviews]
    selections = [e for e in events if e["event"] == "selection"]
    return events, decisions, selections, reviews


def test_normal_success():
    search = _FakeSearch([_rows()])
    sup = _make_supervisor(search)
    events, decisions, selections, _ = _run(sup)
    assert decisions == ["continue"]
    assert sup.run_state.final_state == SupervisorState.DONE
    assert search.calls == 1
    assert selections[-1]["ids"] == [201, 202]
    names = [e["event"] for e in events]
    assert "plan.created" in names and "assistant_delta" in names and "selection" in names
    assert sup._lg_graph is not None  # 已编译 LangGraph


def test_empty_retrieval_revise_retry():
    search = _FakeSearch([[], _rows()])
    sup = _make_supervisor(search)
    _, decisions, selections, _ = _run(sup)
    assert decisions == ["replan", "continue"]
    assert sup.run_state.final_state == SupervisorState.DONE
    assert search.calls == 2
    assert selections[-1]["ids"] == [201, 202]
    attempts = [t["attempt"] for t in sup.run_state.supervisor_trace
                if t["state"] == "execute"]
    assert attempts == [1, 2]


def test_tool_failure_abort_limit():
    search = _FakeSearch([_rows()], fail=True)
    sup = _make_supervisor(search)
    _, decisions, selections, _ = _run(sup)
    assert decisions == ["revise", "abort"]
    assert sup.run_state.final_state == SupervisorState.ABORT
    assert search.calls == 2  # 两次 EXECUTE 后停止（无无限循环）
    assert selections[-1]["ids"] == []
    failures = [e for e in sup.run_state.tool_events if not e.success]
    assert failures and failures[0].tool_name == "search_images"


def test_baseline_compatibility():
    # (a) enhanced=False 仍为 legacy 单次执行（不重试）
    base = _make_supervisor(_FakeSearch([[]]), enhanced=False)
    _, decisions, selections, _ = _run(base)
    assert decisions == ["continue"]
    assert selections[-1]["ids"] == []

    # (b) langgraph 后端与 legacy(enhanced) 后端输出等价
    lg = _make_supervisor(_FakeSearch([[], _rows()]), backend="langgraph")
    leg = _make_supervisor(_FakeSearch([[], _rows()]), backend="legacy")
    _, d1, s1, _ = _run(lg)
    _, d2, s2, _ = _run(leg)
    assert d1 == d2 == ["replan", "continue"]
    assert s1[-1]["ids"] == s2[-1]["ids"] == [201, 202]
    assert lg.run_state.final_state == leg.run_state.final_state == SupervisorState.DONE


def test_state_persistence_checkpoint():
    search = _FakeSearch([_rows()])
    sup = _make_supervisor(search)
    for _ in sup.run_stream([{"role": "user", "content": "找城市建筑图片"}]):
        pass
    st = sup._lg_graph.get_state(sup._lg_graph_config)
    values = st.values
    assert values["final_state"] == "done"
    assert values["iteration_count"] == 1
    assert values["plan"] is not None
    assert len(values["tool_events"]) == 1
    assert any(e["event"] == "selection" for e in values["events"])


def test_replan_planner_failure_falls_back_not_empty():
    """replan 阶段 planner 失败 → 硬底线 fallback retrieve→synth，不交空计划。"""
    state = {"n": 0}

    def flaky_llm(messages):
        state["n"] += 1
        if state["n"] > 1:
            raise RuntimeError("planner boom")
        return json.dumps(_plan_json(), ensure_ascii=False)

    search = _FakeSearch([_rows()[:1]])
    sup = _make_supervisor(search, min_candidates=3,
                           planner=PlannerAgent(llm=flaky_llm))
    events = [json.loads(raw) for raw in sup.run_stream(
        [{"role": "user", "content": "找城市建筑图片"}])]
    names = [e["event"] for e in events]
    assert "plan.failed" not in names
    assert sup.run_state.plan is not None
    assert any(
        t.task_type.value == "retrieve_images" for t in sup.run_state.plan.tasks
    )
    assert state["n"] >= 2


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

