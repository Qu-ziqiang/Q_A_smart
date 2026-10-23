"""SmartImager Agent 闭环：Observe → Replan → Validate → Execute。

覆盖：
  1. Empty Retrieval → Observation → 第二次 Planner → query 改变 → Validator → Execute
  2. Low Relevance（有候选 + 低 CLIP + tag mismatch）→ replan
  3. Execution Failure → revise（同 query），不误判低相关
  4. Replan Invalid → Validator 拒绝，不部分执行
  5. 最多两轮：failure → replan → failure → abort，无第三次 Planner
  6. 首轮成功不二次规划
  7. Legacy / ComplexityGate 单步检索不受影响

零真实 DB / LLM。运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_agent_loop.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    ComplexityGate,
    PlanValidator,
    PlannerAgent,
    SupervisorAgent,
    SupervisorState,
    ToolRegistry,
    build_default_agent_registry,
    build_registry_backed_workers,
    register_builtin_tools,
)
from backend.ai_func.recommendation.orchestration.observation import (
    ReplanAction,
    TriggerType,
    build_observation,
)
from backend.ai_func.recommendation.orchestration.models import TaskType
from backend.ai_func.recommendation.orchestration.workers import (
    DeterministicSynthesisWorker,
    ExpandQueryWorker,
    ImageCandidate,
    RetrievalOutput,
    TaskStatus,
    WorkerResult,
)
from backend.tests.agent_team.loop_support import rewrite_aware_llm


def _ok_rows():
    return [
        {"id": 201, "score": 0.91, "title": "城市建筑A", "description": "a", "tags": ["城市建筑"],
         "metadata": {"location": "北京"}, "width": 1920, "height": 1280,
         "file_type": "jpg", "created_at": "2025-06-01T10:00:00Z",
         "public_url": "/static/201.jpg", "filepath": ""},
        {"id": 202, "score": 0.84, "title": "城市建筑B", "description": "b", "tags": ["城市建筑"],
         "metadata": {"location": "上海"}, "width": 1920, "height": 1080,
         "file_type": "jpg", "created_at": "2025-06-02T10:00:00Z",
         "public_url": "/static/202.jpg", "filepath": ""},
    ]


def _low_rel_rows():
    return [
        {"id": 301, "score": 0.12, "title": "办公桌杂物", "description": "desk",
         "tags": ["office", "desk"], "metadata": {}, "width": 800, "height": 600,
         "file_type": "jpg", "created_at": "2025-01-01T00:00:00Z",
         "public_url": "/static/301.jpg", "filepath": ""},
    ]


def _plan_json(query="城市建筑"):
    return {
        "goal": "找城市建筑图片",
        "synthesis_task_id": "t2",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索城市建筑图片",
             "input": {"query": query, "limit": 20}, "depends_on": []},
            {"task_id": "t2", "task_type": "synthesize", "objective": "汇总推荐",
             "input": {"recommendation_limit": 10}, "depends_on": ["t1"]},
        ],
    }


def _invalid_replan_json():
    data = _plan_json("城市建筑 改写")
    data["tasks"][1]["depends_on"] = ["t1", "unknown_dep"]
    return data


class _FakeSearch:
    def __init__(self, results_by_call, fail=False, queries=None):
        self.results = list(results_by_call)
        self.calls = 0
        self.fail = fail
        self.queries = list(queries or [])

    def __call__(self, query=None, vector_targets=None, filters=None, limit=20):
        self.queries.append(query)
        if self.fail:
            self.calls += 1
            raise RuntimeError("search tool down")
        idx = min(self.calls, len(self.results) - 1) if self.results else 0
        self.calls += 1
        return list(self.results[idx])


def _make_supervisor(search_impl, *, planner=None, plan_json=None, backend="legacy",
                     enhanced=True, max_iterations=2, min_candidates=1):
    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={"search_images": search_impl})
    workers = build_registry_backed_workers(tool_reg, with_vision=False)
    workers[TaskType.EXPAND_QUERY] = ExpandQueryWorker(
        llm_fn=lambda q, vocab: f"{q} 街景"
    )
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=tool_reg)
    if planner is None:
        planner = PlannerAgent(llm=rewrite_aware_llm(plan_json or _plan_json()))
    return SupervisorAgent(
        planner=planner,
        validator=PlanValidator(),
        workers=workers,
        agent_registry=agent_reg,
        tool_registry=tool_reg,
        synthesize_worker=DeterministicSynthesisWorker(),
        enhanced=enhanced,
        max_iterations=max_iterations,
        min_candidates=min_candidates,
        backend=backend,
    )


def _run(supervisor, query="找城市建筑图片"):
    events = [json.loads(raw) for raw in supervisor.run_stream(
        [{"role": "user", "content": query}])]
    reviews = [e for e in events if e["event"] == "agent.review"]
    observes = [e for e in events if e["event"] == "agent.observe"]
    plans = [e for e in events if e["event"] == "plan.created"]
    failed = [e for e in events if e["event"] == "plan.failed"]
    selections = [e for e in events if e["event"] == "selection"]
    decisions = [r["decision"] for r in reviews]
    return events, decisions, selections, reviews, observes, plans, failed


def _retrieve_query(plan):
    for t in plan.tasks:
        if t.task_type.value == "retrieve_images":
            return getattr(t.input, "query", None)
    return None


def test_empty_retrieval_replans_with_new_query():
    search = _FakeSearch([[], _ok_rows()])
    supervisor = _make_supervisor(search)
    events, decisions, selections, reviews, observes, plans, _ = _run(supervisor)
    assert decisions == ["replan", "continue"]
    assert supervisor.run_state.final_state == SupervisorState.DONE
    assert supervisor.planner_call_count == 1
    assert search.calls == 2
    assert any(t.task_type == TaskType.EXPAND_QUERY for t in supervisor.run_state.plan.tasks)
    assert observes[0]["trigger"] == TriggerType.EMPTY_RETRIEVAL.value
    assert observes[0]["REPLAN_REASON"] == "EMPTY_RETRIEVAL"
    assert observes[0]["REPLAN_ACTION"] == "RETRY_REWRITE"
    assert observes[0]["retrieval"]["candidate_count"] == 0
    assert observes[0]["retrieval"]["empty"] is True
    assert search.queries[0] != search.queries[1]
    assert selections[-1]["ids"] == [201, 202]
    assert len(plans) == 2
    assert plans[0]["iteration"] == 1
    assert plans[1]["iteration"] == 2


def test_low_relevance_replans():
    search = _FakeSearch([_low_rel_rows(), _ok_rows()])
    supervisor = _make_supervisor(search)
    from unittest.mock import patch
    with patch(
        "backend.ai_func.recommendation.orchestration.observation.vision_available",
        return_value=False,
    ):
        _, decisions, selections, _, observes, _, _ = _run(supervisor)
    assert decisions == ["replan", "continue"]
    assert observes[0]["trigger"] == TriggerType.LOW_RELEVANCE.value
    assert observes[0]["REPLAN_REASON"] == "LOW_RELEVANCE"
    assert observes[0]["retrieval"]["candidate_count"] == 1
    assert observes[0]["retrieval"]["empty"] is False
    assert observes[0]["retrieval"]["top1_score"] < 0.4
    assert observes[0]["retrieval"]["tag_hit"] is False
    assert supervisor.planner_call_count == 1
    assert any(t.task_type == TaskType.EXPAND_QUERY for t in supervisor.run_state.plan.tasks)
    assert selections[-1]["ids"] == [201, 202]


def test_execution_failure_retries_same_query_not_low_relevance():
    search = _FakeSearch([_ok_rows()], fail=True)
    supervisor = _make_supervisor(search)
    _, decisions, _, reviews, observes, _, _ = _run(supervisor)
    assert decisions == ["revise", "abort"]
    assert observes[0]["trigger"] == TriggerType.EXECUTION_FAILURE.value
    assert observes[0]["REPLAN_REASON"] == "EXECUTION_FAILURE"
    assert observes[0]["suggested_action"] == ReplanAction.RETRY_SAME.value
    assert observes[0]["trigger"] != TriggerType.LOW_RELEVANCE.value
    assert supervisor.planner_call_count == 1
    assert search.calls == 2
    assert search.queries[0] == search.queries[1]


def test_replan_invalid_rejected_no_partial_execute():
    state = {"n": 0}

    def flaky_llm(messages):
        state["n"] += 1
        if state["n"] == 1:
            return json.dumps(_plan_json(), ensure_ascii=False)
        return json.dumps(_invalid_replan_json(), ensure_ascii=False)

    search = _FakeSearch([[]])
    supervisor = _make_supervisor(search, planner=PlannerAgent(llm=flaky_llm))
    events, decisions, selections, _, _, plans, failed = _run(supervisor)
    assert not failed
    plan = supervisor.run_state.plan
    assert plan is not None
    deps = [d for t in plan.tasks for d in (t.depends_on or [])]
    assert "unknown_dep" not in deps
    assert any(t.task_type.value == "retrieve_images" for t in plan.tasks)


def test_maximum_two_iterations_no_third_planner():
    search = _FakeSearch([[], []])
    supervisor = _make_supervisor(search)
    _, decisions, _, _, observes, _, _ = _run(supervisor)
    assert decisions == ["replan", "abort"]
    assert supervisor.run_state.final_state == SupervisorState.ABORT
    assert supervisor.planner_call_count == 1
    assert search.calls == 2
    assert observes[0]["trigger"] == TriggerType.EMPTY_RETRIEVAL.value
    assert observes[1]["trigger"] == TriggerType.EMPTY_RETRIEVAL.value
    execute_attempts = [t["attempt"] for t in supervisor.run_state.supervisor_trace
                        if t["state"] == "execute"]
    assert execute_attempts == [1, 2]


def test_normal_success_single_planner():
    search = _FakeSearch([_ok_rows()])
    supervisor = _make_supervisor(search)
    events, decisions, selections, _, observes, plans, _ = _run(supervisor)
    assert decisions == ["continue"]
    assert supervisor.planner_call_count == 1
    assert search.calls == 1
    assert len(plans) == 1
    assert observes[0]["trigger"] == TriggerType.NONE.value
    assert observes[0]["REPLAN_REASON"] == ""
    assert selections[-1]["ids"] == [201, 202]
    assert supervisor.run_state.final_state == SupervisorState.DONE


def test_legacy_single_step_unaffected():
    gate = ComplexityGate()
    simple = gate.decide(query="你好")
    assert simple.route.value == "legacy"
    complex_dec = gate.decide(query="找出旅行照片，按地点分类，然后推荐最合适的几张")
    assert complex_dec.route.value == "team"

    search = _FakeSearch([[]])
    supervisor = _make_supervisor(search, enhanced=False)
    events, decisions, _, _, observes, plans, _ = _run(supervisor)
    assert decisions == ["continue"]
    assert supervisor.planner_call_count == 1
    assert search.calls == 1
    assert observes == []
    assert len(plans) == 1
    names = [e["event"] for e in events]
    assert "plan.created" in names
    assert names.count("plan.created") == 1

    from backend.ai_func.recommendation import _RecommendationService

    class FakeAgent:
        def run_stream(self, messages, vector_targets=None, filters=None, limit=20):
            yield json.dumps({"event": "assistant_delta", "delta": "legacy"}, ensure_ascii=False) + "\n\n"
            yield json.dumps({"event": "selection", "ids": [7], "images": []}, ensure_ascii=False) + "\n\n"

    service = _RecommendationService()
    service._agent = FakeAgent()
    evs = [json.loads(raw) for raw in service.stream_core(
        [{"role": "user", "content": "你好"}])]
    assert "plan.created" not in [e["event"] for e in evs]
    assert evs[-1]["event"] == "selection"
    assert evs[-1]["ids"] == [7]


def test_empty_and_low_relevance_are_distinct():
    empty_obs = build_observation(
        _validated_plan(),
        {
            "t1": WorkerResult(
                task_id="t1", task_type="retrieve_images", status=TaskStatus.SUCCEEDED,
                output=RetrievalOutput(candidates=[], query="城市建筑", total=0),
            ),
        },
        iteration=1, goal="找城市建筑图片",
    )
    assert empty_obs.trigger == TriggerType.EMPTY_RETRIEVAL

    low = build_observation(
        _validated_plan(),
        {
            "t1": WorkerResult(
                task_id="t1", task_type="retrieve_images", status=TaskStatus.SUCCEEDED,
                output=RetrievalOutput(
                    candidates=[ImageCandidate(id=1, score=0.11, title="desk", tags=["office"])],
                    query="城市建筑", total=1,
                ),
            ),
        },
        iteration=1, goal="找城市建筑图片",
    )
    assert low.trigger == TriggerType.LOW_RELEVANCE
    assert low.retrieval.candidate_count == 1


def _validated_plan():
    return PlanValidator().validate(
        PlannerAgent(llm=lambda m: json.dumps(_plan_json(), ensure_ascii=False)).create_plan(
            goal="找城市建筑图片"
        )
    ).plan


def test_langgraph_empty_replan_changes_query():
    search = _FakeSearch([[], _ok_rows()])
    supervisor = _make_supervisor(search, backend="langgraph")
    _, decisions, selections, _, observes, plans, _ = _run(supervisor)
    assert decisions == ["replan", "continue"]
    assert supervisor.planner_call_count == 1
    assert observes[0]["REPLAN_REASON"] == "EMPTY_RETRIEVAL"
    assert search.queries[0] != search.queries[1]
    assert selections[-1]["ids"] == [201, 202]
    assert len(plans) == 2


def test_planner_first_prompt_has_no_observation():
    captured = []

    def llm(messages):
        captured.append(messages)
        return json.dumps(_plan_json(), ensure_ascii=False)

    PlannerAgent(llm=llm).create_plan(goal="找城市建筑图片")
    assert "上一轮执行观察" not in captured[0][1]["content"]


def test_planner_second_prompt_consumes_observation():
    captured = []

    def llm(messages):
        captured.append(messages)
        data = copy.deepcopy(_plan_json())
        data["tasks"][0]["input"]["query"] = "城市建筑 夜景"
        return json.dumps(data, ensure_ascii=False)

    obs = build_observation(
        _validated_plan(),
        {
            "t1": WorkerResult(
                task_id="t1", task_type="retrieve_images", status=TaskStatus.SUCCEEDED,
                output=RetrievalOutput(candidates=[], query="城市建筑", total=0),
            ),
        },
        iteration=1, goal="找城市建筑图片",
    )
    PlannerAgent(llm=llm).create_plan(goal="找城市建筑图片", observation=obs)
    text = captured[0][1]["content"]
    assert "上一轮执行观察" in text
    assert "EMPTY_RETRIEVAL" in text
    assert "expand_query" in text


# ---------------------------------------------------------------------------
def _main():
    failed = 0
    passed = 0
    for name in sorted(globals()):
        if name.startswith("test_"):
            try:
                globals()[name]()
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
