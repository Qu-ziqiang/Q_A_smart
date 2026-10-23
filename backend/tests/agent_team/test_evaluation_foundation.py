"""Phase 4.1 — Evaluation Foundation 单元测试。

覆盖：
  - EvaluationCase 数据模型
  - EvaluationRunner 执行 Supervisor runtime → EvaluationTrace（RunState/Collector 同源）
  - Recovery case（empty → revise → retry）的 trace / decisions
  - 预期违例（缺 agent / 缺 tool / decisions 不符）标记失败
  - runtime 抛错被捕获进 trace.error
  - 批量 Summary 计数

零真实 DB / LLM。运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_evaluation_foundation.py
"""

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.evaluation import (
    EvaluationCase,
    EvaluationRunner,
    EvaluationTrace,
    expected_violations,
)
from backend.ai_func.recommendation.orchestration import (
    PlanValidator,
    PlannerAgent,
    SupervisorAgent,
    ToolRegistry,
    build_default_agent_registry,
    build_registry_backed_workers,
    register_builtin_tools,
)
from backend.ai_func.recommendation.orchestration.workers import DeterministicSynthesisWorker
from backend.tests.agent_team.loop_support import rewrite_aware_llm


def _rows():
    return [
        {"id": 601, "score": 0.9, "title": "风景A", "description": "d", "tags": ["风景"],
         "metadata": {"location": "北京"}, "width": 1600, "height": 1200, "file_type": "jpg",
         "created_at": "2025-06-01T10:00:00Z", "public_url": "/static/601.jpg", "filepath": ""},
        {"id": 602, "score": 0.86, "title": "风景B", "description": "d", "tags": ["风景"],
         "metadata": {"location": "上海"}, "width": 1600, "height": 1200, "file_type": "jpg",
         "created_at": "2025-06-02T10:00:00Z", "public_url": "/static/602.jpg", "filepath": ""},
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


class _FakeSearch:
    def __init__(self, results_by_call):
        self.results = list(results_by_call)
        self.calls = 0

    def __call__(self, query=None, vector_targets=None, filters=None, limit=20):
        self.calls += 1
        idx = min(self.calls - 1, len(self.results) - 1)
        return [dict(r) for r in self.results[idx]]


def _make_supervisor(search_impl):
    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={"search_images": search_impl})
    workers = build_registry_backed_workers(tool_reg, with_vision=False)
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=tool_reg)
    planner = PlannerAgent(llm=rewrite_aware_llm(_plan_json()))
    return SupervisorAgent(
        planner=planner, validator=PlanValidator(), workers=workers,
        agent_registry=agent_reg, tool_registry=tool_reg,
        synthesize_worker=DeterministicSynthesisWorker(),
        enhanced=True, max_iterations=2,
    )


def _case(case_id, tags=None, **over):
    defaults = dict(
        case_id=case_id, name=case_id, query="找风景图片",
        expected_agents=["retrieval"], expected_tools=["search_images"],
        expected_final_state="done", expect_selection=True,
    )
    defaults.update(over)
    defaults["tags"] = tags or []
    return EvaluationCase(**defaults)


def _runner(search_per_case=None):
    def factory(case):
        if search_per_case is not None:
            impl = search_per_case(case)
        else:
            impl = _FakeSearch([_rows()])
        return _make_supervisor(impl)
    return EvaluationRunner(runtime_factory=factory)


def test_evaluation_case_model_defaults():
    case = EvaluationCase(case_id="c1", query="找图")
    assert case.case_id == "c1"
    assert case.name == ""
    assert case.expected_final_state == "done"
    assert case.expect_selection is True
    assert case.expected_agents == [] and case.expected_tools == [] and case.tags == []


def test_runner_success_trace_and_summary():
    runner = _runner()
    case = _case("c_success")
    (traces, summary) = runner.run([case])
    trace = traces[0]
    assert isinstance(trace, EvaluationTrace)
    assert trace.plan_ok is True
    assert trace.final_state == "done"
    assert trace.decisions == ["continue"]
    assert "retrieval" in {m["sender"] for m in trace.agent_messages}
    assert trace.tool_summary["names"] == ["search_images"]
    assert trace.tool_summary["success"] == 1
    assert trace.selection_ids == [601, 602]
    assert trace.metrics["succeeded"] >= 2
    assert trace.duration_ms >= 0.0
    assert summary.total == 1 and summary.passed == 1 and summary.failed == 0
    assert summary.results[0]["passed"] is True


def test_runner_recovery_case_decisions_trace():
    def search_for(case):
        if "recover" in case.tags:
            return _FakeSearch([[], _rows()])
        return _FakeSearch([_rows()])

    runner = EvaluationRunner(runtime_factory=lambda case: _make_supervisor(search_for(case)))
    case = _case("c_recover", tags=["recover"], expected_decisions=["replan", "continue"])
    trace = runner.run_case(case)
    assert trace.decisions == ["replan", "continue"]
    assert "replan" in trace.supervisor_states or "plan" in trace.supervisor_states
    assert trace.final_state == "done"
    assert trace.selection_ids == [601, 602]
    assert expected_violations(case, trace) == []


def test_expectation_violations_mark_failure():
    runner = _runner()
    case = _case("c_bad_expect", expected_agents=["vision"],
                 expected_tools=["analyze_image"], expected_decisions=["replan"])
    _, summary = runner.run([case])
    assert summary.failed == 1
    mismatches = summary.results[0]["mismatches"]
    assert any("vision" in m for m in mismatches)
    assert any("analyze_image" in m for m in mismatches)
    assert any("decisions" in m for m in mismatches)


def test_runner_captures_runtime_error_in_trace():
    class _BrokenRuntime:
        last_plan_ok = False

        def run_stream(self, messages, **kwargs):
            raise RuntimeError("boom")

    runner = EvaluationRunner(runtime_factory=lambda case: _BrokenRuntime())
    case = _case("c_error")
    trace = runner.run_case(case)
    assert trace.error is not None and "boom" in trace.error
    assert trace.final_state is None


def test_batch_summary_counts():
    runner = _runner()
    cases = [
        _case("ok_1", expected_decisions=["continue"]),
        _case("ok_2"),
    ]
    _, summary = runner.run(cases)
    assert summary.total == 2 and summary.passed == 2 and summary.failed == 0


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

