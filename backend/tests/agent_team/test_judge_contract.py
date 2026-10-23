"""Phase 4.2-A — Judge Contract 单元测试。

覆盖：
  - JudgeInput / JudgeResult 数据模型与 build_judge_input(trace) 映射
  - Judge 抽象不可直接实例化
  - MockJudge：success → pass；recovery(revise) → flag；异常 trace → fail
  - EvaluationRunner 可选 judge：results 含 judge、judge_results 记录；无 judge 时不产出

零真实 LLM。运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_judge_contract.py
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
    Judge,
    JudgeInput,
    MockJudge,
    build_judge_input,
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
        {"id": 701, "score": 0.9, "title": "风景A", "description": "d", "tags": ["风景"],
         "metadata": {"location": "北京"}, "width": 1600, "height": 1200, "file_type": "jpg",
         "created_at": "2025-06-01T10:00:00Z", "public_url": "/static/701.jpg", "filepath": ""},
        {"id": 702, "score": 0.85, "title": "风景B", "description": "d", "tags": ["风景"],
         "metadata": {"location": "上海"}, "width": 1600, "height": 1200, "file_type": "jpg",
         "created_at": "2025-06-02T10:00:00Z", "public_url": "/static/702.jpg", "filepath": ""},
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
    defaults = dict(case_id=case_id, query="找风景图片",
                    expected_final_state="done", expect_selection=True)
    defaults.update(over)
    defaults["tags"] = tags or []
    return EvaluationCase(**defaults)


def _trace_for(tags=None, error=False):
    if error:
        class _Broken:
            last_plan_ok = False

            def run_stream(self, messages, **kwargs):
                raise RuntimeError("boom")

        runner = EvaluationRunner(runtime_factory=lambda case: _Broken())
        return runner.run_case(_case("c_error"))
    if tags and "recover" in tags:
        runner = EvaluationRunner(runtime_factory=lambda case:
                                  _make_supervisor(_FakeSearch([[], _rows()])))
    else:
        runner = EvaluationRunner(runtime_factory=lambda case: _make_supervisor(_FakeSearch([_rows()])))
    return runner.run_case(_case("c_trace", tags=tags or []))


def test_judge_input_built_from_trace():
    trace = _trace_for()
    judge_input = build_judge_input(trace)
    assert isinstance(judge_input, JudgeInput)
    assert judge_input.case_id == trace.case_id
    assert judge_input.plan_ok is True
    assert judge_input.final_state == "done"
    assert judge_input.selection_ids == [701, 702]
    assert judge_input.metrics["succeeded"] >= 2
    assert judge_input.tool_summary["names"] == ["search_images"]


def test_judge_abstract_cannot_instantiate():
    try:
        Judge()  # type: ignore[abstract]
    except TypeError:
        pass
    else:
        raise AssertionError("Judge 是抽象类，不应可直接实例化")


def test_mock_judge_pass_on_success():
    judge = MockJudge()
    trace = _trace_for()
    result = judge.evaluate_trace(trace)
    assert result.verdict == "pass"
    assert result.criterion_scores["execution"] == 5.0
    assert result.criterion_scores["tool_health"] == 5.0
    assert result.criterion_scores["selection_coverage"] == 5.0
    assert result.reasons and isinstance(result.reasons, list)


def test_mock_judge_flag_on_recovery_revise():
    judge = MockJudge()
    trace = _trace_for(tags=["recover"])  # decisions=[revise, continue]
    result = judge.evaluate_trace(trace)
    assert result.verdict == "flag"
    assert any("replan" in r or "revise" in r for r in result.reasons)


def test_mock_judge_fail_on_runtime_error():
    judge = MockJudge()
    trace = _trace_for(error=True)
    result = judge.evaluate_trace(trace)
    assert result.verdict == "fail"
    assert any("执行异常" in r for r in result.reasons) or result.criterion_scores["execution"] <= 1.0


def test_runner_optional_judge_integration():
    judge = MockJudge()
    runner = EvaluationRunner(
        runtime_factory=lambda case: _make_supervisor(_FakeSearch([_rows()])),
        judge=judge,
    )
    _, summary = runner.run([_case("judge_ok")])
    assert summary.results[0]["judge"]["verdict"] == "pass"
    assert len(runner.judge_results) == 1

    runner_no_judge = EvaluationRunner(
        runtime_factory=lambda case: _make_supervisor(_FakeSearch([_rows()])),
    )
    _, summary2 = runner_no_judge.run([_case("no_judge")])
    assert "judge" not in summary2.results[0]


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

