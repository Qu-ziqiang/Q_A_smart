"""Phase 4.2-B — LLMJudge / Rubric / Prompt / MockLLMClient 单元测试。

覆盖：
  - JudgePromptBuilder：JudgeInput → 精简上下文（query/agents/tools/...）
  - MockLLMClient → LLMJudge → JudgeResult 链路
  - pass / flag / invalid JSON / markdown 围栏 输出
  - EvaluationRunner(judge=LLMJudge(MockLLMClient())) 运行成功

零真实 LLM / DB。运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_llm_judge.py
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
    JudgePromptBuilder,
    LLMJudge,
    MockLLMClient,
    build_judge_input,
    criterion_keys,
    parse_judge_response,
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

_CRITERIA = ["planning_quality", "agent_routing", "tool_usage",
             "recovery_behavior", "final_result_quality"]

_PASS_RESPONSE = json.dumps({
    "verdict": "pass",
    "criterion_scores": {k: 5.0 for k in _CRITERIA},
    "reasons": ["符合目标"],
}, ensure_ascii=False)

_FLAG_RESPONSE = json.dumps({
    "verdict": "flag",
    "criterion_scores": {k: 3.5 for k in _CRITERIA},
    "reasons": ["经历 revise 恢复"],
}, ensure_ascii=False)


def _rows():
    return [
        {"id": 801, "score": 0.9, "title": "风景A", "description": "d", "tags": ["风景"],
         "metadata": {"location": "北京"}, "width": 1600, "height": 1200, "file_type": "jpg",
         "created_at": "2025-06-01T10:00:00Z", "public_url": "/static/801.jpg", "filepath": ""},
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
    def __call__(self, query=None, vector_targets=None, filters=None, limit=20):
        return [dict(r) for r in _rows()]


def _make_supervisor():
    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={"search_images": _FakeSearch()})
    workers = build_registry_backed_workers(tool_reg, with_vision=False)
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=tool_reg)
    planner = PlannerAgent(llm=lambda messages: json.dumps(_plan_json(), ensure_ascii=False))
    return SupervisorAgent(
        planner=planner, validator=PlanValidator(), workers=workers,
        agent_registry=agent_reg, tool_registry=tool_reg,
        synthesize_worker=DeterministicSynthesisWorker(),
        enhanced=True, max_iterations=2,
    )


def _trace():
    runner = EvaluationRunner(runtime_factory=lambda case: _make_supervisor())
    return runner.run_case(EvaluationCase(case_id="c_llm", query="找风景图片"))


def test_criteria_keys_align_rubric():
    assert criterion_keys() == _CRITERIA


def test_prompt_builder_compact_context():
    trace = _trace()
    judge_input = build_judge_input(trace)
    builder = JudgePromptBuilder()
    context = builder.build_context(judge_input)
    for key in ("query", "plan_status", "agents", "tools", "recovery", "final_state", "selection"):
        assert key in context, key
    assert context["query"] == "找风景图片"
    assert context["agents"] == ["retrieval"]
    assert context["plan_status"]["plan_ok"] is True
    user_prompt = builder.build_user_prompt(judge_input)
    assert '"query"' in user_prompt and '"final_state"' in user_prompt
    # 精简上下文：不 dump 全量 trace
    assert "plan_tasks" not in context and "tool_events" not in context


def test_mock_llm_client_records_prompt():
    client = MockLLMClient(response=_PASS_RESPONSE)
    judge = LLMJudge(client=client)
    trace = _trace()
    result = judge.evaluate(build_judge_input(trace))
    assert result.verdict == "pass"
    assert client.prompts, "MockLLMClient 应记录 prompt"
    assert all(k in client.prompts[0] for k in _CRITERIA)


def test_llm_judge_pass_response():
    judge = LLMJudge(client=MockLLMClient(response=_PASS_RESPONSE))
    trace = _trace()
    result = judge.evaluate_trace(trace)
    assert result.verdict == "pass"
    assert set(result.criterion_scores.keys()) == set(_CRITERIA)
    assert all(v == 5.0 for v in result.criterion_scores.values())
    assert result.error is None


def test_llm_judge_flag_response():
    judge = LLMJudge(client=MockLLMClient(response=_FLAG_RESPONSE))
    trace = _trace()
    result = judge.evaluate_trace(trace)
    assert result.verdict == "flag"
    assert result.criterion_scores["tool_usage"] == 3.5


def test_llm_judge_invalid_json_records_error():
    judge = LLMJudge(client=MockLLMClient(response="this is not json {"))
    trace = _trace()
    result = judge.evaluate_trace(trace)
    assert result.verdict == "flag"
    assert result.error == "invalid judge response"


def test_parse_judge_response_handles_markdown_and_plain():
    plain = parse_judge_response(_PASS_RESPONSE)
    assert plain["verdict"] == "pass"
    fenced = parse_judge_response("```json\n" + _PASS_RESPONSE + "\n```")
    assert fenced["verdict"] == "pass"
    try:
        parse_judge_response("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 JSON 应抛 ValueError")


def test_runner_with_llm_judge_succeeds():
    judge = LLMJudge(client=MockLLMClient(response=_PASS_RESPONSE))
    runner = EvaluationRunner(
        runtime_factory=lambda case: _make_supervisor(),
        judge=judge,
    )
    _, summary = runner.run([EvaluationCase(case_id="c_r", query="找风景图片")])
    assert summary.results[0]["judge"]["verdict"] == "pass"
    assert len(runner.judge_results) == 1


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

