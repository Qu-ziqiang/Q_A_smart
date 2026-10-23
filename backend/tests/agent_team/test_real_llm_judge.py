"""Phase 4.3 — Real LLM-as-a-Judge（OpenAICompatibleClient + Benchmark）测试。
不依赖真实 API：OpenAI client 用 monkeypatch；LLMJudge/Benchmark 用 FakeLLMClient。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_real_llm_judge.py
"""

import json
import os
import sys
import traceback
from types import SimpleNamespace
from typing import List, Optional

import httpx

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.evaluation import (
    EvaluationCase,
    LLMClient,
    LLMClientError,
    LLMJudge,
    OpenAICompatibleClient,
    aggregate_judge_results,
    load_benchmark_cases,
    parse_judge_response,
    run_agent_benchmark,
)
from backend.ai_func.recommendation.evaluation.judge import JudgeInput, JudgeResult
from backend.ai_func.recommendation.evaluation.prompts.judge_prompt import (
    JudgePromptBuilder,
)
from backend.ai_func.recommendation.evaluation.benchmark_harness import (
    build_case_runtime,
)

_DATASET = os.path.join(_REPO_ROOT, "backend", "ai_func", "recommendation",
                        "evaluation", "datasets", "agent_benchmark_cases.json")
_PASS_JSON = (
    '{"verdict": "pass", "criterion_scores": {"planning_quality": 5.0, '
    '"agent_routing": 5.0, "tool_usage": 5.0, "recovery_behavior": 5.0, '
    '"final_result_quality": 5.0}, "reasons": ["ok"]}')


class FakeLLMClient(LLMClient):
    """测试用 Fake：按顺序返回预设响应，并记录收到的 prompt。"""

    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = list(responses) if responses else [_PASS_JSON]
        self.prompts: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        idx = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[idx]


_SENTINEL = object()


class _patch_attr:
    """简易 monkeypatch（不依赖 pytest fixture）。"""

    def __init__(self, obj, name, value):
        self.obj, self.name, self.value = obj, name, value
        self.old = None

    def __enter__(self):
        self.old = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)
        return self.value

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self.old)


class _no_env:
    """临时删除一组环境变量。"""

    def __init__(self, keys):
        self.keys = list(keys)
        self.saved = {}

    def __enter__(self):
        for k in self.keys:
            self.saved[k] = os.environ.pop(k, _SENTINEL)

    def __exit__(self, *exc):
        for k in self.keys:
            if self.saved.get(k, _SENTINEL) is _SENTINEL:
                os.environ.pop(k, None)
            else:
                os.environ[k] = self.saved[k]


def _sample_cases(limit: int = 8) -> List[EvaluationCase]:
    cases = load_benchmark_cases(_DATASET)
    idx = [0, 2, 5, 7, 10, 14, 19, 24]  # retrieval/empty/toolfail/org/vision/multi/complex
    return [cases[i] for i in idx if i < len(cases)][:limit]


def test_prompt_contains_role_and_bounded_context():
    ji = JudgeInput(
        case_id="c1", request_id="r1", query="找城市建筑", goal="城市建筑推荐",
        plan_tasks=[{"task_id": "t1", "task_type": "retrieve_images"}],
        agent_names=["retrieval"], tool_summary={"total": 1, "success": 1,
                                                 "failure": 0, "names": ["search_images"]},
        execution_status={"t1": "succeeded"}, metrics={"executed_tasks": 2},
        decisions=["continue"], final_state="done", selection_ids=[601, 602],
    )
    full = JudgePromptBuilder().build_full_prompt(ji)
    assert "Agent System Evaluation Expert" in full
    assert "planning_quality" in full and "final_result_quality" in full
    context = JudgePromptBuilder().build_context(ji)
    assert set(context) >= {"query", "goal", "plan", "agents", "tools",
                            "execution", "recovery", "final"}
    assert "agent_messages" not in context and "tool_events" not in context
    assert "assistant_delta" not in json.dumps(context, ensure_ascii=False)


def test_llm_judge_with_fake_client_and_parse():
    client = FakeLLMClient()
    judge = LLMJudge(client=client)
    result = judge.evaluate(JudgeInput(case_id="c1", request_id="r1", query="q",
                                       plan_ok=True, final_state="done"))
    assert isinstance(result, JudgeResult) and result.verdict == "pass"
    assert set(result.criterion_scores) == {
        "planning_quality", "agent_routing", "tool_usage",
        "recovery_behavior", "final_result_quality"}
    assert "Agent System Evaluation Expert" in client.prompts[0]
    fenced = json.loads(client.responses[0])
    assert parse_judge_response("```json\n" + json.dumps(fenced) + "\n```") == fenced

def test_openai_compatible_client_chat_and_endpoint():
    client = OpenAICompatibleClient(
        api_key="sk-test-not-real", base_url="https://llm.example.test/v1",
        model="judge-model", max_retries=1)
    seen = {}

    def fake_post(payload, attempt):
        seen["payload"] = payload
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": '{"verdict":"pass"}'}}],
                          "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        )

    old = client._post
    client._post = fake_post
    try:
        text = client.generate("hello judge")
    finally:
        client._post = old
    assert json.loads(text)["verdict"] == "pass"
    assert client._endpoint() == "https://llm.example.test/v1/chat/completions"
    assert seen["payload"]["model"] == "judge-model"
    assert seen["payload"]["messages"] == [{"role": "user", "content": "hello judge"}]
    assert "sk-test-not-real" not in repr(client)


def test_openai_compatible_client_retry_then_success():
    client = OpenAICompatibleClient(
        api_key="sk-test-not-real", base_url="https://llm.example.test/v1",
        model="m", max_retries=2, timeout_seconds=1.0)
    calls = {"n": 0}

    def flaky_post(payload, attempt):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.TransportError("boom transport")
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "ok"}}]})

    old = client._post
    client._post = flaky_post
    try:
        out = client.chat([{"role": "user", "content": "hi"}])
    finally:
        client._post = old
    assert out == "ok"
    assert calls["n"] == 3  # 2 次重试后成功


def test_openai_compatible_client_all_retries_fail_no_key_leak():
    client = OpenAICompatibleClient(
        api_key="sk-SUPER-SECRET", base_url="https://llm.example.test/v1",
        model="m", max_retries=1, timeout_seconds=1.0)

    def always_fail(payload, attempt):
        raise httpx.TransportError("down")

    old = client._post
    client._post = always_fail
    try:
        try:
            client.chat([{"role": "user", "content": "hi"}])
        except LLMClientError as e:
            assert "sk-SUPER-SECRET" not in str(e)
        else:
            raise AssertionError("应抛 LLMClientError")
    finally:
        client._post = old
    assert "sk-SUPER-SECRET" not in repr(client)


def test_openai_client_missing_env_raises():
    with _no_env(("LLM_JUDGE_API_KEY", "LLM_JUDGE_BASE_URL", "LLM_JUDGE_MODEL")):
        try:
            OpenAICompatibleClient(api_key=None, model="m")
        except LLMClientError as e:
            assert "LLM_JUDGE_API_KEY" in str(e)
        else:
            raise AssertionError("缺少 API Key 应抛错")


def test_benchmark_pipeline_with_fake_client():
    cases = _sample_cases(8)
    client = FakeLLMClient()
    traces, result, judge_results = run_agent_benchmark(
        cases=cases, runtime_factory=build_case_runtime, client=client)
    assert result.total_cases == len(cases) == 8
    assert result.verdict["pass"] == 8
    assert result.overall == 5.0
    assert all(v == 5.0 for v in result.scores.values())
    assert result.deterministic["total"] == 8
    assert result.deterministic["passed"] == 8
    assert len(client.prompts) == 8


def test_benchmark_verdict_statistics():
    from backend.ai_func.recommendation.evaluation import criterion_keys

    results = [
        JudgeResult(case_id="a", request_id="r", verdict="pass",
                    criterion_scores={k: 4.0 for k in criterion_keys()}, reasons=[]),
        JudgeResult(case_id="b", request_id="r", verdict="flag",
                    criterion_scores={k: 3.0 for k in criterion_keys()}, reasons=[]),
        JudgeResult(case_id="c", request_id="r", verdict="fail",
                    criterion_scores={k: 1.0 for k in criterion_keys()}, reasons=[]),
    ]
    agg = aggregate_judge_results(results, total=3)
    assert agg.verdict == {"pass": 1, "flag": 1, "fail": 1}
    assert agg.pass_rate == round(1 / 3, 4)
    assert agg.fail_rate == round(1 / 3, 4)
    assert agg.scores["planning_quality"] == round((4 + 3 + 1) / 3, 3)


# ---------------------------------------------------------------------------
def _main():
    failed = 0
    passed = 0
    for name in sorted(globals()):
        if name.startswith("test_"):
            fn = globals()[name]
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

