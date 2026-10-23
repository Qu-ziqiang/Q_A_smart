"""Phase 4.1/4.2/4.3 — Evaluation 包。

对外 API：
  - models: EvaluationCase / EvaluationTrace / EvaluationSummary
  - runner: EvaluationRunner / build_evaluation_trace / expected_violations
  - judge:  JudgeInput / JudgeResult / Judge / MockJudge / build_judge_input
  - llm_judge: LLMJudge / parse_judge_response
  - llm_client: LLMClient / MockLLMClient / LLMClientError
  - providers: OpenAICompatibleClient（真实外部 LLM，OpenAI-compatible）
  - rubric: JudgeCriterion / JUDGE_CRITERIA / criterion_keys / rubric_for_prompt
  - prompts: JudgePromptBuilder
  - benchmark: BenchmarkResult / run_agent_benchmark / load_benchmark_cases /
               save_benchmark_result / aggregate_judge_results
  - benchmark_harness: 确定性场景运行时（Fake 工具，供 script/测试/演示）
"""

from .models import EvaluationCase, EvaluationSummary, EvaluationTrace
from .runner import EvaluationRunner, build_evaluation_trace, expected_violations
from .judge import Judge, JudgeInput, JudgeResult, MockJudge, build_judge_input
from .llm_client import LLMClient, LLMClientError, MockLLMClient
from .llm_judge import LLMJudge, parse_judge_response
from .providers import OpenAICompatibleClient
from .rubric import JUDGE_CRITERIA, JudgeCriterion, criterion_keys, rubric_for_prompt
from .prompts.judge_prompt import JudgePromptBuilder
from .benchmark import (
    BenchmarkResult,
    aggregate_judge_results,
    load_benchmark_cases,
    run_agent_benchmark,
    save_benchmark_result,
)

__all__ = [
    "EvaluationCase",
    "EvaluationTrace",
    "EvaluationSummary",
    "EvaluationRunner",
    "build_evaluation_trace",
    "expected_violations",
    "Judge",
    "JudgeInput",
    "JudgeResult",
    "MockJudge",
    "build_judge_input",
    "LLMClient",
    "LLMClientError",
    "MockLLMClient",
    "LLMJudge",
    "parse_judge_response",
    "OpenAICompatibleClient",
    "JudgeCriterion",
    "JUDGE_CRITERIA",
    "criterion_keys",
    "rubric_for_prompt",
    "JudgePromptBuilder",
    "BenchmarkResult",
    "aggregate_judge_results",
    "load_benchmark_cases",
    "run_agent_benchmark",
    "save_benchmark_result",
]
