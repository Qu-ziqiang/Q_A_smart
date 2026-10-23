"""Phase 4.3 — Agent Quality Benchmark（evaluation 纯聚合层，不 import Agent Runtime）。

流程：
  case (EvaluationCase)
    → runtime_factory(case) → run_stream  → EvaluationTrace
    → LLMJudge（真实 client 或 Fake/Mock）→ JudgeResult
    → BenchmarkResult（criterion 均值 / overall / verdict 计数 / 通过率）

runtim 场景由调用方（scripts/run_agent_benchmark.py 或测试）经 runtime_factory 注入，
保持本包与 Agent Runtime 解耦。
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from .judge import JudgeResult
from .llm_client import LLMClient, MockLLMClient
from .llm_judge import LLMJudge
from .models import EvaluationCase, EvaluationTrace
from .runner import EvaluationRunner
from .rubric import criterion_keys


class BenchmarkResult(BaseModel):
    """一次 Agent Benchmark 的统计结果。"""

    model_config = ConfigDict(extra="forbid")

    total_cases: int = 0
    scores: Dict[str, float] = {}          # 五个 rubric 维度的平均分
    criterion_average: Dict[str, float] = {}  # 与 scores 相同（别名，便于消费方）
    overall: float = 0.0
    verdict: Dict[str, int] = {"pass": 0, "flag": 0, "fail": 0}
    # 通过率定义：pass = LLM Judge verdict == "pass"；fail_rate 同理（flag 单列）
    pass_rate: float = 0.0
    fail_rate: float = 0.0
    # 确定性框架校验（expected_violations）：Framework validation 与 Judge 分开报告
    deterministic: Dict[str, Any] = {"total": 0, "passed": 0, "failed": 0}
    per_case: List[Dict[str, Any]] = []


def _avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def aggregate_judge_results(results: List[JudgeResult], total: int) -> BenchmarkResult:
    """把 LLM Judge 输出聚合成 BenchmarkResult。"""
    scores = {key: [] for key in criterion_keys()}
    verdict: Dict[str, int] = {"pass": 0, "flag": 0, "fail": 0}
    for r in results:
        verdict[r.verdict if r.verdict in verdict else "flag"] += 1
        for key in scores:
            scores[key].append(float(r.criterion_scores.get(key, 0.0)))

    score_avg = {key: round(_avg(vals), 3) for key, vals in scores.items()}
    overall = round(_avg(list(score_avg.values())), 3) if score_avg else 0.0
    passes = verdict.get("pass", 0)
    fails = verdict.get("fail", 0)
    return BenchmarkResult(
        total_cases=total,
        scores=score_avg,
        criterion_average=dict(score_avg),
        overall=overall,
        verdict=verdict,
        pass_rate=round(passes / total, 4) if total else 0.0,
        fail_rate=round(fails / total, 4) if total else 0.0,
    )


def run_agent_benchmark(
    cases: List[EvaluationCase],
    runtime_factory: Callable[[EvaluationCase], Any],
    client: Optional[LLMClient] = None,
    *,
    request_id_prefix: str = "bench",
) -> Tuple[List[EvaluationTrace], BenchmarkResult, List[JudgeResult]]:
    """运行 benchmark：每个 case 一次 runtime 执行 + LLMJudge 评分。"""
    judge = LLMJudge(client=client if client is not None else MockLLMClient())
    runner = EvaluationRunner(
        runtime_factory=runtime_factory,
        request_id_prefix=request_id_prefix,
        judge=judge,
    )
    traces, summary = runner.run(cases)

    result = aggregate_judge_results(runner.judge_results, len(cases))
    result.deterministic = {
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
    }
    result.per_case = [
        {
            "case_id": t.case_id,
            "final_state": t.final_state,
            "selection_count": len(t.selection_ids),
            "decisions": list(t.decisions),
        }
        for t in traces
    ]
    return traces, result, runner.judge_results


def load_benchmark_cases(path: str) -> List[EvaluationCase]:
    """加载 agent_benchmark_cases.json → EvaluationCase 列表。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("dataset 必须是 JSON 数组")

    cases = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cases.append(EvaluationCase(
            case_id=item.get("case_id", ""),
            name=item.get("name", ""),
            query=item.get("query", ""),
            goal=item.get("goal"),
            expected_agents=list(item.get("expected_agents") or []),
            expected_tools=list(item.get("expected_tools") or []),
            expected_decisions=item.get("expected_decisions"),
            expected_final_state=item.get("expected_final_state", "done"),
            expect_selection=bool(item.get("expect_selection", True)),
            tags=list(item.get("tags") or []),
        ))
    return cases


def save_benchmark_result(result: BenchmarkResult, path: str) -> str:
    """把 BenchmarkResult 写成 JSON（目录自动创建）。"""
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    return path
