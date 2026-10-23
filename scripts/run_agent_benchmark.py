"""Phase 4.3-B — SmartImager 200-Case Agent Benchmark Runner.

流程（每 case）：
  case → Runtime Execute (确定性场景) → EvaluationTrace → LLMJudge → JudgeResult

支持：
  --provider openai|fake
  --dataset（默认 evaluation/datasets/smartimager_agent_benchmark_200.json）
  --output（benchmark_results.json 路径；同目录产出 failure_analysis.json / benchmark_summary.md）

真实运行（DeepSeek 等 OpenAI 兼容）：
  set LLM_JUDGE_API_KEY=xxx
  set LLM_JUDGE_BASE_URL=https://api.deepseek.com
  set LLM_JUDGE_MODEL=deepseek-chat
  python scripts/run_agent_benchmark.py --provider openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.evaluation import (
    LLMClientError,
    MockLLMClient,
    OpenAICompatibleClient,
    criterion_keys,
    load_benchmark_cases,
)
from backend.ai_func.recommendation.evaluation.judge import JudgeResult
from backend.ai_func.recommendation.evaluation.llm_judge import LLMJudge
from backend.ai_func.recommendation.evaluation.runner import (
    EvaluationRunner,
    expected_violations,
)
from backend.ai_func.recommendation.evaluation.benchmark_harness import (
    build_case_runtime,
)

_DATASET = os.path.join(_REPO_ROOT, "backend", "ai_func", "recommendation",
                        "evaluation", "datasets", "smartimager_agent_benchmark_200.json")
_DEFAULT_OUT = os.path.join(_REPO_ROOT, "backend", "ai_func", "recommendation",
                            "evaluation", "results", "benchmark_results.json")

_CANONICAL = [("retrieval", "retrieval"), ("organization", "organization"),
              ("vision", "vision"), ("multi_step", "multi_agent"),
              ("failure_recovery", "recovery"), ("complex_planning", "complex")]


def category_of(tags) -> str:
    for marker, name in _CANONICAL:
        if marker in tags:
            return name
    return "other"


def _make_client(provider: str):
    if provider == "openai":
        return OpenAICompatibleClient()   # 环境变量 LLM_JUDGE_*
    if provider == "fake":
        return MockLLMClient()
    raise ValueError(f"未知 provider: {provider!r}")


def _case_score(result: JudgeResult) -> float:
    vals = [float(v) for k, v in result.criterion_scores.items() if k in criterion_keys()]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _possible_issue(rec) -> str:
    tool = rec.get("tool_summary") or {}
    issues = []
    if tool.get("failure"):
        issues.append(f"工具失败 {tool.get('failure')}/{tool.get('total')}")
    if rec.get("final_state") == "abort":
        issues.append("最终 abort（超迭代预算或不可恢复失败）")
    if rec.get("judge_error"):
        issues.append("Judge 输出异常：" + str(rec["judge_error"])[:100])
    if not issues:
        issues.append("Judge 判定低分；需结合 judge_reason 人工复核")
    return "; ".join(issues)


def main() -> int:
    ap = argparse.ArgumentParser(description="SmartImager Agent Benchmark (200 cases)")
    ap.add_argument("--provider", choices=["openai", "fake"], default="openai")
    ap.add_argument("--dataset", default=_DATASET)
    ap.add_argument("--output", default=_DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 个（调试）")
    args = ap.parse_args()

    cases = load_benchmark_cases(args.dataset)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    try:
        client = _make_client(args.provider)
    except LLMClientError as e:
        print(f"[benchmark] Judge LLM 初始化失败：{e}", file=sys.stderr)
        print("  请先设置 LLM_JUDGE_API_KEY / LLM_JUDGE_BASE_URL / LLM_JUDGE_MODEL。",
              file=sys.stderr)
        return 2

    provider_name = "openai" if args.provider == "openai" else "mock/fake"
    judge_model = getattr(client, "model", "") or ""
    judge_base = getattr(client, "base_url", "") or ""
    judge = LLMJudge(client=client)
    runner = EvaluationRunner(runtime_factory=build_case_runtime,
                              request_id_prefix="bench200")
    print(f"[benchmark] cases={len(cases)} provider={provider_name} model={judge_model}",
          flush=True)

    started = time.perf_counter()
    records = []
    judge_results = []
    det_pass = 0
    det_total = len(cases)

    for idx, case in enumerate(cases, start=1):
        trace = runner.run_case(case)
        violations = expected_violations(case, trace)
        if not violations:
            det_pass += 1
        try:
            jr = judge.evaluate_trace(trace)
        except Exception as e:  # 网络/解析失败 → flag + error，不中断整批
            jr = JudgeResult(case_id=case.case_id,
                             request_id=getattr(trace, "request_id", ""),
                             verdict="flag", error=f"{type(e).__name__}: {str(e)[:200]}")
        judge_results.append(jr)
        records.append({
            "case_id": case.case_id,
            "category": category_of(case.tags),
            "query": case.query,
            "goal": case.goal,
            "tags": list(case.tags),
            "score": _case_score(jr),
            "dimension_scores": {
                k: float(jr.criterion_scores.get(k, 0.0)) for k in criterion_keys()},
            "verdict": jr.verdict,
            "judge_reason": (jr.reasons or [])[:3],
            "final_state": trace.final_state,
            "decisions": list(trace.decisions),
            "tool_summary": dict(trace.tool_summary),
            "deterministic_passed": not violations,
            "judge_error": jr.error,
        })
        if idx % 20 == 0 or idx == len(cases):
            print(f"  [progress] {idx}/{len(cases)} done", flush=True)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _aggregate_and_write(args, records, provider_name, judge_model,
                                judge_base, det_pass, det_total, elapsed_ms)


def _aggregate_and_write(args, records, provider_name, judge_model, judge_base,
                         det_pass, det_total, elapsed_ms) -> int:
    """聚合统计并写出 benchmark_results.json / failure_analysis.json / benchmark_summary.md。"""
    total = len(records)
    verdict = {"pass": 0, "flag": 0, "fail": 0}
    for rec in records:
        verdict[rec["verdict"] if rec["verdict"] in verdict else "flag"] += 1

    dim_scores = _dimension_averages(records)
    scores = dim_scores
    overall = round(sum(scores.values()) / len(scores), 3) if scores else 0.0
    pass_count = verdict.get("pass", 0)
    fail_count = verdict.get("fail", 0)

    cats = defaultdict(list)
    for rec in records:
        cats[rec["category"]].append(rec["score"])
    category_breakdown = {
        cat: {"count": len(vals), "score": round(sum(vals) / len(vals), 3)}
        for cat, vals in sorted(cats.items())
    }

    result = {
        "meta": {
            "dataset": os.path.basename(args.dataset),
            "total_cases": total,
            "judge": {"provider": provider_name, "model": judge_model,
                      "base_url": judge_base,
                      "temperature": 0.0},
        },
        "overall": {
            "total_cases": total,
            "success_cases": pass_count,
            "failed_cases": fail_count,
            "pass_rate": round(pass_count / total, 4) if total else 0.0,
            "fail_rate": round(fail_count / total, 4) if total else 0.0,
            "overall_score": overall,
        },
        "scores": scores,
        "category_breakdown": category_breakdown,
        "verdict": verdict,
        "deterministic": {"total": det_total, "passed": det_pass,
                          "failed": det_total - det_pass},
        "elapsed_ms": round(elapsed_ms, 1),
    }

    worst = sorted(records, key=lambda r: r["score"])[:20]
    failure_analysis = {
        "total": len(worst),
        "note": "score = 该 case 五个 rubric 维度的平均分（LLM Judge 输出）",
        "items": [
            {
                "case_id": rec["case_id"],
                "query": rec["query"],
                "category": rec["category"],
                "score": rec["score"],
                "verdict": rec["verdict"],
                "judge_reason": rec["judge_reason"],
                "possible_issue": _possible_issue(rec),
            }
            for rec in worst
        ],
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    failure_path = os.path.join(out_dir, "failure_analysis.json")
    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failure_analysis, f, ensure_ascii=False, indent=2)
    summary_path = os.path.join(out_dir, "benchmark_summary.md")
    _write_summary(result, len(worst), summary_path)

    print(json.dumps({
        "total_cases": total,
        "overall_score": overall,
        "scores": scores,
        "verdict": verdict,
        "deterministic": result["deterministic"],
        "category_breakdown": category_breakdown,
    }, ensure_ascii=False, indent=2))
    print(f"[benchmark] results -> {args.output}")
    print(f"[benchmark] failure_analysis -> {failure_path}")
    print(f"[benchmark] summary -> {summary_path}")
    print(f"[benchmark] elapsed_ms={elapsed_ms:.0f}")
    return 0


def _dimension_averages(records) -> dict:
    """records 中只存 case 平均分；逐维分数需额外记录。作为兜底，这里从
    per-dimension 数据补算——本函数从 record['score'] 不可还原，因此要求外部在
    records 里已带 'dimension_scores'；若无则退回按 case 平均分聚合同一值。"""
    keys = list(criterion_keys())
    out = {k: [] for k in keys}
    missing = False
    for rec in records:
        dims = rec.get("dimension_scores")
        if isinstance(dims, dict) and dims:
            for k in keys:
                out[k].append(float(dims.get(k, 0.0)))
        else:
            missing = True
            break
    if missing:
        # 兼容旧行为：只用 case 平均分近似（真实维度分在正常 run 中总会写入）
        return {}
    return {k: round(sum(vs) / len(vs), 3) for k, vs in out.items() if vs}


def _write_summary(result, failure_count, path) -> None:
    lines = ["# SmartImager Agent Benchmark Summary", ""]
    lines.append(f"- total_cases: {result['meta']['total_cases']}")
    lines.append(f"- judge: {result['meta']['judge']['provider']} "
                 f"model={result['meta']['judge']['model'] or '-'}")
    lines.append(f"- overall_score: {result['overall']['overall_score']}")
    lines.append(f"- verdict: pass={result['verdict']['pass']} "
                 f"flag={result['verdict']['flag']} fail={result['verdict']['fail']}")
    lines.append("")
    lines.append("## Scores")
    for k, v in result["scores"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Category Breakdown")
    for cat, info in result["category_breakdown"].items():
        lines.append(f"- {cat}: count={info['count']} avg_score={info['score']}")
    lines.append("")
    lines.append(f"failure cases analyzed: {failure_count}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
