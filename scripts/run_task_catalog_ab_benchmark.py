"""Task-catalog A/B：旧 5 类 Task vs 现 10 类 Task，真实 Planner + 真实 Judge。

默认：200 条冻结集 + 40 条扩充（共 240）。不覆盖旧 benchmark_results.json。

必须提供 Key（CLI 或环境变量），写入仓库根 .env 后不会自动删除，方便之后改。

  set LLM_JUDGE_API_KEY=...
  set LLM_JUDGE_BASE_URL=https://api.deepseek.com
  set LLM_JUDGE_MODEL=deepseek-chat
  python scripts/run_task_catalog_ab_benchmark.py --provider openai

或：
  python scripts/run_task_catalog_ab_benchmark.py --provider openai --api-key ... --base-url https://api.deepseek.com --model deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.ai_func.recommendation.evaluation import (
    LLMClientError,
    MockLLMClient,
    OpenAICompatibleClient,
    criterion_keys,
    load_benchmark_cases,
)
from backend.ai_func.recommendation.evaluation.benchmark_harness import (
    build_real_planner_runtime,
)
from backend.ai_func.recommendation.evaluation.judge import JudgeResult
from backend.ai_func.recommendation.evaluation.llm_judge import LLMJudge
from backend.ai_func.recommendation.evaluation.runner import EvaluationRunner

_BASE_200 = os.path.join(
    _REPO_ROOT, "backend", "ai_func", "recommendation",
    "evaluation", "datasets", "smartimager_agent_benchmark_200.json",
)
_EXTRA = os.path.join(
    _REPO_ROOT, "backend", "ai_func", "recommendation",
    "evaluation", "datasets", "smartimager_task_catalog_ab_extra.json",
)
_DEFAULT_OUT_DIR = os.path.join(
    _REPO_ROOT, "backend", "ai_func", "recommendation",
    "evaluation", "results", "task_catalog_ab",
)
_ENV_PATH = os.path.join(_REPO_ROOT, ".env")

_CANONICAL = [
    ("retrieval", "retrieval"),
    ("organization", "organization"),
    ("vision", "vision"),
    ("multi_step", "multi_agent"),
    ("failure_recovery", "recovery"),
    ("complex_planning", "complex"),
]


def category_of(tags) -> str:
    for marker, name in _CANONICAL:
        if marker in tags:
            return name
    return "other"


def load_ab_cases(include_extra: bool = True) -> list:
    cases = load_benchmark_cases(_BASE_200)
    if include_extra and os.path.isfile(_EXTRA):
        extra = load_benchmark_cases(_EXTRA)
        seen = {c.case_id for c in cases}
        for c in extra:
            if c.case_id not in seen:
                cases.append(c)
                seen.add(c.case_id)
    return cases


def persist_env(api_key: str, base_url: str, model: str, path: str = _ENV_PATH) -> str:
    """写入 .env（gitignore），不删除已有其他键。"""
    rows: Dict[str, str] = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                k, _, v = raw.partition("=")
                rows[k.strip()] = v.strip()
    rows["LLM_JUDGE_API_KEY"] = api_key
    rows["LLM_JUDGE_BASE_URL"] = base_url
    rows["LLM_JUDGE_MODEL"] = model
    rows["OPENAI_API_KEY"] = api_key
    rows["OPENAI_API_BASE"] = base_url
    if model:
        rows["CHAT_MODEL"] = model
    with open(path, "w", encoding="utf-8") as f:
        f.write("# local LLM credentials — do not commit\n")
        for k, v in rows.items():
            f.write(f"{k}={v}\n")
    return path


def load_dotenv(path: str = _ENV_PATH) -> None:
    """把 .env 灌进 os.environ；已存在的环境变量不覆盖。"""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            k, _, v = raw.partition("=")
            k, v = k.strip(), v.strip()
            if k and k not in os.environ:
                os.environ[k] = v


def resolve_credentials(args) -> Dict[str, str]:
    api_key = (
        args.api_key
        or os.environ.get("LLM_JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("SMARTIMAGER_LLM_API_KEY")
        or ""
    ).strip()
    base_url = (
        args.base_url
        or os.environ.get("LLM_JUDGE_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "https://api.deepseek.com"
    ).strip()
    model = (
        args.model
        or os.environ.get("LLM_JUDGE_MODEL")
        or os.environ.get("CHAT_MODEL")
        or "deepseek-chat"
    ).strip()
    if args.provider == "openai" and not api_key:
        raise LLMClientError(
            "缺少 API Key。请用 --api-key 传入，或设置 LLM_JUDGE_API_KEY / OPENAI_API_KEY。"
        )
    return {"api_key": api_key, "base_url": base_url, "model": model}


def _make_judge(provider: str, creds: Dict[str, str]):
    if provider == "fake":
        return LLMJudge(client=MockLLMClient()), "mock/fake", "", ""
    client = OpenAICompatibleClient(
        api_key=creds["api_key"],
        base_url=creds["base_url"],
        model=creds["model"],
        temperature=0.0,
        timeout_seconds=90.0,
        max_retries=4,
    )
    return LLMJudge(client=client), "openai", creds["model"], creds["base_url"]


def _planner_client(creds: Dict[str, str]):
    from openai import OpenAI

    base = creds["base_url"].rstrip("/")
    return OpenAI(api_key=creds["api_key"], base_url=base, timeout=90.0)


def _case_score(jr: JudgeResult) -> float:
    vals = [float(jr.criterion_scores.get(k, 0.0)) for k in criterion_keys()]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _dimension_averages(records: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = list(criterion_keys())
    buckets = {k: [] for k in keys}
    for rec in records:
        dims = rec.get("dimension_scores") or {}
        for k in keys:
            buckets[k].append(float(dims.get(k, 0.0)))
    return {k: round(sum(vs) / len(vs), 3) for k, vs in buckets.items() if vs}


def _aggregate(records: List[Dict[str, Any]], *, arm: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    verdict = {"pass": 0, "flag": 0, "fail": 0}
    for rec in records:
        verdict[rec["verdict"] if rec["verdict"] in verdict else "flag"] += 1
    scores = _dimension_averages(records)
    overall = round(sum(scores.values()) / len(scores), 3) if scores else 0.0
    cats: Dict[str, List[float]] = defaultdict(list)
    types: Dict[str, int] = defaultdict(int)
    for rec in records:
        cats[rec["category"]].append(rec["score"])
        for t in rec.get("plan_task_types") or []:
            types[t] += 1
    return {
        "arm": arm,
        "meta": meta,
        "overall": {
            "total_cases": len(records),
            "overall_score": overall,
            "pass_rate": round(verdict["pass"] / len(records), 4) if records else 0.0,
            "fail_rate": round(verdict["fail"] / len(records), 4) if records else 0.0,
            "plan_ok_rate": round(
                sum(1 for r in records if r.get("plan_ok")) / len(records), 4
            ) if records else 0.0,
        },
        "scores": scores,
        "verdict": verdict,
        "category_breakdown": {
            cat: {"count": len(vals), "score": round(sum(vals) / len(vals), 3)}
            for cat, vals in sorted(cats.items())
        },
        "task_type_counts": dict(sorted(types.items())),
        "records": records,
    }


def _compare(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    dims = {}
    for k in criterion_keys():
        b = float((before.get("scores") or {}).get(k, 0.0))
        a = float((after.get("scores") or {}).get(k, 0.0))
        dims[k] = {"before": b, "after": a, "delta": round(a - b, 3)}
    bo = before["overall"]
    ao = after["overall"]
    return {
        "overall_score": {
            "before": bo["overall_score"],
            "after": ao["overall_score"],
            "delta": round(ao["overall_score"] - bo["overall_score"], 3),
        },
        "pass_rate": {
            "before": bo["pass_rate"],
            "after": ao["pass_rate"],
            "delta": round(ao["pass_rate"] - bo["pass_rate"], 4),
        },
        "plan_ok_rate": {
            "before": bo["plan_ok_rate"],
            "after": ao["plan_ok_rate"],
            "delta": round(ao["plan_ok_rate"] - bo["plan_ok_rate"], 4),
        },
        "scores": dims,
        "note": (
            "After = 10 TaskTypes（含 rerank/dedup/sort/select/verify）；"
            "Before = 5 TaskTypes（retrieve/filter/group/analyze_visual/synthesize）。"
            "同一 Judge、同一 Fake 工具、真实 Planner。"
        ),
    }


def _write_comparison_md(cmp: Dict[str, Any], path: str) -> None:
    lines = ["# Task Catalog A/B — LLM Judge", ""]
    lines.append(cmp["note"])
    lines.append("")
    lines.append(
        f"- overall: {cmp['overall_score']['before']} → {cmp['overall_score']['after']} "
        f"(Δ {cmp['overall_score']['delta']:+})"
    )
    lines.append(
        f"- pass_rate: {cmp['pass_rate']['before']} → {cmp['pass_rate']['after']} "
        f"(Δ {cmp['pass_rate']['delta']:+})"
    )
    lines.append(
        f"- plan_ok_rate: {cmp['plan_ok_rate']['before']} → {cmp['plan_ok_rate']['after']} "
        f"(Δ {cmp['plan_ok_rate']['delta']:+})"
    )
    lines.append("")
    lines.append("## Dimensions")
    for k, v in cmp["scores"].items():
        lines.append(f"- {k}: {v['before']} → {v['after']} (Δ {v['delta']:+})")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_arm(
    arm: str,
    cases,
    judge: LLMJudge,
    creds: Dict[str, str],
    provider: str,
    planner_llm=None,
) -> List[Dict[str, Any]]:
    planner_client = None
    planner_model = None
    if planner_llm is None and provider == "openai":
        planner_client = _planner_client(creds)
        planner_model = creds["model"]

    def factory(case, _arm=arm):
        llm = planner_llm
        if llm is None and provider == "fake":
            from backend.ai_func.recommendation.evaluation.benchmark_harness import (
                _plan_json,
            )
            llm = lambda messages, c=case: json.dumps(_plan_json(c), ensure_ascii=False)
        return build_real_planner_runtime(
            case,
            _arm,
            planner_client=planner_client,
            planner_model=planner_model,
            planner_llm=llm,
        )

    runner = EvaluationRunner(runtime_factory=factory, request_id_prefix=f"ab_{arm}")
    records = []
    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        trace = runner.run_case(case)
        try:
            jr = judge.evaluate_trace(trace)
        except Exception as e:
            jr = JudgeResult(
                case_id=case.case_id,
                request_id=getattr(trace, "request_id", ""),
                verdict="flag",
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
        records.append({
            "case_id": case.case_id,
            "arm": arm,
            "category": category_of(case.tags),
            "query": case.query,
            "score": _case_score(jr),
            "dimension_scores": {
                k: float(jr.criterion_scores.get(k, 0.0)) for k in criterion_keys()
            },
            "verdict": jr.verdict,
            "judge_reason": (jr.reasons or [])[:3],
            "judge_error": jr.error,
            "plan_ok": bool(trace.plan_ok),
            "final_state": trace.final_state,
            "plan_task_types": [t.get("task_type") for t in (trace.plan_tasks or [])],
            "selection_count": len(trace.selection_ids or []),
        })
        if idx % 10 == 0 or idx == total:
            print(f"  [{arm}] {idx}/{total}", flush=True)
        if provider == "openai":
            time.sleep(0.25)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Task catalog A/B (real Planner + real Judge)")
    ap.add_argument("--provider", choices=["openai", "fake"], default="openai")
    ap.add_argument("--api-key", default="", help="写入 .env，用完不删")
    ap.add_argument("--base-url", default="", help="默认 https://api.deepseek.com")
    ap.add_argument("--model", default="", help="默认 deepseek-chat")
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 条（调试）")
    ap.add_argument("--no-extra", action="store_true", help="只用冻结的 200 条")
    ap.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    ap.add_argument("--no-persist-env", action="store_true", help="不把 Key 写入 .env")
    ap.add_argument("--arm", choices=["both", "before", "after"], default="both",
                    help="只重跑某一臂时保留另一臂已有 json")
    args = ap.parse_args()
    load_dotenv()

    try:
        creds = resolve_credentials(args)
    except LLMClientError as e:
        print(f"[ab] {e}", file=sys.stderr)
        return 2

    if args.provider == "openai" and not args.no_persist_env and creds["api_key"]:
        path = persist_env(creds["api_key"], creds["base_url"], creds["model"])
        print(f"[ab] credentials saved to {path} (gitignored, not deleted after run)", flush=True)
        os.environ["LLM_JUDGE_API_KEY"] = creds["api_key"]
        os.environ["LLM_JUDGE_BASE_URL"] = creds["base_url"]
        os.environ["LLM_JUDGE_MODEL"] = creds["model"]
        os.environ["OPENAI_API_KEY"] = creds["api_key"]
        os.environ["OPENAI_API_BASE"] = creds["base_url"]

    cases = load_ab_cases(include_extra=not args.no_extra)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    try:
        judge, provider_name, judge_model, judge_base = _make_judge(args.provider, creds)
    except LLMClientError as e:
        print(f"[ab] Judge 初始化失败：{e}", file=sys.stderr)
        return 2

    print(
        f"[ab] cases={len(cases)} provider={provider_name} model={judge_model or '-'}",
        flush=True,
    )
    started = time.perf_counter()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    before_path = os.path.join(out_dir, "before.json")
    after_path = os.path.join(out_dir, "after.json")

    before_records = None
    after_records = None
    if args.arm in ("both", "before"):
        before_records = run_arm("before", cases, judge, creds, args.provider)
    if args.arm in ("both", "after"):
        after_records = run_arm("after", cases, judge, creds, args.provider)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    meta = {
        "total_cases": len(cases),
        "judge": {
            "provider": provider_name,
            "model": judge_model,
            "base_url": judge_base,
            "temperature": 0.0,
        },
        "elapsed_ms": round(elapsed_ms, 1),
        "include_extra": not args.no_extra,
        "arm": args.arm,
    }
    if before_records is not None:
        before = _aggregate(before_records, arm="before", meta=dict(meta, catalog="5-task"))
    elif os.path.isfile(before_path):
        with open(before_path, "r", encoding="utf-8") as f:
            before = json.load(f)
    else:
        print("[ab] 缺少 before.json，无法对比。请先跑 --arm before 或 both。", file=sys.stderr)
        return 2
    if after_records is not None:
        after = _aggregate(after_records, arm="after", meta=dict(meta, catalog="10-task"))
    elif os.path.isfile(after_path):
        with open(after_path, "r", encoding="utf-8") as f:
            after = json.load(f)
    else:
        print("[ab] 缺少 after.json，无法对比。", file=sys.stderr)
        return 2
    comparison = _compare(before, after)
    cmp_path = os.path.join(out_dir, "comparison.json")
    md_path = os.path.join(out_dir, "comparison.md")
    with open(before_path, "w", encoding="utf-8") as f:
        json.dump(before, f, ensure_ascii=False, indent=2)
    with open(after_path, "w", encoding="utf-8") as f:
        json.dump(after, f, ensure_ascii=False, indent=2)
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    _write_comparison_md(comparison, md_path)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"[ab] before -> {before_path}")
    print(f"[ab] after  -> {after_path}")
    print(f"[ab] comparison -> {cmp_path}")
    print(f"[ab] summary -> {md_path}")
    print(f"[ab] elapsed_ms={elapsed_ms:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
