"""P1 After-only 复测：DeepSeek Planner + Fake 工具，不打 Judge。

产出 evaluation/results/p1_after_retest/（不覆盖旧 A/B JSON）。
对照基线：task_catalog_ab_deepseek/p1_plan_bloat.json（简单均 Task 4.85）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts._load_deepseek_env import load_deepseek_env
from scripts.measure_p1_plan_bloat import _OPTIONAL, _is_simple, analyze
from scripts.run_task_catalog_ab_benchmark import (
    _planner_client,
    category_of,
    load_ab_cases,
)

from backend.ai_func.recommendation.evaluation.benchmark_harness import (
    build_real_planner_runtime,
)
from backend.ai_func.recommendation.evaluation.latency import StageTimer, format_summary
from backend.ai_func.recommendation.evaluation.runner import EvaluationRunner
from backend.ai_func.recommendation.orchestration.opt_flags import STATS, reset_opt_stats

_BASELINE = {
    "simple_avg_tasks": 4.85,
    "simple_pure_retrieve_rate": 0.085,
    "plan_ok_rate": 0.9833,
    "optional_full": {
        "deduplicate_images": 187,
        "select_images": 132,
        "rerank_candidates": 68,
        "verify_visual_condition": 20,
    },
    "source": "evaluation/results/task_catalog_ab_deepseek/p1_plan_bloat.json",
}


def _stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [len(r.get("plan_task_types") or []) for r in rows]
    pure = 0
    optional_hits = Counter()
    for r in rows:
        types = [t for t in (r.get("plan_task_types") or []) if t]
        core = [t for t in types if t != "synthesize"]
        if core == ["retrieve_images"]:
            pure += 1
        for t in types:
            if t in _OPTIONAL:
                optional_hits[t] += 1
    n = len(rows) or 1
    return {
        "n": len(rows),
        "avg_tasks": round(sum(lengths) / n, 3) if rows else 0.0,
        "pure_retrieve_rate": round(pure / n, 4) if rows else 0.0,
        "optional_counts": dict(optional_hits),
        "plan_ok_rate": round(
            sum(1 for r in rows if r.get("plan_ok")) / n, 4
        ) if rows else 0.0,
    }


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 After-only DeepSeek retest (no Judge)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--simple-only", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--lint", action="store_true", help="开 SM_PLAN_LINT（软剥离）")
    ap.add_argument(
        "--out-dir",
        default=os.path.join(
            _REPO, "backend", "ai_func", "recommendation",
            "evaluation", "results", "p1_after_retest",
        ),
    )
    args = ap.parse_args()

    prefix = load_deepseek_env()
    print(f"[p1] deepseek key prefix={prefix} model=deepseek-chat", flush=True)
    if args.lint:
        os.environ["SM_PLAN_LINT"] = "1"
    else:
        os.environ.pop("SM_PLAN_LINT", None)
    reset_opt_stats()

    cases = load_ab_cases(include_extra=True)
    if args.simple_only:
        cases = [
            c for c in cases
            if _is_simple({"category": category_of(c.tags), "query": c.query})
        ]
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    creds = {
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.environ["OPENAI_API_BASE"],
        "model": os.environ.get("CHAT_MODEL") or "deepseek-chat",
    }
    planner_client = _planner_client(creds)
    planner_model = creds["model"]

    timer = StageTimer()

    def factory(case):
        runtime = build_real_planner_runtime(
            case,
            "after",
            planner_client=planner_client,
            planner_model=planner_model,
        )
        inner = getattr(runtime, "_inner", runtime)
        planner = inner.planner
        orig = planner.create_plan

        def timed_create_plan(*a, **kw):
            with timer.span("planner"):
                return orig(*a, **kw)

        planner.create_plan = timed_create_plan
        return runtime

    runner = EvaluationRunner(runtime_factory=factory, request_id_prefix="p1_after")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    records_path = os.path.join(out_dir, "after_retest.json")
    summary_path = os.path.join(out_dir, "p1_after_summary.json")

    records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    total = len(cases)
    print(f"[p1] cases={total} lint={bool(args.lint)} out={out_dir}", flush=True)

    for idx, case in enumerate(cases, start=1):
        trace = runner.run_case(case)
        types = [t.get("task_type") for t in (trace.plan_tasks or [])]
        rec = {
            "case_id": case.case_id,
            "category": category_of(case.tags),
            "query": case.query,
            "plan_ok": bool(trace.plan_ok),
            "final_state": trace.final_state,
            "plan_task_types": types,
            "error": trace.error,
        }
        records.append(rec)
        if idx % 10 == 0 or idx == total:
            elapsed = time.perf_counter() - started
            print(f"  [p1] {idx}/{total} elapsed={elapsed:.0f}s", flush=True)
            _write_json(records_path, {
                "arm": "after",
                "lint": bool(args.lint),
                "n": len(records),
                "records": records,
            })
        if args.sleep > 0:
            time.sleep(args.sleep)

    simple = [r for r in records if _is_simple(r)]
    summary = {
        "arm": "after",
        "lint": bool(args.lint),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "baseline": _BASELINE,
        "all": _stats(records),
        "simple_subset": _stats(simple),
        "delta": {
            "simple_avg_tasks": round(
                _stats(simple)["avg_tasks"] - _BASELINE["simple_avg_tasks"], 3
            ),
            "simple_pure_retrieve_rate": round(
                _stats(simple)["pure_retrieve_rate"] - _BASELINE["simple_pure_retrieve_rate"], 4
            ),
            "plan_ok_rate": round(
                _stats(records)["plan_ok_rate"] - _BASELINE["plan_ok_rate"], 4
            ),
        },
        "plan_lint_stripped": int(getattr(STATS, "plan_lint_stripped", 0)),
        "plan_floor_retry": int(getattr(STATS, "plan_floor_retry", 0)),
        "plan_floor_fallback": int(getattr(STATS, "plan_floor_fallback", 0)),
        "stage_latency": timer.summary(),
        "pass": None,
        "note": (
            "通过：简单均 Task 显著下降 + plan_ok 不降。"
            "单独 4.85 不进简历；只写 4.85→X。"
        ),
        "paths": {
            "records": records_path,
            "summary": summary_path,
            "baseline": _BASELINE["source"],
        },
    }
    avg = summary["simple_subset"]["avg_tasks"]
    plan_ok = summary["all"]["plan_ok_rate"]
    summary["pass"] = bool(avg < 4.0 and plan_ok >= _BASELINE["plan_ok_rate"] - 0.005)
    summary["still_bloated"] = bool(avg > 4.0)

    _write_json(records_path, {
        "arm": "after",
        "lint": bool(args.lint),
        "meta": {"n": len(records), "elapsed_s": summary["elapsed_s"]},
        "records": records,
    })
    _write_json(summary_path, summary)
    # 复用 measure 脚本口径再写一份
    bloat = analyze(records_path)
    bloat["retest_summary"] = {
        "simple_avg_tasks": avg,
        "vs_baseline": f"4.85→{avg}",
        "plan_ok": plan_ok,
        "lint": bool(args.lint),
    }
    _write_json(os.path.join(out_dir, "p1_plan_bloat_retest.json"), bloat)

    print(json.dumps({
        "simple_avg": avg,
        "vs_baseline": f"4.85→{avg}",
        "pure": summary["simple_subset"]["pure_retrieve_rate"],
        "plan_ok": plan_ok,
        "pass": summary["pass"],
        "still_bloated": summary["still_bloated"],
        "lint_stripped": summary["plan_lint_stripped"],
        "planner_latency": (summary.get("stage_latency") or {}).get("planner"),
        "out": out_dir,
    }, ensure_ascii=False, indent=2), flush=True)
    lat = summary.get("stage_latency") or {}
    if lat:
        print(format_summary(lat), flush=True)
    return 0 if summary["pass"] or not summary["still_bloated"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
