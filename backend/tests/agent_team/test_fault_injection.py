"""故障注入：Supervisor 恢复的客观度量（Fake 工具，零真实 API）。

复用 test_agent_loop 场景，输出成功率，不进 Judge。
"""

from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.tests.agent_team import test_agent_loop as loop


def _run_case(name, fn):
    try:
        fn()
        return {"name": name, "ok": True, "error": None}
    except Exception as e:
        return {"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"}


def collect_recovery_metrics():
    cases = [
        ("empty_retrieval_replan", loop.test_empty_retrieval_replans_with_new_query),
        ("low_relevance_replan", loop.test_low_relevance_replans),
        ("exec_failure_revise_abort", loop.test_execution_failure_retries_same_query_not_low_relevance),
        ("invalid_replan_rejected", loop.test_replan_invalid_rejected_no_partial_execute),
        ("two_round_budget_abort", loop.test_maximum_two_iterations_no_third_planner),
        ("legacy_untouched", loop.test_legacy_single_step_unaffected),
    ]
    rows = [_run_case(n, f) for n, f in cases]
    ok = sum(1 for r in rows if r["ok"])
    return {
        "n": len(rows),
        "success_count": ok,
        "success_rate": round(ok / len(rows), 4),
        "cases": rows,
        "note": "确定性断言，不用 Judge recovery_behavior。",
    }


def test_fault_injection_all_pass():
    report = collect_recovery_metrics()
    out = os.path.join(
        _REPO_ROOT, "backend", "ai_func", "recommendation",
        "evaluation", "results", "fault_injection.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    assert report["success_rate"] == 1.0, report
