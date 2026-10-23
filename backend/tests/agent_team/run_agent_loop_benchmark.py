"""Observe/Replan 最小 Before/After 对照（全 Fake，不改测试数据、不硬调阈值）。

Before：旧 REVIEW（空检索 → revise / limit+10；无 LOW_RELEVANCE）。
After ：当前 Observation → Replan（空检索改 query；低相关单独触发）。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/run_agent_loop_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    PlanValidator,
    PlannerAgent,
    SupervisorAgent,
    SupervisorDecision,
    TaskType,
    ToolRegistry,
    build_default_agent_registry,
    build_registry_backed_workers,
    register_builtin_tools,
    revise_plan,
)
from backend.ai_func.recommendation.orchestration.supervisor import (
    MIN_CANDIDATE_THRESHOLD,
    _retrieval_stats,
    review_execution as review_execution_after,
)
import backend.ai_func.recommendation.orchestration.supervisor as supervisor_mod
from backend.ai_func.recommendation.orchestration.workers import (
    DeterministicSynthesisWorker,
    TaskStatus,
)
from backend.tests.agent_team.loop_support import rewrite_aware_llm
from backend.tests.agent_team.test_agent_loop import (
    _FakeSearch,
    _low_rel_rows,
    _ok_rows,
    _plan_json,
)


def review_execution_before(
    plan, results, *, attempt, max_iterations, min_candidates=MIN_CANDIDATE_THRESHOLD,
    enhanced=True, **_ignored,
) -> SupervisorDecision:
    """升级前 REVIEW：空检索与工具失败都走 revise；无 LOW_RELEVANCE。"""
    if not enhanced:
        return SupervisorDecision(decision="continue", reason="enhanced disabled")
    failed_tasks: List[str] = []
    empty_tasks: List[str] = []
    for t in plan.tasks:
        r = results.get(t.task_id)
        if r is None:
            continue
        if getattr(r, "status", None) == TaskStatus.FAILED:
            failed_tasks.append(t.task_id)
        elif t.task_type == TaskType.RETRIEVE_IMAGES and getattr(r, "status", None) == TaskStatus.SUCCEEDED:
            if _retrieval_stats(t, results) == 0:
                empty_tasks.append(t.task_id)
    if failed_tasks or empty_tasks:
        affected = failed_tasks + empty_tasks
        if attempt >= max_iterations:
            return SupervisorDecision(decision="abort", reason="max", affected_tasks=affected)
        return SupervisorDecision(decision="revise", reason="failed/empty", affected_tasks=affected)
    synth = results.get(plan.synthesis_task_id)
    out = getattr(synth, "output", None) if synth is not None else None
    has_selection = bool(getattr(out, "image_ids", None) if out is not None else False)
    retrieval_count = sum(
        max(0, _retrieval_stats(t, results)) for t in plan.tasks
        if t.task_type == TaskType.RETRIEVE_IMAGES
    )
    if (not has_selection) or retrieval_count < min_candidates:
        if attempt >= max_iterations:
            return SupervisorDecision(decision="abort", reason="synth/min")
        return SupervisorDecision(decision="replan", reason="synth/min")
    return SupervisorDecision(decision="continue", reason="ok")


def _supervisor(search, review_fn=None):
    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={"search_images": search})
    workers = build_registry_backed_workers(tool_reg, with_vision=False)
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=tool_reg)
    sup = SupervisorAgent(
        planner=PlannerAgent(llm=rewrite_aware_llm(_plan_json())),
        validator=PlanValidator(),
        workers=workers,
        agent_registry=agent_reg,
        tool_registry=tool_reg,
        synthesize_worker=DeterministicSynthesisWorker(),
        enhanced=True,
        max_iterations=2,
        backend="legacy",
    )
    supervisor_mod.review_execution = review_fn or review_execution_after
    return sup


def _run(label, search, review_fn=None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    sup = _supervisor(search, review_fn=review_fn)
    events = [json.loads(raw) for raw in sup.run_stream(
        [{"role": "user", "content": "找城市建筑图片"}])]
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    reviews = [e for e in events if e["event"] == "agent.review"]
    observes = [e for e in events if e["event"] == "agent.observe"]
    plans = [e for e in events if e["event"] == "plan.created"]
    failed = [e for e in events if e["event"] == "plan.failed"]
    queries = list(getattr(search, "queries", []) or [])
    query_changed = len(queries) >= 2 and (queries[0] or "") != (queries[1] or "")
    trigger = observes[0]["trigger"] if observes else ""
    return {
        "label": label,
        "decisions": [r["decision"] for r in reviews],
        "final": getattr(getattr(sup.run_state, "final_state", None), "value", None),
        "planner_calls": sup.planner_call_count,
        "search_calls": search.calls,
        "plan_events": len(plans),
        "observe_events": len(observes),
        "plan_failed": bool(failed),
        "trigger": trigger,
        "action": observes[0].get("REPLAN_ACTION", "") if observes else "",
        "query_changed": query_changed,
        "queries": queries,
        "selection": next((e["ids"] for e in reversed(events) if e["event"] == "selection"), []),
        "latency_ms": latency_ms,
        "replan_reason": observes[0].get("REPLAN_REASON", "") if observes else "",
    }


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows) or 1
    recovered = [
        r for r in rows
        if r["final"] == "done" and any(d in ("revise", "replan") for d in r["decisions"])
    ]
    aborted = [r for r in rows if r["final"] == "abort"]
    triggered = [r for r in rows if r["decisions"] and r["decisions"][0] != "continue"]
    second_plan = [r for r in rows if r["planner_calls"] >= 2]
    action_changed = [r for r in rows if r.get("query_changed")]
    invalid = [r for r in rows if r.get("plan_failed")]
    return {
        "n": len(rows),
        "recovery_trigger_rate": round(len(triggered) / n, 3),
        "second_plan_rate": round(len(second_plan) / n, 3),
        "action_change_rate": round(len(action_changed) / n, 3),
        "successful_recovery_rate": round(len(recovered) / n, 3),
        "abort_rate": round(len(aborted) / n, 3),
        "planner_calls_per_request": round(sum(r["planner_calls"] for r in rows) / n, 3),
        "latency_ms_mean": round(sum(r["latency_ms"] for r in rows) / n, 2),
        "invalid_replan_rate": round(len(invalid) / n, 3),
    }


def main() -> int:
    scenarios = [
        ("success", lambda: _FakeSearch([_ok_rows()])),
        ("empty_then_hit", lambda: _FakeSearch([[], _ok_rows()])),
        ("low_relevance_then_hit", lambda: _FakeSearch([_low_rel_rows(), _ok_rows()])),
        ("exec_fail", lambda: _FakeSearch([_ok_rows()], fail=True)),
        ("empty_then_empty", lambda: _FakeSearch([[], []])),
    ]
    before_rows = []
    after_rows = []
    print("======== BEFORE (old review: empty→revise) ========")
    for name, factory in scenarios:
        row = _run(name, factory(), review_fn=review_execution_before)
        before_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    print("======== AFTER (observe→replan) ========")
    for name, factory in scenarios:
        row = _run(name, factory(), review_fn=None)
        after_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    print("======== METRICS ========")
    report = {"before": _metrics(before_rows), "after": _metrics(after_rows)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "agent_loop_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"before_rows": before_rows, "after_rows": after_rows, "metrics": report},
                  f, ensure_ascii=False, indent=2)
    print("wrote", out_path)

    supervisor_mod.review_execution = review_execution_after
    search = _FakeSearch([[], _ok_rows()])
    sup = _supervisor(search, review_fn=None)
    events = [json.loads(raw) for raw in sup.run_stream(
        [{"role": "user", "content": "找城市建筑图片"}])]
    slim = []
    for e in events:
        name = e.get("event")
        if name == "plan.created":
            slim.append({
                "event": name,
                "iteration": e.get("iteration"),
                "tasks": [t.get("task_type") for t in e.get("tasks", [])],
            })
        elif name == "agent.observe":
            slim.append({
                "event": name,
                "iteration": e.get("iteration"),
                "trigger": e.get("trigger"),
                "REPLAN_REASON": e.get("REPLAN_REASON"),
                "REPLAN_ACTION": e.get("REPLAN_ACTION"),
                "retrieval": {
                    "empty": (e.get("retrieval") or {}).get("empty"),
                    "candidate_count": (e.get("retrieval") or {}).get("candidate_count"),
                    "query": (e.get("retrieval") or {}).get("query"),
                    "top1_score": (e.get("retrieval") or {}).get("top1_score"),
                },
            })
        elif name == "agent.review":
            slim.append({
                "event": name,
                "iteration": e.get("iteration"),
                "decision": e.get("decision"),
                "trigger": e.get("trigger"),
                "REPLAN_REASON": e.get("REPLAN_REASON"),
                "REPLAN_ACTION": e.get("REPLAN_ACTION"),
            })
        elif name in ("assistant_delta", "selection"):
            slim.append({"event": name, "ids": e.get("ids")})
    trace_path = os.path.join(out_dir, "agent_loop_trace_example.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "planner_calls": sup.planner_call_count,
            "queries": search.queries,
            "supervisor_trace": getattr(sup.run_state, "supervisor_trace", []),
            "events": slim,
        }, f, ensure_ascii=False, indent=2)
    print("wrote", trace_path)
    supervisor_mod.review_execution = review_execution_after
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
