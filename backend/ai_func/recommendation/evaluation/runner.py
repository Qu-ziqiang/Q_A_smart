"""Phase 4.1 — 离线 EvaluationRunner。

职责：
  - 输入：List[EvaluationCase]
  - 执行：对每个 case 调用 runtime_factory(case) 生成 runtime（SupervisorAgent 风格：
          run_stream(messages)->Iterable[str]，且暴露 .run_state / .last_plan_ok），
          消费其事件流；
  - 输出：List[EvaluationTrace] + EvaluationSummary（确定性预期检查）。

不引入 LLM Judge；EvaluationTrace 直接基于 RunState（agent_messages /
tool_events / supervisor_trace / execution_results）与 EventCollector.snapshot() 派生，
保证与既有可观测层同源。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from .judge import JudgeResult
from .models import EvaluationCase, EvaluationSummary, EvaluationTrace


def build_evaluation_trace(
    case: EvaluationCase,
    runtime: Any,
    run_state: Any,
    selection_ids: List[int],
    duration_ms: float,
    request_id: str,
    error: Optional[str] = None,
) -> EvaluationTrace:
    """从 runtime / RunState 派生 EvaluationTrace。"""
    statuses: Dict[str, str] = {}
    if run_state is not None and getattr(run_state, "execution_results", None):
        statuses = {
            tid: (r.status.value if hasattr(r, "status") else "?")
            for tid, r in run_state.execution_results.items()
        }

    plan_tasks: List[Dict[str, Any]] = []
    plan = getattr(run_state, "plan", None) if run_state is not None else None
    if plan is not None:
        plan_tasks = [{"task_id": t.task_id, "task_type": t.task_type.value}
                      for t in plan.tasks]

    states: List[str] = []
    decisions: List[str] = []
    if run_state is not None:
        for step in getattr(run_state, "supervisor_trace", []) or []:
            states.append(step.get("state", ""))
            # 语义上的“REVIEW 决策”只来自 review 步骤；revise/replan 步骤只是
            # 同一决策的执行化记录，done/abort 终态亦不重复计入。
            if step.get("decision") and step.get("state") == "review":
                decisions.append(step["decision"])

    messages = [
        {"message_id": m.message_id, "sender": m.sender,
         "receiver": m.receiver, "message_type": m.message_type}
        for m in (getattr(run_state, "agent_messages", None) or [])
    ] if run_state is not None else []

    tool_events = [
        {"tool_name": getattr(e, "tool_name", None),
         "agent_id": getattr(e, "agent_id", None),
         "success": bool(getattr(e, "success", False)),
         "latency_ms": getattr(e, "latency_ms", None),
         "error": getattr(e, "error", None)}
        for e in (getattr(run_state, "tool_events", None) or [])
    ] if run_state is not None else []

    ok_tools = [e["tool_name"] for e in tool_events if e["success"]]
    tool_summary: Dict[str, Any] = {
        "total": len(tool_events),
        "success": len(ok_tools),
        "failure": len(tool_events) - len(ok_tools),
        "names": list(dict.fromkeys(e["tool_name"] for e in tool_events)),
        "total_latency_ms": round(sum(float(e["latency_ms"] or 0.0) for e in tool_events), 3),
    }
    # 注入 EventCollector.snapshot()（若有），保证与 Collector 同源
    collector = getattr(run_state, "collector", None) if run_state is not None else None
    if collector is not None and hasattr(collector, "snapshot"):
        tool_summary["collector_snapshot"] = collector.snapshot()

    metrics: Dict[str, Any] = {
        "executed_tasks": len(statuses),
        "succeeded": sum(1 for s in statuses.values() if s == "succeeded"),
        "failed": sum(1 for s in statuses.values() if s == "failed"),
        "skipped": sum(1 for s in statuses.values() if s == "skipped"),
        "selected_count": len(selection_ids),
    }

    final_state = None
    if run_state is not None and getattr(run_state, "final_state", None) is not None:
        final_state = run_state.final_state.value

    plan_ok = bool(getattr(runtime, "last_plan_ok", False))
    if plan is not None:
        plan_ok = True

    return EvaluationTrace(
        case_id=case.case_id,
        request_id=request_id,
        query=case.query,
        goal=case.goal or None,
        duration_ms=round(duration_ms, 3),
        plan_ok=plan_ok,
        final_state=final_state,
        error=error,
        plan_tasks=plan_tasks,
        execution_status=statuses,
        supervisor_states=states,
        decisions=decisions,
        agent_messages=messages,
        tool_events=tool_events,
        tool_summary=tool_summary,
        selection_ids=list(selection_ids),
        metrics=metrics,
    )


def expected_violations(case: EvaluationCase, trace: EvaluationTrace) -> List[str]:
    """确定性预期检查：返回违例列表（空 = 通过）。"""
    violations: List[str] = []
    if case.expected_final_state and trace.final_state != case.expected_final_state:
        violations.append(
            f"final_state: expect {case.expected_final_state!r}, got {trace.final_state!r}"
        )
    if case.expect_selection and not trace.selection_ids:
        violations.append("selection: 期望非空，实际为空")
    if case.expected_decisions is not None:
        if trace.decisions != list(case.expected_decisions):
            violations.append(
                f"decisions: expect {case.expected_decisions!r}, got {trace.decisions!r}"
            )
    observed_senders = {m.get("sender") for m in trace.agent_messages}
    for agent in case.expected_agents:
        if agent not in observed_senders:
            violations.append(f"agent: 期望执行 {agent!r}，实际 senders={sorted(observed_senders)}")
    observed_tools = set(trace.tool_summary.get("names") or [])
    for tool in case.expected_tools:
        if tool not in observed_tools:
            violations.append(f"tool: 期望调用 {tool!r}，实际 tools={sorted(observed_tools)}")
    return violations


class EvaluationRunner:
    """离线 runner：case → runtime 执行 → EvaluationTrace → Summary。"""

    def __init__(
        self,
        runtime_factory: Optional[Callable[[EvaluationCase], Any]] = None,
        request_id_prefix: str = "eval",
        judge: Optional[Any] = None,
    ):
        # runtime_factory(case) 应返回具备 run_stream(messages)->Iterable[str] 的 runtime，
        # 并暴露 .run_state（SupervisorAgent 风格）。
        self.runtime_factory = runtime_factory
        self.request_id_prefix = request_id_prefix
        self.judge = judge                 # 可选 Judge（evaluate_trace(trace) -> JudgeResult）
        self.judge_results: List[JudgeResult] = []

    # -- 单条 ---------------------------------------------------------------
    def run_case(self, case: EvaluationCase, request_id: Optional[str] = None) -> EvaluationTrace:
        if self.runtime_factory is None:
            raise RuntimeError("EvaluationRunner 需要 runtime_factory（case -> runtime）")
        if not case.query.strip():
            raise ValueError(f"case {case.case_id} 缺少 query")
        request_id = request_id or f"{self.request_id_prefix}_{case.case_id}_{uuid.uuid4().hex[:8]}"
        messages = list(case.messages) if case.messages else []
        if not any(m.get("role") == "user" for m in messages):
            messages.append({"role": "user", "content": case.query})
        runtime = self.runtime_factory(case)
        kwargs: Dict[str, Any] = {}
        if case.goal:
            kwargs["goal_override"] = case.goal

        selection_ids: List[int] = []
        error: Optional[str] = None
        started = time.perf_counter()
        try:
            for raw in runtime.run_stream(messages, **kwargs):
                try:
                    event = json.loads(raw)
                except Exception:
                    continue
                if event.get("event") == "selection":
                    ids = event.get("ids") or []
                    if ids:
                        selection_ids = [int(i) for i in ids]
        except Exception as e:  # 记录失败并继续产出 trace（不中断整批）
            error = f"{type(e).__name__}: {str(e)[:200]}"
        duration_ms = (time.perf_counter() - started) * 1000.0

        return build_evaluation_trace(
            case=case,
            runtime=runtime,
            run_state=getattr(runtime, "run_state", None),
            selection_ids=selection_ids,
            duration_ms=duration_ms,
            request_id=request_id,
            error=error,
        )

    # -- 批量 ---------------------------------------------------------------
    def run(self, cases: List[EvaluationCase]) -> Tuple[List[EvaluationTrace], EvaluationSummary]:
        traces: List[EvaluationTrace] = []
        results: List[Dict[str, Any]] = []
        for case in cases:
            trace = self.run_case(case)
            traces.append(trace)
            violations = expected_violations(case, trace)
            entry: Dict[str, Any] = {
                "case_id": case.case_id,
                "passed": not violations,
                "final_state": trace.final_state,
                "mismatches": violations,
            }
            if self.judge is not None:
                judge_result = self.judge.evaluate_trace(trace)
                self.judge_results.append(judge_result)
                entry["judge"] = judge_result.model_dump()
            results.append(entry)
        summary = EvaluationSummary(
            total=len(cases),
            passed=sum(1 for r in results if r["passed"]),
            failed=sum(1 for r in results if not r["passed"]),
            results=results,
        )
        return traces, summary

