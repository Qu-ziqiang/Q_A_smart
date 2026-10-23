"""Phase 6 — Supervisor LangGraph：StateGraph 构建（execute/review/revise/finalize + router）。

LangGraph 仅承载状态转移；节点内复用的执行/评审/修订逻辑与 legacy Supervisor 完全一致
（DAGExecutor._execute、review_execution、revise_plan、finalize_output），避免双实现漂移。
"""

from __future__ import annotations

from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from ..models import PlannerFailure
from ..observation import ReplanAction
from ..supervisor import (
    SupervisorState,
    finalize_output,
    inject_expand_query,
    review_execution,
    revise_plan,
)
from .state import AgentRuntimeState

# 路由表：decision -> 目标节点（对应 PHASE6 Task4）
_ROUTE_TO_NODE = {
    "continue": "finalize",
    "revise": "revise",
    "replan": "execute",
    "abort": "finalize",
}


def _decision_dict(state: AgentRuntimeState) -> Dict[str, Any]:
    return dict(state.get("supervisor_decision") or {})


def _build_nodes(supervisor: Any) -> Dict[str, Any]:
    """生成绑定到 SupervisorAgent 的图节点闭包。"""
    sup = supervisor

    # ------------------------------------------------------------------
    def execute_node(state: AgentRuntimeState) -> Dict[str, Any]:
        """EXECUTE：复用 SupervisorAgent._execute（DAGExecutor + AgentMessage +
        ToolCallEvent 收集），并把事件收集进 state.events。"""
        run_state = sup.run_state
        plan = state.get("plan")
        if plan is None:  # 防御：无计划不执行
            raise RuntimeError("LangGraph execute_node: 缺少 plan")
        attempt = int(state.get("iteration_count", 0)) + 1
        run_state.trace_step(SupervisorState.EXECUTE, attempt, None)
        events = [ev for ev in sup._execute(plan, attempt)]
        return {
            "iteration_count": attempt,
            "execution_results": dict(run_state.execution_results),
            "tool_events": list(run_state.tool_events),
            "agent_messages": list(run_state.agent_messages),
            "events": events,
        }

    # ------------------------------------------------------------------
    def _replan_events(state: AgentRuntimeState, decision: Any) -> List[Dict[str, Any]]:
        """decision==replan 时重新规划（Observation → Planner → Validator）。"""
        run_state = sup.run_state
        attempt = int(state.get("iteration_count", 0))
        run_state.trace_step(SupervisorState.PLAN, attempt, None)
        action = getattr(decision, "suggested_action", None)
        if action == ReplanAction.RETRY_REWRITE.value and state.get("plan") is not None:
            plan = inject_expand_query(state.get("plan"), state.get("goal") or "")
            run_state.plan = plan
            sup.last_plan_ok = True
            state["plan"] = plan
            return [sup._plan_created_event(plan, iteration=attempt + 1)]
        observation = getattr(run_state, "last_observation", None)
        try:
            vres = sup.ensure_plan(
                state.get("goal", ""),
                state.get("messages") or [],
                observation=observation,
            )
        except PlannerFailure as e:
            print(f"[Supervisor] planner failure: {e}")
            run_state.final_state = SupervisorState.ABORT
            return [{"event": "plan.failed", "reason": "planner_failure", "errors": [str(e)]}]
        if not vres.valid or vres.plan is None:
            print(f"[Supervisor] validation failed: {vres.errors}")
            run_state.final_state = SupervisorState.ABORT
            return [{"event": "plan.failed", "reason": "validation_failed",
                     "errors": list(vres.errors[:5])}]
        plan = vres.plan
        run_state.plan = plan
        sup.last_plan_ok = True
        state["plan"] = plan
        return [sup._plan_created_event(plan, iteration=attempt + 1)]

    def review_node(state: AgentRuntimeState) -> Dict[str, Any]:
        """REVIEW：复用 review_execution（确定性规则）；replan 决策在此同步重规划。"""
        run_state = sup.run_state
        plan = state.get("plan")
        attempt = int(state.get("iteration_count", 0))
        decision = review_execution(
            plan,
            run_state.execution_results,
            attempt=attempt,
            max_iterations=sup.max_iterations,
            min_candidates=sup.min_candidates,
            enhanced=sup.enhanced,
            request_id=str(state.get("request_id") or ""),
            goal=str(state.get("goal") or ""),
        )
        observation = sup.observation_from_decision(decision)
        run_state.last_decision = decision
        run_state.last_observation = observation
        if observation is not None:
            run_state.observations.append(observation)
        run_state.trace_step(SupervisorState.REVIEW, attempt, decision)
        events = []
        if sup.enhanced:
            events.append(sup._observe_event(observation, attempt))
        events.append(sup._review_event(decision, attempt, SupervisorState.REVIEW.value))

        update: Dict[str, Any] = {
            "supervisor_decision": {
                "decision": decision.decision,
                "reason": decision.reason,
                "affected_tasks": list(decision.affected_tasks),
                "trigger": decision.trigger,
                "suggested_action": decision.suggested_action,
            }
        }
        if decision.decision == "replan":
            replan_events = _replan_events(state, decision)
            events.extend(replan_events)
            if run_state.final_state == SupervisorState.ABORT:
                update["planner_failed"] = True
                update["final_state"] = "abort"
        update["events"] = events
        update["plan"] = state.get("plan")
        return update

    # ------------------------------------------------------------------
    def revise_node(state: AgentRuntimeState) -> Dict[str, Any]:
        """REVISE：复用 revise_plan()（retrieve limit 扩大），然后重新执行。"""
        run_state = sup.run_state
        plan = state.get("plan")
        attempt = int(state.get("iteration_count", 0))
        decision = _decision_dict(state)
        run_state.trace_step(SupervisorState.REVISE, attempt,
                             run_state.last_decision)
        new_plan = revise_plan(plan, decision.get("affected_tasks") or [])
        run_state.plan = new_plan
        return {
            "plan": new_plan,
            "events": [sup._plan_created_event(new_plan)],
        }

    # ------------------------------------------------------------------
    def finalize_node(state: AgentRuntimeState) -> Dict[str, Any]:
        """FINALIZE：continue/abort → finalize_output → assistant_delta + selection。"""
        run_state = sup.run_state
        plan = state.get("plan")
        attempt = int(state.get("iteration_count", 0))
        decision = _decision_dict(state)
        kind = decision.get("decision", "continue")
        if kind == "continue":
            run_state.trace_step(SupervisorState.DONE, attempt, run_state.last_decision)
            run_state.final_state = SupervisorState.DONE
            final_value = "done"
        else:
            run_state.trace_step(SupervisorState.ABORT, attempt, run_state.last_decision)
            run_state.final_state = SupervisorState.ABORT
            final_value = "abort"
        text, ids, images = finalize_output(plan, run_state.execution_results)
        return {
            "final_state": final_value,
            "events": [
                {"event": "assistant_delta", "delta": text},
                {"event": "selection", "ids": ids, "images": images},
            ],
        }

    # ------------------------------------------------------------------
    def review_router(state: AgentRuntimeState) -> str:
        """条件路由：continue→finalize；revise→revise；replan→execute；abort→finalize。"""
        if state.get("planner_failed"):
            return END
        decision = _decision_dict(state).get("decision", "continue")
        node = _ROUTE_TO_NODE.get(decision, "finalize")
        # 防御：REVISE/REPLAN 但已到执行上限 → 终止（杜绝无限循环）
        if node in ("revise", "execute") and \
                int(state.get("iteration_count", 0)) >= int(sup.max_iterations):
            return "finalize"
        return node

    return {
        "execute_node": execute_node,
        "review_node": review_node,
        "revise_node": revise_node,
        "finalize_node": finalize_node,
        "review_router": review_router,
    }


def build_supervisor_graph(supervisor: Any) -> StateGraph:
    """构建未编译的 StateGraph（START→execute→review→router→…→finalize→END）。"""
    nodes = _build_nodes(supervisor)
    graph = StateGraph(AgentRuntimeState)
    graph.add_node("execute", nodes["execute_node"])
    graph.add_node("review", nodes["review_node"])
    graph.add_node("revise", nodes["revise_node"])
    graph.add_node("finalize", nodes["finalize_node"])

    graph.add_edge(START, "execute")
    graph.add_edge("execute", "review")
    graph.add_conditional_edges(
        "review",
        nodes["review_router"],
        {
            "execute": "execute",
            "revise": "revise",
            "finalize": "finalize",
            END: END,
        },
    )
    graph.add_edge("revise", "execute")
    graph.add_edge("finalize", END)
    return graph

