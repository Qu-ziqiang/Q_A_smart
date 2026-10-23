"""Phase 6 — AgentRuntimeState：LangGraph StateGraph 的共享状态。

字段与 RunState / Supervisor 语义一一对应（request 级，无跨请求共享）。
集合类字段使用 reducer 追加（Annotated + _merge），便于每轮 superstep 增量收集。
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict


def _merge(left: List[Any], right: List[Any]) -> List[Any]:
    """reducer：拼接追加型字段（events / tool_events / agent_messages）。"""
    return list(left or []) + list(right or [])


class AgentRuntimeState(TypedDict, total=False):
    """LangGraph Supervisor 共享状态。

    - plan：校验后的 TaskPlan（None 表示需要初始/重新规划）
    - supervisor_decision：review 决策 {decision, reason, affected_tasks}
    - events：待下发为 SSE 的事件 dict 缓冲（plan.created / task.* / agent.review /
      assistant_delta / selection）
    - final_state：done | abort（None = 仍在执行）
    - planner_failed：规划/校验失败标记（直接终止，不产出 selection，与 legacy 一致）
    """

    request_id: str
    goal: str
    messages: List[Dict[str, str]]
    plan: Any  # TaskPlan（类型仅为避免重依赖）

    execution_results: Dict[str, Any]
    agent_messages: Annotated[List[Any], _merge]
    tool_events: Annotated[List[Any], _merge]
    events: Annotated[List[Dict[str, Any]], _merge]

    supervisor_decision: Optional[Dict[str, Any]]
    iteration_count: int
    final_state: Optional[str]
    planner_failed: bool


def new_state(
    *,
    request_id: str,
    goal: str,
    messages: Optional[List[Dict[str, str]]] = None,
    plan: Any = None,
) -> AgentRuntimeState:
    """构造初始 LangGraph 状态。"""
    return {
        "request_id": request_id,
        "goal": goal,
        "messages": list(messages or []),
        "plan": plan,
        "execution_results": {},
        "agent_messages": [],
        "tool_events": [],
        "events": [],
        "supervisor_decision": None,
        "iteration_count": 0,
        "final_state": None,
        "planner_failed": False,
    }
