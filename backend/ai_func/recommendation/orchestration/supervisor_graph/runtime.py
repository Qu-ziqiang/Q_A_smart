"""Phase 6 — LangGraphSupervisorRuntime：以 LangGraph 为后端的 Supervisor 流式封装。

职责（与 legacy run_stream 的 SSE 事件语义保持一致）：
  1. 初始规划（planner → validator → plan.created / plan.failed）在图外一次完成；
  2. 将已规划 plan 作为初始 AgentRuntimeState 输入 StateGraph；
  3. 逐 superstep 取 node 增量 events 并原序 yield（plan.created / task.* /
     agent.review / assistant_delta / selection）；
  4. replan 在 review_node 内同步重规划（等价 legacy loop-head）；其余语义与 legacy 一致。
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterator, List, Optional

from langgraph.checkpoint.memory import MemorySaver

from ..models import PlannerFailure
from ..supervisor import SupervisorState
from .graph import build_supervisor_graph
from .state import AgentRuntimeState, new_state


class LangGraphSupervisorRuntime:
    """绑定某个 SupervisorAgent 实例的 LangGraph 运行时。"""

    def __init__(self, supervisor: Any):
        self.sup = supervisor
        self.graph: Any = None            # compiled StateGraph（checkpointer=MemorySaver）
        self.graph_config: Dict[str, Any] = None
        self.applied_edges: List[str] = []  # 审计用：记录走过的路由

    # ------------------------------------------------------------------
    def _compile(self) -> None:
        if self.graph is None:
            self.graph = build_supervisor_graph(self.sup).compile(
                checkpointer=MemorySaver())
            self.graph_config = {
                "recursion_limit": 64,
                "configurable": {"thread_id": "lg_" + uuid.uuid4().hex[:12]},
            }

    def _initial_plan(self, goal: str, messages: List[Dict[str, str]]) -> Iterator[Dict[str, Any]]:
        """初始规划：plan.created / plan.failed（与 legacy 相同分支）。"""
        run_state = self.sup.run_state
        run_state.trace_step(SupervisorState.PLAN, 0, None)
        try:
            vres = self.sup.ensure_plan(goal, messages, observation=None)
        except PlannerFailure as e:
            print(f"[Supervisor] planner failure: {e}")
            run_state.final_state = SupervisorState.ABORT
            yield {"event": "plan.failed", "reason": "planner_failure", "errors": [str(e)]}
            return
        if not vres.valid or vres.plan is None:
            print(f"[Supervisor] validation failed: {vres.errors}")
            run_state.final_state = SupervisorState.ABORT
            yield {"event": "plan.failed", "reason": "validation_failed",
                   "errors": list(vres.errors[:5])}
            return
        plan = vres.plan
        run_state.plan = plan
        self.sup.last_plan_ok = True
        self._initial_plan_result = plan
        yield self.sup._plan_created_event(plan)

    # ------------------------------------------------------------------
    def run_stream(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        *,
        request_id: str = "",
        goal: str = "",
    ) -> Iterator[Dict[str, Any]]:
        """产出与 legacy run_stream 同语义的事件 dict 流（由调用方 _encode 下发）。"""
        messages = list(messages or [])
        goal = (goal or "").strip()

        self._initial_plan_result: Any = None
        for ev in self._initial_plan(goal, messages):
            yield ev
        plan = self._initial_plan_result
        if plan is None:  # planner/validator 失败已 yield plan.failed
            return

        self._compile()
        sup = self.sup
        sup._lg_graph = self.graph
        sup._lg_graph_config = self.graph_config

        initial = new_state(request_id=request_id, goal=goal,
                            messages=messages, plan=plan)
        for chunk in self.graph.stream(initial, self.graph_config):
            for node_name, updates in (chunk or {}).items():
                if not isinstance(updates, dict):
                    continue
                self.applied_edges.append(node_name)
                events = updates.get("events")
                if events:
                    for ev in events:
                        yield ev
        # 图结束后同步兜底（finalize 节点通常已设置）
        if self.sup.run_state.final_state is None:
            st = self.graph.get_state(self.graph_config).values or {}
            fs = st.get("final_state")
            if fs:
                self.sup.run_state.final_state = (
                    SupervisorState.DONE if fs == "done" else SupervisorState.ABORT)
