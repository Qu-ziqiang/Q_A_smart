"""Supervisor：多 Agent 编排的控制面（有限状态机），不是第四个推理 Agent。

TEAM 路径默认走本控制器（ENABLE_SUPERVISOR_RUNTIME 显式关闭时回退 Orchestrator）。
状态机：PLAN → EXECUTE → OBSERVE/REVIEW → (REVISE | REPLAN | DONE | ABORT)

原则：
  - 有界：最多 2 轮 EXECUTE，使用 for-range，禁止 while True；
  - REVIEW 不调用 LLM，从 WorkerResult 构造 Observation 并分类触发；
  - 空召回或低相关（RETRY_REWRITE）由 Supervisor 直接插入 expand_query，不再请 Planner 决定是否改写；
  - 其余 Replan 仍回到 Planner；执行失败才允许同 query REVISE；
  - 下面的角色 Agent 由 Registry 分派；LLM 在 Planner 与 Synthesizer；
  - enhanced=True 时状态转移优先 LangGraph，缺依赖回退本 FSM。
"""

from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .executor import DAGExecutor
from .messages import TYPE_REPORT, TYPE_TASK, AgentMessage
from .models import (
    ExpandQueryInput,
    PlanValidationResult,
    PlanValidator,
    PlannerFailure,
    Task,
    TaskPlan,
    TaskType,
)
from .observation import (
    ReplanAction,
    RoundObservation,
    TriggerType,
    assert_replan_changed,
    build_observation,
)
from .planner import PlannerAgent

# 资源上限（集中常量，禁止散落 magic number）
MAX_SUPERVISOR_ITERATIONS = 2          # 最大 EXECUTE 次数（含首次；达到后 REVIEW→ABORT）
MIN_CANDIDATE_THRESHOLD = 1            # 候选数下限（低于则触发 replan 类决策）
RETRIEVE_LIMIT_STEP = 10               # REVISE 时 retrieve limit 增量
RETRIEVE_LIMIT_CAP = 40                # REVISE 时 retrieve limit 上限（与 models 一致）


class SupervisorState(str, Enum):
    PLAN = "plan"
    EXECUTE = "execute"
    REVIEW = "review"
    REVISE = "revise"
    REPLAN = "replan"
    DONE = "done"
    ABORT = "abort"


class SupervisorDecision(BaseModel):
    """确定性 REVIEW 决策输出。"""

    model_config = {"extra": "forbid"}
    decision: Literal["continue", "revise", "replan", "abort"]
    reason: str = ""
    affected_tasks: List[str] = Field(default_factory=list)
    trigger: str = TriggerType.NONE.value
    suggested_action: str = ReplanAction.NONE.value
    observation: Optional[Dict[str, Any]] = None


class SupervisorAgent:
    """受控 Supervisor：PLAN → EXECUTE → REVIEW → (REVISE|REPLAN|DONE|ABORT)。

    - enhanced=False（默认）：等价既有“一次规划→执行→合成”基线；
    - enhanced=True：REVIEW 走确定性规则，可触发受限 REVISE / REPLAN；
    - max_iterations 限制 EXECUTE 次数（默认 2），杜绝无限循环。
    """

    def __init__(
        self,
        planner: Optional[PlannerAgent] = None,
        validator: Optional[PlanValidator] = None,
        *,
        workers: Optional[Dict[TaskType, Any]] = None,
        agent_registry: Any = None,
        tool_registry: Any = None,
        synthesize_worker: Any = None,
        enhanced: bool = False,
        max_iterations: int = MAX_SUPERVISOR_ITERATIONS,
        min_candidates: int = MIN_CANDIDATE_THRESHOLD,
        collector: Any = None,
        backend: str = "langgraph",
    ):
        self.planner = planner if planner is not None else PlannerAgent()
        self.validator = validator if validator is not None else PlanValidator()
        self._workers = workers
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry
        self._synthesize_worker = synthesize_worker
        self.enhanced = bool(enhanced)
        self.max_iterations = int(max_iterations)
        self.min_candidates = int(min_candidates)
        self.collector = collector  # EventCollector（None → run_stream 时默认 InMemory）
        self.backend = str(backend or "legacy").lower()  # legacy | langgraph
        self.executor: Optional[DAGExecutor] = None
        self.run_state: Optional[RunState] = None
        self.last_plan_ok: bool = False
        self.planner_call_count: int = 0

    # -- 组件装配 ------------------------------------------------------------
    def _default_agent_registry(self) -> Any:
        from .agent_registry import build_default_agent_registry
        from .workers import default_workers

        if self.agent_registry is not None:
            return self.agent_registry
        base = self._workers if self._workers is not None else default_workers()
        return build_default_agent_registry(base, tool_registry=self.tool_registry)

    def _default_executor(self) -> DAGExecutor:
        from .agent_registry import build_dispatched_workers
        from .synthesizer import SynthesizerAgent
        from .workers import default_workers

        base = self._workers if self._workers is not None else default_workers()
        agent_reg = self._default_agent_registry()
        synth = self._synthesize_worker if self._synthesize_worker is not None else SynthesizerAgent()
        return DAGExecutor(workers=build_dispatched_workers(base, agent_reg,
                                                            synthesize_worker=synth))

    def _get_executor(self) -> DAGExecutor:
        if self.executor is None:
            self.executor = self._default_executor()
        return self.executor

    # -- 工具注册中心（惰性默认共享 registry） --------------------------------
    def _get_tool_registry(self) -> Any:
        if self.tool_registry is None:
            from .tools.builtin_tools import default_tool_registry

            self.tool_registry = default_tool_registry()
        return self.tool_registry

    # -- 消息/事件辅助 -------------------------------------------------------
    @staticmethod
    def _encode(event: Dict[str, Any]) -> str:
        return json.dumps(event, ensure_ascii=False) + "\n\n"

    @staticmethod
    def _last_user_query(messages: Optional[List[Dict[str, str]]]) -> str:
        if not messages:
            return ""
        for m in reversed(messages):
            if m.get("role") == "user":
                return (m.get("content") or "").strip()
        return ""

    def _agent_id_for(self, task_type: TaskType) -> str:
        agent = self._default_agent_registry().get_agent_for_task(task_type)
        return agent.agent_id if agent is not None else "synthesizer"

    def _append_task_message(self, task_id: str, task_type: TaskType, plan: TaskPlan) -> None:
        task = next((t for t in plan.tasks if t.task_id == task_id), None)
        if task is None or self.run_state is None:
            return
        payload: Dict[str, Any] = {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "objective": task.objective,
            "input": task.input.model_dump(mode="json") if hasattr(task.input, "model_dump") else {},
            "depends_on": list(task.depends_on),
        }
        msg = AgentMessage(
            message_id="msg_" + uuid.uuid4().hex[:12],
            sender="supervisor",
            receiver=self._agent_id_for(task.task_type),
            message_type=TYPE_TASK,
            payload=payload,
        )
        if self.run_state is not None:
            self.run_state.add_agent_message(msg)

    def _append_report_message(self, event: Dict[str, Any], task_type: TaskType) -> None:
        if self.run_state is None:
            return
        msg = AgentMessage(
            message_id="msg_" + uuid.uuid4().hex[:12],
            sender=self._agent_id_for(task_type),
            receiver="supervisor",
            message_type=TYPE_REPORT,
            payload={k: v for k, v in event.items() if k != "event"},
        )
        self.run_state.add_agent_message(msg)

    @staticmethod
    def _plan_created_event(plan: TaskPlan, iteration: int = 1) -> Dict[str, Any]:
        return {
            "event": "plan.created",
            "plan_id": plan.plan_id,
            "plan_version": plan.plan_version,
            "goal": plan.goal,
            "iteration": iteration,
            "synthesis_task_id": plan.synthesis_task_id,
            "tasks": [{"task_id": t.task_id, "task_type": t.task_type.value,
                       "objective": t.objective} for t in plan.tasks],
        }

    @staticmethod
    def _observe_event(observation: Optional[RoundObservation], attempt: int) -> Dict[str, Any]:
        if observation is None:
            return {"event": "agent.observe", "iteration": attempt, "trigger": TriggerType.NONE.value}
        return {
            "event": "agent.observe",
            "iteration": attempt,
            "original_query": observation.original_goal,
            "trigger": observation.trigger.value,
            "suggested_action": observation.suggested_action.value,
            "REPLAN_REASON": (
                observation.trigger.value
                if observation.trigger != TriggerType.NONE else ""
            ),
            "REPLAN_ACTION": (
                observation.suggested_action.value
                if observation.suggested_action != ReplanAction.NONE else ""
            ),
            "retrieval": observation.retrieval.model_dump(mode="json"),
            "organizer": observation.organizer.model_dump(mode="json"),
            "vision": observation.vision.model_dump(mode="json"),
            "failure": observation.failure.model_dump(mode="json"),
            "current_plan_summary": observation.current_plan_summary,
        }

    @staticmethod
    def _review_event(decision: SupervisorDecision, attempt: int, state: str) -> Dict[str, Any]:
        return {
            "event": "agent.review",
            "state": state,
            "attempt": attempt,
            "iteration": attempt,
            "decision": decision.decision,
            "reason": decision.reason,
            "affected_tasks": list(decision.affected_tasks),
            "trigger": decision.trigger,
            "suggested_action": decision.suggested_action,
            "REPLAN_REASON": decision.trigger if decision.trigger != TriggerType.NONE.value else "",
            "REPLAN_ACTION": (
                decision.suggested_action
                if decision.suggested_action != ReplanAction.NONE.value else ""
            ),
        }

    def invoke_planner(
        self,
        goal: str,
        messages: Optional[List[Dict[str, str]]] = None,
        observation: Any = None,
        repair_errors: Optional[List[str]] = None,
    ):
        """调用 Planner。observation 仅第二轮传入；Planner 不执行任何业务工具。

        SM_PLAN_CACHE=1 且无 observation 时先查计划缓存（精确 + 0.95 相似）。
        命中不计 planner_call_count。工具注册变更会使缓存失效。
        """
        if observation is None and not repair_errors:
            from .plan_cache import lookup_plan, store_plan

            cached = lookup_plan(goal)
            if cached is not None:
                return cached
        self.planner_call_count += 1
        kwargs = {
            "goal": goal,
            "recent_messages": list(messages or []),
            "memory_hints": getattr(self, "_memory_hints", None),
        }
        try:
            draft = self.planner.create_plan(
                observation=observation, repair_errors=repair_errors, **kwargs
            )
        except TypeError:
            draft = self.planner.create_plan(**kwargs)
        from .plan_lint import maybe_lint_plan

        draft = maybe_lint_plan(goal, draft)
        if observation is None and not repair_errors:
            from .plan_cache import store_plan

            store_plan(goal, draft)
        return draft

    def ensure_plan(
        self,
        goal: str,
        messages: Optional[List[Dict[str, str]]] = None,
        observation: Any = None,
    ) -> PlanValidationResult:
        """Planner + Validator；空计划重试一次后 fallback retrieve→synthesize。"""
        from .plan_floor import ensure_executable_plan

        def _create(repair):
            return self.invoke_planner(
                goal, messages, observation=observation, repair_errors=repair,
            )

        vres = ensure_executable_plan(
            create_draft=_create,
            validate=lambda d: self.accept_plan(d, observation),
            goal=goal,
            previous_queries=_previous_retrieve_queries(observation),
        )
        return vres

    def accept_plan(
        self,
        draft: Any,
        observation: Optional[RoundObservation] = None,
    ) -> PlanValidationResult:
        """Validator 全量再入；Replan 未改策略视为校验失败，禁止部分执行。"""
        vres = self.validator.validate(draft)
        if not vres.valid or vres.plan is None:
            return vres
        if observation is not None:
            err = assert_replan_changed(observation, vres.plan)
            if err:
                return PlanValidationResult(valid=False, errors=[err], plan=None)
        return vres

    @staticmethod
    def observation_from_decision(decision: Optional[SupervisorDecision]) -> Optional[RoundObservation]:
        if decision is None or not decision.observation:
            return None
        try:
            return RoundObservation.model_validate(decision.observation)
        except Exception:
            return None

    # -- 主入口：PLAN → EXECUTE → REVIEW → (REVISE|REPLAN|DONE|ABORT) -------
    def run_stream(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        *,
        request_id: Optional[str] = None,
        goal_override: Optional[str] = None,
        conversation_id: Optional[str] = None,
        memory_service: Optional[Any] = None,
    ) -> Any:
        """产出与 AdaptiveTeamOrchestrator 兼容的 SSE JSON 事件流。"""
        request_id = request_id or ("req_" + uuid.uuid4().hex[:12])
        collector = self.collector
        if collector is None:
            from .observability.event_collector import InMemoryEventCollector

            collector = InMemoryEventCollector()
        self.run_state = RunState(request_id, messages, collector=collector)
        goal = (goal_override or "").strip() or self._last_user_query(messages)
        self.last_plan_ok = False
        self.planner_call_count = 0
        self._memory_hints = None
        if memory_service is not None and conversation_id:
            try:
                episodes = memory_service.recall_episodes(conversation_id, goal)
                self._memory_hints = memory_service.render_memory_hints(
                    episodes, conversation_id=conversation_id
                ) or None
            except Exception as e:
                print(f"[Memory] recall failed: {type(e).__name__}: {e}")

        # ---- Phase 6：enhanced + backend=langgraph 走 LangGraph 状态机 ----
        if self.enhanced and self.backend == "langgraph":
            try:
                from .supervisor_graph.runtime import LangGraphSupervisorRuntime
            except Exception as e:  # pragma: no cover - 依赖缺失时回退 legacy
                print(f"[Supervisor] langgraph backend unavailable, fallback legacy: {e}")
            else:
                runtime = LangGraphSupervisorRuntime(self)
                self._lg_runtime = runtime  # 审计用：节点访问序列 / checkpoint
                for raw in runtime.run_stream(messages=messages, request_id=request_id,
                                              goal=goal):
                    yield self._encode(raw)
                self._remember_execution(
                    memory_service, conversation_id, request_id, goal
                )
                return

        plan: Optional[TaskPlan] = None
        decision: Optional[SupervisorDecision] = None
        attempt = 0

        for _round in range(self.max_iterations):
            kind = decision.decision if decision else None
            action = decision.suggested_action if decision else None
            observation = self.run_state.last_observation if kind == "replan" else None

            # ---- 检索不好：Supervisor 自己插入意图扩写，不再请 Planner ----
            if kind == "replan" and action == ReplanAction.RETRY_REWRITE.value and plan is not None:
                self.run_state.trace_step(SupervisorState.PLAN, attempt, decision)
                plan = inject_expand_query(plan, goal)
                self.run_state.plan = plan
                self.last_plan_ok = True
                yield self._encode(self._plan_created_event(plan, iteration=attempt + 1))
            # ---- PLAN（首次或其余 replan）----
            elif plan is None or kind == "replan":
                self.run_state.trace_step(SupervisorState.PLAN, attempt, None)
                vres = self.ensure_plan(goal, messages, observation=observation)
                if not vres.valid or vres.plan is None:
                    print(f"[Supervisor] validation failed: {vres.errors}")
                    self.run_state.final_state = SupervisorState.ABORT
                    yield self._encode({"event": "plan.failed", "reason": "validation_failed",
                                        "errors": vres.errors[:5]})
                    return
                plan = vres.plan
                self.run_state.plan = plan
                self.last_plan_ok = True
                yield self._encode(self._plan_created_event(plan, iteration=attempt + 1))

            # ---- REVISE（执行失败：同 query 确定性重试）----
            elif kind == "revise":
                self.run_state.trace_step(SupervisorState.REVISE, attempt, decision)
                plan = revise_plan(plan, decision.affected_tasks)
                self.run_state.plan = plan
                yield self._encode(self._plan_created_event(plan, iteration=attempt + 1))

            decision = None  # 决策已消费

            # ---- EXECUTE ----
            attempt += 1
            self.run_state.trace_step(SupervisorState.EXECUTE, attempt, None)
            for ev in self._execute(plan, attempt):
                yield self._encode(ev)

            # ---- OBSERVE + REVIEW ----
            decision = review_execution(
                plan,
                self.run_state.execution_results,
                attempt=attempt,
                max_iterations=self.max_iterations,
                min_candidates=self.min_candidates,
                enhanced=self.enhanced,
                request_id=request_id,
                goal=goal,
            )
            observation = self.observation_from_decision(decision)
            self.run_state.last_decision = decision
            self.run_state.last_observation = observation
            if observation is not None:
                self.run_state.observations.append(observation)
            self.run_state.trace_step(SupervisorState.REVIEW, attempt, decision)
            if self.enhanced:
                yield self._encode(self._observe_event(observation, attempt))
            yield self._encode(self._review_event(decision, attempt, SupervisorState.REVIEW.value))

            if decision.decision == "continue":
                self.run_state.trace_step(SupervisorState.DONE, attempt, decision)
                self.run_state.final_state = SupervisorState.DONE
                text, ids, images = finalize_output(plan, self.run_state.execution_results)
                self._remember_execution(
                    memory_service, conversation_id, request_id, goal
                )
                yield self._encode({"event": "assistant_delta", "delta": text})
                yield self._encode({"event": "selection", "ids": ids, "images": images})
                return
            if decision.decision == "abort":
                self.run_state.trace_step(SupervisorState.ABORT, attempt, decision)
                self.run_state.final_state = SupervisorState.ABORT
                text, ids, images = finalize_output(plan, self.run_state.execution_results)
                yield self._encode({"event": "assistant_delta", "delta": text})
                yield self._encode({"event": "selection", "ids": ids, "images": images})
                return
            # revise / replan → 下一轮（已由 max_iterations 封顶）

        self.run_state.final_state = SupervisorState.ABORT
        text, ids, images = finalize_output(plan, self.run_state.execution_results)
        yield self._encode({"event": "assistant_delta", "delta": text})
        yield self._encode({"event": "selection", "ids": ids, "images": images})

    def _remember_execution(
        self,
        memory_service: Optional[Any],
        conversation_id: Optional[str],
        request_id: str,
        goal: str,
    ) -> None:
        """将 Supervisor 最终成功执行写入既有 episodic memory。"""
        if (
            memory_service is None
            or not conversation_id
            or self.run_state is None
            or self.run_state.final_state != SupervisorState.DONE
        ):
            return
        executor = self.executor
        context = getattr(executor, "context", None) if executor is not None else None
        if context is None:
            return
        # revise/replan 后以最终 plan/goal 为准，保持与 Orchestrator 写入契约一致。
        context.plan = self.run_state.plan
        context.goal = goal
        try:
            memory_service.remember(
                context,
                route="team",
                conversation_id=conversation_id,
                request_id=request_id,
            )
        except Exception as e:
            print(f"[Memory] write failed: {type(e).__name__}: {e}")

    # -- EXECUTE：Executor 事件 + AgentMessage + ToolCallEvent 收集 -----------
    def _execute(self, plan: TaskPlan, attempt: int) -> Any:
        """运行 DAGExecutor；产出事件 dict，并同步 RunState 中的消息/结果/工具轨迹。"""
        executor = self._get_executor()
        tool_reg = self._get_tool_registry()
        before = len(tool_reg.trace()) if tool_reg is not None else 0
        for ev in executor.execute_stream(plan):
            if self.run_state is not None:
                self.run_state.execution_results = dict(executor.context.results)
            if ev["event"] == "task.started":
                self._append_task_message(ev["task_id"], TaskType(ev["task_type"]), plan)
            elif ev["event"] in ("task.completed", "task.failed", "task.skipped"):
                self._append_report_message(ev, TaskType(ev["task_type"]))
            yield ev
        if self.run_state is not None:
            self.run_state.execution_results = dict(executor.context.results)
            if tool_reg is not None:
                for event in tool_reg.trace()[before:]:
                    self.run_state.add_tool_event(event)
        return None




class RunState:
    """单次 Supervisor 运行的状态容器（request 级隔离）。"""

    def __init__(self, request_id: str, messages: Optional[List[Dict[str, str]]] = None,
                 collector: Any = None):
        self.request_id = request_id
        self.messages = list(messages or [])
        self.collector = collector  # EventCollector（None=不收集）
        self.plan: Optional[TaskPlan] = None
        self.execution_results: Dict[str, Any] = {}
        self.agent_messages: List[AgentMessage] = []
        self.tool_events: List[Any] = []
        self.supervisor_trace: List[Dict[str, Any]] = []
        self.last_decision: Optional[SupervisorDecision] = None
        self.last_observation: Optional[RoundObservation] = None
        self.observations: List[RoundObservation] = []
        self.final_state: Optional[SupervisorState] = None

    def add_agent_message(self, message: AgentMessage) -> None:
        """记录 Agent 消息（内部列表 + 注入的 EventCollector）。"""
        self.agent_messages.append(message)
        if self.collector is not None:
            self.collector.record_agent_message(message)

    def add_tool_event(self, event: Any) -> None:
        """记录工具调用事件（内部列表 + 注入的 EventCollector）。"""
        self.tool_events.append(event)
        if self.collector is not None:
            self.collector.record_tool_event(event)

    def trace_step(self, state: SupervisorState, attempt: int,
                   decision: Optional[SupervisorDecision], note: str = "") -> None:
        self.supervisor_trace.append({
            "state": state.value if isinstance(state, SupervisorState) else str(state),
            "attempt": attempt,
            "iteration": attempt,
            "decision": decision.decision if decision else None,
            "reason": decision.reason if decision else (note or ""),
            "affected_tasks": list(decision.affected_tasks) if decision else [],
            "trigger": decision.trigger if decision else None,
            "suggested_action": decision.suggested_action if decision else None,
            "REPLAN_REASON": (decision.trigger if decision and decision.trigger != TriggerType.NONE.value else ""),
            "REPLAN_ACTION": (
                decision.suggested_action
                if decision and decision.suggested_action != ReplanAction.NONE.value else ""
            ),
        })

    def to_dict(self) -> Dict[str, Any]:
        plan_tasks = [
            {"task_id": t.task_id, "task_type": t.task_type.value}
            for t in (self.plan.tasks if self.plan else [])
        ]
        return {
            "request_id": self.request_id,
            "plan_tasks": plan_tasks,
            "synthesis_task_id": self.plan.synthesis_task_id if self.plan else None,
            "execution_status": {
                tid: (r.status.value if hasattr(r, "status") else "?")
                for tid, r in self.execution_results.items()
            },
            "agent_messages": [
                {"message_id": m.message_id, "sender": m.sender, "receiver": m.receiver,
                 "message_type": m.message_type} for m in self.agent_messages
            ],
            "tool_event_count": len(self.tool_events),
            "tool_events": [
                {"tool_name": e.tool_name, "agent_id": e.agent_id, "success": e.success,
                 "latency_ms": e.latency_ms} for e in self.tool_events
            ],
            "supervisor_trace": self.supervisor_trace,
            "final_state": self.final_state.value if self.final_state else None,
            "planner_observations": [
                o.model_dump(mode="json") for o in self.observations
            ],
        }


# ---------------------------------------------------------------------------
# Deterministic REVIEW（不调用 LLM）
# ---------------------------------------------------------------------------
def _previous_retrieve_queries(observation: Any) -> List[str]:
    if observation is None:
        return []
    fn = getattr(observation, "retrieval", None)
    q = getattr(fn, "query", None) if fn is not None else None
    if isinstance(q, str) and q.strip():
        return [q]
    try:
        from .observation import retrieve_queries_from_obs

        return list(retrieve_queries_from_obs(observation) or [])
    except Exception:
        return []


def _result_status(task: Task, results: Dict[str, Any]):
    r = results.get(task.task_id)
    return r


def _retrieval_stats(task: Task, results: Dict[str, Any]) -> int:
    """统计某 retrieve 任务的成功候选数；失败/未执行返回 -1。"""
    r = results.get(task.task_id)
    if r is None or getattr(r, "status", None) is None:
        return -1
    from .workers import TaskStatus

    if r.status != TaskStatus.SUCCEEDED:
        return -1
    out = getattr(r, "output", None)
    if out is None:
        return 0
    cands = getattr(out, "candidates", None)
    if isinstance(cands, list):
        return len(cands)
    total = getattr(out, "total", None)
    return int(total) if isinstance(total, int) else 0


def review_execution(
    plan: TaskPlan,
    results: Dict[str, Any],
    *,
    attempt: int,
    max_iterations: int,
    min_candidates: int = MIN_CANDIDATE_THRESHOLD,
    enhanced: bool = False,
    request_id: str = "",
    goal: str = "",
) -> SupervisorDecision:
    """确定性 REVIEW：构造 Observation 并分类触发。

    优先级：
      1) EXECUTION_FAILURE → revise（同 query 重试）/ abort
      2) EMPTY_RETRIEVAL   → replan（必须改 query）/ abort
      3) LOW_RELEVANCE     → replan（改 query 或 ADD_VISION）/ abort
      4) INSUFFICIENT_CANDIDATES → replan（扩大召回或改写 query，保留筛选）
         预算用尽后返回已匹配的图，不把部分结果丢掉
      5) SYNTHESIS_FAILURE → replan / abort
    非增强模式一律 continue（保留基线行为，不额外规划）。
    Supervisor 不改 query、不重写 filter、不调用视觉模型。
    """
    if not enhanced:
        return SupervisorDecision(decision="continue", reason="enhanced disabled（基线模式）")

    obs = build_observation(
        plan,
        results,
        iteration=attempt,
        request_id=request_id,
        goal=goal,
        min_candidates=min_candidates,
    )
    payload = obs.model_dump(mode="json")
    affected = list(obs.failed_task_ids or obs.empty_retrieve_task_ids or [])
    if obs.trigger == TriggerType.SYNTHESIS_FAILURE:
        affected = [plan.synthesis_task_id]

    if obs.trigger == TriggerType.NONE:
        return SupervisorDecision(
            decision="continue",
            reason="执行结果满足要求",
            trigger=obs.trigger.value,
            suggested_action=obs.suggested_action.value,
            observation=payload,
        )

    if attempt >= max_iterations:
        if obs.trigger == TriggerType.INSUFFICIENT_CANDIDATES:
            return SupervisorDecision(
                decision="continue",
                reason=(
                    f"补检索后仍只有 {obs.matched_count}/{obs.requested_count} 张匹配，"
                    "返回已匹配结果"
                ),
                trigger=obs.trigger.value,
                suggested_action=ReplanAction.NONE.value,
                observation=payload,
            )
        return SupervisorDecision(
            decision="abort",
            reason=f"达到最大执行次数({max_iterations})，trigger={obs.trigger.value}",
            affected_tasks=affected,
            trigger=obs.trigger.value,
            suggested_action=ReplanAction.ABORT.value,
            observation=payload,
        )

    if obs.trigger == TriggerType.EXECUTION_FAILURE:
        return SupervisorDecision(
            decision="revise",
            reason=f"EXECUTION_FAILURE: {obs.failure.code or obs.failure.message}",
            affected_tasks=affected or list(obs.failed_task_ids),
            trigger=obs.trigger.value,
            suggested_action=ReplanAction.RETRY_SAME.value,
            observation=payload,
        )

    return SupervisorDecision(
        decision="replan",
        reason=f"{obs.trigger.value}: action={obs.suggested_action.value}",
        affected_tasks=affected,
        trigger=obs.trigger.value,
        suggested_action=obs.suggested_action.value,
        observation=payload,
    )


# ---------------------------------------------------------------------------
# REVISE：确定性参数修订（同计划，改输入后整体重跑）
# ---------------------------------------------------------------------------
def inject_expand_query(plan: TaskPlan, goal: str) -> TaskPlan:
    """检索为空或相关度低时，由 Supervisor 插入 expand_query，retrieve 改为依赖它。"""
    tasks: List[Task] = []
    if any(t.task_type == TaskType.EXPAND_QUERY for t in plan.tasks):
        for t in plan.tasks:
            if t.task_type == TaskType.EXPAND_QUERY:
                prev = str(getattr(t.input, "query", "") or goal or "")
                t = t.model_copy(update={
                    "input": t.input.model_copy(update={"previous_query": prev}),
                    "objective": "Supervisor 触发：检索不好，LLM 扩写",
                    "status": "pending",
                })
            else:
                t = t.model_copy(update={"status": "pending"})
            tasks.append(t)
    else:
        retrieve = next((t for t in plan.tasks if t.task_type == TaskType.RETRIEVE_IMAGES), None)
        query = str(getattr(getattr(retrieve, "input", None), "query", "") or goal or "").strip() or "找相关图片"
        dataset_id = None
        scope_ids: List[str] = []
        for t in plan.tasks:
            if t.task_type == TaskType.SCOPE_DATASET:
                scope_ids.append(t.task_id)
                raw = getattr(t.input, "dataset_id", None)
                if raw is not None:
                    dataset_id = int(raw)
        used = {t.task_id for t in plan.tasks}
        expand_id = "expand_1"
        n = 1
        while expand_id in used:
            n += 1
            expand_id = f"expand_{n}"
        expand = Task(
            task_id=expand_id,
            task_type=TaskType.EXPAND_QUERY,
            objective="Supervisor 触发：检索不好，LLM 扩写",
            input=ExpandQueryInput(query=query, dataset_id=dataset_id, previous_query=query),
            depends_on=scope_ids[:1],
        )
        inserted = False
        for t in plan.tasks:
            if t.task_type == TaskType.RETRIEVE_IMAGES and not inserted:
                tasks.append(expand)
                inserted = True
            if t.task_type == TaskType.RETRIEVE_IMAGES:
                deps = list(t.depends_on)
                if expand_id not in deps:
                    deps.append(expand_id)
                t = t.model_copy(update={"depends_on": deps, "status": "pending"})
            else:
                t = t.model_copy(update={"status": "pending"})
            tasks.append(t)
        if not inserted:
            tasks.insert(0, expand)
    return TaskPlan(
        plan_version=plan.plan_version,
        plan_id="plan_" + uuid.uuid4().hex[:12],
        goal=plan.goal,
        tasks=tasks,
        synthesis_task_id=plan.synthesis_task_id,
    )


def revise_plan(plan: TaskPlan, affected_tasks: List[str]) -> TaskPlan:
    """对受影响任务做确定性参数修订（当前仅作用于 retrieve_images）。"""
    tasks: List[Task] = []
    for t in plan.tasks:
        if t.task_id in affected_tasks and t.task_type == TaskType.RETRIEVE_IMAGES:
            new_limit = min(int(getattr(t.input, "limit", 20) or 20) + RETRIEVE_LIMIT_STEP,
                            RETRIEVE_LIMIT_CAP)
            new_input = t.input.model_copy(update={"limit": new_limit})
            t = t.model_copy(update={
                "input": new_input,
                "objective": f"{t.objective}（Supervisor 重试：扩大候选）",
                "status": "pending",
            })
        elif t.task_id in affected_tasks and t.task_type != TaskType.RETRIEVE_IMAGES:
            t = t.model_copy(update={"objective": f"{t.objective}（Supervisor 重试）",
                                     "status": "pending"})
        else:
            t = t.model_copy(update={"status": "pending"})
        tasks.append(t)
    return TaskPlan(
        plan_version=plan.plan_version,
        plan_id="plan_" + uuid.uuid4().hex[:12],
        goal=plan.goal,
        tasks=tasks,
        synthesis_task_id=plan.synthesis_task_id,
    )


# ---------------------------------------------------------------------------
# ToolRegistry 集成（adapter，不改 Worker 核心逻辑）
# ---------------------------------------------------------------------------
def make_registry_retrieval_adapter(registry: Any):
    """把 RetrievalWorker 的 search_tool 注入位接到 ToolRegistry('search_images')。"""
    def _search(query: str, targets: Optional[List[str]] = None,
                filters: Optional[Dict[str, Any]] = None, limit: int = 20):
        return registry.call("search_images", {
            "query": query,
            "vector_targets": list(targets or []),
            "filters": filters or {},
            "limit": limit,
        })
    return _search


class RegistryAnalyzer:
    """把 VisionWorker 的 analyzer 注入位接到 ToolRegistry('analyze_image')。

    提供 .client（非 None）以通过 VisionWorker 客户端门禁；analyze_image 只转发
    ToolDefinition.input_schema 声明的键。
    """

    client = True

    def __init__(self, registry: Any):
        self.registry = registry

    def analyze_image(self, image_url: Optional[str] = None, local_image_path: Optional[str] = None,
                      model: Optional[str] = None, detail: str = "low",
                      prompt: Optional[str] = None, timeout: Optional[float] = None,
                      **ignored) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        if image_url is not None:
            args["image_url"] = image_url
        if local_image_path is not None:
            args["local_image_path"] = local_image_path
        if model is not None:
            args["model"] = model
        if detail is not None:
            args["detail"] = detail
        if prompt is not None:
            args["prompt"] = prompt
        if timeout is not None:
            args["timeout"] = timeout
        return self.registry.call("analyze_image", args)


def build_registry_backed_workers(registry: Any, *, with_vision: Optional[bool] = None):
    """构造“工具经 ToolRegistry”的 Worker 表（adapter 方案，Worker 逻辑不变）。"""
    from .workers import (
        DedupWorker,
        GroupImagesWorker,
        MetadataWorker,
        ExpandQueryWorker,
        RerankWorker,
        RetrievalWorker,
        SelectWorker,
        SortWorker,
        FocusedVisionWorker,
        VerifyVisualWorker,
        VisionWorker,
    )

    workers: Dict[TaskType, Any] = {
        TaskType.RETRIEVE_IMAGES: RetrievalWorker(search_tool=make_registry_retrieval_adapter(registry)),
        TaskType.EXPAND_QUERY: ExpandQueryWorker(),
        TaskType.RERANK_CANDIDATES: RerankWorker(),
        TaskType.FILTER_METADATA: MetadataWorker(),
        TaskType.DEDUPLICATE_IMAGES: DedupWorker(),
        TaskType.GROUP_IMAGES: GroupImagesWorker(),
        TaskType.SORT_IMAGES: SortWorker(),
        TaskType.SELECT_IMAGES: SelectWorker(),
    }
    if with_vision is None:
        tool = registry.get("analyze_image")
        with_vision = bool(tool is not None and tool.enabled)
    if with_vision:
        analyzer = RegistryAnalyzer(registry)
        workers[TaskType.ANALYZE_VISUAL] = VisionWorker(analyzer=analyzer)
        workers[TaskType.VERIFY_VISUAL_CONDITION] = VerifyVisualWorker(analyzer=analyzer)
        for facet in (
            TaskType.READ_VISIBLE_TEXT,
            TaskType.LIST_OBJECTS,
            TaskType.COUNT_NAMED_OBJECT,
            TaskType.JUDGE_MAIN_SUBJECT,
            TaskType.VERIFY_SPATIAL,
        ):
            workers[facet] = FocusedVisionWorker(facet, analyzer=analyzer)
    return workers


# ---------------------------------------------------------------------------
# 最终输出（与 AdaptiveTeamOrchestrator 兼容的 text / selection）
# ---------------------------------------------------------------------------
def _public_images(candidates: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for c in candidates:
        out.append({
            "id": c.id,
            "score": c.score,
            "title": c.title,
            "tags": list(c.tags) if isinstance(getattr(c, "tags", None), list) else [],
            "public_url": c.public_url,
        })
    return out


def finalize_output(plan: TaskPlan, results: Dict[str, Any]) -> tuple:
    """从 synthesize 结果构建 (text, image_ids, images)；失败时退化为确定性摘要。"""
    from .workers import SynthesisOutput, TaskStatus, status_summary

    synth = results.get(plan.synthesis_task_id)
    if (synth is not None and getattr(synth, "status", None) == TaskStatus.SUCCEEDED
            and isinstance(getattr(synth, "output", None), SynthesisOutput)
            and (getattr(synth.output, "answer", "") or "").strip()):
        out: SynthesisOutput = synth.output
        return out.answer.strip(), list(out.image_ids), _public_images(out.candidates)

    ids: List[int] = []
    images: List[Dict[str, Any]] = []
    lines = ["已按任务计划执行（Supervisor 模式）。"]
    for t in plan.tasks:
        r = results.get(t.task_id)
        if r is None:
            continue
        status = getattr(r, "status", None)
        if status == TaskStatus.SUCCEEDED:
            lines.append(f"- 任务 {t.task_id}（{t.task_type.value}）：{status_summary(r)}")
        elif status == TaskStatus.FAILED:
            code = r.error.code if r.error else "UNKNOWN"
            lines.append(f"- 任务 {t.task_id}（{t.task_type.value}）：失败（{code}）")
    if synth is not None and getattr(synth, "output", None) is not None:
        out = synth.output
        if getattr(out, "image_ids", None):
            ids = list(out.image_ids)
            images = _public_images(getattr(out, "candidates", None) or [])
    lines.append(f"共推荐 {len(ids)} 张。" if ids else "未找到满足条件的图片。")
    return "\n".join(lines), ids, images



