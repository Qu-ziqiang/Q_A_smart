"""Phase 1 Specialist — AgentRegistry（Specialist Agent 管理 + 任务分发）。

能力：
  - register_agent(agent)
  - get_agent(agent_id)
  - list_agents()
  - get_agent_for_task(task_type)

并提供一个执行链分发辅助 build_dispatched_workers()，把 Orchestrator 的
“WorkerRegistry 直连”升级为：
    Task → AgentRegistry.get_agent_for_task(task_type)
        → SpecialistAgent.execute(task, context)
        → Worker.run(task, context)
未注册 Specialist 的任务类型回退原 Worker —— 保证 DAGExecutor 行为不变。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agent_cards import AgentCard, build_default_agent_cards
from .models import TaskType
from .specialist import SpecialistAgent
from .workers import default_workers

# 默认构建三个 Specialist 使用的卡 id（顺序即注册顺序）
_DEFAULT_ROLE_IDS = ("retrieval", "organizer", "vision")


class AgentRegistry:
    """按 agent_id 注册/查询 Specialist Agent，并按 task_type 建立分发索引。"""

    def __init__(self) -> None:
        self._agents: Dict[str, SpecialistAgent] = {}
        self._task_index: Dict[TaskType, List[str]] = {}

    # -- 注册 ----------------------------------------------------------------
    def register_agent(self, agent: SpecialistAgent) -> SpecialistAgent:
        """注册一个 SpecialistAgent；重复注册同一 agent_id 时覆盖并重建索引。"""
        if not isinstance(agent, SpecialistAgent):
            raise TypeError("register_agent 只接受 SpecialistAgent 实例")
        self._agents[agent.card.agent_id] = agent
        self._rebuild_index()
        return agent

    def _rebuild_index(self) -> None:
        index: Dict[TaskType, List[str]] = {}
        for agent in self._agents.values():
            if not agent.card.enabled:
                continue  # 禁用的 Agent 不参与任务分发（仍可查询/列举）
            for ttype in agent.card.supported_tasks:
                index.setdefault(ttype, []).append(agent.card.agent_id)
        self._task_index = index

    # -- 查询 ----------------------------------------------------------------
    def get_agent(self, agent_id: str) -> Optional[SpecialistAgent]:
        """按 agent_id 查询（含 disabled Agent）。"""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[SpecialistAgent]:
        """列出全部已注册 Agent。"""
        return list(self._agents.values())

    def get_agent_for_task(self, task_type: TaskType) -> Optional[SpecialistAgent]:
        """返回能处理该 task_type 的第一个 enabled Agent；无则 None。"""
        agent_ids = self._task_index.get(task_type) or []
        for agent_id in agent_ids:
            agent = self._agents.get(agent_id)
            if agent is not None and agent.card.enabled:
                return agent
        return None

    def list_cards(self) -> List[AgentCard]:
        return [a.card for a in self._agents.values()]

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"AgentRegistry(agents={sorted(self._agents)}, tasks={len(self._task_index)})"


# ---------------------------------------------------------------------------
# 默认三个 Specialist 构建（Retrieval / Organizer / Vision）
# ---------------------------------------------------------------------------
def build_default_agent_registry(
    workers: Optional[Dict[TaskType, Any]] = None,
    tool_registry: Optional[Any] = None,
) -> AgentRegistry:
    """用默认/注入的 Worker 构建默认三个 Specialist Agent。

    workers 缺省时使用 default_workers()（与 Planner/Validator 同源的能力口径）。
    仅对 workers 中实际存在的任务类型注册 Agent —— 视觉能力关闭时自动不注册
    VisionAgent，保证与现有 default_workers() 的能力门禁一致。

    tool_registry：Phase 2 注入的 ToolRegistry（None → 各 Agent 惰性共享默认 registry）。
    """
    base_workers = dict(workers if workers is not None else default_workers())
    cards = {c.agent_id: c for c in build_default_agent_cards(
        vision_ok=(
            TaskType.ANALYZE_VISUAL in base_workers
            or TaskType.VERIFY_VISUAL_CONDITION in base_workers
        )
    )}

    registry = AgentRegistry()
    retrieval = cards.get("retrieval")
    retrieval_workers = {
        t: base_workers[t]
        for t in (
            TaskType.REWRITE_QUERY,
            TaskType.EXPAND_QUERY,
            TaskType.SCOPE_DATASET,
            TaskType.FILTER_BY_LABELS,
            TaskType.RETRIEVE_IMAGES,
            TaskType.RETRIEVE_SIMILAR,
            TaskType.RERANK_CANDIDATES,
        )
        if t in base_workers
    }
    if retrieval is not None and retrieval_workers:
        card = retrieval.model_copy(update={"supported_tasks": list(retrieval_workers)})
        registry.register_agent(SpecialistAgent(card, retrieval_workers, registry=tool_registry))

    organizer = cards.get("organizer")
    organizer_workers = {
        t: base_workers[t]
        for t in (
            TaskType.FILTER_METADATA,
            TaskType.DEDUPLICATE_IMAGES,
            TaskType.GROUP_IMAGES,
            TaskType.SORT_IMAGES,
            TaskType.SELECT_IMAGES,
        )
        if t in base_workers
    }
    if organizer is not None and organizer_workers:
        card = organizer.model_copy(update={"supported_tasks": list(organizer_workers)})
        registry.register_agent(SpecialistAgent(card, organizer_workers, registry=tool_registry))

    vision = cards.get("vision")
    vision_workers = {
        t: base_workers[t]
        for t in (
            TaskType.ANALYZE_VISUAL,
            TaskType.VERIFY_VISUAL_CONDITION,
            TaskType.READ_VISIBLE_TEXT,
            TaskType.LIST_OBJECTS,
            TaskType.COUNT_NAMED_OBJECT,
            TaskType.JUDGE_MAIN_SUBJECT,
            TaskType.VERIFY_SPATIAL,
        )
        if t in base_workers
    }
    if vision is not None and vision_workers:
        card = vision.model_copy(update={"supported_tasks": list(vision_workers)})
        registry.register_agent(SpecialistAgent(card, vision_workers, registry=tool_registry))

    return registry


# ---------------------------------------------------------------------------
# 执行链分发辅助（Requirement 5：Orchestrator 默认 Executor 使用）
# ---------------------------------------------------------------------------
def build_dispatched_workers(
    base_workers: Dict[TaskType, Any],
    registry: AgentRegistry,
    *,
    synthesize_worker: Any = None,
) -> Dict[TaskType, Any]:
    """把 base_workers 升级为“经 AgentRegistry 分发”的 Executor workers 表。

    规则：
      - 每个 task_type 优先取 registry.get_agent_for_task(task_type)（SpecialistAgent，
        其 execute/run 会委托底层 Worker）；
      - 未注册 Agent 的 task_type 回退原 Worker（防御性，保持既有行为）；
      - SYNTHESIZE 由 synthesize_worker 覆盖（Orchestrator 传入 LLM SynthesizerAgent；
        未传则沿用 base_workers 中既有实现）。
    """
    dispatched: Dict[TaskType, Any] = {}
    for ttype, worker in base_workers.items():
        if ttype == TaskType.SYNTHESIZE:
            dispatched[ttype] = synthesize_worker if synthesize_worker is not None else worker
            continue
        agent = registry.get_agent_for_task(ttype)
        dispatched[ttype] = agent if agent is not None else worker
    if synthesize_worker is not None:
        dispatched[TaskType.SYNTHESIZE] = synthesize_worker
    return dispatched
