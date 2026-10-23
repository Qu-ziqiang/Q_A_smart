"""角色 Agent 执行入口（Supervisor–Specialist 编排里的 worker 节点）。

对外名称：RetrievalAgent / OrganizerAgent / VisionAgent（见 AgentCard.name）。
实现：AgentCard + 既有 Worker。没有独立 LLM loop，不与其他角色协商。

原则：
  - Worker 负责领域执行（检索/筛选/分组/视觉）；
  - 本类负责角色身份、工具白名单与统一 execute 入口；
  - 决策权在 Planner / Supervisor，不在本类。

SpecialistAgent 支持两种 Worker 形态：
  1) 单个 Worker（RetrievalAgent / VisionAgent）：所有 supported_tasks 共用该 Worker；
  2) Dict[TaskType, Worker]（OrganizerAgent）：按 task.task_type 解析对应 Worker。

execute() 每次执行打印角色日志（供运行时可观测 / Runtime Verification 断言），
例如："[Specialist:organizer] OrganizerAgent executed task t3 (group_images) via GroupImagesWorker"
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, Optional

from .agent_cards import AgentCard
from .models import TaskType

if TYPE_CHECKING:  # 类型标注专用；运行时避免 import workers（启动开销/循环）
    from .workers import Worker  # noqa: F401


class SpecialistAgent:
    """Worker 的角色化包装：执行仍委托给既有 Worker，行为零改动。"""

    def __init__(
        self,
        card: AgentCard,
        worker: Any,
        registry: Any = None,
    ):
        if not isinstance(card, AgentCard):
            raise TypeError("card 必须是 AgentCard 实例")
        if worker is None:
            raise ValueError(f"agent '{card.agent_id}' 缺少底层 Worker")

        self.card = card
        # 原样保留（单一 Worker 或 {TaskType: Worker} 映射）
        self.worker = worker
        self._task_to_worker: Dict[TaskType, Any] = self._index_workers(card, worker)
        # Phase 2 Tool Binding：注入 ToolRegistry（None → 惰性使用默认内置 registry）
        self._registry = registry

    # -- 内部：worker 索引 --------------------------------------------------
    @staticmethod
    def _index_workers(card: AgentCard, worker: Any) -> Dict[TaskType, Any]:
        if isinstance(worker, Mapping):
            missing = [t for t in card.supported_tasks if t not in worker]
            if missing:
                raise ValueError(
                    f"agent '{card.agent_id}' 的 supported_tasks 缺少对应 Worker: "
                    f"{[t.value for t in missing]}"
                )
            # 仅登记 card.supported_tasks 声明的任务类型
            return {t: worker[t] for t in card.supported_tasks}
        if card.supported_tasks:
            return {t: worker for t in card.supported_tasks}
        raise ValueError(f"agent '{card.agent_id}' 的 supported_tasks 为空，无法索引 Worker")

    def _worker_for(self, task_type: TaskType) -> Any:
        worker = self._task_to_worker.get(task_type)
        if worker is None:
            raise TypeError(
                f"agent '{self.card.agent_id}' 不支持任务类型 {getattr(task_type, 'value', task_type)!r}"
            )
        return worker

    # -- 执行入口 ------------------------------------------------------------
    def execute(self, task: Any, context: Any) -> Any:
        """Phase 1 指定执行入口：Task → SpecialistAgent.execute() → Worker.run()。"""
        ttype = getattr(task, "task_type", None)
        worker = self._worker_for(ttype)
        print(
            f"[Specialist:{self.card.agent_id}] {self.card.name} executed task "
            f"{getattr(task, 'task_id', '?')} ({getattr(ttype, 'value', ttype)}) "
            f"via {type(worker).__name__}"
        )
        return worker.run(task, context)

    # -- Worker 协议兼容 -----------------------------------------------------
    def run(self, task: Any, context: Any) -> Any:
        """Worker Protocol 兼容入口（DAGExecutor 直接以 run() 分派时使用）。"""
        return self.execute(task, context)

    # -- Phase 2 Tool Binding -------------------------------------------------
    @property
    def registry(self) -> Any:
        """Agent 的 ToolRegistry（未注入时惰性创建共享默认 registry）。"""
        if self._registry is None:
            from .tools.builtin_tools import default_tool_registry

            self._registry = default_tool_registry()
        return self._registry

    @property
    def tools(self) -> list:
        """Agent Card 绑定的工具名列表。"""
        return self.card.tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Agent 经 ToolRegistry 统一入口调用工具（受 Agent Card 绑定约束）。

        SpecialistAgent → ToolRegistry → ToolDefinition(校验) → implementation → Worker/既有能力
        """
        if not self.card.can_use(tool_name):
            raise PermissionError(
                f"agent '{self.card.agent_id}' 未绑定工具 {tool_name!r}；"
                f"可用工具: {self.card.tools}"
            )
        return self.registry.call(tool_name, arguments, agent_id=self.agent_id)

    # -- 便捷只读视图 --------------------------------------------------------
    @property
    def agent_id(self) -> str:
        return self.card.agent_id

    @property
    def task_type(self) -> Optional[TaskType]:
        """兼容 Worker 形态：单一支持任务时返回该任务；多任务返回 None。"""
        if len(self._task_to_worker) == 1:
            return next(iter(self._task_to_worker))
        return None

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"SpecialistAgent(agent_id={self.card.agent_id!r}, "
            f"name={self.card.name!r}, tasks="
            f"{[t.value for t in self.card.supported_tasks]})"
        )
