"""Adaptive Agent Team — orchestration 包（Phase 1 / 1-A / 1-B / 1-C / Specialist）。

对外 API：
  - models:    TaskType / Task / TaskPlan / TaskPlanDraft / PlanValidator / 常量
  - gate:      ComplexityGate（简单 / 复杂路由）
  - planner:   PlannerAgent（一次性 Typed 规划）
  - workers:   Retrieval / Rerank / Metadata / Dedup / Group / Sort / Select /
               Vision / Verify / DeterministicSynthesis
  - synthesizer: SynthesizerAgent（LLM 合成；内部回退确定性合成）+ validate_selection
  - executor:  DAGExecutor（串行 Bounded DAG 执行）
  - orchestrator: AdaptiveTeamOrchestrator（Planner→Validator→Executor→Synthesizer→fallback）
  - agent_cards: AgentCard（角色元数据）+ 默认三张 Agent Card
  - specialist:  SpecialistAgent（Worker → Specialist 角色化包装层）
  - agent_registry: AgentRegistry（注册/查询/任务分发）+ build_dispatched_workers
  - messages:   AgentMessage（Supervisor ↔ Specialist 通信信封雏形，本阶段不通信）
  - tools:      MCP-style Tool Registry（ToolDefinition / ToolRegistry / MCP schema / 内置工具）
"""

from .models import (
    MAX_PLAN_DEPTH,
    MAX_TASKS,
    PLANNER_TIMEOUT_SECONDS,
    PlanValidationResult,
    PlanValidator,
    PlannerFailure,
    RouteType,
    Task,
    TaskDraft,
    TaskPlan,
    TaskPlanDraft,
    TaskStatus,
    TaskType,
    vision_available,
)
from .gate import ComplexityGate, GateDecision
from .planner import PlannerAgent, TASK_PLAN_DRAFT_SCHEMA
from .workers import (
    DedupWorker,
    DeterministicSynthesisWorker,
    GroupImagesWorker,
    MetadataWorker,
    RerankWorker,
    RetrievalWorker,
    SelectWorker,
    SortWorker,
    VerifyVisualWorker,
    VisionWorker,
    WorkerResult,
    default_workers,
)
from .executor import DAGExecutor, ExecutionResult, MAX_EXECUTED_TASKS
from .orchestrator import AdaptiveTeamOrchestrator
from .synthesizer import SynthesisResult, SynthesizerAgent, SynthesizerFailure, validate_selection
from .agent_cards import AgentCard, build_default_agent_cards, default_agent_card_map
from .agent_registry import (
    AgentRegistry,
    build_default_agent_registry,
    build_dispatched_workers,
)
from .messages import (
    TYPE_DECISION,
    TYPE_EVENT,
    TYPE_REPORT,
    TYPE_TASK,
    AgentMessage,
)
from .specialist import SpecialistAgent
from .observation import (
    EXPLORATORY_CLIP_TOP1_THRESHOLD,
    ReplanAction,
    RoundObservation,
    TriggerType,
    build_observation,
    low_relevance_threshold,
)
from .supervisor import (
    MAX_SUPERVISOR_ITERATIONS,
    MIN_CANDIDATE_THRESHOLD,
    RegistryAnalyzer,
    RunState,
    SupervisorAgent,
    SupervisorDecision,
    SupervisorState,
    build_registry_backed_workers,
    make_registry_retrieval_adapter,
    revise_plan,
    review_execution,
)
from .tools import (
    CATEGORY_ORGANIZATION,
    CATEGORY_RETRIEVAL,
    CATEGORY_VISION,
    ToolCallEvent,
    ToolDefinition,
    ToolRegistry,
    builtin_tool_definitions,
    default_implementations,
    default_tool_registry,
    export_tool_schema,
    export_tool_schemas,
    normalize_candidates,
    register_builtin_tools,
    validate_arguments,
)
from .observability import EventCollector, InMemoryEventCollector

__all__ = [
    "TaskType",
    "TaskStatus",
    "RouteType",
    "Task",
    "TaskDraft",
    "TaskPlan",
    "TaskPlanDraft",
    "PlanValidationResult",
    "PlanValidator",
    "PlannerFailure",
    "GateDecision",
    "ComplexityGate",
    "PlannerAgent",
    "TASK_PLAN_DRAFT_SCHEMA",
    "RetrievalWorker",
    "RerankWorker",
    "MetadataWorker",
    "DedupWorker",
    "GroupImagesWorker",
    "SortWorker",
    "SelectWorker",
    "DeterministicSynthesisWorker",
    "VisionWorker",
    "VerifyVisualWorker",
    "WorkerResult",
    "default_workers",
    "DAGExecutor",
    "ExecutionResult",
    "MAX_EXECUTED_TASKS",
    "AdaptiveTeamOrchestrator",
    "SynthesizerAgent",
    "SynthesizerFailure",
    "SynthesisResult",
    "validate_selection",
    "MAX_TASKS",
    "MAX_PLAN_DEPTH",
    "PLANNER_TIMEOUT_SECONDS",
    "vision_available",
    # Phase 1 Specialist（Worker → Specialist Agent 角色化封装）
    "AgentCard",
    "build_default_agent_cards",
    "default_agent_card_map",
    "AgentRegistry",
    "build_default_agent_registry",
    "build_dispatched_workers",
    "SpecialistAgent",
    "AgentMessage",
    "TYPE_TASK",
    "TYPE_REPORT",
    "TYPE_DECISION",
    "TYPE_EVENT",
    # Phase 2 MCP-style Tool Registry
    "ToolDefinition",
    "ToolCallEvent",
    "ToolRegistry",
    "validate_arguments",
    "CATEGORY_RETRIEVAL",
    "CATEGORY_VISION",
    "CATEGORY_ORGANIZATION",
    "builtin_tool_definitions",
    "default_implementations",
    "default_tool_registry",
    "register_builtin_tools",
    "export_tool_schema",
    "export_tool_schemas",
    # Phase 3 Supervisor Agent
    "MAX_SUPERVISOR_ITERATIONS",
    "MIN_CANDIDATE_THRESHOLD",
    "SupervisorState",
    "SupervisorDecision",
    "RunState",
    "SupervisorAgent",
    "review_execution",
    "revise_plan",
    "RoundObservation",
    "TriggerType",
    "ReplanAction",
    "build_observation",
    "low_relevance_threshold",
    "EXPLORATORY_CLIP_TOP1_THRESHOLD",
    "build_registry_backed_workers",
    "make_registry_retrieval_adapter",
    "RegistryAnalyzer",
    # Phase 3.2 observability / adapters
    "EventCollector",
    "InMemoryEventCollector",
    "normalize_candidates",
]
