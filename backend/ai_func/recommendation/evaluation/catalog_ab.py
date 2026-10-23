"""Task-catalog A/B：旧 5 类 Task vs 现 10 类 Task。

只用于离线评测。Before 收窄 Planner schema / Validator 注册表 / few-shot，
避免新 TaskType 从示例泄漏进对照臂。
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Sequence

from ..orchestration.models import MAX_PLAN_DEPTH, MAX_TASKS, TaskType
from ..orchestration.planner import (
    TASK_CAPABILITY_CATALOG,
    PlannerAgent,
    _render_capability_semantics,
    _render_input_contracts,
)

BEFORE_CORE: List[str] = [
    TaskType.RETRIEVE_IMAGES.value,
    TaskType.FILTER_METADATA.value,
    TaskType.GROUP_IMAGES.value,
    TaskType.SYNTHESIZE.value,
]
BEFORE_VISION: List[str] = [TaskType.ANALYZE_VISUAL.value]

AFTER_CORE: List[str] = [
    TaskType.RETRIEVE_IMAGES.value,
    TaskType.RERANK_CANDIDATES.value,
    TaskType.FILTER_METADATA.value,
    TaskType.DEDUPLICATE_IMAGES.value,
    TaskType.GROUP_IMAGES.value,
    TaskType.SORT_IMAGES.value,
    TaskType.SELECT_IMAGES.value,
    TaskType.SYNTHESIZE.value,
]
AFTER_VISION: List[str] = [
    TaskType.ANALYZE_VISUAL.value,
    TaskType.VERIFY_VISUAL_CONDITION.value,
]

_BEFORE_MAX_TASKS = 6
_BEFORE_MAX_DEPTH = 4

_BEFORE_ROUTING = (
    "1. 能力选择遵循“最小必要”：能用便宜的检索/元数据满足就不调用昂贵能力。\n"
    "2. filter_metadata 只能使用数据库已有结构化字段；它读不到图片像素。\n"
    "3. 颜色、物体、场景、构图等可见事实必须由 analyze_visual 从像素验证。\n"
    "4. analyze_visual 只能消费已有候选；使用前必须存在 retrieve_images。\n"
    "5. analyze_visual 昂贵；仅当用户明确要求视觉核验时使用，"
    "不要因为出现“视觉/颜色/图片”一词就无条件加入。\n"
    "6. 每个计划必须且只能有一个 synthesize 作为终点。\n"
)

_BEFORE_DAG = (
    "retrieve_images → synthesize\n"
    "retrieve_images → filter_metadata → synthesize\n"
    "retrieve_images → group_images → synthesize\n"
    "retrieve_images → filter_metadata → group_images → synthesize\n"
    "retrieve_images → analyze_visual → synthesize\n"
)

_BEFORE_FEW = (
    "例1 (仅检索)：「找几张海边日落照片」→ retrieve_images → synthesize\n"
    "例2 (仅元数据)：「找 2025 年上传的照片」→ retrieve_images → filter_metadata → synthesize\n"
    "例3 (分组)：「找旅行照片，按地点分组」→ retrieve_images → group_images → synthesize\n"
    "例4 (检索 + 视觉)：「找派对照片，再确认画面中是否有生日蛋糕」→ "
    "retrieve_images → analyze_visual → synthesize\n"
    "反例：「找红色花朵照片」不自动需要 Vision。\n"
)

_PATCH_MODULES = (
    "backend.ai_func.recommendation.orchestration.models",
    "backend.ai_func.recommendation.orchestration.planner",
    "backend.ai_func.recommendation.orchestration.workers",
)


def enabled_task_types(arm: str, vision: bool) -> List[str]:
    """arm = before | after。"""
    if arm == "before":
        if vision:
            return [
                TaskType.RETRIEVE_IMAGES.value,
                TaskType.FILTER_METADATA.value,
                TaskType.GROUP_IMAGES.value,
                TaskType.ANALYZE_VISUAL.value,
                TaskType.SYNTHESIZE.value,
            ]
        return list(BEFORE_CORE)
    if arm != "after":
        raise ValueError(f"未知对照臂: {arm!r}")
    if vision:
        return [
            TaskType.RETRIEVE_IMAGES.value,
            TaskType.RERANK_CANDIDATES.value,
            TaskType.FILTER_METADATA.value,
            TaskType.DEDUPLICATE_IMAGES.value,
            TaskType.GROUP_IMAGES.value,
            TaskType.SORT_IMAGES.value,
            TaskType.SELECT_IMAGES.value,
            TaskType.ANALYZE_VISUAL.value,
            TaskType.VERIFY_VISUAL_CONDITION.value,
            TaskType.SYNTHESIZE.value,
        ]
    return list(AFTER_CORE)


def capabilities_for(arm: str, vision: bool) -> Dict[str, Any]:
    enabled = enabled_task_types(arm, vision)
    max_tasks = _BEFORE_MAX_TASKS if arm == "before" else MAX_TASKS
    max_depth = _BEFORE_MAX_DEPTH if arm == "before" else MAX_PLAN_DEPTH
    return {
        "catalog_mode": arm,
        "enabled_task_types": enabled,
        "vision_available": bool(vision),
        "max_tasks": max_tasks,
        "max_plan_depth": max_depth,
        "task_capabilities": [
            {**TASK_CAPABILITY_CATALOG[t], "task_type": t}
            for t in enabled if t in TASK_CAPABILITY_CATALOG
        ],
    }


def _enum_types(values: Sequence[str]) -> set:
    return {TaskType(v) for v in values if v in TaskType._value2member_map_}


class CatalogPinnedPlanner(PlannerAgent):
    """真实 Planner，但 capabilities 钉死在某一对照臂。"""

    def __init__(self, arm: str, vision: bool, **kwargs: Any):
        super().__init__(**kwargs)
        self.arm = arm
        self.vision = bool(vision)
        self._caps = capabilities_for(arm, self.vision)

    def create_plan(self, *args: Any, **kwargs: Any):
        caps = dict(self._caps)
        goal = kwargs.get("goal")
        if not goal and args:
            goal = args[0]
        from ..orchestration.planner import routing_mode_for_query

        caps["routing_mode"] = routing_mode_for_query(str(goal or ""))
        kwargs["capabilities"] = caps
        return super().create_plan(*args, **kwargs)

    def _parse_draft(self, raw: str):
        """评测专用：若只有一个 synthesize，把 synthesis_task_id 对齐，避免 Flash 漂移。"""
        draft = super()._parse_draft(raw)
        synth = [t for t in draft.tasks if getattr(t.task_type, "value", t.task_type) == "synthesize"]
        if len(synth) == 1 and draft.synthesis_task_id != synth[0].task_id:
            return draft.model_copy(update={"synthesis_task_id": synth[0].task_id})
        return draft

    @staticmethod
    def _build_system_prompt(capabilities: Dict[str, Any]) -> str:
        from ..prompts import PLANNER_SYSTEM_PROMPT_TEMPLATE

        enabled = capabilities.get("enabled_task_types") or []
        vision_ok = bool(capabilities.get("vision_available"))
        max_tasks = capabilities.get("max_tasks", MAX_TASKS)
        max_depth = capabilities.get("max_plan_depth", MAX_PLAN_DEPTH)
        if capabilities.get("catalog_mode") == "before":
            routing = _BEFORE_ROUTING
            dag = _BEFORE_DAG
            few = _BEFORE_FEW
            vision_note = (
                "可用，可以使用 analyze_visual"
                if vision_ok
                else "不可用，不得使用 analyze_visual"
            )
        else:
            from ..orchestration.planner import (
                _CAPABILITY_DAG_EXAMPLES,
                _CAPABILITY_FEW_SHOT,
                routing_policy_for_mode,
            )
            routing = routing_policy_for_mode(
                str(capabilities.get("routing_mode") or "simple")
            )
            dag = _CAPABILITY_DAG_EXAMPLES
            few = _CAPABILITY_FEW_SHOT
            vision_note = (
                "可用，可以使用 analyze_visual / verify_visual_condition"
                if vision_ok
                else "不可用，不得使用 analyze_visual / verify_visual_condition"
            )
        return PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
            enabled=enabled,
            max_tasks=max_tasks,
            max_plan_depth=max_depth,
            vision_note=vision_note,
            capability_semantics=_render_capability_semantics(enabled),
            input_contracts=_render_input_contracts(enabled),
            routing_policy=routing,
            dag_examples=dag,
            few_shot=few,
        )


@contextmanager
def pin_registered_catalog(arm: str, vision: bool) -> Iterator[List[str]]:
    """临时收窄注册表：Validator 只能看见本臂允许的 TaskType。"""
    enabled = enabled_task_types(arm, vision)
    pinned = _enum_types(enabled)
    core_values = [t for t in enabled if t not in AFTER_VISION]
    saved: Dict[str, Any] = {}
    models_mod = importlib.import_module(
        "backend.ai_func.recommendation.orchestration.models"
    )
    saved["_registered"] = models_mod.REGISTERED_TASK_TYPES
    saved["_registered_fn"] = models_mod.registered_task_types
    models_mod.REGISTERED_TASK_TYPES = _enum_types(core_values)
    models_mod.registered_task_types = lambda vision_available_flag=None: set(pinned)

    vision_fn = (lambda: True) if vision else (lambda: False)
    for name in _PATCH_MODULES:
        mod = importlib.import_module(name)
        saved[name] = getattr(mod, "vision_available", None)
        mod.vision_available = vision_fn
    try:
        yield enabled
    finally:
        models_mod.REGISTERED_TASK_TYPES = saved.pop("_registered")
        models_mod.registered_task_types = saved.pop("_registered_fn")
        for name, value in saved.items():
            importlib.import_module(name).vision_available = value
