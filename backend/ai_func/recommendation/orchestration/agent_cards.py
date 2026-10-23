"""Phase 1 Specialist — Agent Card（角色元数据，Multi-Agent Ready 第一步）。

Agent Card 只负责“描述 Agent”：
  - 身份（agent_id / name）
  - 能力范围（description）
  - 支持任务类型（supported_tasks）
  - 可使用工具（available_tools）
  - 输入 / 输出协议（input_schema / output_schema）
  - 成本级别（cost_level）与启用状态（enabled）

不绑定任何业务逻辑：Worker 实现、Tool 实现、执行流程均不在此模块。
设计依据：SMARTIMAGER_AGENT_UPGRADE_DESIGN.md §3.4.1 / Phase 1 Requirements。
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import TaskType, vision_available


class AgentCard(BaseModel):
    """Agent 角色元数据（纯描述，无行为）。"""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    name: str
    description: str
    supported_tasks: List[TaskType] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=list)
    input_schema: str = ""
    output_schema: str = ""
    cost_level: str = "cheap"
    enabled: bool = True

    # -- Phase 2 Tool Binding -------------------------------------------------
    @property
    def tools(self) -> List[str]:
        """Agent 绑定的工具名（与 available_tools 同义，供 ToolRegistry 绑定校验使用）。"""
        return self.available_tools

    def can_use(self, tool_name: str) -> bool:
        """该 Agent 是否被授权调用指定工具。"""
        return tool_name in self.available_tools


# ---------------------------------------------------------------------------
# 默认 Agent Card（Retrieval / Organizer / Vision 三角色）
# vision_ok: 视觉能力门禁（None → 读真实配置 vision_available()）。
# 卡片始终返回三张（便于描述与注册）；VisionAgent.enabled 跟随视觉能力。
# ---------------------------------------------------------------------------
def build_default_agent_cards(vision_ok: Optional[bool] = None) -> List[AgentCard]:
    """构造默认三张 Agent Card。"""
    vision = vision_available() if vision_ok is None else bool(vision_ok)
    return [
        AgentCard(
            agent_id="retrieval",
            name="RetrievalAgent",
            description=(
                "检索角色 Agent：规则改写、LLM 扩写、数据集域、类别筛选、"
                "文搜、以图搜图、重排。不负责视觉开放描述。"
            ),
            supported_tasks=[
                TaskType.REWRITE_QUERY,
                TaskType.EXPAND_QUERY,
                TaskType.SCOPE_DATASET,
                TaskType.FILTER_BY_LABELS,
                TaskType.RETRIEVE_IMAGES,
                TaskType.RETRIEVE_SIMILAR,
                TaskType.RERANK_CANDIDATES,
            ],
            available_tools=["search_images", "rerank"],
            input_schema=(
                "Rewrite/Expand/Scope/FilterByLabels/Retrieve/Similar/Rerank Inputs"
            ),
            output_schema="Rewrite/Expand/Scope/Retrieval/Rerank Outputs",
            cost_level="cheap",
            enabled=True,
        ),
        AgentCard(
            agent_id="organizer",
            name="OrganizerAgent",
            description=(
                "整理角色 Agent：元数据筛选、去重、分组、排序与子集选择。"
                "一个 Agent 承接多类整理 Task，不召回、不改 query。"
            ),
            supported_tasks=[
                TaskType.FILTER_METADATA,
                TaskType.DEDUPLICATE_IMAGES,
                TaskType.GROUP_IMAGES,
                TaskType.SORT_IMAGES,
                TaskType.SELECT_IMAGES,
            ],
            available_tools=["filter_images", "group_images"],
            input_schema="FilterMetadataInput / DeduplicateImagesInput / GroupImagesInput / SortImagesInput / SelectImagesInput",
            output_schema="MetadataOutput / DedupOutput / GroupOutput / SortOutput / SelectOutput",
            cost_level="cheap",
            enabled=True,
        ),
        AgentCard(
            agent_id="vision",
            name="VisionAgent",
            description=(
                "视觉角色 Agent：开放描述、条件核验、读字、列物体、计数、主体和空间关系。"
                "enabled 跟随视觉模型配置。只消费已有候选。"
            ),
            supported_tasks=[
                TaskType.ANALYZE_VISUAL,
                TaskType.VERIFY_VISUAL_CONDITION,
                TaskType.READ_VISIBLE_TEXT,
                TaskType.LIST_OBJECTS,
                TaskType.COUNT_NAMED_OBJECT,
                TaskType.JUDGE_MAIN_SUBJECT,
                TaskType.VERIFY_SPATIAL,
            ],
            available_tools=["analyze_image"],
            input_schema="AnalyzeVisualInput / VerifyVisualInput",
            output_schema="VisionTaskOutput / VerifyTaskOutput",
            cost_level="expensive",
            enabled=bool(vision),
        ),
    ]


def default_agent_card_map(vision_ok: Optional[bool] = None) -> dict:
    """agent_id → AgentCard 的默认映射（供 Registry 使用）。"""
    return {card.agent_id: card for card in build_default_agent_cards(vision_ok=vision_ok)}
