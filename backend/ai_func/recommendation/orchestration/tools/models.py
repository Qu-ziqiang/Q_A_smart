"""Phase 2 MCP-style Tool — 数据模型。

ToolDefinition：工具元数据（纯描述，不含实现）。
ToolCallEvent：一次工具调用的执行记录（供未来 Agent Evaluation / LLM Judge / Tool Optimization）。

不引入 MCP SDK；ToolDefinition 字段对齐 MCP Tool 最小映射所需
（name / description / inputSchema），保证未来可直接导出。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# 工具分类（category）约定词表
CATEGORY_RETRIEVAL = "retrieval"
CATEGORY_VISION = "vision"
CATEGORY_ORGANIZATION = "organization"


class ToolDefinition(BaseModel):
    """MCP-style 工具定义（元数据层，实现由 ToolRegistry 单独持有）。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: str = ""          # 返回结构说明（文本描述 / 类名 / JSON Schema 描述）
    category: str = "general"        # retrieval / vision / organization / mcp / ...
    cost_level: str = "cheap"        # cheap / medium / expensive
    source: str = "local"            # local（内置实现）| remote（MCP Server）
    enabled: bool = True


class ToolCallEvent(BaseModel):
    """一次工具调用的执行记录（Trace 载荷）。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    agent_id: Optional[str] = None   # 调用方 Agent（无则 None，表示系统/外部调用）
    success: bool = False
    latency_ms: Optional[float] = None
    error: Optional[str] = None
