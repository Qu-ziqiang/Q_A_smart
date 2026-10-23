"""Phase 2 MCP Compatibility Layer — ToolDefinition → MCP Tool schema 映射。

目的：保证 ToolDefinition 可以映射到标准 MCP Tool 格式；
注意：本阶段不安装 MCP SDK、不启动 MCP Server。

MCP Tool 格式（2024-11-05 / 2025 各版本核心字段）：
    { name, description, inputSchema }
我们只导出这 3 个核心字段（inputSchema 即 ToolDefinition.input_schema，
它是一个 JSON Schema 对象，符合 MCP inputSchema 的形态）。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .models import ToolDefinition


def export_tool_schema(tool: ToolDefinition) -> Dict[str, Any]:
    """把单个 ToolDefinition 映射为 MCP Tool 格式字典。"""
    if not isinstance(tool, ToolDefinition):
        raise TypeError("export_tool_schema 只接受 ToolDefinition 实例")
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    }


def export_tool_schemas(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
    """批量导出（保持顺序）。"""
    return [export_tool_schema(t) for t in tools]
