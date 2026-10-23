"""Tool Registry 子包（Phase 2 MCP-style + Phase 5 Real MCP）。

对外 API：
  - models:        ToolDefinition / ToolCallEvent / CATEGORY_* 常量
  - registry:      ToolRegistry / validate_arguments
  - mcp_schema:    export_tool_schema / export_tool_schemas（本地 JSON Schema 导出层）
  - builtin_tools: register_builtin_tools / default_tool_registry / builtin_tool_definitions
  - mcp_client:    MCPToolClient / MemoryMCPClient / discover_fastmcp / call_fastmcp /
                   register_remote_tools（Phase 5 真实 MCP 客户端适配）

数据流：Agent → ToolRegistry.call(name, arguments) → ToolDefinition 校验
      → implementation → 既有能力（Worker / Repository / ImageAnalysis / Rerank）
                        → 或 MCP Server（source="remote" 远端工具）
"""

from .models import (
    CATEGORY_ORGANIZATION,
    CATEGORY_RETRIEVAL,
    CATEGORY_VISION,
    ToolCallEvent,
    ToolDefinition,
)
from .adapters import normalize_candidates
from .registry import ToolRegistry, validate_arguments
from .mcp_schema import export_tool_schema, export_tool_schemas
from .mcp_client import (
    MCPToolClient,
    MCPToolError,
    MemoryMCPClient,
    call_fastmcp,
    discover_fastmcp,
    mcp_tool_to_definition,
    register_remote_tools,
)
from .builtin_tools import (
    builtin_tool_definitions,
    default_implementations,
    default_tool_registry,
    register_builtin_tools,
)

__all__ = [
    "CATEGORY_ORGANIZATION",
    "CATEGORY_RETRIEVAL",
    "CATEGORY_VISION",
    "ToolCallEvent",
    "ToolDefinition",
    "normalize_candidates",
    "ToolRegistry",
    "validate_arguments",
    "export_tool_schema",
    "export_tool_schemas",
    "builtin_tool_definitions",
    "default_implementations",
    "default_tool_registry",
    "register_builtin_tools",
    # Phase 5 MCP Client Adapter
    "MCPToolClient",
    "MCPToolError",
    "MemoryMCPClient",
    "discover_fastmcp",
    "call_fastmcp",
    "mcp_tool_to_definition",
    "register_remote_tools",
]
