"""Phase 5 — MCP Client Adapter（orchestration/tools/mcp_client.py）。

职责：
  - 连接 MCP Server（stdio 传输，FastMCP / official MCP SDK）
  - tools/list 发现工具
  - 把 MCP Tool 转换为 ToolDefinition（可注册进 ToolRegistry，实现无感知）
  - tools/call 调用工具
  - 记录 ToolCallEvent 风格 trace（tool_name/arguments/success/latency_ms/error）

设计：
  - 每次 list/call 独立连接（简单、无长生命周期状态；进程内多次调用有 spawn 开销，
    记入 performance 评估）；
  - 工具失败（MCP isError / 服务器异常）统一抛 MCPToolError；
  - 与既有 SpecialistAgent.call_tool 兼容：远程工具注册进 ToolRegistry 后走同一入口。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from .models import ToolDefinition


def _result_payload(result: Any, name: str) -> Any:
    text = "".join(str(getattr(c, "text", "") or "") for c in (result.content or []))
    if getattr(result, "isError", False) or not text.strip():
        raise MCPToolError(text or f"MCP tool {name!r} returned error")
    try:
        return json.loads(text)
    except Exception:
        return {"text": text}


def discover_fastmcp(server: Any) -> List[ToolDefinition]:
    """in-process memory 传输：对 FastMCP server 执行 tools/list。"""

    async def _run() -> List[Any]:
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            res = await session.list_tools()
            return res.tools

    return [mcp_tool_to_definition(t) for t in asyncio.run(_run())]


def call_fastmcp(server: Any, name: str, arguments: Dict[str, Any]) -> Any:
    """in-process memory 传输：调用 FastMCP server 工具。"""
    async def _run() -> Any:
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            return await session.call_tool(name, arguments=arguments)

    return _result_payload(asyncio.run(_run()), name)


def list_resource_templates_fastmcp(server: Any) -> List[Any]:
    """in-process memory 传输：列出 FastMCP server 的 resource 模板。"""
    async def _run() -> List[Any]:
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            res = await session.list_resource_templates()
            return list(res.resourceTemplates)

    return asyncio.run(_run())


def read_fastmcp_resource(server: Any, uri: str) -> str:
    """in-process memory 传输：读取 MCP resource（拼接 text 内容）。"""
    from pydantic import AnyUrl

    async def _run() -> str:
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            res = await session.read_resource(AnyUrl(uri))
            return "".join(str(getattr(c, "text", "") or "") for c in res.contents)

    return asyncio.run(_run())



class MCPToolError(RuntimeError):
    """MCP 工具调用失败（isError 或服务器异常）。"""


def mcp_tool_to_definition(tool: Any, *, source: str = "remote") -> ToolDefinition:
    """把 MCP tools/list 返回的 Tool 转换为 ToolDefinition。"""
    input_schema = getattr(tool, "inputSchema", None) or {}
    if hasattr(input_schema, "model_dump"):
        input_schema = input_schema.model_dump()
    return ToolDefinition(
        name=getattr(tool, "name", ""),
        description=getattr(tool, "description", "") or "",
        input_schema=dict(input_schema),
        output_schema="MCP CallToolResult (JSON text)",
        category="mcp",
        cost_level="medium",
        enabled=True,
        source=source,
    )


class MCPToolClient:
    """stdio 版 MCP 客户端（每次操作独立连接）。"""

    def __init__(
        self,
        module: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        command: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self.module = module
        self.command = command if command is not None else sys.executable
        self.cwd = cwd
        self.env = env
        self.name = name or module
        self._trace: List[Dict[str, Any]] = []
        self.on_call: Optional[Callable[[Dict[str, Any]], None]] = None

    # -- 内部 --------------------------------------------------------------
    def _params(self) -> StdioServerParameters:
        kwargs: Dict[str, Any] = {"command": self.command, "args": ["-m", self.module]}
        if self.cwd:
            kwargs["cwd"] = self.cwd
        if self.env is not None:
            merged = dict(os.environ)
            merged.update({k: v for k, v in self.env.items() if v is not None})
            kwargs["env"] = merged
        return StdioServerParameters(**kwargs)

    async def _session_call(self, fn: Callable[[ClientSession], Any]) -> Any:
        async with stdio_client(self._params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)

    # -- 同步 API ----------------------------------------------------------
    def list_tools(self) -> List[Any]:
        """返回 MCP tools/list 原始结果。"""
        return asyncio.run(self._session_call(lambda s: s.list_tools())).tools

    def discover(self) -> List[ToolDefinition]:
        """发现并把 MCP 工具转换为 ToolDefinition 列表。"""
        return [mcp_tool_to_definition(t) for t in self.list_tools()]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用 MCP 工具并返回 JSON 解析结果；失败抛 MCPToolError。"""
        started = time.perf_counter()
        try:
            result = asyncio.run(self._session_call(
                lambda s: s.call_tool(name, arguments=arguments)))
            latency_ms = (time.perf_counter() - started) * 1000.0
            text = "".join(str(getattr(c, "text", "") or "")
                           for c in (result.content or []))
            if getattr(result, "isError", False) or not text.strip():
                raise MCPToolError(text or f"MCP tool {name!r} returned error")
            try:
                payload = json.loads(text)
            except Exception:
                payload = {"text": text}
            self._record(name, arguments, True, latency_ms, None)
            return payload
        except MCPToolError:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record(name, arguments, False, latency_ms, "isError")
            raise
        except Exception as e:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record(name, arguments, False, latency_ms, str(e)[:200])
            raise MCPToolError(f"MCP call failed: {type(e).__name__}: {e}") from e

    # -- trace -------------------------------------------------------------
    def _record(self, name, arguments, success, latency_ms, error) -> None:
        event = {
            "tool_name": name,
            "arguments": dict(arguments),
            "agent_id": None,
            "success": bool(success),
            "latency_ms": round(latency_ms, 3),
            "error": error,
        }
        self._trace.append(event)
        if self.on_call is not None:
            self.on_call(event)

    def trace(self) -> List[Dict[str, Any]]:
        return list(self._trace)


class MemoryMCPClient:
    """in-process memory 版 MCP 客户端（同 MCPToolClient 接口，用于协议/运行时测试）。"""

    def __init__(self, server: Any, name: Optional[str] = None):
        self.server = server
        self.name = name or getattr(server, "name", "fastmcp")
        self._trace: List[Dict[str, Any]] = []

    def discover(self) -> List[ToolDefinition]:
        return discover_fastmcp(self.server)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        started = time.perf_counter()
        try:
            payload = call_fastmcp(self.server, name, arguments)
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._trace.append({"tool_name": name, "arguments": dict(arguments),
                                "agent_id": None, "success": True,
                                "latency_ms": round(latency_ms, 3), "error": None})
            return payload
        except MCPToolError as e:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._trace.append({"tool_name": name, "arguments": dict(arguments),
                                "agent_id": None, "success": False,
                                "latency_ms": round(latency_ms, 3),
                                "error": str(e)[:200]})
            raise

    def trace(self) -> List[Dict[str, Any]]:
        return list(self._trace)


# ---------------------------------------------------------------------------
# 注册为 ToolRegistry 远端工具（本地 + 远端并存，无破坏）
# ---------------------------------------------------------------------------
def register_remote_tools(registry: Any, client: MCPToolClient,
                          names: Optional[List[str]] = None) -> List[str]:
    """把 MCP 客户端发现的工具注册进 ToolRegistry；返回已注册工具名。"""
    registered: List[str] = []
    for definition in client.discover():
        if names is not None and definition.name not in names:
            continue
        props = set((definition.input_schema or {}).get("properties", {}).keys())

        def _impl(_client=client, _name=definition.name, _props=props, **kwargs):
            args: Dict[str, Any] = {}
            for key in _props:
                if key in kwargs:
                    args[key] = kwargs[key]
            # 本地 worker 常以 limit 表达 top-k；远端 schema 用 top_k 时做映射
            if "top_k" in _props and "top_k" not in args and "limit" in kwargs:
                args["top_k"] = kwargs["limit"]
            payload = _client.call_tool(_name, args)
            # MCP Server 返回 {ok, count, rows|candidates, ...} 包装 → 解包为行列表，
            # 使结果与内置 LocalTool 对 worker 的返回形状一致
            if isinstance(payload, dict):
                for key in ("rows", "candidates"):
                    if isinstance(payload.get(key), list):
                        return payload[key]
            return payload

        registry.register(definition, implementation=_impl)
        registered.append(definition.name)
    return registered

