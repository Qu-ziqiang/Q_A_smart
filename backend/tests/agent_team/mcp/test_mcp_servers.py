"""Phase 5 — MCP Server / Client 协议测试（official MCP SDK，in-process memory 传输）。

说明：MCP Server 以 FastMCP（stdio 生产入口见各 server.py 的 __main__）；本套测试使用
mcp.shared.memory 建立真实 JSON-RPC 会话（tools/list / tools/call），稳定且无需 OS 进程。
stdio 通道已单独冒烟验证（见 PHASE5 报告 runtime evidence）。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/mcp/test_mcp_servers.py
"""

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

# 先设夹具环境再导入 server 模块（每个测试文件独立进程）
os.environ["SMARTIMAGER_MCP_RETRIEVAL_SOURCE"] = "fake"
os.environ["SMARTIMAGER_MCP_VISION_SOURCE"] = "fake"

import backend.mcp_servers.retrieval_server.server as retrieval_server
import backend.mcp_servers.vision_server.server as vision_server

from backend.ai_func.recommendation.orchestration import ToolRegistry, register_builtin_tools
from backend.ai_func.recommendation.orchestration.tools.mcp_client import (
    MemoryMCPClient,
    MCPToolError,
    call_fastmcp,
    discover_fastmcp,
    list_resource_templates_fastmcp,
    read_fastmcp_resource,
    register_remote_tools,
)


def test_retrieval_discovery_and_schema():
    tools = discover_fastmcp(retrieval_server.mcp)
    names = [t.name for t in tools]
    assert names == ["search_images", "rerank_candidates"]
    search = next(t for t in tools if t.name == "search_images")
    assert set((search.input_schema.get("properties") or {}).keys()) >= {"query", "top_k"}
    assert search.source == "remote"


def test_retrieval_search_call():
    payload = call_fastmcp(retrieval_server.mcp, "search_images",
                           {"query": "城市建筑", "top_k": 2})
    assert payload["ok"] is True
    assert payload["count"] == 2
    assert [r["id"] for r in payload["rows"]] == [901, 902]


def test_vision_discovery_and_call():
    tools = discover_fastmcp(vision_server.mcp)
    assert [t.name for t in tools] == ["analyze_image"]
    payload = call_fastmcp(vision_server.mcp, "analyze_image",
                           {"image_id": "901", "task": "画面内容"})
    assert payload["ok"] is True
    assert payload["title"]


def test_rerank_tool_call():
    candidates = [{"id": 1, "score": 0.9, "metadata": {}},
                  {"id": 2, "score": 0.8, "metadata": {}}]
    payload = call_fastmcp(retrieval_server.mcp, "rerank_candidates",
                           {"candidates": candidates, "strategy": "default"})
    assert payload["ok"] is True
    assert payload["count"] == 2


def test_mcp_failure_handling():
    try:
        call_fastmcp(retrieval_server.mcp, "search_images", {"query": "fail", "top_k": 2})
    except MCPToolError:
        pass
    else:
        raise AssertionError("服务器错误应抛 MCPToolError")
    try:
        call_fastmcp(vision_server.mcp, "analyze_image", {"image_id": "fail", "task": "x"})
    except MCPToolError:
        pass
    else:
        raise AssertionError("vision 服务器错误应抛 MCPToolError")


def test_remote_tools_register_and_local_coexist():
    reg = ToolRegistry()
    register_builtin_tools(reg)  # 本地工具先注册（LocalTool）
    client = MemoryMCPClient(retrieval_server.mcp)
    registered = register_remote_tools(reg, client, names=["search_images"])  # MCPRemoteTool
    assert registered == ["search_images"]
    assert reg.get("search_images").source == "remote"
    payload = reg.call("search_images", {"query": "城市建筑", "top_k": 2})
    assert isinstance(payload, list) and len(payload) == 2
    assert payload[0]["id"] == 901
    # 本地 + 远端并存：其他本地工具仍可用
    assert reg.get("group_images").source == "local"
    assert client.trace() and client.trace()[-1]["success"] is True


def test_mcp_resources():
    templates = list_resource_templates_fastmcp(retrieval_server.mcp)
    uris = [str(t.uriTemplate) for t in templates]
    assert "image://{image_id}" in uris and "metadata://{image_id}" in uris
    text = read_fastmcp_resource(retrieval_server.mcp, "image://901")
    assert json.loads(text)["id"] == 901


# ---------------------------------------------------------------------------
def _main():
    failed = 0
    passed = 0
    mod_globals = globals()
    for name in sorted(mod_globals):
        if name.startswith("test_"):
            fn = mod_globals[name]
            try:
                fn()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print(f"\n== {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
