"""Phase 2 MCP-style Tool — ToolDefinition / MCP schema 单元测试。

覆盖：
  - ToolDefinition 创建与默认值
  - extra=forbid（未知字段拒绝）
  - export_tool_schema / export_tool_schemas（MCP 格式映射：name/description/inputSchema）
  - 默认五个内置工具的定义口径（name/category/cost_level/enabled）

零真实依赖。运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/tools/test_tool_schema.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from pydantic import ValidationError

from backend.ai_func.recommendation.orchestration import (
    ToolDefinition,
    builtin_tool_definitions,
    default_tool_registry,
    export_tool_schema,
    export_tool_schemas,
)
from backend.ai_func.recommendation.orchestration.tools.models import (
    CATEGORY_ORGANIZATION,
    CATEGORY_RETRIEVAL,
    CATEGORY_VISION,
)

_INPUT = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}


def test_create_minimal_definition():
    tool = ToolDefinition(name="x", description="d", input_schema=_INPUT)
    assert tool.name == "x"
    assert tool.description == "d"
    assert tool.input_schema == _INPUT
    assert tool.output_schema == ""
    assert tool.category == "general"
    assert tool.cost_level == "cheap"
    assert tool.enabled is True


def test_definition_rejects_unknown_fields():
    try:
        ToolDefinition(name="x", description="d", input_schema=_INPUT, evil=1)
    except ValidationError:
        pass
    else:
        raise AssertionError("extra=forbid 未生效")


def test_export_tool_schema_maps_mcp_core_fields():
    tool = ToolDefinition(name="search_images", description="desc", input_schema=_INPUT)
    exported = export_tool_schema(tool)
    assert set(exported.keys()) == {"name", "description", "inputSchema"}
    assert exported["name"] == "search_images"
    assert exported["description"] == "desc"
    assert exported["inputSchema"] == _INPUT


def test_export_tool_schema_rejects_non_definition():
    try:
        export_tool_schema({"name": "x"})
    except TypeError:
        pass
    else:
        raise AssertionError("export_tool_schema 应拒绝非 ToolDefinition")


def test_export_tool_schemas_batch_preserves_order():
    tools = [
        ToolDefinition(name="a", description="", input_schema={}),
        ToolDefinition(name="b", description="", input_schema={}),
    ]
    names = [item["name"] for item in export_tool_schemas(tools)]
    assert names == ["a", "b"]


def test_builtin_definitions_cover_five_tools_with_categories():
    defs = builtin_tool_definitions()
    assert set(defs.keys()) == {
        "search_images", "rerank", "analyze_image", "filter_images", "group_images",
    }
    assert defs["search_images"].category == CATEGORY_RETRIEVAL
    assert defs["rerank"].category == CATEGORY_RETRIEVAL
    assert defs["analyze_image"].category == CATEGORY_VISION
    assert defs["filter_images"].category == CATEGORY_ORGANIZATION
    assert defs["group_images"].category == CATEGORY_ORGANIZATION
    assert defs["analyze_image"].cost_level == "expensive"
    assert defs["search_images"].cost_level == "cheap"
    assert all(t.enabled for t in defs.values())
    # input_schema 必须有可映射的 properties（含 query/candidates/image 源字段）
    for t in defs.values():
        props = t.input_schema.get("properties") or {}
        assert any(k in props for k in
                   ("query", "candidates", "image_url", "local_image_path", "group_by"))


def test_default_registry_mcp_export_all_tools():
    reg = default_tool_registry()
    names = [t.name for t in reg.list_tools()]
    assert names == ["search_images", "rerank", "analyze_image", "filter_images", "group_images"]
    exported = export_tool_schemas(reg.list_tools())
    assert len(exported) == 5
    for item in exported:
        assert set(item.keys()) == {"name", "description", "inputSchema"}


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
