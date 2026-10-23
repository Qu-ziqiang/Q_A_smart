"""Phase 2 MCP-style Tool — Agent Tool Binding 单元测试。

覆盖：
  - AgentCard.tools 绑定（与 available_tools 同义）+ can_use
  - 默认三张卡片的绑定 ⊆ ToolRegistry 已注册工具
  - SpecialistAgent 注入 ToolRegistry；call_tool 走统一入口并记录 agent_id
  - 未绑定工具 → PermissionError
  - organizer 多工具绑定（filter_images / group_images）
  - 调用失败也留下 Trace（供 LLM Judge / 调优）

零真实 DB / API（Fake 实现）。运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/tools/test_agent_tool_binding.py
"""

import contextlib
import io
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    AgentCard,
    AgentRegistry,
    SpecialistAgent,
    TaskType,
    ToolRegistry,
    build_default_agent_cards,
    default_tool_registry,
    register_builtin_tools,
)
from backend.ai_func.recommendation.orchestration.models import TaskType as TT


class _FW:
    """占位 Worker：绑定测试不触发 Worker，仅满足构造约束。"""

    def __init__(self, ttype):
        self.task_type = ttype

    def run(self, task, context):
        raise AssertionError("binding 测试不应执行 Worker")


def _registry_with_fakes():
    reg = ToolRegistry()
    register_builtin_tools(reg, overrides={
        "search_images": lambda query, **kw: [{"id": 1, "query": query}],
        "analyze_image": lambda **kw: {"title": "fake", "description": "fake"},
    })
    return reg


def test_card_tools_property_aliases_available_tools():
    card = AgentCard(agent_id="retrieval", name="RetrievalAgent", description="r",
                     supported_tasks=[TaskType.RETRIEVE_IMAGES],
                     available_tools=["search_images", "rerank"])
    assert card.tools == ["search_images", "rerank"]
    assert card.can_use("search_images") is True
    assert card.can_use("analyze_image") is False


def test_default_cards_binding_matches_registry():
    cards = {c.agent_id: c for c in build_default_agent_cards(vision_ok=True)}
    registry = default_tool_registry()
    available = {t.name for t in registry.list_tools()}
    expected = {
        "retrieval": ["search_images", "rerank"],
        "organizer": ["filter_images", "group_images"],
        "vision": ["analyze_image"],
    }
    for agent_id, tools in expected.items():
        assert set(cards[agent_id].tools) == set(tools)
        # 绑定必须能被 registry 发现（Tool Binding 一致性）
        assert set(tools) <= available


def test_specialist_call_tool_through_registry_logs_and_traces():
    reg = _registry_with_fakes()
    agent = SpecialistAgent(
        AgentCard(agent_id="retrieval", name="RetrievalAgent", description="r",
                  supported_tasks=[TaskType.RETRIEVE_IMAGES],
                  available_tools=["search_images", "rerank"]),
        _FW(TaskType.RETRIEVE_IMAGES),
        registry=reg,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.call_tool("search_images", {"query": "旅行照片", "limit": 5})
    assert result == [{"id": 1, "query": "旅行照片"}]
    text = buf.getvalue()
    assert "[ToolRegistry:OK]" in text
    assert "agent=retrieval" in text and "tool=search_images" in text
    assert len(reg.trace()) == 1
    ev = reg.trace()[0]
    assert ev.agent_id == "retrieval"
    assert ev.tool_name == "search_images"
    assert ev.success is True


def test_specialist_denies_unbound_tool():
    reg = _registry_with_fakes()
    agent = SpecialistAgent(
        AgentCard(agent_id="retrieval", name="RetrievalAgent", description="r",
                  supported_tasks=[TaskType.RETRIEVE_IMAGES],
                  available_tools=["search_images", "rerank"]),
        _FW(TaskType.RETRIEVE_IMAGES),
        registry=reg,
    )
    try:
        agent.call_tool("analyze_image", {"image_url": "http://x"})
    except PermissionError:
        pass
    else:
        raise AssertionError("未绑定工具应抛 PermissionError")
    assert reg.trace() == []  # 绑定校验发生在 registry 之前，不产生调用


def test_organizer_multi_tool_binding():
    reg = _registry_with_fakes()
    organizer = SpecialistAgent(
        AgentCard(agent_id="organizer", name="OrganizerAgent", description="o",
                  supported_tasks=[TaskType.FILTER_METADATA, TaskType.GROUP_IMAGES],
                  available_tools=["filter_images", "group_images"]),
        {TaskType.FILTER_METADATA: _FW(TaskType.FILTER_METADATA),
         TaskType.GROUP_IMAGES: _FW(TaskType.GROUP_IMAGES)},
        registry=reg,
    )
    rows = [{"id": 1, "metadata": {"location": "北京"}}, {"id": 2, "metadata": {"location": "上海"}}]
    filtered = organizer.call_tool("filter_images", {
        "candidates": rows,
        "predicates": [{"field": "location", "operator": "eq", "value": "北京"}],
    })
    assert filtered["kept_ids"] == [1]
    grouped = organizer.call_tool("group_images", {
        "candidates": rows, "group_by": "location",
    })
    assert [g["key"] for g in grouped["groups"]] == ["北京", "上海"]
    tool_names = {ev.tool_name for ev in reg.trace()}
    assert tool_names == {"filter_images", "group_images"}


def test_call_tool_failure_records_trace_for_future_judge():
    reg = ToolRegistry()
    register_builtin_tools(reg, overrides={
        "search_images": lambda query, **kw: (_ for _ in ()).throw(RuntimeError("db down")),
    })
    agent = SpecialistAgent(
        AgentCard(agent_id="retrieval", name="RetrievalAgent", description="r",
                  supported_tasks=[TaskType.RETRIEVE_IMAGES],
                  available_tools=["search_images"]),
        _FW(TaskType.RETRIEVE_IMAGES),
        registry=reg,
    )
    try:
        agent.call_tool("search_images", {"query": "x"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("实现失败应 re-raise")
    ev = reg.trace()[-1]
    assert ev.agent_id == "retrieval"
    assert ev.success is False
    assert "db down" in (ev.error or "")


def test_agents_in_registry_share_tool_registry():
    reg = _registry_with_fakes()
    workers = {
        TT.RETRIEVE_IMAGES: _FW(TT.RETRIEVE_IMAGES),
        TT.FILTER_METADATA: _FW(TT.FILTER_METADATA),
        TT.GROUP_IMAGES: _FW(TT.GROUP_IMAGES),
        TT.ANALYZE_VISUAL: _FW(TT.ANALYZE_VISUAL),
    }
    agent_reg = AgentRegistry()
    agent_reg.register_agent(SpecialistAgent(
        AgentCard(agent_id="retrieval", name="RetrievalAgent", description="r",
                  supported_tasks=[TT.RETRIEVE_IMAGES], available_tools=["search_images", "rerank"]),
        workers[TT.RETRIEVE_IMAGES], registry=reg,
    ))
    agent_reg.register_agent(SpecialistAgent(
        AgentCard(agent_id="vision", name="VisionAgent", description="v",
                  supported_tasks=[TT.ANALYZE_VISUAL], available_tools=["analyze_image"]),
        workers[TT.ANALYZE_VISUAL], registry=reg,
    ))
    retrieval = agent_reg.get_agent_for_task(TT.RETRIEVE_IMAGES)
    vision = agent_reg.get_agent_for_task(TT.ANALYZE_VISUAL)
    assert retrieval.registry is reg and vision.registry is reg  # 共享同一 registry


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

