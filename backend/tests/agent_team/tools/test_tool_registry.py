"""Phase 2 MCP-style Tool — ToolRegistry 单元测试。

覆盖：
  - register / get / list_tools / has
  - 同名覆盖（幂等、可注入 Fake 实现）
  - call 成功：校验 + 实现调用 + 统计 + ToolCallEvent 轨迹（含 agent_id / latency / success）
  - call 失败：缺少必填参数 / 类型错误 / 实现抛错 → success=False 轨迹并 re-raise
  - disabled 工具不可调用 / enabled_only 过滤
  - 未注册 / 缺少实现 的错误分支

零真实 DB / API（Fake 实现）。运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/tools/test_tool_registry.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import ToolDefinition, ToolRegistry


def _schema(name="x"):
    return {
        "type": "object",
        "properties": {"value": {"type": "integer"}, "text": {"type": "string"}},
        "required": [name],
        "additionalProperties": False,
    }


def _reg_with(fake=None, enabled=True):
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="echo", description="echo tool",
                       input_schema=_schema("value"), output_schema="int",
                       category="test", cost_level="cheap", enabled=enabled),
        implementation=fake or (lambda value, **_: value),
    )
    return reg


def test_register_get_list_has():
    reg = ToolRegistry()
    definition = ToolDefinition(name="echo", description="d", input_schema=_schema("value"))
    reg.register(definition, implementation=lambda value: value)
    assert reg.get("echo") is definition
    assert reg.has("echo") is True
    assert reg.has("nope") is False
    names = [t.name for t in reg.list_tools()]
    assert names == ["echo"]
    assert len(reg) == 1


def test_register_rejects_non_definition():
    reg = ToolRegistry()
    try:
        reg.register("echo")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("register 应拒绝非 ToolDefinition")


def test_duplicate_register_overrides_definition():
    reg = ToolRegistry()
    a = ToolDefinition(name="echo", description="a", input_schema=_schema("value"))
    b = ToolDefinition(name="echo", description="b", input_schema=_schema("value"))
    reg.register(a, implementation=lambda value: 1)
    reg.register(b, implementation=lambda value: 2)
    assert reg.get("echo") is b
    assert reg.call("echo", {"value": 0}) == 2


def test_call_success_records_stats_and_trace():
    reg = _reg_with(fake=lambda value: value * 2)
    result = reg.call("echo", {"value": 21}, agent_id="retrieval")
    assert result == 42
    stat = reg.stats("echo")
    assert stat["calls"] == 1 and stat["successes"] == 1 and stat["failures"] == 0
    events = reg.trace()
    assert len(events) == 1
    ev = events[0]
    assert ev.tool_name == "echo"
    assert ev.agent_id == "retrieval"
    assert ev.success is True
    assert ev.error is None
    assert ev.latency_ms is not None and ev.latency_ms >= 0
    assert ev.arguments == {"value": 21}


def test_call_validation_missing_required_raises_and_traces_failure():
    reg = _reg_with(fake=lambda value: value)
    try:
        reg.call("echo", {})
    except ValueError:
        pass
    else:
        raise AssertionError("缺少必填参数应抛 ValueError")
    events = reg.trace()
    assert len(events) == 1
    assert events[0].success is False
    assert events[0].error is not None
    assert reg.stats("echo")["failures"] == 1


def test_call_validation_wrong_type_raises():
    reg = _reg_with(fake=lambda value: value)
    try:
        reg.call("echo", {"value": "not-an-int"})
    except TypeError:
        pass
    else:
        raise AssertionError("类型错误应抛 TypeError")


def test_call_implementation_error_records_failure_and_reraises():
    def boom(**kwargs):
        raise RuntimeError("impl boom")

    reg = _reg_with(fake=boom)
    try:
        reg.call("echo", {"value": 1})
    except RuntimeError:
        pass
    else:
        raise AssertionError("实现异常应 re-raise")
    ev = reg.trace()[-1]
    assert ev.success is False
    assert "impl boom" in (ev.error or "")
    assert reg.stats("echo")["failures"] == 1


def test_disabled_tool_not_callable_and_filtered():
    reg = _reg_with(fake=lambda value: value, enabled=False)
    assert reg.get("echo") is not None             # 定义仍可查询
    assert reg.list_tools(enabled_only=True) == []  # enabled 过滤
    assert len(reg.list_tools(enabled_only=False)) == 1
    try:
        reg.call("echo", {"value": 1})
    except RuntimeError:
        pass
    else:
        raise AssertionError("disabled 工具调用应抛 RuntimeError")


def test_call_unknown_tool_raises_keyerror():
    reg = _reg_with(fake=lambda value: value)
    try:
        reg.call("missing", {"value": 1})
    except KeyError:
        pass
    else:
        raise AssertionError("未注册工具应抛 KeyError")


def test_call_missing_implementation_raises():
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="echo", description="d", input_schema=_schema("value")))
    try:
        reg.call("echo", {"value": 1})
    except RuntimeError:
        pass
    else:
        raise AssertionError("缺少实现应抛 RuntimeError")


def test_stats_summary_and_reset_trace():
    reg = _reg_with(fake=lambda value: value)
    reg.call("echo", {"value": 1}, agent_id="organizer")
    summary = reg.stats_summary()
    assert summary["echo"]["calls"] == 1
    reg.reset_trace()
    assert reg.trace() == []


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

