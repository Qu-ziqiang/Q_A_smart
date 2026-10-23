"""Phase 2 MCP-style Tool — ToolRegistry（注册 / 发现 / 校验 / 调用 / 统计）。

职责：
  - 工具注册：register(tool_definition, implementation)
  - 工具发现：get / list_tools
  - 参数校验：基于 ToolDefinition.input_schema（required + 基础 JSON 类型 + enum）
  - 调用统计：ToolCallEvent 轨迹 + 每工具调用计数 / 成功失败 / 总延迟
  - 统一入口：call(name, arguments, agent_id) —— 禁止 Agent 直接调用 Python 函数

设计要点：
  - ToolDefinition 是纯元数据；实现（Callable）由 Registry 单独持有并按键隔离；
  - call() 只接受 dict 参数（MCP 风格），内部按 input_schema 校验后展开调用；
  - 任何调用都会先记录 Trace（失败也记录并 re-raise），供 LLM Judge / 调优使用。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .models import ToolCallEvent, ToolDefinition

# JSON Schema type → Python 类型映射（仅基础类型校验，不做重量级 JSON Schema 引擎）
_JSON_TYPE_CHECK: Dict[str, tuple] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _is_missing(v: Any) -> bool:
    return v is None


def validate_arguments(arguments: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """按 JSON Schema（受限子集）校验参数；返回原参数字典。

    覆盖：required 存在性 + 顶层属性基础类型 + enum 成员 + array 元素基础类型。
    额外键不拒绝（保持宽容，具体工具/Worker 内部仍有白名单等既有防线）。
    """
    args = dict(arguments or {})
    if not isinstance(args, dict):
        raise TypeError(f"arguments 必须是 dict，实际 {type(args).__name__}")

    props = (schema or {}).get("properties") or {}
    required = (schema or {}).get("required") or []
    for key in required:
        if key not in args or _is_missing(args[key]):
            raise ValueError(f"缺少必填参数: {key!r}")

    for key, value in args.items():
        spec = props.get(key) or {}
        jtype = spec.get("type")
        if jtype and jtype in _JSON_TYPE_CHECK:
            if not isinstance(value, _JSON_TYPE_CHECK[jtype]):
                raise TypeError(
                    f"参数 {key!r} 类型错误: 期望 {jtype}，实际 {type(value).__name__}"
                )
        if isinstance(value, list):
            items_type = (spec.get("items") or {}).get("type")
            if items_type and items_type in _JSON_TYPE_CHECK:
                for item in value:
                    if not isinstance(item, _JSON_TYPE_CHECK[items_type]):
                        raise TypeError(
                            f"参数 {key!r} 的数组元素必须为 {items_type}，实际 {type(item).__name__}"
                        )
        enum = spec.get("enum")
        if enum is not None and value not in enum:
            raise ValueError(f"参数 {key!r} 取值 {value!r} 不在允许集合 {enum} 内")
    return args


class ToolRegistry:
    """MCP-style 工具注册中心（进程内，无共享可变全局实现）。"""

    def __init__(self) -> None:
        self._definitions: Dict[str, ToolDefinition] = {}
        self._implementations: Dict[str, Callable] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._trace: List[ToolCallEvent] = []

    # -- 注册 / 发现 --------------------------------------------------------
    def register(self, tool_definition: ToolDefinition, implementation: Optional[Callable] = None):
        """注册工具定义；同名注册视为覆盖（幂等，便于测试注入 Fake 实现）。"""
        if not isinstance(tool_definition, ToolDefinition):
            raise TypeError("register 只接受 ToolDefinition 实例")
        name = tool_definition.name
        self._definitions[name] = tool_definition
        if implementation is not None:
            self._implementations[name] = implementation
        self._stats.setdefault(name, {
            "calls": 0, "successes": 0, "failures": 0, "total_latency_ms": 0.0,
        })
        return tool_definition

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._definitions.get(name)

    def list_tools(self, enabled_only: bool = True) -> List[ToolDefinition]:
        tools = list(self._definitions.values())
        if enabled_only:
            return [t for t in tools if t.enabled]
        return tools

    def has(self, name: str) -> bool:
        return name in self._definitions

    # -- 调用 ----------------------------------------------------------------
    def call(self, name: str, arguments: Dict[str, Any], agent_id: Optional[str] = None) -> Any:
        """统一调用入口：校验 → 执行 → 记录 ToolCallEvent / 统计 → 返回结果。

        执行失败：记录 success=False 事件后 re-raise（调用方决定降级策略）。
        """
        definition = self._definitions.get(name)
        if definition is None:
            raise KeyError(f"未注册的工具: {name!r}")
        if not definition.enabled:
            raise RuntimeError(f"工具已被禁用: {name!r}")
        implementation = self._implementations.get(name)
        if implementation is None:
            raise RuntimeError(f"工具 {name!r} 缺少实现（仅注册了定义）")

        arguments = dict(arguments or {})
        started = time.perf_counter()
        try:
            validated = validate_arguments(arguments, definition.input_schema)
            result = implementation(**validated)
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record(name, arguments, agent_id, True, latency_ms, None)
            return result
        except Exception as e:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record(name, arguments, agent_id, False, latency_ms, str(e)[:200])
            raise

    def _record(self, name, arguments, agent_id, success, latency_ms, error) -> None:
        event = ToolCallEvent(
            tool_name=name, arguments=arguments, agent_id=agent_id,
            success=bool(success), latency_ms=latency_ms, error=error,
        )
        self._trace.append(event)
        stat = self._stats.setdefault(name, {
            "calls": 0, "successes": 0, "failures": 0, "total_latency_ms": 0.0,
        })
        stat["calls"] += 1
        stat["total_latency_ms"] += latency_ms
        if success:
            stat["successes"] += 1
        else:
            stat["failures"] += 1
        tag = "OK" if success else "FAIL"
        print(f"[ToolRegistry:{tag}] agent={agent_id or 'system'} tool={name} "
              f"latency_ms={latency_ms:.1f}")

    # -- 统计 / 轨迹 --------------------------------------------------------
    def stats(self, name: str) -> Dict[str, Any]:
        return dict(self._stats.get(name, {}))

    def stats_summary(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self._stats.items()}

    def trace(self) -> List[ToolCallEvent]:
        return list(self._trace)

    def reset_trace(self) -> None:
        self._trace.clear()

    def __len__(self) -> int:
        return len(self._definitions)

