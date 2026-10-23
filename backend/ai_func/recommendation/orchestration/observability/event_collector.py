"""Phase 3.2 — Event Collector（可观测性收口，为 Phase 4 Judge 预留接口）。

作用：把 Agent Runtime 的事件（工具调用、Agent 消息）从“随请求消失”的 RunState 列表
升级为“可注入收集器”的抽象；本阶段仅实现 InMemory 默认实现，不落库。

未来（Phase 4）：
  SQLiteEventCollector（复用既有表/连接池）→ Judge Pipeline
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class EventCollector:
    """事件收集器抽象（默认实现为空操作）。"""

    def record_tool_event(self, event: Any) -> None:
        """记录一次工具调用事件（ToolCallEvent）。"""

    def record_agent_message(self, message: Any) -> None:
        """记录一条 Agent 通信消息（AgentMessage）。"""

    def snapshot(self) -> Dict[str, Any]:
        """导出供评估/Judge 消费的统计快照（默认空）。"""
        return {}


class InMemoryEventCollector(EventCollector):
    """进程内收集器：事件存活于当前请求生命周期，可 snapshot() 供上层使用。"""

    def __init__(self) -> None:
        self._tool_events: List[Any] = []
        self._agent_messages: List[Any] = []

    def record_tool_event(self, event: Any) -> None:
        self._tool_events.append(event)

    def record_agent_message(self, message: Any) -> None:
        self._agent_messages.append(message)

    # -- 只读视图 ------------------------------------------------------------
    @property
    def tool_events(self) -> List[Any]:
        return list(self._tool_events)

    @property
    def agent_messages(self) -> List[Any]:
        return list(self._agent_messages)

    def reset(self) -> None:
        self._tool_events.clear()
        self._agent_messages.clear()

    def snapshot(self) -> Dict[str, Any]:
        """面向 Judge 的轻量统计（不含完整 arguments/候选，防止大对象泄漏）。"""
        tool_failures = [e for e in self._tool_events if not getattr(e, "success", False)]
        return {
            "agent_message_count": len(self._agent_messages),
            "tool_event_count": len(self._tool_events),
            "tool_success_count": len(self._tool_events) - len(tool_failures),
            "tool_failure_count": len(tool_failures),
            "tool_names": [getattr(e, "tool_name", None) for e in self._tool_events],
            "tool_latency_ms": [getattr(e, "latency_ms", None) for e in self._tool_events],
            "tool_failures": [
                {"tool_name": getattr(e, "tool_name", None),
                 "agent_id": getattr(e, "agent_id", None),
                 "error": getattr(e, "error", None)} for e in tool_failures
            ],
            "agent_senders": [getattr(m, "sender", None) for m in self._agent_messages],
            "agent_receivers": [getattr(m, "receiver", None) for m in self._agent_messages],
        }
