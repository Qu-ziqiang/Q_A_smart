"""Phase 3.2 — observability 子包（事件收集器抽象）。

对外 API：
  - EventCollector：抽象（record_tool_event / record_agent_message / snapshot）
  - InMemoryEventCollector：默认进程内实现（供 Supervisor RunState 使用）
"""

from .event_collector import EventCollector, InMemoryEventCollector

__all__ = ["EventCollector", "InMemoryEventCollector"]
