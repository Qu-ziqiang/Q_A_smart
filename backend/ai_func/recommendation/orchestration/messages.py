"""Phase 1 Specialist — Agent Message Protocol（信封雏形）。

为后续 Supervisor ↔ Specialist Agent 通信准备的基础消息结构。

注意：本阶段不实现真实消息通信，仅冻结消息信封契约。
后续 Phase 3（Supervisor Agent）将基于该信封承载：
  - task（Supervisor 下发指令，payload = Typed Task / TaskPlan）
  - report（Specialist 回传结果，payload = WorkerResult / AgentReport）
  - decision（Supervisor 复盘决策，payload = SupervisorDecision）

与既有 SSE 事件的关系：AgentMessage 是“系统内/进程内”信封；
SSE 事件（plan.created / task.* / assistant_delta / selection）是它的
对外可观测投影（Event Bus），两者字段语义保持一致、只增不改。
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class AgentMessage(BaseModel):
    """Agent 间基础消息信封（纯数据，本阶段不路由、不投递）。"""

    model_config = ConfigDict(extra="forbid")

    message_id: str
    sender: str          # "user" / "supervisor" / "retrieval" / "organizer" / "vision" / "synthesizer"
    receiver: str        # 目标 Agent id；"*" 表示广播（预留给未来 Event Bus）
    message_type: str    # "task" / "report" / "decision" / "event"（Phase 3 扩展）
    payload: Dict[str, Any] = Field(default_factory=dict)


# 预定义 message_type（约定词汇表，后续阶段扩展）
TYPE_TASK = "task"
TYPE_REPORT = "report"
TYPE_DECISION = "decision"
TYPE_EVENT = "event"
