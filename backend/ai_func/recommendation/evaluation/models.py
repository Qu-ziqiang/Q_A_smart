"""Phase 4.1 — Evaluation Foundation 数据模型。

- EvaluationCase：一条离线测试用例的“预期规格”（纯声明，不含夹具/实现）。
- EvaluationTrace：一次 runtime 执行的结构化轨迹（由 Supervisor RunState +
  EventCollector 派生），供确定性断言与后续 Phase 4 LLM-as-a-Judge 消费。
- EvaluationSummary：一组用例的确定性通过/失败汇总。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCase(BaseModel):
    """离线评估用例（输入规格 + 确定性预期）。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    name: str = ""
    query: str
    goal: Optional[str] = None          # 显式 goal（缺省用 query）
    messages: List[Dict[str, str]] = Field(default_factory=list)  # 可选前置历史
    # 确定性预期（无 LLM Judge；Phase 4.2 再叠加评分）
    expected_agents: List[str] = Field(default_factory=list)      # 应执行的 Specialist
    expected_tools: List[str] = Field(default_factory=list)       # 应调用的工具
    expected_decisions: Optional[List[str]] = None                # REVIEW decisions 序列
    expected_final_state: str = "done"
    expect_selection: bool = True
    tags: List[str] = Field(default_factory=list)


class EvaluationTrace(BaseModel):
    """一次执行的评估轨迹（源自 RunState / EventCollector / SSE selection）。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    request_id: str
    query: str = ""
    goal: Optional[str] = None             # Phase 4.3：目标（来自 EvaluationCase.goal）
    duration_ms: float = 0.0
    plan_ok: bool = False
    final_state: Optional[str] = None
    error: Optional[str] = None

    plan_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    execution_status: Dict[str, str] = Field(default_factory=dict)
    supervisor_states: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)

    agent_messages: List[Dict[str, Any]] = Field(default_factory=list)
    tool_events: List[Dict[str, Any]] = Field(default_factory=list)
    tool_summary: Dict[str, Any] = Field(default_factory=dict)

    selection_ids: List[int] = Field(default_factory=list)

    metrics: Dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    """用例集合的确定性汇总（pass = 无预期违例）。"""

    total: int = 0
    passed: int = 0
    failed: int = 0
    results: List[Dict[str, Any]] = Field(default_factory=list)
