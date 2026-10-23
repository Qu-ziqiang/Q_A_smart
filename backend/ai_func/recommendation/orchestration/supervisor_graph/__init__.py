"""SmartImager — Supervisor LangGraph backend（Phase 6）。

包结构：
  - state.py    AgentRuntimeState（LangGraph StateGraph 的共享状态）
  - graph.py    StateGraph 构建 + 四个节点 + review_router
  - runtime.py  LangGraphSupervisorRuntime（SSE 事件兼容的流式封装）

LangGraph 只替换 Supervisor 的状态机执行；Planner / DAGExecutor / Specialist /
Worker / ToolRegistry / MCP / Evaluation 均不替换。
"""
