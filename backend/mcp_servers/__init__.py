"""SmartImager — MCP Servers（外部工具能力层，不触碰 Agent Runtime）。

通过 stdio + MCP JSON-RPC 暴露真实 MCP 工具：
  - backend/mcp_servers/retrieval_server：search_images / rerank_candidates
  - backend/mcp_servers/vision_server：analyze_image
"""
