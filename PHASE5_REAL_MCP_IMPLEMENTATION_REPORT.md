# Phase 5 — Real MCP Implementation Report（真实 Model Context Protocol 集成）

> ⚠️ **历史快照（迁移前口径）**：文中的测试门禁数字（448 passed 等）为迁移前口径。当前数据层已是 **PostgreSQL + pgvector（HNSW/cosine）**、图库约 **1000 张 VOC2010**、门禁 **463 passed**（2026-09-10 复测）。MCP 相关结论（stdio ≈1.5s/call vs in-process 3–30ms、未进默认链路）**不受迁移影响，仍然有效**。


> 状态：完成 | 范围：MCP 仅作为外部工具能力层，未改写 Agent Runtime
> 结论：`MCP Client/Server` 链路已验证（stdio 冒烟 + in-process 协议/运行时套件全绿）

---

## 1. Task 1 — 研究：MCP 集成选型

### 1.1 SDK 选型（结论先行）

| 项 | 选择 | 理由 |
|---|---|---|
| MCP Python SDK | `mcp`（official，modelcontextprotocol/python-sdk）`1.29.1` | 官方维护、与 Anthropic Reference Server 兼容；先评估 2.x（API 大改、FastMCP 更名 MCPServer）后**固定 1.x** 以获得稳定 FastMCP + stdio 实现 |
| 传输 | **stdio**（生产入口）+ **mcp.shared.memory**（协议/运行时测试） | stdio = MCP 标准本地进程传输（initialize → JSON-RPC 消息帧）；memory 会话复用同一 `ClientSession` 语义、无 OS 进程，适合确定性 CI |
| 服务器框架 | `FastMCP`（`@mcp.tool()` / `@mcp.resource()`） | 类型签名 → JSON Schema 自动生成（inputSchema 与 MCP Tool 完全一致） |
| 集成方式 | SpecialistAgent → ToolRegistry → **远端 Tool impl（薄包装）** → MCPToolClient → MCP Server | 不改 `SpecialistAgent.call_tool`、不动 `Worker`/`DAGExecutor`/Planner；远端工具与本地工具**同注册表并存**，实现无感知 |
| 依赖变化 | 新引入唯一运行时依赖：`mcp<2` | 其余 anyio/httpx/pydantic 已是既有依赖 |

> 备选（未采用）：`langchain-mcp-adapters`（引入 LangChain 约束，违反约束项）；自研 JSON-RPC 客户端（协议兼容风险高）。**未使用 LangGraph。**

### 1.2 兼容性风险与规避

- **MCP 2.x API 迁移风险**：`mcp.server.fastmcp` 在 2.x 移除 → 固定 `mcp<2`（venv 已装 `1.29.1`）。
- **字段命名差异**：协议结果字段驼峰（`resourceTemplates`），已在 client helper 中显式处理。
- **Windows stdio 多进程稳定性**：多次子进程连接在捕获式跑批下不稳定 → 自动化套件走 in-process memory 会话（仍是真实 JSON-RPC / tools·call），stdio 单独冒烟通过（见 §5 evidence）。
- **既有回归风险**：对 `ToolDefinition` 仅**追加可选字段** `source: str = "local"`（远端标 `"remote"`），MCP 相关代码全部集中在 `tools/mcp_client.py` 与 `backend/mcp_servers/`；运行时默认行为不变。

---

## 2. Before / After 架构

### Before（Phase 2 之后）
```
Agent Runtime
  └ PlannerAgent → PlanValidator → DAGExecutor → SupervisorAgent(FSM)
        ├ AgentRegistry → SpecialistAgent → ToolRegistry ──► BuiltinTools(Local)
        │                                   └ search/rerank/analyze/filter/group
        └ Registry 工具事件 → ToolCallEvent
```
（Phase 2 的 “MCP-style” 仅为本地 ToolDefinition + JSON Schema，**并非 MCP 协议**。）

### After（Phase 5）
```
Agent Runtime（不变）
  └ PlannerAgent → PlanValidator → DAGExecutor → SupervisorAgent
        ├ AgentRegistry → SpecialistAgent → ToolRegistry
        │        ├─ LocalTool  (source="local")            → 内置实现（原逻辑原样）
        │        └─ MCPRemoteTool (source="remote")        → impl 薄包装
        │                └ MCPToolClient / MemoryMCPClient
        │                      └ MCP 协议 (initialize / tools·list / tools·call)
        │                            ├ backend/mcp_servers/retrieval_server（stdio 入口）
        │                            │     search_images / rerank_candidates / resources
        │                            └ backend/mcp_servers/vision_server（stdio 入口）
        │                                  analyze_image
        └ Worker / VisionWorker 注入位、Supervisor review/revise 均复用既有机制
```
目标验收链：
```
User Query → Planner → Validator → DAGExecutor → SpecialistAgent
           → MCP Client → MCP Server → Tool Execution → Supervisor Review → Final Answer
```

---

## 3. MCP 协议集成要点

| 层 | 文件 | 说明 |
|---|---|---|
| Retrieval Server | `backend/mcp_servers/retrieval_server/server.py` | FastMCP + stdio；`search_images(query,top_k)` 内部调用 `tool_search_images`（复用 sqlite-vec，无重复实现）；`rerank_candidates(candidates,strategy)` 复用 `rerank_candidates()`（JSON→ImageCandidate）；resources `image://{id}` / `metadata://{id}` |
| Vision Server | `backend/mcp_servers/vision_server/server.py` | `analyze_image(image_id, task)`，真实路径经 `SearchRepository` 定位文件后调用 `ImageAnalysis`；未修改 VisionWorker |
| Client Adapter | `orchestration/tools/mcp_client.py` | `MCPToolClient`（stdio，单次连接 per 操作）、`MemoryMCPClient`（memory 会话）、`discover()→ToolDefinition`、`call_tool()` 解析/失败聚合、trace（ToolCallEvent 风格）、`register_remote_tools()`（按远端 inputSchema 过滤/映射参数并解包 `rows|candidates`） |
| 向后兼容 | 仅 `ToolDefinition` 增 `source` 可选字段 | 本地工具 + MCP 远端工具同 `ToolRegistry` 并存；`SpecialistAgent.call_tool()` 原入口不变 |

### 运行方式（stdio 生产入口）
```
venv\Scripts\python.exe -m backend.mcp_servers.retrieval_server.server   # stdio MCP
venv\Scripts\python.exe -m backend.mcp_servers.vision_server.server
```
测试夹具：`SMARTIMAGER_MCP_RETRIEVAL_SOURCE=fake`、`SMARTIMAGER_MCP_VISION_SOURCE=fake`（离线确定性；
fake 还提供故障注入：query 含 `fail` 抛错、含 `recover` 且 top_k<30 抛错、image_id=`fail` 抛错）。

---

## 4. MCP 工具清单

| Server | Tool | Input (JSON Schema) | 复用实现 | 返回 |
|---|---|---|---|---|
| retrieval | `search_images` | `query:string` `top_k:integer=20` | `tool_search_images` | `{ok,count,rows[]}`（client 解包 rows） |
| retrieval | `rerank_candidates` | `candidates:array` `strategy:string` | `rerank_candidates` | `{ok,count,candidates[]}` |
| vision | `analyze_image` | `image_id:string` `task:string` | `SearchRepository` + `ImageAnalysis` | `{ok,tool,image_id,task,title,…}` |
| retrieval | resources `image://{image_id}` | — | 检索库行记录 | 图片元数据 JSON |
| retrieval | resources `metadata://{image_id}` | — | 检索库行记录 | 元数据 JSON |

<!-- SPLIT2 -->

---

## 5. Runtime Traces（Evidence / 验收日志）

### 5.1 stdio 冒烟（真实子进程，单次连接）
```
discover tools: ['search_images', 'rerank_candidates']
schema search_images keys: ['properties', 'required', 'title', 'type']
call result ok=True count=2 ids=[901, 902]
discover_ms=1509 call_ms=1566
failure handled: MCPToolError Error executing tool search_images: retrieval tool down (fake)
```

### 5.2 自动化运行（3 个场景全绿；详细 trace 落盘）
`backend/tests/agent_team/mcp/mcp_runtime_traces.log`（节选）

**场景 A — image retrieval（DAG→Specialist→MCP search）**
```
SCENARIO A: query=城市建筑
  agent.review attempt=1 decision=continue reason="执行结果满足要求"
  selection ids=[901, 902]
  MCP client: retrieval-client search_images success=true latency_ms=8.825
  registry: tool_call search_images success=True
```

**场景 B — retrieval + vision（search_images + analyze_image ×2 走远端）**
```
  agent.review decision=continue
  MCP client: retrieval-client search_images  success latency_ms=5.5
             vision-client  analyze_image image_id=901 success latency_ms=4.7
             vision-client  analyze_image image_id=902 success latency_ms=5.5
  registry: tool_call analyze_image success=True (×2)
```

**场景 C — tool failure recovery（Supervisor REVISE → retry）**
```
  agent.review attempt=1 decision=revise reason="检测到失败/空检索任务: failed=['t1'] empty=[]"
  MCP client: search_images top_k=20 success=false error="retrieval transient failure (fake, top_k<30)"
  agent.review attempt=2 decision=continue
  MCP client: search_images top_k=30 success=true
  selection ids=[903]
```

### 5.3 验证套件
| 套件 | 结果 |
|---|---|
| `backend/tests/agent_team/mcp/test_mcp_servers.py`（discovery/schema/call/failure/resources/registry 并存） | 7 passed |
| `backend/tests/agent_team/mcp/test_mcp_runtime.py`（场景 A/B/C） | 3 passed |
| 既有回归 `backend/tests/agent_team/`（pytest，含 30 项 tools 套件） | 448 passed |

---

## 6. Performance Overhead

| 路径 | 单次工具调用开销（本机 Windows 实测） | 说明 |
|---|---|---|
| 内置 LocalTool（基线） | ~µs–ms | 无进程/协议开销 |
| MCP memory 会话 | ~3–30 ms（registry 记录 2.7–28 ms） | 协议 JSON-RPC + 序列化 |
| MCP stdio 子进程 | ~1.5 s（discover 1509 ms / call 1566 ms，含 FastMCP/backend 导入与进程拉起） | 主要成本是**进程冷启动**，非协议本身 |

结论：作为**外部工具能力层**，stdio 单次调用 1.5 s 级开销可接受（本质等价“调一次外部服务”）；若部署在常驻服务（streamable HTTP/SSE）或复用长会话，可降到 memory 量级。当前为保持零运行时状态，客户端每次调用新建连接——已记入文档。

---

## 7. Limitations（局限）

- **自动测试走 memory 会话**：Windows 下 stdio 多进程连接在捕获式批跑不稳定，自动化套件用 `mcp.shared.memory`（协议一致）；stdio 已单独冒烟验证。
- **远端工具参数为受控白名单**：worker 本地扩展字段（`vector_targets`/`filters` 等）在 client 侧按远端 inputSchema 过滤，`limit`→`top_k` 做映射；多 schema 差异适配需后续扩展。
- **每次调用独立连接**：无连接池 / 长会话复用；面向频繁调用需做持久化 stdio/HTTP 客户端（后续可做）。
- **Server 进程生命周期**：stdio 服务器为单工具进程（FastMCP）；需进程管理器拉起（systemd/supervisor），未捆绑到运行时。
- **fake 模式范围**：离线 fixture 仅覆盖确定性输入；真实路径依赖既有 DB/模型环境。
- **Resources 为最小实现**：暴露 `image://{id}` / `metadata://{id}` 元数据读取，未做订阅/变化通知（MCP 订阅语义未用）。

---

## 8. 交付物对照

| # | 要求 | 状态 |
|---|---|---|
| Task 1 | MCP 选型研究（SDK/传输/集成/风险） | ✅ §1 |
| Task 2 | `backend/mcp_servers/retrieval_server/`（search_images / rerank_candidates 复用既有实现） | ✅ |
| Task 3 | `backend/mcp_servers/vision_server/`（analyze_image，不改 VisionWorker） | ✅ |
| Task 4 | `orchestration/tools/mcp_client.py`（连接/发现/转 ToolDefinition/调用/事件 trace） | ✅ |
| Task 5 | 向后兼容：LocalTool + MCPRemoteTool 同注册表、`call_tool()` 不变 | ✅（测试断言 local+remote 并存） |
| Task 6 | MCP Resources（可选） | ✅ 最小实现 + 测试 |
| Task 7 | Server/Client/Runtime 测试 + A/B/C 场景 | ✅ 7+3 passed + 回归 448 passed |
| 报告 | 本文件 | ✅ |

**变更文件**
新增：`backend/mcp_servers/{__init__,retrieval_server,vision_server}/*`、`orchestration/tools/mcp_client.py`、
`backend/tests/agent_team/mcp/test_mcp_servers.py`、`backend/tests/agent_team/mcp/test_mcp_runtime.py`、
`backend/tests/agent_team/mcp/mcp_runtime_traces.log`。
修改：`orchestration/tools/models.py`（+`source`）、`orchestration/tools/__init__.py`（导出 MCP client 符号）。
依赖：`mcp<2`（1.29.1）。
