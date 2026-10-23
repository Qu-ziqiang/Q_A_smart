# PHASE2_TOOL_REGISTRY_IMPLEMENTATION_REPORT

> 项目：SmartImager — 多模态智能图库与对话推荐系统
> 阶段：Phase 2 Upgrade —— MCP-style Tool Registry and Agent Tool Calling
> 日期：2026-09-07
> 代码基准：`Smartlmager-sqlite-vec/backend/ai_func/recommendation/orchestration/`

---

## 1. 概述

Phase 1 完成 Worker → Specialist Agent 角色化后，工具层仍是“Agent 直接依赖 Python 函数”
（工具定义分散、schema 不统一、缺发现与治理）。本阶段引入 **MCP-style Tool Registry**，
把既有 5 个能力统一为 `ToolDefinition`（MCP 可映射元数据）+ 统一调用入口：

```text
之前：  Agent → Python Function（search_images() / rerank() / analyze_image() / ...）
现在：  Agent → ToolRegistry.call(name, arguments)
                → ToolDefinition（校验）
                → implementation（薄适配，复用既有能力）
                → Worker / Repository / ImageAnalysis / Rerank
```

并新增 **ToolCallEvent** 执行轨迹（tool_name / arguments / agent_id / success / latency），
为 Phase 4（LLM-as-a-Judge / Tool Optimization）提供数据基础。

**稳定性红线**：不修改 Planner / DAGExecutor / Task Model / Worker 核心逻辑 / DB / 前端。

---

## 2. 需求覆盖矩阵

| Requirement | 内容 | 状态 | 落地位置 |
|-------------|------|------|----------|
| 1 | ToolDefinition | ✅ | `orchestration/tools/models.py` |
| 2 | ToolRegistry（注册/发现/校验/统计） | ✅ | `orchestration/tools/registry.py` |
| 3 | 注册既有工具（5 个内置） | ✅ | `orchestration/tools/builtin_tools.py` |
| 4 | Agent Tool Binding（AgentCard） | ✅ | `agent_cards.py`（`tools` 属性 + `can_use`） |
| 5 | SpecialistAgent 注入 ToolRegistry | ✅ | `specialist.py`（`registry` 注入 + `call_tool`） |
| 6 | MCP Compatibility Layer | ✅ | `orchestration/tools/mcp_schema.py`（不装 MCP SDK） |
| 7 | Tool Calling Trace（ToolCallEvent） | ✅ | `tools/models.py::ToolCallEvent` + registry 记录 |
| 8 | 新增测试（tools/） | ✅ | `tests/agent_team/tools/` 3 个文件 |
| 9 | Runtime Verification | ✅ | `demo_tool_registry.py`（离线） |
| 10 | 实施报告 | ✅ | 本文件 |

---

## 3. 修改 / 新增文件

**新增模块**
- `orchestration/tools/__init__.py`
- `orchestration/tools/models.py`（ToolDefinition / ToolCallEvent / CATEGORY_*）
- `orchestration/tools/registry.py`（ToolRegistry / validate_arguments）
- `orchestration/tools/mcp_schema.py`（export_tool_schema / export_tool_schemas）
- `orchestration/tools/builtin_tools.py`（5 个内置定义 + 实现 + register_builtin_tools / default_tool_registry）

**新增测试 / Demo**
- `backend/tests/agent_team/tools/test_tool_registry.py`
- `backend/tests/agent_team/tools/test_tool_schema.py`
- `backend/tests/agent_team/tools/test_agent_tool_binding.py`
- `backend/tests/agent_team/demo_tool_registry.py`

**修改（行为向后兼容）**
- `orchestration/agent_cards.py`：`AgentCard.tools`（与 `available_tools` 同义的绑定属性）+ `can_use()`
- `orchestration/specialist.py`：构造函数新增可选 `registry`；新增 `registry` 属性 / `tools` 属性 / `call_tool()`
- `orchestration/agent_registry.py`：`build_default_agent_registry(..., tool_registry=None)` 注入共享 registry
- `orchestration/__init__.py`：导出 tools API

**未改动（约束确认）**：`planner.py` / `executor.py` / `workers.py` 核心 / `models.py` / DB / 前端 / SSE。

---

## 4. Tool Registry 设计

### 4.1 数据流
```text
SpecialistAgent.call_tool(name, arguments)
   │ ① AgentCard.can_use(name) —— Agent Tool Binding 授权检查（未绑定 → PermissionError，不产生调用）
   ▼
ToolRegistry.call(name, arguments, agent_id)
   │ ② 定义存在性 / enabled / 实现存在性
   │ ③ validate_arguments(arguments, input_schema)
   │      required 存在性 + 顶层基础 JSON 类型 + array 元素类型 + enum 成员
   │ ④ implementation(**validated)   ← 薄适配：惰性导入复用既有能力
   ▼
成功/失败 → 记录 ToolCallEvent + 统计（calls/successes/failures/total_latency_ms）
          → 打印 [ToolRegistry:OK|FAIL] agent=... tool=... latency_ms=...
```

### 4.2 注册的既有能力（薄适配，不改 Worker）
| Tool | category | cost_level | implementation 复用来源 |
|------|----------|------------|------------------------|
| search_images | retrieval | cheap | `tool_search_images` → `SearchRepository.unified_search` |
| rerank | retrieval | medium | `rerank_candidates`（R1 确定性重排；R2 见 model_reranker） |
| analyze_image | vision | expensive | `ImageAnalysis().analyze_image`（VisionWorker 使用的 provider 客户端） |
| filter_images | organization | cheap | 复用 `MetadataWorker._field_value / _apply_op`（确定性过滤语义一致） |
| group_images | organization | cheap | 复用 `GroupImagesWorker._group_key`（确定性分组语义一致） |

注：实现一律“函数内惰性导入”，模块 import 零重依赖；测试 / Demo 通过 `overrides`
替换 Fake 实现，不触碰真实 DB / 视觉 API。

### 4.3 治理能力
- 工具注册/发现：`register / get / list_tools(enabled_only) / has`
- 参数校验：`validate_arguments`（轻量，无额外依赖）
- 调用统计与轨迹：`stats(name) / stats_summary() / trace() / reset_trace()`
- 禁用工具：`enabled=False` 时不可调用、可从发现列表过滤，但仍可查询定义
- 幂等注册：同名覆盖（便于测试注入 / 后续动态 Skill 扩展）

---

## 5. MCP 兼容性说明

- `export_tool_schema(tool)` 返回 MCP Tool 核心三元组 `{name, description, inputSchema}`，
  其中 `inputSchema` 即 `ToolDefinition.input_schema`（标准 JSON Schema 对象）；
- `export_tool_schemas(tools)` 批量导出；
- **本阶段不安装 MCP SDK、不启动 MCP Server**；仅保证“ToolDefinition 可 1:1 映射为 MCP
  Tool 格式”。未来 Phase（MCP 外化）可直接基于 registry 自动生成工具列表并接入
  stdio / streamable-http 传输，而无需改动 Agent 层。

---

## 6. Agent Tool 绑定方式

- **声明（Agent Card）**：`RetrievalAgent.available_tools=["search_images","rerank"]`、
  `OrganizerAgent=["filter_images","group_images"]`、`VisionAgent=["analyze_image"]`；
  `AgentCard.tools` 为绑定视图，`can_use(name)` 做授权判定。
- **执行（SpecialistAgent）**：
  - 构造时注入 `registry`（或惰性共享 `default_tool_registry()`）；
  - `call_tool(tool_name, arguments)` 先做 Agent 绑定校验 → `ToolRegistry.call(..., agent_id=...)`；
  - `call_tool` 与 `execute`（DAG Worker 路径）**并存**：Worker 路径保持不变以守住执行稳定性，
    ToolRegistry 作为 Agent 工具的“统一入口 / 授权 / Trace 层”。
- **绑定一致性约束**：默认卡片绑定的工具 ⊆ 默认 registry 已注册工具（测试 `test_default_cards_binding_matches_registry` 强制）。

---

## 7. 测试结果汇总

运行方式（repo 根目录）：`venv\Scripts\python.exe -X utf8 backend/tests/agent_team/...`

**Phase 2 新增测试（tools/）**
| 测试文件 | 用例数 | 结果 |
|----------|-------:|------|
| tools/test_tool_registry.py | 11 | 11 passed |
| tools/test_tool_schema.py | 7 | 7 passed |
| tools/test_agent_tool_binding.py | 7 | 7 passed |

小计：**25 passed / 0 failed**。

**Phase 1 新增测试回归（Agent 层，本阶段改动涉及）**
| 测试文件 | 用例数 | 结果 |
|----------|-------:|------|
| test_agent_cards.py | 6 | 6 passed |
| test_agent_registry.py | 7 | 7 passed |
| test_specialist_agent.py | 10 | 10 passed |

小计：23 passed。

**既有核心回归（Phase 2 改动后复跑）**
| 测试文件 | 用例数 | 结果 |
|----------|-------:|------|
| test_registration.py | 19 | 19 passed |
| test_orchestrator.py | 6 | 6 passed |
| test_executor.py | 8 | 8 passed |
| test_workers.py | 12 | 12 passed |
| test_sse_contract.py | 9 | 9 passed |
| test_dag_propagation.py | 14 | 14 passed |

小计：68 passed。

> Phase 2 改动不触及 planner/validator/synthesizer/vision/gate/intent/legacy 等模块
> （此前基线：planner 11 / plan_validation 20 / synthesizer 12 / vision 49 / gate 13 /
> intent 23 / integration_full_chain 15 / demo 均全绿）。

**本阶段验证合计：25（新增）+ 23（Agent 层回归）+ 68（核心回归）= 116 passed / 0 failed。**

---

## 8. Runtime Verification 证据（Requirement 9）

运行：`venv\Scripts\python.exe -X utf8 backend/tests/agent_team/demo_tool_registry.py`（离线）

```text
== 1) 默认 ToolRegistry 视图 + MCP schema 导出 ==
  search_images   category=retrieval    cost=cheap     enabled=True
  rerank          category=retrieval    cost=medium    enabled=True
  analyze_image   category=vision       cost=expensive enabled=True
  filter_images   category=organization cost=cheap     enabled=True
  group_images    category=organization cost=cheap     enabled=True
  MCP export (search_images): {"name":"search_images","description":"…",
                               "inputSchema_type":"object","required":["query"]}

== 2) Agent → ToolRegistry → Tool 统一调用（Fake 实现，离线）==
[ToolRegistry:OK] agent=retrieval tool=search_images latency_ms=0.0
    retrieval_agent -> search_images => 2 行候选
[ToolRegistry:OK] agent=retrieval tool=rerank        latency_ms=0.0
[ToolRegistry:OK] agent=vision tool=analyze_image    latency_ms=0.0
[ToolRegistry:OK] agent=organizer tool=filter_images latency_ms=0.0
[ToolRegistry:OK] agent=organizer tool=group_images  latency_ms=0.0

  ToolCallEvent Trace（供 LLM Judge / 调优）：
    [ToolRegistry] retrieval -> search_images success=True ...
    [ToolRegistry] retrieval -> rerank ...
    [ToolRegistry] vision -> analyze_image ...
    [ToolRegistry] organizer -> filter_images ...
    [ToolRegistry] organizer -> group_images ...

== 3) 既有执行链稳定路径核对 ==
  executor.workers[retrieve_images] → RetrievalAgent
  executor.workers[filter_metadata] → OrganizerAgent
  executor.workers[group_images]    → OrganizerAgent
  executor.workers[synthesize]      → SynthesizerAgent
```

日志体现了 `[ToolRegistry] retrieval_agent -> search_images` 与
`[ToolRegistry] vision_agent -> analyze_image` 等 Agent→Tool 调用轨迹。

---

## 9. 约束合规性检查

| 约束 | 状态 |
|------|------|
| 不引入 MCP SDK | ✅ 仅 MCP 格式映射（mcp_schema.py），无新依赖 |
| 不修改 Planner | ✅ planner.py / prompts.py 未动 |
| 不修改 DAGExecutor | ✅ executor.py 未动 |
| 不修改数据库 | ✅ 无 DDL / repository 改动 |
| 不修改前端 | ✅ 未触及 frontend/ |
| 不创建新的 LLM Agent | ✅ ToolRegistry 无 LLM 调用 |
| Worker 核心逻辑不变 | ✅ workers.py 未动；工具以薄适配复用既有实现 |

---

## 10. 当前距离“完整 Agent 系统”的差距

**本阶段已具备**
- 统一工具描述（ToolDefinition）+ 注册 / 发现 / 校验 / 计量 / 轨迹（ToolRegistry / ToolCallEvent）；
- MCP 可映射工具层（mcp_schema）；
- Agent Card → 工具绑定与授权（can_use / call_tool）；
- SpecialistAgent 通过注册中心调用工具（日志 + Trace 可见）。

**仍未具备（后续阶段）**
1. **真实 MCP Server / 外部工具**：目前工具全部 in-process；跨进程 / 第三方 MCP 工具需
   Phase（MCP 外化）引入传输层与鉴权（见 SMARTIMAGER_AGENT_UPGRADE_DESIGN.md §5.1）。
2. **执行链强制经 registry**：DAG 内 Worker 直连既有实现是稳定性红线；真正“Worker 统一走
   registry”需要先冻结 Worker 注入位（search_tool/analyzer 等已是注入位），可作为后续优化项。
3. **Supervisor Agent（Phase 3）**：re-plan / revise / abort 决策回路仍缺失；
   AgentMessage 信封与 ToolCallEvent 已为其备好输入。
4. **Tool 动态扩展 / Skill**：ToolRegistry 支持运行时同名覆盖注册，但还没有“按用户/租户启用
   某组工具”的策略层。
5. **LLM-as-a-Judge（Phase 4）**：ToolCallEvent（arguments/success/latency/agent_id）已可作为
   Judge 输入；回答质量 / 达成度评估待建。

---

## 11. 后续阶段建议

1. **Phase 3 — Supervisor Agent**：基于 AgentRegistry + ToolRegistry + AgentMessage，实现
   plan → execute → review → revise/replan/abort 五态小状态机（默认 flag 关闭），
   review 决策可消费 ToolCallEvent 统计（失败工具 / 延迟 / 空结果）。
2. **MCP 外化（按需）**：出现第二个消费方或能力独立部署需求时，用
   `export_tool_schemas(registry.list_tools())` 生成 MCP server 工具列表，加 stdio/http 传输。
3. **工具审计增强**：ToolCallEvent 落库（复用 request_sessions / agent_runs 思路），
   供离线调优与安全审计。
4. **绑定一致性治理**：把“AgentCard.tools ⊆ ToolRegistry 注册工具”约束从测试提升为
   运行时启动自检（Registry 与 AgentRegistry 同源校验），防止 drift。

