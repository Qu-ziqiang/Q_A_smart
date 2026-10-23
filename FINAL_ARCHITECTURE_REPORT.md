# FINAL ARCHITECTURE REPORT — SmartImager Agent Team

> 范围：Phase 1–5 全部架构审计（Multi-Agent → ToolRegistry → Supervisor → LLM-as-a-Judge Evaluation → Real MCP）。
> 原则：停止功能开发；仅审计既有实现，无任何生产代码修改。
> 验证基准：`pytest backend/tests/agent_team -q` → **463 passed**（2026-09-10 复测；PG 迁移前终验为 448 passed）。

---

## 1. 系统全貌（Component Map）

SmartImager = 图片检索 + 视觉核验 + 结构化推荐的 Agent 后端，最终形态为五层：

```
[API/SSE]  FastAPI → SSE JSON 事件流（对外契约，Phase 5 不改动）
        │ user / assistant_delta / selection
[Agent Team]  ComplexityGate(legacy|team) → PlannerAgent → PlanValidator
        │     SupervisorAgent(FSM) ── DAGExecutor(串行有界 DAG)
[Specialist]  AgentRegistry → SpecialistAgent = AgentCard + Adapter + Worker
        │     retrieval / organizer / vision（三角色）
[Tool]       ToolRegistry（ToolDefinition + ToolCallEvent）
        │     ├ LocalTool：builtin 5 个（search_images/rerank/analyze_image/filter_images/group_images）
        │     └ MCPRemoteTool：MCPToolClient → MCP Server（stdio / in-process 两种会话）
[Capability] SearchRepository(PostgreSQL + pgvector, HNSW/cosine) / ImageAnalysis(GLM-Vision)
        │     rerank_candidates(R1) / model_reranker(R2) / Memory / Intent
```

横切观测与离线评估：`EventCollector`（agent 消息 + 工具事件）、`RunState`（request 级执行轨迹）、
`evaluation/`（`EvaluationTrace` → 确定性断言 → LLM-as-a-Judge 评分，LLMJudge/MockJudge 同一接口）。


## 2. 运行时序图（Runtime Sequence）

成功路径（Supervisor REVIEW→DONE，事件与 SSE 契约一致）：

```
User → PlannerAgent.create_plan() → 草稿 → PlanValidator(类型化+语义+上限)
     → TaskPlan 通过
     → SupervisorAgent(EXECUTE) → DAGExecutor.execute_stream(plan)
          task.started ──▶ SpecialistAgent.execute(task, context)（角色日志）
               └─ Worker 注入位 → ToolRegistry.call(tool, args)
                    └ validate(ToolDefinition.input_schema) → impl → 结果
          task.completed/failed（WorkerResult；失败→REVIEW 决策）
          AgentMessage(task 下行 / report 上行) + ToolCallEvent → RunState/Collector
     → review_execution()（确定性：failed/empty/synthesis/候选阈值）
          continue → DONE → finalize_output()
              → assistant_delta（正文）→ selection {ids, images}
     → SSE 逐行 JSON 下发（对外契约，只增不改）
```

失败恢复路径（Phase 3/5 已验证并落盘 `mcp_runtime_traces.log`）：

```
EXECUTE#1 → review → revise（受影响 retrieve limit +=10，cap 40）→ EXECUTE#2 → continue/DONE
EXECUTE#1 → review → replan（synthesis 输入不足/候选低于阈值且有预算）→ 回 PLAN
attempt ≥ max_iterations(默认 2) → abort（受控终止，杜绝无限循环）
MCP/工具调用失败 → isError → MCPToolError → task.failed → 同上确定性 REVIEW
```


## 3. Agent 生命周期

1. **注册**：`build_default_agent_cards()` 产出三张 AgentCard（retrieval/organizer/vision）；
   `AgentRegistry.register_agent()` 按 `supported_tasks` 重建 task 分发索引；视觉能力关闭时
   VisionAgent `enabled=False`，不参与分发但可查询/枚举。
2. **实例化**：`SpecialistAgent = AgentCard(角色元数据) + Adapter(_task_to_worker 索引) + 既有 Worker`；
   Worker 代码零改动，Specialist 只增加角色日志与执行入口；Phase 2 起注入 ToolRegistry。
3. **分发**：默认 `default_workers()`；Supervisor/Orchestrator 可用 `build_dispatched_workers()` 把
   task_type 分发升级为 `AgentRegistry.get_agent_for_task(task_type)` → `SpecialistAgent`，
   未注册类型回退原 Worker（防御）。
4. **执行**：`SpecialistAgent.execute(task, context)` 打印 `[Specialist:<id>] … via <Worker>`
   后委托 `worker.run()`；工具调用必须走 `Agent.card.can_use()` 白名单 + `registry.call()`。
5. **信封通信**：`AgentMessage(message_id, sender, receiver, type=task|report|decision|event)`，
   进程内契约；SSE 事件是它的对外可观测投影，字段只增不改。
6. **回收**：`RunState` request 级隔离，随请求结束丢弃；无跨请求共享 Agent 可变状态。

---

## 4. Planner / DAGExecutor / Supervisor 的关系

| 组件 | 职责 | 关键机制 |
|---|---|---|
| PlannerAgent | 只规划不执行，产出一次性 `TaskPlanDraft` | 强制 JSON Schema 函数调用 + Pydantic 解析；失败抛 PlannerFailure → Legacy Fast Path 回退 |
| PlanValidator | 确定性计划校验 | Typed input 转换；task_id 正则；依赖引用/环/唯一 synthesize/上限（MAX_TASKS=6、MAX_PLAN_DEPTH=4）；校验后 Task 不可变 |
| DAGExecutor | 有界串行 DAG 执行 | 依赖图拓扑执行；依赖全失败→SKIPPED(DEPENDENCY_UNAVAILABLE)；Partial Failure Isolation；MAX_EXECUTED_TASKS=16 防御；task.* 事件 |
| SupervisorAgent | 受控监督层（TEAM 默认控制层） | PLAN→EXECUTE→REVIEW→(REVISE/REPLAN)→DONE/ABORT；生产默认 `enhanced=True`（`ENABLE_SUPERVISOR_RUNTIME` 未设置即开启），`enhanced=False` 时等价原基线 |
| ToolRegistry | 统一工具入口 | Worker 注入位经 adapter 走 registry；`ToolDefinition` 纯元数据 + impl 隔离 + ToolCallEvent |

确定性 REVIEW 规则（`review_execution()`，不调用 LLM）：
1. 存在 FAILED / 检索空结果 → `revise`（受影响任务；retrieve limit +10，cap 40）；
2. synthesis 无有效选择 / 输入不足 → `replan`（有预算时）；
3. 总检索候选数 < min_candidates(1) → `replan`；
4. 达 max_iterations(2) 仍未解决 → `abort`；否则 `continue` → DONE + `finalize_output()`。

Supervisor 是 TEAM 路径的**默认控制层**（`recommendation/__init__.py:38-45`：`ENABLE_SUPERVISOR_RUNTIME` 未设置 → True；`SupervisorAgent(enhanced=True, backend="langgraph")`），显式设 `ENABLE_SUPERVISOR_RUNTIME=false` 才回退 `AdaptiveTeamOrchestrator`。它运行 `DAGExecutor.execute_stream`，增量收集 ToolCallEvent 到
`RunState`/`EventCollector`，并把决策序列写入 `supervisor_trace`。


## 5. MCP Client / Server Flow（Phase 5，真实 MCP 协议）

架构定位：MCP 仅作为外部工具能力层，Planner/DAGExecutor/Supervisor/Worker 均未改动。

```
Agent Runtime
   └ SpecialistAgent.call_tool(name, args)           （原入口不变）
         → ToolRegistry.call(...)                      （AgentCard 绑定白名单）
              → remote impl（schema 过滤 + limit→top_k 映射 + rows/candidates 解包）
                   → MCPToolClient（stdio）/ MemoryMCPClient（in-process）
                        → MCP ClientSession.initialize()
                        → tools/list（发现 → 转 ToolDefinition, source="remote"）
                        → tools/call（JSON-RPC 消息）
                             → MCP Server（FastMCP）
                                  ├ retrieval_server: search_images / rerank_candidates
                                  │     + resources image://{id} / metadata://{id}
                                  └ vision_server:  analyze_image(image_id, task)
                             → 复用既有能力（tool_search_images / rerank_candidates / ImageAnalysis）
        ← 文本 content（isError → MCPToolError）
        → ToolCallEvent（registry） + client trace（latency/error）
```

- **协议**：official `mcp` SDK（固定 `mcp<2`，本机 1.29.1）；stdio 为生产入口
  （`python -m backend.mcp_servers.<server>`）；测试/CI 用 `mcp.shared.memory` 会话（同为真实 JSON-RPC）。
- **Local 与 Remote 并存**：ToolRegistry 允许同一 name 覆盖实现；`ToolDefinition.source`
  ∈ {local, remote}。本地工具路径零改动。
- **错误恢复**：服务器异常 → FastMCP isError content → client 抛 `MCPToolError` →
  registry 记 failure 事件并 re-raise → Worker 捕获 → task.failed → Supervisor REVIEW 决策。
- 实测开销：stdio 单次 ~1.5 s（进程冷启动，discover≈1509 ms/call≈1566 ms）；
  in-process memory 单次 3–30 ms（registry 记录 2.7–28 ms）。

---

## 6. ToolRegistry 设计（Phase 2 MCP-style + Phase 5 Remote）

数据模型（`tools/models.py`，47 行内聚）：
- `ToolDefinition`：name / description / input_schema(JSON Schema) / output_schema / category /
  cost_level / **source**(local|remote) / enabled —— 纯元数据。
- `ToolCallEvent`：tool_name / arguments / agent_id / success / latency_ms / error。
- 常量：CATEGORY_RETRIEVAL / CATEGORY_VISION / CATEGORY_ORGANIZATION。

注册中心（`tools/registry.py`）：
- `register(definition, implementation)`：同名覆盖注册（幂等，便于测试注入 Fake）；
- `list_tools(enabled_only)` / `get`：发现与查询；
- `call(name, arguments, agent_id=None)`：**唯一入口**——validate → implementation(**validated)
  → 成功/失败都记录 ToolCallEvent 与 per-tool 统计（calls/successes/failures/total_latency_ms），
  失败 re-raise 交由上层降级；
- `validate_arguments`：JSON Schema 受限子集校验（required + 顶层类型 + array 元素类型 + enum），
  额外键宽容（保留 Worker 内部白名单等既有防线）。

内置工具（`tools/builtin_tools.py`）与内置能力薄适配：

| Tool | Category | 复用实现 |
|---|---|---|
| search_images | retrieval | tool_search_images / SearchRepository.unified_search |
| rerank | retrieval | rerank_candidates（R1）；model_reranker（R2） |
| analyze_image | vision | ImageAnalysis（惰性导入） |
| filter_images | organization | MetadataWorker 确定性过滤语义（静态复用） |
| group_images | organization | GroupImagesWorker 确定性分组语义（静态复用） |

适配器（`tools/adapters.py` + `supervisor.py`）：`make_registry_retrieval_adapter` / `RegistryAnalyzer` /
`build_registry_backed_workers` 把 Worker 注入位接到 registry，不改 Worker 核心；
检索/视觉能力经 registry 事件进入 RunState 轨迹，供 Judge 与调优消费。
远程注册：`register_remote_tools(registry, client)` 从 MCP tools/list 构建 ToolDefinition 注册。


## 7. Evaluation Pipeline（Phase 4.1 / 4.2-A / 4.2-B）

```
EvaluationCase（离线用例声明：query/goal + 预期 agents/tools/decisions/final_state/selection）
        │ EvaluationRunner.run_case(case)
        ▼
runtime_factory(case) → SupervisorAgent（run_stream + run_state）
        │ 逐事件收集 selection_ids / error / duration
        ▼
build_evaluation_trace() → EvaluationTrace（plan_ok / execution_status / supervisor_states /
        decisions / agent_messages / tool_events / tool_summary / metrics / selection_ids）
        ├─ 确定性：expected_violations() 对照预期 → pass/fail 汇总（EvaluationSummary）
        └─ 质量：Judge.evaluate_trace(trace)
              ├ JudgeInput（结构化证据，由 build_judge_input 派生；无 LLM 生成物）
              ├ Judge(ABC)：evaluate(JudgeInput) → JudgeResult
              │      verdict = pass | flag | fail
              │      criterion_scores（0–5）+ reasons[] + error
              ├ MockJudge（确定性启发式：execution/tool_health/selection_coverage）
              └ LLMJudge（Phase 4.2-B，厂商无关）
                    JudgePromptBuilder（精简上下文 + rubric 段）→ LLMClient.generate(prompt)
                    → parse_judge_response（剥离 ```json 围栏，严格 JSON）
                    → 非法输出收敛为 verdict=flag + error="invalid judge response"
```

Rubric（`rubric.py`）五个评分维度：`planning_quality`、`agent_routing`、`tool_usage`、
`recovery_behavior`、`final_result_quality`，各 0–5 分；`LLMJudge.criteria = criterion_keys()`
与 rubric 一致，`MockJudge` 则为轻量确定性启发式（execution/tool_health/selection_coverage）——
两者共用同一 `Judge` 接口与 `JudgeResult`，可互相替换。
`LLMClient` 为 ABC（MockLLMClient 测试用；未来可接 OpenAI/Azure/vLLM），Phase 4.2-B 未改 Runtime。

评估链路不变式：确定性断言（无 LLM）负责“结构正确”，LLM-as-a-Judge 负责“质量判定”，两者解耦。

---

## 8. 关键设计不变式与验证

不变式：
1. **不推翻既有运行时**：Planner/Validator/DAGExecutor/Synthesizer 全程未被替换；
   Supervisor 是叠加控制层：生产默认 `enhanced=True`（可触发受限 REVISE/REPLAN），`enhanced=False` 时与基线行为等价（用于零回归验证与紧急回退）。
2. **Worker 核心零改动**：Specialist/工具/MCP/视觉复用都通过“包装/适配/注入位”实现。
3. **有限状态机**：REVIEW 确定性规则，max_iterations 限制迭代；Executor 有 MAX_EXECUTED_TASKS 防御。
4. **类型化输入**：Task.input 为 Typed Union；TaskPlanDraft→Task 必须经 PlanValidator，任务不可变。
5. **SSE 契约稳定**：事件只增不改；MCP/评估均为离线或外部能力层，不改对外契约。
6. **双轨评估**：确定性 expected_violations + Judge 接口统一（MockJudge ↔ LLMJudge 可替换）。

验证证据（全部本机实测）：
- `pytest backend/tests/agent_team -q` → 463 passed（2026-09-10 复测）；
- MCP 协议/运行时：test_mcp_servers 7 passed + test_mcp_runtime 3 passed（A 检索 / B 检索+视觉 / C 失败恢复→REVISE→done），trace 见 `backend/tests/agent_team/mcp/mcp_runtime_traces.log`；
- 早期阶段门禁：Phase 1 报告（Specialist）、Phase 2 报告（ToolRegistry + 集成验证）、
  Phase 3 closure（Supervisor/生产开关）、Phase 4.1/4.2-A/4.2-B（Evaluation + Judge 契约 + LLMJudge）
  逐阶段报告位于仓库根目录。

审计结论：SmartImager 达到“受控多 Agent 运行时 + 统一工具层 + 真实 MCP 外部工具能力 + 双轨评估”的
既定目标；剩余为生产化扩展点（MCP 连接池/常驻会话、真实 LLM Judge 客户端接入等；数据层已于 2026-09-10 完成
PostgreSQL + pgvector 迁移，不再是 SQLite），详见各阶段报告 Limitation 章节。

相关文档：`PHASE5_REAL_MCP_IMPLEMENTATION_REPORT.md` / `SMARTIMAGER_AGENT_UPGRADE_DESIGN.md` /
各 `PHASE{1..4}*_REPORT.md`（仓库根目录）。
