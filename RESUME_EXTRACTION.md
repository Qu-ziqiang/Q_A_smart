# RESUME EXTRACTION — SmartImager Multi-Agent Platform

> 项目一句话：**SmartImager** — 将单体“图片检索+视觉核验”工作流演进为**受控多 Agent 平台**
> （Specialist Agent + ToolRegistry + Supervisor FSM + LLM-as-a-Judge + 真实 MCP 外部工具），
> Python 3.12 / FastAPI / PostgreSQL + pgvector / Pydantic v2，463 项 pytest 门禁全绿。
> **2026-09-10 更新**：数据层已由 sqlite-vec 迁至 PostgreSQL + pgvector（HNSW/cosine），图库 66 张 → 约 1000 张 VOC2010；门禁由 448 → **463 passed**。本文数字已按此口径更新。

可用事实底座（所有子弹建议锚定这些数字）：
- 6 阶段演进、零回归基线：`enhanced=False` 等价原路径；Worker 核心代码零改动。
- 运行时：3 个 Specialist 角色 / 5 类 Task / 7 个 Supervisor 状态 / 5 个内置工具 +
  3 个 MCP 远端工具 + 2 个 MCP resources / 5 维 Judge rubric。
- 质量：确定性 REVIEW（0 LLM 依赖、可复现）+ 双轨离线评估（确定性断言 + Judge pass/flag/fail，
  0–5 分）。测试门禁最终 `pytest backend/tests/agent_team` **463 passed**（阶段门禁 276→296→448→463，
  463 为 2026-09-10 PG 迁移后复测）。
- MCP：官方 SDK `mcp 1.29.1`；stdio 生产入口单次 ~1.5 s（进程冷启动），in-process 会话 3–30 ms。

---

## A. Backend Engineer（后端工程师版）

### 简历子弹（3 选 1，可微调职责行）
1. **后端架构演进**：将单体图片检索工作流重构为分层的多 Agent 平台（Planner→Validator→DAG→
   Specialist→ToolRegistry），全程保持既有 SSE 契约与推荐逻辑兼容（`enhanced=False` 基线等价），
   Worker 核心零改动，累计 463 项 pytest 门禁通过。
2. **后端工程化**：用 Pydantic v2 建立 Typed Task 模型 + JSON Schema 受限的 PlanValidator 拦截
   LLM 计划输出（5 类 TaskType、task_id 白名单、依赖/环/上限校验），并在 DAGExecutor 中内置
   资源上限（MAX_TASKS=6 / MAX_EXECUTED_TASKS=16）等有界执行保护。
3. **后端稳定性与恢复**：实现确定性 Supervisor FSM（PLAN→EXECUTE→REVIEW→REVISE/REPLAN→DONE/
   ABORT，默认 max_iterations=2），把空检索/工具失败/候选不足转化为可测试的 revise/replan 策略，
   工具失败恢复场景由运行日志验证。

### Technical Highlights
- 分层架构设计：`AgentCard(元数据) + SpecialistAgent(包装) + AgentRegistry(分发) + 既有 Worker`，
  分发失败回退原 Worker；新增能力不触碰执行内核。
- 统一工具层：ToolDefinition(JSON Schema) 与 implementation 分离、AgentCard 工具绑定白名单、
  `registry.call()` 唯一入口、成功/失败均产出 ToolCallEvent（含 latency）。
- 可观测性：SSE 事件流 / AgentMessage / RunState 三层投影，request 级隔离。
- Python 工程质量：常量集中（无散落 magic number）、函数内惰性导入规避循环依赖、
  用 Fake/overrides 注入做离线确定性测试（不触真实 DB/模型）。

### Measurable Metrics
- **463 passed** 最终测试门禁（阶段化 276 → 282 → 288 → 296 → 448 → 463）；
- 5 类 TaskType、6 个内置/远端工具 + 2 resources、7 个 Supervisor 状态、默认 2 次迭代上限、
  REVISE 步长 +10（cap 40）；
- 回归风险隔离：`enhanced=False` 与基线行为等价；Worker 核心文件零修改。

### Interview Discussion Points
- 如何在“叠加监督层”与“不重写运行时”之间做架构决策？如何证明零回归？
- Pydantic Typed Union 校验 + Validator 拦截 LLM 计划的错误处理链（PlannerFailure→Legacy fallback）；
- DAGExecutor 的依赖可用性/部分失败隔离/执行上限设计；为什么串行而预留 ExecutionContext？
- ToolRegistry schema 校验为什么用受限子集而非完整 JSON Schema 引擎？


## B. AI Application Engineer（AI 应用工程师版）

### 简历子弹（3 选 1，可微调职责行）
1. **LLM 驱动的 Agent 应用**：把“用户一句话 → 结构化图片推荐”做成受控 Agent 平台：LLM
   Planner 生成受限计划（JSON Schema + 确定性校验）→ 确定性执行 → 确定性 REVIEW 恢复，
   让 LLM 只负责“规划与合成”，执行/恢复完全确定可测。
2. **LLM-as-a-Judge 评估体系**：设计离线 Evaluation 管道（EvaluationCase→Trace→Summary），
   用统一 Judge 接口（JudgeInput→JudgeResult，verdict pass/flag/fail + 5 维 0–5 分 rubric：
   planning_quality/agent_routing/tool_usage/recovery_behavior/final_result_quality）接入
   MockJudge 与真实 LLMJudge（厂商无关 LLMClient 抽象 + 严格 JSON 解析/兜底）。
3. **视觉多模态应用**：串起 PostgreSQL + pgvector（HNSW/cosine）语义检索 + GLM-Vision 视觉核验 + R1/R2 重排，
   视觉能力以功能门禁（vision_available）控制注册与计划，按需注入测试。

### Technical Highlights
- 规划层：Planner 一次性 Typed 规划；PlanValidator 强制任务类型/依赖/上限约束，防 schema 漂移。
- 执行层：`SpecialistAgent`（retrieval/organizer/vision）包装既有 Worker；视觉/检索均经
  ToolRegistry 走统一工具调用并可 trace。
- 恢复层：确定性 REVIEW 把 empty retrieval / 工具失败 / 候选不足映射为 revise/replan，LLM 不介入
  线上控制路径（可解释、0 额外成本）。
- 评估层：确定性 expected_violations（结构）+ LLM-as-a-Judge（质量）双轨；Judge 结果含
  reasons[] 与 criterion_scores，支持后续回归分析；非法 Judge 输出收敛为 flag+invalid。
- MCP（Agent 能力即服务）：真实 MCP 服务器暴露 search_images/rerank_candidates/analyze_image，
  复用既有检索/视觉实现，无重复造轮子。

### Measurable Metrics
- 463 项 pytest（2026-09-10 复测；PG 迁移前末次为 448）；Phase4.2-B 终点 296 项（evaluation/judge/llm_judge 全绿）；
- 5 维 Judge rubric 0–5 分；3 档 verdict（pass/flag/fail）；
- 2 个 MCP Server / 3 个远端工具 + 2 resources；stdio ≈1.5 s/call vs memory 3–30 ms/call；
- 3 个 Specialist 角色覆盖 5 类任务；视觉候选上限 8、synthesis 上限 20。

### Interview Discussion Points
- 为什么不把 LLM 用在 Supervisor 线上 REVIEW（第一阶段确定性）？何时才值得上 LLM judge？
- LLM 计划可靠性：如何限制自由度、validator 拦截哪些错、失败如何回退？
- Judge rubric 如何设计成“可解释 + 可量化”（每维 description/evaluation_focus/0–5）？
- 多模态链路里哪些环节是确定性的、哪些是模型推理？如何在测试中隔离模型？
- MCP 集成“外部工具能力层”与 Agent runtime 的边界在哪？


## C. Agent Infrastructure Engineer（Agent 基础设施工程师版）

### 简历子弹（3 选 1，可微调职责行）
1. **Agent 平台基础设施**：从单体 workflow 演进出“角色化 Agent + 注册表 + 受控监督层 +
   统一工具层 + 双轨评估”的多 Agent 基础设施；用叠加（控制层与 Worker 解耦，
   `enhanced=False` 时与基线行为等价，作为零回归验证与紧急回退）而非
   重写的方式保证演进零回归。
2. **工具调用/协议层**：实现 MCP-style ToolRegistry（ToolDefinition/JSON Schema/校验/ToolCallEvent
   统计），并将官方 MCP SDK（mcp 1.x）接入为外部工具能力层——stdio 服务器 + MCPToolClient +
   远端工具自动发现/参数白名单适配/rows 解包，LocalTool 与 MCPRemoteTool 同注册表并存。
3. **可观测与评估基础设施**：设计 RunState/EventCollector/SSE 三层可观测投影与离线 Evaluation
   管道（EvaluationCase→Trace→Summary），Judge 接口化（Mock/LLM 可替换、verdict+5 维评分），
   使“agent 质量”可离线量化与回归。

### Technical Highlights
- 控制面：Supervisor FSM（7 状态）确定性 REVIEW（0 LLM）→ revise/replan/abort；双层防无限循环
  （max_iterations=2、DAG MAX_EXECUTED_TASKS=16）。
- 身份/权限面：AgentCard（角色、能力、工具绑定白名单、cost、enabled）+ AgentRegistry task 分发；
  `SpecialistAgent.call_tool` 越权抛 PermissionError；禁用 Agent 不参与分发。
- 消息面：AgentMessage 信封（task/report/decision/event）+ SSE 事件流投影，字段只增不改。
- 工具面：registry 唯一入口；schema 受限子集校验；每次调用成功/失败都记 ToolCallEvent + 统计
  （calls/successes/failures/latency）；同名注册幂等覆盖（测试注入 Fake 友好）。
- MCP 面：MCPToolClient(stdio)/MemoryMCPClient(同协议 in-process)；discover→ToolDefinition 转换；
  isError 聚合为 MCPToolError → registry failure 事件 → Supervisor 恢复；资源模板
  image://{id}/metadata://{id}。
- 测试面：确定性 Fake/夹具（env 注入）避免真实 DB/模型；协议测试 7 项 + 端到端场景 3 项
  （retrieval / retrieval+vision / 失败恢复→REVISE→done），trace 落盘日志。

### Measurable Metrics
- 463 项 pytest 门禁；阶段门禁 276→282→288→296→448→463；
- 端到端验证：Planner→Validator→DAG→Specialist→MCP→Server→Supervisor 全链在 3 个场景通过并留档；
- MCP：discover/call schema 正确、failure 收敛为 MCPToolError；stdio ≈1.5 s/call、memory 3–30 ms/call；
- 收敛保障：Supervisor 默认 max_iterations=2、REVISE limit cap 40；工具调用 100% 事件化。

### Interview Discussion Points
- 设计“可证明有界”的 agent loop：iteration cap、执行上限、validator 拦截分别防什么？
- 为什么让 REVIEW 先确定性、把 LLM 放到离线 judge？两者各自的价值与成本？
- 统一工具层最难的点：schema 校验边界、Agent 权限绑定、事件化的失败/延迟观测。
- MCP 作为“外部工具能力层”的边界决策：为什么不动 Planner/DAG/Worker？兼容性风险怎么控
  （SDK 版本固定、memory vs stdio、schema 白名单适配）？
- 如何让一套运行时同时支持“真实模型”与“测试 Fake”而不改生产代码？

---

## 附：术语对照与延伸（避免面试口径不一致）
- “Agent”在本项目 = **确定性执行 + 有限决策**：Specialist 无自主 loop，决策集中在
  Planner（规划）与 Supervisor（恢复策略）；LLM 另用于 Synthesizer 与 LLMJudge。
- “MCP-style（Phase 2）”与“Real MCP（Phase 5）”的区别：前者是本地 JSON Schema 工具层；
  后者通过官方 SDK 走真实 tools/list、tools/call（stdio/in-process 会话）。
- 测试口径：仓库内不同阶段跑法不同（自研 runner / pytest），引用数字请带范围：
  463 = `pytest backend/tests/agent_team` 当前门禁（2026-09-10 PG 迁移后复测；迁移前末次 448，含 tools 30 + mcp 10）；
  阶段报告的 276/282/288/296 为当时各自 runner 口径。
