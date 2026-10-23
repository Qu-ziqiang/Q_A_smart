# PROJECT TECHNICAL SUMMARY — SmartImager Agent Team

> 定位：面向面试/交接的项目技术总结（工程演进、关键决策、取舍、局限）。
> 关联：`FINAL_ARCHITECTURE_REPORT.md`（架构）、`RESUME_EXTRACTION.md`（简历表述）。
> **2026-09-10 更新**：数据层已由 SQLite/sqlite-vec 迁至 PostgreSQL + pgvector（HNSW/cosine），图库约 1000 张 VOC2010；测试门禁由 448 → **463 passed**。

---

## 1. 原始问题（Original Problem）

SmartImager 是一个**图片语义检索与视觉核验应用后端**。初始形态是“单体 workflow”：

- 用户一句话请求（混合检索意图、元数据约束、展示偏好）；
- 后端用一套**硬编码流水线**执行：意图改写 → 向量检索（PostgreSQL + pgvector）→ 确定性重排 →
  元数据过滤/分组 →（可选）视觉核验 → LLM 合成推荐；
- 路由由 `ComplexityGate` 在 legacy fast path 与 team path 之间二选一。

核心痛点：
1. **不可组合**：逻辑分散在 worker/函数内联，新增“检索→分析→汇总”类组合需要改大段控制流；
2. **不可观测/不可评估**：执行过程没有结构化轨迹，无法判断一次推荐“为什么好/坏”；
3. **无恢复机制**：检索空结果 / 视觉失败 / 工具异常 → 直接失败，无重试或修正；
4. **能力绑定进程内**：检索、视觉、工具都与 Agent 进程强耦合，无法作为外部服务/工具暴露；
5. **缺少 Agent 语义**：虽然已有 worker 团队，但没有角色/注册表/信封/监督层的“agent 化”分层。

目标：在不破坏既有推荐质量与 SSE 契约的前提下，把它演进成
**受控多 Agent 运行时 + 统一工具层 + 真实 MCP 外部工具能力 + 离线/LLM 双轨评估**，
并全程用测试门禁与运行日志验证。

---

## 2. 架构演进（Architecture Evolution）

| 阶段 | 交付 | 关键结构变化 | 验证 |
|---|---|---|---|
| 既有基线 | PlannerAgent → PlanValidator → DAGExecutor → Synthesizer；ComplexityGate 双路径；SSE 契约 | — | 既有测试保持绿 |
| Phase 1 Specialist | AgentCard / SpecialistAgent / AgentRegistry / AgentMessage 信封 | Worker 之上加“角色化包装层”，不改 Worker | Phase1 报告 + 回归 |
| Phase 2 ToolRegistry | ToolDefinition / ToolRegistry / JSON Schema / ToolCallEvent / 5 个内置工具 | 工具统一入口 + 事件轨迹；AgentCard 工具绑定 | ToolRegistry 套件 + 集成验证报告 |
| Phase 3 Supervisor | SupervisorAgent FSM、RunState、确定性 review/revise/replan、registry-backed workers、生产开关（**默认开启 enhanced=True / backend=langgraph**） | 在 Executor 之上加“受控监督层” | Supervisor state-machine + 生产集成报告 |
| Phase 4.1/4.2 | evaluation 包（case/trace/summary/runner）、Judge ABC/JudgeInput/JudgeResult、MockJudge、LLMJudge + Rubric(5 维) + LLMClient 抽象 | 离线执行→评估管道；Runtime 零改动 | Evaluation/Judge/LLMJudge 套件 |
| Phase 5 MCP | `mcp_servers/`（retrieval/vision）+ `tools/mcp_client.py` + resources | MCP 作为外部工具能力层，Local/Remote 工具并存 | mcp 7+3 用例 + 回归 448 passed（后复测 463） |

演进原则（贯穿全程）：**叠加不推翻**、**Worker 核心零改动**、**SSE 契约只增不改**、
**新增能力以开关控制、可一键回退**（当前状态：Supervisor/增强 REVIEW **默认开启**；Vision、MCP、GLM-4V 重排仍默认关闭）、**阶段验收用报告 + 测试门禁**。


## 3. 关键工程决策（Key Engineering Decisions）

**D1. 用“叠加控制层”而非重写运行时**
决策：Supervisor 状态机架在既有 DAGExecutor 之上；生产默认 `enhanced=True`（`ENABLE_SUPERVISOR_RUNTIME` 未设置即开启），`enhanced=False` 与基线等价，作为零回归验证与紧急回退手段。
理由：基线推荐已被既有测试/体验锁定，任何行为差异都可量化；REVIEW 规则确定性可测。
取舍：控制力不如“重写 orchestrator”，但换来零回归风险与渐进上线。

**D2. Typed Task 模型 + 确定性 Validator 拦截 LLM 计划**
决策：Planner 只产 JSON Schema 受限的 TaskPlanDraft，Pydantic 解析 + PlanValidator 语义校验后
才生成不可变 Task；LLM 不能自由创建任务类型。
理由：把“LLM 自由度”压缩到最小可信任面，防止 schema 漂移与执行越权。
取舍：表达能力受限（预定义 5 类 TaskType），换取计划稳定性。

**D3. Worker → Specialist 只加“角色化包装”**
决策：SpecialistAgent = AgentCard(元数据) + adapter(索引) + 原 Worker(执行)。
理由：Agent 语义（角色/注册/白名单/通信信封）与执行实现解耦；可单测包装层。
取舍：无独立 LLM agent loop；Specialist 不“自主决策”，决策权集中在 Supervisor/Planner。

**D4. ToolRegistry 统一工具入口（MCP-style schema，本地优先）**
决策：所有工具以 ToolDefinition(input_schema) + implementation 注册；Agent 只能经
`registry.call()`（受 AgentCard 绑定约束）调用；成功/失败均产生 ToolCallEvent。
理由：工具调用可校验、可观测、可统计，为 LLM-Judge 提供工具健康证据。
取舍：参数受受限 JSON Schema 校验（额外键宽容），完整 JSON Schema 引擎未引入。

**D5. Supervisor REVIEW 第一阶段“不调用 LLM”**
决策：failed/empty/候选阈值/synthesis 输入不足 → 确定性 revise/replan/abort；LLM 留到
Evaluation 阶段做离线 judge。
理由：线上控制路径要可解释、无额外成本、无重复随机；LLM 质量判断放到离线管道。
取舍：策略不如 RL/LLM 控制器自适应强，但 100% 可复现、可测。

**D6. 评估双轨：确定性断言 + Judge 接口（Mock → LLM 可替换）**
决策：`EvaluationTrace` 统一“一次执行的证据”；确定性 `expected_violations` 判结构，
`Judge(ABC)`（JudgeInput→JudgeResult, verdict pass/flag/fail + 0–5 criterion_scores）
判质量；MockJudge 与 LLMJudge 同接口。
理由：结构正确与质量判定解耦；不绑定厂商（LLMClient ABC）。
取舍：Judge 输出是“参考信号”，未做在线强化闭环。

**D7. MCP 只作为“外部工具能力层”**
决策：官方 mcp SDK（1.x）stdio 服务器 + MCPToolClient/register_remote_tools；
远端工具注册进同一 ToolRegistry（source=remote），与 LocalTool 并存；schemas 白名单适配。
理由：真实 MCP 协议但零侵入 Agent Runtime；检索/视觉逻辑复用既有能力，不重复实现。
取舍：stdio 单次调用 1.5 s 进程冷启动；自动化测试用 in-process memory 会话（协议一致）。

**D8. 可观测性贯穿 request 生命周期**
决策：RunState(request 级) + EventCollector + SSE 事件流三层投影；ToolCallEvent 统一工具轨迹。
理由：评估、调试、验收（日志证据）都依赖同一事实源。
取舍：请求级状态不跨请求持久化（评估需离线持久化时才引入存储，属后续扩展）。

**D9. 严格资源上限**
决策：MAX_TASKS=6 / MAX_PLAN_DEPTH=4 / MAX_EXECUTED_TASKS=16 / 检索 40 / 视觉 8 /
synthesis 20 / Supervisor max_iterations=2 等常量集中定义。
理由：把“有界性”做成结构约束，防御无限执行/资源爆炸。
取舍：硬上限牺牲极端长链能力，换取可证明的终止性。


## 4. 局限（Limitations）

1. **Supervisor 决策为确定性启发式**：REVISE 仅作用于 retrieve limit（+10/cap 40）；其他任务
   类型只标记重试，不做语义级计划重排；REPLAN 需要重新规划（受 max_iterations 预算约束）。
2. **无跨请求持久状态**：RunState/EventCollector 均为 in-memory、request 级；评估用例如需
   跨会话分析（趋势/回归），需额外落库（Phase 4.3 待办）。
3. **Judge 未接真实厂商**：LLMJudge 已有契约+Mock 验证，`LLMClient` 只到抽象层；
   真实 OpenAI/Azure/vLLM 客户端未实现，Judge 对线上行为无在线反馈闭环。
4. **MCP 连接为“每次调用新建”**：stdio 冷启动 ~1.5 s；无长会话/连接池、无 streamable-HTTP
   服务器入口（仅 stdio 生产入口）；自动测试用 in-process memory 会话（协议一致但非跨进程）。
5. **Tool schema 校验为受限子集**：无完整 JSON Schema 引擎（组合/嵌套引用等）；依赖 Worker 层
   既有白名单兜底。
6. **Worker 串行执行**：DAGExecutor 为串行实现（ExecutionContext 为并发预留），视觉多图
   串行逐图；未启用并行调度。
7. **Agent 无独立规划循环**：Specialist 不自主决策；系统控制点集中在 Planner/Supervisor，
   自由度低但可控性强——对“自主 agent”类需求属于有意取舍。
8. **Windows/进程相关限制**：stdio 多进程在批量捕获下不稳定（已用 memory 会话规避）；
   SSE/HTTP 的服务器部署形态需另行验证。
9. **领域范围**：TaskType 固定为检索/过滤/分组/视觉/合成 5 类；新增领域能力需扩展枚举、
   schema、Validator、Worker、Card 多处（有集中常量与注册表支撑）。

## 5. 工程数据快照（供摘要引用）

- 测试门禁：`pytest backend/tests/agent_team -q` → **463 passed**（2026-09-10 PG 迁移后复测）；
  阶段报告：Phase 3 closure 276、Phase 4.1 282、4.2-A 288、4.2-B 296、Phase 5 448（随阶段递增，口径见各报告）。
- 数据层：PostgreSQL + pgvector（`pgvector/pgvector:pg16`；三张 `vector(512)` 表 + HNSW(cosine)）；
  千级图库（约 1000 张 VOC2010）实测检索均值 65.0ms（`r1_eval_pg1000.json`）。
- 代码形态：Python 3.12；核心包 `backend/ai_func/recommendation/orchestration/`
  （supervisor 655 行 / workers 772 / planner 421 / executor 234 / tools registry 174 +
  builtin 284 / mcp_client 265）+ `evaluation/`（models 68 / runner 232 / judge 173 /
  llm_judge 113 / rubric 75）+ `backend/mcp_servers/`（retrieval/vision stdio servers）。
- 运行时组件数：3 张默认 AgentCard、5 类 TaskType、5 个内置工具 + 3 个 MCP 远端工具
  （search_images/rerank_candidates/analyze_image）+ 2 类 MCP resources、7 个 SupervisorState、
  5 维 Judge rubric。
