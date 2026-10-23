# SMARTIMAGER_AGENT_UPGRADE_DESIGN

> 项目：SmartImager — 多模态智能图库与对话推荐系统
> 文档性质：**只读架构审计 + 演进设计**（不修改生产代码）
> 代码基准：`Smartlmager-sqlite-vec/`（FastAPI + React；Agent Runtime 位于 `backend/ai_func/recommendation/`）
> 参考文档：`PHASE1A/B/C_IMPLEMENTATION_REPORT.md`、`AGENT_MEMORY_AUDIT_AND_DESIGN.md`、仓库内各 `PHASE2_STAGE*`/`PHASE1_SPECIALIST*`/`PHASE3_*`/`PHASE4_*`/`PHASE5_*`/`PHASE6_*` 与 Benchmark 报告
> 日期：2026-09-07（**2026-09-10 更新：数据层已迁至 PostgreSQL + pgvector；文中"复用/替代 SQLite"的表述以现状为准**）

---

## 0. TL;DR（结论先行）

1. **当前 Runtime 不是“无架构”，而是一个强约束的「单 Planner 一次性规划 + Typed DAG 串行执行 + LLM 终端合成」流水线。** 它已经把 Multi-Agent 系统里最难的 *可控性层*（类型化 Task、确定性 Validator、有界 Executor、能力目录、失败隔离）建好了。
2. **真正缺的不是“智能体个数”，而是三样东西：**
   - (a) **角色化封装**：现在 `Worker`（Retrieval/Metadata/Group/Vision/Synthesis）是“函数执行单元”，不是“可被描述、可被监督、可选工具”的 Specialist Agent；
   - (b) **一个能复述/复盘/终止的 Supervisor 决策点**：现在 `Planner → Executor → Synthesizer` 是一次性的单向流水线，没有“看结果→决定继续/重做/换工具”的回路（recovery 目前靠整条 fallback 到 Legacy）；
   - (c) **统一的 Tool 描述与注册**：工具 schema 目前散落在 `agent.py`（inline OpenAI function）与 `planner.py`（submit_task_plan）两处，没有单一 ToolRegistry。
3. **建议的目标形态**：`SupervisorAgent（= Gate+Intent+Planner+质量决策）` 编排 `RetrievalAgent / OrganizerAgent / VisionAgent` 三个 Specialist；**Planner/DAG/Validator 一律保留**，只在其上增加“执行→汇报→（必要时）再决定”的最小循环。首个版本甚至可以是「无自由对话的受控多智能体」——多 Agent 之间不闲聊，只交换 Typed 消息。
4. **技术选型结论（详见 §5）**：
   - **MCP Tool Protocol**：值得引入其 *形态*（tool name/description/inputSchema 单点注册），真正接入 MCP SDK 建议放到 M4 之后、以“能力外部化”为触发条件，不要在 M1 强上；
   - **LangGraph StateGraph**：**不建议整体引入**。现有 Pydantic Typed 模型 + DAG 语义与 StateGraph 的 dict-state 模型重叠又冲突；收益可被“自研小型 Supervisor 状态机”覆盖，代价是重写编排层并丢失现有契约；
   - **Agent Memory**：已有 Episodic（确定性 episode）应保留并作为地基；扩展**用户偏好/画像级长期记忆**值得做（可复用现有向量底座；该底座已于 2026-09-10 由 sqlite-vec 换成 PostgreSQL + pgvector），但要用 flag + 预算约束渐进式开放；
   - **LLM-as-a-Judge**：值得引入，但定位为 **offline 评估第二层**（补充现有确定性 Gate/Planner/Rerank benchmark），不做在线阻塞。
5. **落地路径**：M0（保持现状）→ M1（Tool Registry 化，纯重构）→ M2（Worker→Specialist 角色化，纯加层）→ M3（Supervisor 最小决策回路）→ M4（可选 MCP 外化）→ M5（Memory/Judge 增强）。每一步独立可合入、可回滚，且不推翻 Planner/DAG。


---

## 1. 现状架构审计（As-Is Analysis）

### 1.1 一句话现状

当前系统是一条**带路由开关的分阶段流水线**，而不是“多智能体协作”：

```text
User Query
   │  POST /api/v1/ai/recommend/chat[/stream]        backend/routers/ai/recommendation.py
   ▼
_RecommendationService.stream_core()                 backend/ai_func/recommendation/__init__.py
   │ ① IntentRewrite（确定性规则）→ NormalizedIntent
   │ ② ComplexityGate.decide（确定性正则，默认保守 LEGACY）
   ▼
┌─ LEGACY ────────────────────────────────────────────────┐
│ RecommendAgent.run_stream()                            │
│   → 单 LLM 对话智能体，inline tools:                    │
│     search_images / choose_images（OpenAI function）    │
│   → SSE: assistant_delta / selection                    │
└─────────────────────────────────────────────────────────┘
   │ TEAM
   ▼
AdaptiveTeamOrchestrator.run_stream()                   orchestration/orchestrator.py
   │③ PlannerAgent（LLM，tool_choice=submit_task_plan 强约束，只出计划不执行）
   │④ PlanValidator（确定性：结构/语义/DAG/能力门禁）
   │⑤ DAGExecutor.execute_stream（串行 Kahn；依赖失败传播；MAX_EXECUTED_TASKS=16）
   │    每 Task → WorkerRegistry[TaskType].run(task, context) → WorkerResult
   │      RetrievalWorker / MetadataWorker / GroupImagesWorker / VisionWorker
   │     终端 SYNTHESIZE → SynthesizerAgent（LLM 合成，失败→确定性 fallback）
   │⑥ Memory（仅成功 Team run + conversation_id 存在时）：episode 写入
   ▼
SSE: plan.created/plan.failed → task.* → assistant_delta → selection → complete
   （plan.failed 时 service 层整体回退 Legacy Fast Path）
```

### 1.2 逐层审计

| # | 层 / 组件 | 主文件 | 已具备 | 局限（与本目标相关） |
|---|-----------|--------|--------|----------------------|
| 1 | HTTP / SSE 会话层 | `routers/ai/recommendation.py` | conversation_id 消息存取、request_sessions 审计、Team 事件原样透传、旧事件兼容 | 消息只有 `role/content/image_ids`；无 Agent/任务级持久化轨迹 |
| 2 | 服务路由 | `ai_func/recommendation/__init__.py`（`_RecommendationService`） | Gate→Team/Legacy 分叉、plan 失败回退 Legacy | 路由决策的“理由”没有结构化留存；Supervisor 语义缺失 |
| 3 | Intent Rewrite | `orchestration/intent.py` | 确定性规范化 + 能力信号抽取（复用 models 词表），无 LLM | 规则人工维护、覆盖率有限（宜作 Supervisor 轻量 pre-step 而非智能体） |
| 4 | Complexity Gate | `orchestration/gate.py` | 确定性高置信复合特征路由；默认 LEGACY；异常安全 | 只给二值路由；不产结构化决策（为何/需要谁）；将来可作为 Supervisor pre-filter |
| 5 | Planner | `orchestration/planner.py` `PlannerAgent` | 一次性 Typed 规划；capability catalog 单一来源；per-call 收窄 schema；tool_choice 强制；超时 | **只规划不执行、不消费结果、不能 re-plan**；角色语义与 Supervisor 高度重合 |
| 6 | Typed Task 模型 | `orchestration/models.py` | TaskType/TaskStatus/受限 Typed Input/Task/TaskPlan；资源上限集中 | 封闭枚举——新能力必须显式扩枚举（Skill/Tool 扩展的松绑点，需权衡） |
| 7 | Validator | `orchestration/models.py` `PlanValidator` | 结构/依赖/环/深度/能力门禁/候选生产者-消费者/单合成终点/类型化 | 校验“计划静态正确”，不校验“结果是否达成目标”（质量层缺失 → §5.4） |
| 8 | Executor | `orchestration/executor.py` `DAGExecutor` | 串行拓扑执行、部分失败隔离、依赖失败传播、执行上限、逐任务事件、ExecutionContext | 串行（并发常量已预留）；“谁以何依据执行”隐式由 context.results 决定 |
| 9 | Workers | `orchestration/workers.py` | Worker Protocol、5 类 Worker、`default_workers()` 注册表、结构化输出 | 无自我描述（能力/工具/成本/契约挂在 catalog 不挂在 Worker）；不能“决定调用别的能力” |
| 10 | Vision | `orchestration/vision.py` + `ai_func/image_analysis.py` | 严格契约（extra=forbid、长度/预算）、逐图失败隔离、systemic 语义、MIME 白名单 | `ImageAnalysis` 是函数式 provider 客户端，距“Vision Agent（带证据汇报）”只差解释封装 |
| 11 | Synthesizer | `orchestration/synthesizer.py` | LLM 合成 + `validate_selection`（ID 白名单）+ 确定性 fallback | 是唯一“汇报者”，但失败只降级确定性文本，无“回报 Supervisor 再决定”路径 |
| 12 | Memory | `orchestration/memory.py` + `repositories/agent_episodes.py` | 确定性 episodic：成功才写、24h 去重、TTL/cap、bounded recall、脱敏 | 仅“策略回忆”；无用户偏好结构化、无向量语义记忆 |
| 13 | Tool 层 | `recommendation/tools.py` `tool_search_images`；`repositories/search.py` `unified_search` | 检索工具已包装 + 向量目标白名单 | **无单点 ToolRegistry**：Legacy 在 `agent.py` inline 声明 function，Planner 在 `planner.py` 另声明 schema |
| 14 | Rerank | `orchestration/rerank.py`（R1）+ `model_reranker.py`（R2） | 同池重排、注入位、失败自动降级 | 属 RetrievalAgent 内部排序工具，被 Worker 直连，未走 Tool 抽象 |
| 15 | 状态 | ExecutionContext + `conversation_messages`/`request_sessions`/`agent_episodes` | 运行内工作记忆 + 会话持久化 + episode 分层清晰 | 无 agent/task 级运行轨迹表（复盘/eval 需从 SSE 重建） |
### 1.3 结构性判断

**强项（升级必须保留的地基）**

- **约束优先**：任何 LLM 产物（plan、vision JSON、synthesis）都过 Pydantic 白名单/结构校验，绝不信任裸输出；
- **能力单一来源**：`registered_task_types()` / `default_workers()` / `TASK_CAPABILITY_CATALOG` 已做到同源，Validator/Planner/Worker 共用一个口径；
- **失败隔离**：Worker crash 不炸 Executor；Vision 逐图失败不阻断下游；Rerank/provider 失败自动降级；Synthesizer 失败回退确定性；
- **预算思维**：MAX_TASKS / MAX_PLAN_DEPTH / MAX_EXECUTED_TASKS / evidence 字符预算 / recall 预算全部集中常量，无散落 magic number。

**结构性短板（与 Multi-Agent 目标的差距）**

1. **单向流水线，无“回路”**：`Plan → Execute → Synthesize → End`。一次计划失败只有整链 fallback；局部失败（如某路检索结果差）没有“让 Supervisor 换 query / 调条件再试一轮”的机制。
2. **Worker 是“哑执行器”**：Worker 只见 `task.input + context.results`，看不见自己的角色、能力边界、可用工具描述；也没有“产出证据 + 置信度 + 建议下一步”的汇报语言。
3. **无 Agent 协议层**：Task/WorkerResult 是数据结构不是消息信封；不同 Specialist 之间没有统一的“请求/响应/证据/置信”封装；将来接 MCP 工具或外部 Agent 必须补这一层。
4. **工具描述与注册不统一**：`agent.py` 的 function tools（search_images/choose_images）与 `planner.py` 的 submit_task_plan schema 各自维护；`tool_search_images` 参数名 `targets` 与 Legacy function 参数名 `vector_targets` 也不一致——接入 MCP 前的最小治理点。
5. **入口对象是进程级单例**：`_RecommendationService` 及其内部 `_orchestrator.last_plan` 是模块级共享；单进程并发请求下靠 ExecutionContext 局部性掩盖共享状态。Multi-Agent 化后应显式“一次 run 一个 RunState”。
6. **质量/达成度评估缺失**：现有 benchmark 验证“计划规范 / 路由正确”，不验证“推荐是否诚实、是否达成用户目标”。

---

## 2. 演进判定：哪些能自然长成目标角色

### 2.1 现状 → 目标映射总表

| 现状组件 | 目标角色/层 | 演进动作 | 复用程度 |
|----------|-------------|----------|----------|
| `ComplexityGate` | Supervisor 的「预过滤器 / 决策输入」 | 保留原判定；输出接入 Supervisor 决策或作为其 first-responder | 直接复用 |
| `IntentRewrite` | Supervisor 的「query 规范化前置」 | 保留原规则；产出 NormalizedIntent 作为 supervisor 上下文 | 直接复用 |
| `PlannerAgent` | **SupervisorAgent 的“规划大脑”** | 原样保留 `create_plan`；Supervisor 在它之上加“结果复盘→re-plan/终止”决策 | 直接复用 |
| `PlanValidator` | Supervisor 计划校验器 | 原样保留；re-plan 产出的新计划同样过校验 | 直接复用 |
| `DAGExecutor` | **受控执行运行时（Scheduler）** | 保留 execute_stream 与事件契约；作为 Supervisor 每次“执行一个计划”的确定性引擎 | 直接复用 |
| `RetrievalWorker` | **RetrievalAgent（Specialist）** | 包一层 Agent Card：描述 + 工具集（search+rerank）+ 汇报格式；内部逻辑不动 | 高（逻辑 100% 复用） |
| `MetadataWorker` + `GroupImagesWorker` | **OrganizerAgent（Specialist）** | 合入同一 Specialist（确定性过滤/分组）；预留“LLM 分组标题”可选能力 | 高 |
| `VisionWorker` + `ImageAnalysis` | **VisionAgent（Specialist）** | 包 Agent Card；能力门禁（vision_available）不变 | 高 |
| `SynthesizerAgent`（LLM+fallback） | **Supervisor 的“汇报器 / 回答器”** | 保留；从“必须终点”变为 Supervisor 可选的 answer 生成器（默认终点，保证向后兼容） | 高 |
| `tools.py tool_search_images` | Tool 层：`retrieve_images` 工具实现 | 收编进 ToolRegistry（单点 schema + 单点实现） | 直接复用 |
| `unified_search` / repositories | Tool 实现（数据面） | 不变，只被 Registry 引用 | 直接复用 |
| `ExecutionContext` | RunState（单次 run 的状态总线） | 扩展字段（agent_trace / messages / budget 记账），不推翻 | 重构（小） |
| `EpisodicMemoryService` | Agent Memory 地基（episodic） | 保留；扩展 user-preference 层（见 §5.3） | 直接复用 |
| SSE 事件（plan.*/task.*） | Agent Communication 的可观测投影 | 作为消息信封的“下行投递”格式复用；新增 agent.report 事件 | 复用 + 扩展 |
| Benchmark harness | Evaluation 框架地基 | 保留确定性层；增加 Judge 层（见 §5.4） | 直接复用 |

### 2.2 谁天然是 Supervisor Agent

**结论：`PlannerAgent + ComplexityGate + IntentRewrite + SynthesizerAgent + DAGExecutor 的组合`，行为上已经是一个“Task-Oriented Supervisor”，只差两层壳：**

- 一个**决策循环壳**（supervisor decide：`plan → execute → review → done | revise | replan | abort`）；
- 一个**角色视图壳**（把“发给 Worker 的 Task + 从 Worker 收的 WorkerResult”显式表达为“派给 Specialist 的指令 + Specialist 的汇报”，即 Agent 协议层）。

不需要新建“另一个 LLM 大脑”来当 Supervisor —— 当前 PlannerAgent 已承担“规划决策”。升级点在**编排控制流**（从单向到可回看），不在模型。

### 2.3 谁天然是 Specialist Agent

**结论：现有 5 类 Worker 已经是“无壳的 Specialist”。** 每个 Worker 拥有：
- 唯一的 `task_type`（角色 ID）；
- 明确的 Typed Input（任务指令格式）；
- 明确的输出模型（汇报格式：RetrievalOutput/MetadataOutput/GroupOutput/VisionTaskOutput/SynthesisOutput）；
- 依赖注入位（search_tool / image_repo / analyzer / reranker 全部可注入，等于“工具槽”）；
- 能力门禁（vision_available 注册 gating）。

天然映射：`RetrievalWorker→RetrievalAgent`、`VisionWorker→VisionAgent`、`MetadataWorker+GroupImagesWorker→OrganizerAgent`。`DeterministicSynthesisWorker` 留在 Supervisor 侧作兜底汇报器。

### 2.4 Agent Communication Protocol 的“已有点”

- Task（指令）与 WorkerResult（汇报）已经结构化，且 Executor 通过 `ExecutionContext.results` 做“值传递”；
- SSE 事件（task.started/completed/failed/skipped、plan.created）已具备“过程可观测”；
- 缺的是**信封与路由语义**：如 `agent_id`、`request_id`、`evidence/confidence/next_suggestion`、以及“事件是写给谁/由谁决策”的字段。这是新协议层要做的最小集合。

### 2.5 Tool Server / MCP 层的“种子”

- `tool_search_images` 已是“工具函数 + 参数 + 白名单”形态；`submit_task_plan`/vision 输出契约已类 MCP `inputSchema/output`；
- 数据面（SearchRepository.unified_search、ImageAnalysis、generate_vector）已按能力边界隔离，天然可作为独立 Tool Server 的能力；
- 缺的：统一 tool 声明（名称/描述/参数 JSON Schema/实现/校验/成本标记）与调用拦截（计数、预算、审计）。

### 2.6 “必须新增”的本质差距（四条）

1. **Agent/角色元数据（Agent Card）**：能力、可用工具、输入输出契约、成本、约束的统一描述对象；
2. **消息信封与总线（AgentMessage/EventBus）**：让 Supervisor 与 Specialist 之间、以及外部观察者，用同一语言交流；
3. **Supervisor 决策循环（review/re-plan/abort）**：最小化的“执行后复盘”状态机；
4. **ToolRegistry 与调用治理**：单点 schema、校验、注入、计量。

---

## 3. 目标架构：Multi-Agent Collaboration System

### 3.1 目标拓扑（保持 Planner/DAG 语义）

```text
User / Frontend
      │  SSE: assistant_delta / selection / complete（向后兼容不变）
      ▼
┌──────────────────────────────  API / 会话层（routers） ─────────────────────────────┐
│  conversation store / request_sessions / SSE 透传（现有，不改契约）                  │
└──────────────────────────────────────┬──────────────────────────────────────────────┘
                                       ▼
┌────────────────────────── Supervisor Agent（新增：决策壳 + 规划 + 复盘） ─────────────┐
│  pre-step: IntentRewrite → ComplexityGate（保留，作为确定性快路）                     │
│  brain:    PlannerAgent.create_plan（保留）→ PlanValidator（保留）                   │
│  control:  execute(plan) → review(partial results)                                 │
│              ├─ PASS    → answer（SynthesizerAgent，保留）                          │
│              ├─ REVISE  → 局部参数修正后重跑受影响子图（受限 N 轮）                   │
│              ├─ REPLAN  → 携带失败摘要再次 create_plan（受限 N 轮）                  │
│              └─ ABORT   → 确定性 fallback / Legacy（现有语义）                      │
└───────────────┬──────────────────────────────┬──────────────────────────────┐
                │ dispatch（AgentMessage：       │                               │
                │  typed Task / TaskPlan 切片）  │                               │
                ▼                               ▼                               ▼
┌───────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
│   RetrievalAgent      │  │   OrganizerAgent         │  │   VisionAgent            │
│  (Specialist)         │  │  (Specialist)            │  │  (Specialist)            │
│  role: 语义召回+排序    │  │  role: 过滤/分组/整理      │  │  role: 像素核验/证据       │
│  impl: RetrievalWorker│  │  impl: MetadataWorker +  │  │  impl: VisionWorker +    │
│  tools: search_images │  │        GroupImagesWorker │  │        ImageAnalysis      │
│         rerank(R1/R2) │  │  tools: filter_by_meta / │  │  tools: analyze_image     │
│  out: RetrievalOutput │  │         group_images      │  │  out: VisionTaskOutput   │
└──────────┬────────────┘  └──────────┬───────────────┘  └──────────┬───────────────┘
           │ 汇报（AgentMessage：evidence + confidence + failures + next）
           └──────────────────────► AgentContext / RunState ◄────────────────┘
                                    │（= ExecutionContext 演进：计划+结果+trace）
                                    ▼
                 ┌───────────────────────────────────────────────┐
                 │  Tool Layer（新增 ToolRegistry + 统一 schema）  │
                 │  retrieve / filter / group / analyze / embed   │
                 │   ← in-process 工具（现 repositories/ImageAnalysis）
                 │   ← 可选 MCP adapter（M4，能力外化）            │
                 └───────────────────────────────────────────────┘
                 ▼
        Persistence：PostgreSQL + pgvector（images / vectors / …）+ conversation_messages
                    + agent_episodes（Memory）+ agent_runs（M3+，additive）
```

**关键点：图中没有任何“Agent 自由聊天”。** Specialist 之间不直接互发消息；一切信息交换经 Supervisor 派发的 typed Task 与回传的 typed WorkerResult（信封化）完成。这最大化复用现有 Executor 的单向数据流，同时把决策权集中到 Supervisor。

### 3.2 设计原则与边界

1. **不推翻**：`TaskType` 枚举、Typed Input、`TaskPlanDraft/TaskPlan`、`PlanValidator`、`DAGExecutor`、`default_workers()` 全部保留，作为“原子执行模型”。
2. **加层不改层**：Multi-Agent 语义作为其上的编排壳，Worker 代码尽量零改动。
3. **协议最小化**：只引入“信封 + 角色描述 + 复盘决策”三类新结构，不做通用 Agent 通信平台。
4. **约束不放松**：Agent 化后所有资源预算（任务数/深度/执行次数/视觉候选/evidence 预算）继续由现有常量与 Validator 强制。
5. **向后兼容**：SSE 事件流对前端不变；`chat_recommend` / `stream_core` 公共入口签名不变。
6. **渐进可用**：新能力默认关闭，feature flag 开启；默认仍走“Gate→Team（一次计划）→合成”，行为等价现状。

### 3.3 模块归属总表（新增 / 重构 / 复用）

**复用（不迁移、零改动或仅 import）**
- `orchestration/planner.py`、`models.py`（Typed 模型 + Validator + 常量）、`executor.py`、`workers.py`、`vision.py`、`synthesizer.py`、`gate.py`、`intent.py`、`memory.py`
- `recommendation/agent.py`（Legacy 路径保留）、`tools.py`（工具实现本体）
- `repositories/*`、`ai_func/image_analysis.py`、`ai_func/generate_vector.py`

**重构（小范围、行为不变）**
- `recommendation/__init__.py` 的 `stream_core()`：从“Gate 分流逻辑”改为“构造 SupervisorAgent 并调用”，Legacy 仍作 fallback；
- `orchestration/orchestrator.py` `AdaptiveTeamOrchestrator`：收敛为 Supervisor 的“默认执行策略（一次 plan 模式）”；旧类名保留别名兼容现有测试；
- `ai_func/recommendation/tools.py`：模块 → 包 `tools/`（`registry.py`+`impl.py`+`schema.py`），`tool_search_images` 函数体不变；若改动面过大可保留文件，另建 `backend/ai_func/toolkit/`（二选一，见 §3.4.4）。

**新增**
- `orchestration/agent_cards.py` —— Agent Card（角色元数据）模型 + 注册表；
- `orchestration/messages.py` —— AgentMessage / 事件信封；
- `orchestration/supervisor.py` —— `SupervisorAgent`（决策循环 + review/re-plan/abort）；
- `orchestration/runtime.py` —— `RunState` / 执行内核（Executor 事件、trace、budget 记账串联）；
- `backend/ai_func/toolkit/`（或 `tools/registry.py`）—— ToolRegistry；
- `backend/db_func/repositories/agent_runs.py`（可选，M3）—— agent/task 级运行轨迹表；
- 测试：`backend/tests/agent_team/test_agent_cards.py`、`test_messages.py`、`test_supervisor.py`、`test_tool_registry.py`、`backend/tests/agent_team/eval/*`（M5 Judge）。

### 3.4 关键设计

#### 3.4.1 Agent 单元：Agent Card + Specialist Adapter

不把 Worker 重写成“自带 LLM 循环的 Agent”，而是给每个 Worker 发一张“名片”并保持其执行器身份：

```python
# 草案（illustrative，非生产代码）
class AgentCard(BaseModel):
    agent_id: str                    # "retrieval" / "organizer" / "vision"
    display_name: str
    task_types: List[TaskType]       # 该 agent 能接的任务类型
    tools: List[str]                 # 拥有/允许调用的工具名（来自 ToolRegistry）
    output_schema: str               # 汇报模型名（RetrievalOutput 等）
    cost_tier: Literal["cheap", "medium", "expensive"]
    enabled: bool = True             # vision_available 等门禁映射到此处


class SpecialistAgent:
    """Adapter：把既有 Worker 装成可被 Supervisor 指派的 Specialist。"""
    def __init__(self, card: AgentCard, worker: Worker): ...
    def handle(self, task: Task, state: RunState) -> AgentReport:
        result = self._worker.run(task, state.context)   # 完全复用原逻辑
        return AgentReport.from_result(task.task_id, result)
```

- `Worker.run()` 契约原封不动 → `test_workers.py` / `test_dag_propagation.py` 继续全绿；
- “能力目录”升级为“Agent 目录”：`registered_task_types()` 依旧作为单一来源；`default_workers()` 继续按 TaskType 分派；新加一张 `agent_cards` 视图把 TaskType 归组到 Agent，供 Supervisor 与观察层使用。

#### 3.4.2 执行与编排：两种模式（默认保留 DAG 模式）

- **模式 A「DAG-Execute（默认，= 今天的行为）」**：Supervisor 一次 plan → Validator → DAGExecutor 串行执行 → SynthesizerAgent 汇总。行为与 `AdaptiveTeamOrchestrator` 完全一致。
- **模式 B「Supervised Re-run（受限，flag `SIF_SUPERVISOR_LOOP=1` 开启）」**：计划执行完后，若满足确定性重试条件（某 RETRIEVE 任务失败、候选为空、synthesize 前候选集过小），Supervisor 生成**修订输入**（改 query/limit/filters），只对受影响子图重跑，最多 `MAX_SUPERVISOR_ITERATIONS`（建议 2）轮。重跑计划仍过 PlanValidator，累计执行数仍受 `MAX_EXECUTED_TASKS` 约束。
- 模式 B 的 review 决策第一阶段用**确定性触发器**（空结果 / 失败码 / 覆盖率阈值），不引入第二个 LLM 决策调用；只有验证确有增益后才考虑 LLM 复盘（放入 M5）。

#### 3.4.3 通信契约草案（信封 + 事件映射）

```python
# 草案
class AgentMessage(BaseModel):
    msg_id: str
    request_id: str
    sender: Literal["user", "supervisor", "retrieval", "organizer", "vision", "synthesizer"]
    kind: Literal["task", "report", "decision", "event"]
    payload: Any          # Task / WorkerResult / SupervisorDecision / 结构化事件
    created_at: str
```

与现有 SSE 的映射（不加新事件也能工作；新增事件可渐进）：
- Supervisor 派发 Task → 现有 `task.started`；
- Specialist 汇报 → 现有 `task.completed`（output_summary 不变，payload 附带 evidence/confidence 摘要）；
- Supervisor 复盘/修订 → 新增 `agent.review {decision: continue|revise|replan|abort, reason}`（旧前端对未知事件静默忽略，天然兼容）；
- 最终回答 → `assistant_delta` + `selection`（不变）。

#### 3.4.4 Tool 抽象与 Registry（M1 先做，成本最低）

目标：**一处声明，多端消费**（Legacy `agent.py`、Planner schema 生成、将来的 Specialist / MCP）。

```python
# 草案
class ToolDef(BaseModel):
    name: str                  # "retrieve_images"
    description: str
    parameters: Dict[str, Any] # JSON Schema（复用 OpenAI function format）
    impl: Callable
    cost_tier: str = "cheap"
    enabled: bool = True

class ToolRegistry:
    def register(self, tool: ToolDef): ...
    def openai_functions(self, names: List[str]) -> List[dict]: ...  # 喂 chat.completions
    def call(self, name: str, arguments: dict) -> Any: ...           # 校验+计量+审计
```

- 首个登记工具即 `retrieve_images`（`tool_search_images` 包装，参数名统一 `query/vector_targets/filters/limit`）；
- `agent.py` 的 inline tool 声明改为从 registry 生成（删除手写重复 schema）；
- `submit_task_plan` 属系统级 schema，可登记到同一 registry 的 `kind="system"` 区。

#### 3.4.5 State 定位（一次 run 一个 RunState）

```python
# 草案：在 ExecutionContext 之上加薄壳，不改其内部
class RunState:
    request_id: str
    conversation_id: Optional[str]
    messages: List[dict]               # 输入（引用，不拷贝大文本）
    plan: Optional[TaskPlan] = None
    context: ExecutionContext          # 复用现有 results
    trace: List[AgentMessage]          # 运行轨迹（内存；可选落库 agent_runs）
    budgets: Dict[str, int]            # 用 models.py 常量初始化
```

- `_RecommendationService` 由共享单例改为 **run 级工厂**：每次请求 new 一个 Supervisor 实例，消除 §1.3-5 的隐式共享；
- 跨运行状态继续使用 conversation_messages / request_sessions / agent_episodes，语义不变。

#### 3.4.6 事件 / SSE 兼容红线

- 保留 `TEAM_EVENT_NAMES` 原样透传；新事件一律“未知即忽略”兼容；
- `assistant_delta` / `selection` / `complete` 结构不变；
- 所有新增字段（如 agent.review）只增不改。

---

## 4. 分阶段迁移路径（M0 → M5）

| 阶段 | 目标 | 动作（复用 / 新增） | 验收标准 | 风险 |
|------|------|---------------------|----------|------|
| **M0** | 基线冻结 | 无 | 现有 agent_team 测试全绿；planner/routing benchmark 可复跑 | - |
| **M1** | Tool Registry 化（纯重构） | 复用：`tools.py`/`repositories/*`/`agent.py` 逻辑；新增：ToolRegistry + ToolDef；重构：inline function 声明改由 registry 生成 | 行为不变：Legacy 与 Team 事件流 diff 为空（或仅顺序）；全测试绿 | 低。参数名对齐（targets→vector_targets）需一次串改 + 回归 |
| **M2** | Worker→Specialist（加壳） | 复用：workers/vision/rerank/synthesizer 零改动；新增：agent_cards.py + messages.py + SpecialistAgent adapter；测试新增 | `registered_task_types` 与 `agent_cards` 同源断言通过；Executor 事件流不变 | 低。纯加层，不改执行路径 |
| **M3** | Supervisor 最小回路 | 复用：planner/validator/executor/synthesizer/gate/intent；重构：`_RecommendationService.stream_core` → 构造 SupervisorAgent；`AdaptiveTeamOrchestrator` 收敛为其默认策略（保留别名）；新增：supervisor.py + runtime.py（RunState）+ 可选 agent_runs 表 | flag 关闭时行为 == M2（回归）；开启时（确定性触发器）仅对空结果/失败码触发受限修订；事件含 agent.review | 中。**默认关闭**，回路轮数硬上限 + 沿用 MAX_EXECUTED_TASKS 防失控 |
| **M4** | （可选）MCP 外化 | 复用：ToolRegistry；新增：`backend/ai_func/mcp_bridge.py`（把 registry 工具导出为 MCP server 或经 MCP client 调用外部工具） | 由 ToolRegistry 自动生成 tool 列表；in-process 调用与 MCP 调用结果一致（同构测试） | 中。引入外部传输/生命周期；若无需外化可跳过本阶段 |
| **M5** | Memory + Judge 增强 | 复用：memory.py/episode repo + benchmark harness + 现有向量底座（PostgreSQL + pgvector）；新增：user-preference 记忆（flag）、eval/judge 合约与 runner | Judge 报告在既有 routing/planner 数据集上稳定复现 + 新增 answer 质量维度；Memory 不写脏数据（脱敏 + 预算 + flag） | 中。Judge 用 token 成本；需小样本校准 |

**关键排序理由**：M1/M2 成本极低且为后续一切提供“统一工具视图”与“统一角色视图”；M3 是唯一动控制流的阶段，因此默认关闭 + 硬上限；M4/M5 属可选增强，与主线解耦。

---

## 5. 候选技术方案评估

> 评估口径统一为：**收益 vs 代价 vs 本项目适配度 vs 结论 vs 最小引入方案**。判断标准：不堆技术名词，只有“能为当前缺口提供结构性收益且代价可控”才引入。

### 5.1 MCP Tool Protocol（Model Context Protocol）

**收益**
- 强制“工具即一等公民”：name / description / inputSchema / 输出结构化，正好解决审计 §1.3-4 的“工具描述与注册不统一”；
- 能力与实现解耦：将来可把 `retrieve / analyze` 放进独立 Tool Server 进程 / 另一台机器，不碰 Agent Runtime；
- 生态互操作：若未来让外部 Agent（或本系统其他服务）调用 SmartImager 检索能力，MCP 是现成标准；
- 社区工具可即插即用——若产品路线引入外部信息源（网图 / 版权库 / 地图地点），收益递增。

**代价**
- 引入 MCP Python SDK 与传输生命周期（stdio / streamable-http）管理，涉及鉴权、进程 / 连接治理；
- 本项目是单进程一体化 FastAPI，能力与数据同库——外部化当前**没有实际需求方**，纯为“将来”买单；
- MCP 的 tool 描述格式与 OpenAI function 格式存在差异，两层 schema 需映射与维护。

**适配度 / 结论**
- **形态值得立刻学（M1 的 ToolDef/ToolRegistry 即“MCP 风格”内部版）**；**协议本体建议 M4 才接入**，触发条件明确：
  (1) 出现第二个消费方（另一 Agent 系统 / 另一服务进程）；或
  (2) 需要把一个能力独立部署 / 扩容。
- 在此之前，registry 预留 `to_mcp_tool()` 序列化钩子即可，不引入依赖。

**最小引入方案（M4）**：`mcp_bridge.py` 将 registry 工具导出为一个最小 MCP Server（仅 2-3 个工具起步），自测 client/server 同构调用；不做鉴权平台化。

### 5.2 LangGraph StateGraph

**收益**
- 对“复杂控制流”场景（条件分支、循环、human-in-the-loop、checkpointer、跨会话恢复、并行 fan-out/fan-in、可视化）提供框架级支持；
- 工作区内 DB-GPT 工程已在 `langgraph_agent` 目录验证过同类用法（有先例可参考）。

**代价**
- **与现有模型体系冲突 / 重叠**：
  - 本项目控制流本质是“**一次规划出 DAG，然后按依赖执行**”，DAG 已显式存在（TaskPlan）；LangGraph 的强项是“**运行中动态构图**”，二者重叠又不同构；
  - 现有状态是 Pydantic 强类型（TaskPlan / Typed Input / WorkerResult），LangGraph 状态是 dict + reducer 语义，迁移意味着把类型擦除再重建；
  - Planner / Validator / Executor / 事件契约是大量测试与 SSE 协议的地基，包进 StateGraph 需重写编排与事件路径；
- 引入一整套新依赖 + checkpointer（本项目 PostgreSQL 已自管会话/记忆，checkpointer 收益低）。

**适配度 / 结论**
- **不建议整体引入。** 当前缺口（supervisor 复盘/修订）只需要一个“最多 2 轮、确定性触发”的小状态机——自研约 150 行可覆盖，且保持 Typed 与事件契约；
- 若未来出现以下任一信号，再评估局部引入或借鉴其 state/reducer 思想：
  - 多轮“用户可见”的澄清式交互（human-in-the-loop）成为主线；
  - 需要跨请求断点续跑任务；
  - 计划图需在执行中被用户/外部事件动态改写。

**最小引入方案（若采纳）**：仅在 `supervisor.py` 内部实现“plan / execute / review / revise / abort”五态小状态机；不引依赖、不加 dict state、不加 checkpointer。

### 5.3 Agent Memory（长期记忆增强）

**收益**
- 现有 Episodic Memory 已覆盖“**该会话之前怎么处理过类似请求**”（策略回忆）；
- 缺口是：**用户画像 / 偏好级记忆**（用户常说“风景 / 横版 / 无人物 / 按地点整理”），跨会话复用可显著提升 Organizer / Retrieval 的参数质量与合成回答的个性化；
- 本项目已有 PostgreSQL + pgvector + 统一 embedding，可低成本实现向量化偏好记忆（复用 memory.py 的脱敏与预算纪律）。

**代价**
- 引入 LLM 摘要（把对话/选择历史提炼为偏好）会增加写路径的 LLM 成本与隐私面；
- 记忆质量差会反向污染推荐（偏好过期 / 错误推断），需要版本与回收策略；
- 评估“记忆增益”需长会话实验，短期 ROI 不明显。

**适配度 / 结论**
- **分层处理：**
  - episodic（现有）——保留，不动；
  - **user-preference**——建议做，但以 flag + 确定性规则先行：从成功 episode 的 `preference_hints`（filter/group/vision/group_by 等已结构化字段）统计“该用户高频策略”，**先不引入 LLM 摘要**；仅在统计信号不足时才考虑 LLM 提炼（M5 后半）。

**最小引入方案**：新表 `user_preferences(conversation_id/user_id, strategy, weight, window)`，写路径复用 `build_episode_from_context` 的成功样本做加权聚合；读路径注入 Planner 的 memory_hints（已有插槽），复用现有 bounded 预算。

### 5.4 LLM-as-a-Judge（生成质量评估）

**收益**
- 现有确定性评估（Gate/Planner/Rerank/Vision 合约）只能回答“**行为对不对**”，不能回答“**回答好不好 / 有没有达成用户目标**”；
- Judge 可统一度量：答案诚实性（只引用候选内事实、不虚构）、推荐与目标一致性、视觉证据引用正确性、输出格式规范、过程效率（是否过度使用昂贵 vision）；
- 与现有 harness 天然互补：同一批 gold queries 先跑确定性层，再跑 Judge 层，形成两层报告。

**代价**
- 每次评估增加 LLM 调用成本（采样批评估可控）；
- Judge 本身有偏差/幻觉，需要：结构化 rubric + 明确“证据可判定”口径 + 小样本人工校准；单一 Judge 打分不稳定时应上 2-Judge + 差异仲裁（成本翻倍）。

**适配度 / 结论**
- **值得引入，但定位为离线评估增强（不阻塞在线路径）**；首期聚焦两个维度（诚实性、一致性），跑在既有 benchmark 数据子集上；
- 不引入“在线 Judge 控制 Supervisor 行为”——在线仍以确定性触发器为准（见 §3.4.2）。

**最小引入方案（合约草案）**
- 输入：`{goal, query, plan, task_results(摘要), synthesis_output, selection_ids, judge_criteria}`
- 输出：`{score_per_criterion: 0-5, verdict: pass|flag|fail, reasons[]（每条引证据）}`
- 存放：`backend/tests/agent_team/eval/judge_contract.py` + `judge_runner.py`，报告 JSON 与既有 planner_benchmark_v2_results.json 同目录。

### 5.5 决策矩阵汇总

| 候选技术 | 解决的本项目缺口 | 收益 | 代价 | 建议 |
|----------|------------------|------|------|------|
| MCP Tool Protocol | 工具描述/注册不统一；能力外部化 | 高（形态）+ 中（协议，未来） | 中 | **M1 学其形态；M4 视外部化需求接入协议** |
| LangGraph StateGraph | 动态复杂控制流 | 低-中（当前控制流已足够简单） | 高（重写 + 类型冲突） | **不引入**；自研五态小状态机 |
| Agent Memory | 跨会话偏好缺失 | 中-高 | 中（LLM 写路径/隐私） | **分层引入**：episodic 保留 + user-preference 确定性优先 |
| LLM-as-a-Judge | 质量/达成度评估缺失 | 高（离线） | 中（token/校准） | **引入**，离线增强，不阻塞在线 |

---

## 6. 风险与治理（Non-Goals 与回滚策略）

1. **默认关闭原则**：M3+ 一切新控制流以 flag 关闭为默认；版本回归 = flag 置 0 即恢复 M2 语义。
2. **预算硬上限**：任何新增回路 / 记忆 / Judge 都必须注册到 `models.py` 常量体系（如 MAX_SUPERVISOR_ITERATIONS、MEMORY_WRITE_CAP、JUDGE_MAX_ITEMS…），禁止散落 magic number。
3. **协议演化纪律**：AgentMessage / SSE 事件遵循“只增字段、不改语义、未知事件忽略”。
4. **明确 Non-Goals**（延续既有 Non-Goals 清单）：不做 Agent 自建 Agent、不做 Specialist 间自由对话、不做无限自主循环、不引入第二套状态/事务系统替代现有 PostgreSQL 持久层。
5. **回滚路径**：每个里程碑独立 PR；M1/M2 无行为差异，M3 由 flag 控制，M4/M5 完全可选，可整体跳过。

---

## 附录 A 源码位置索引（以代码基准为根）

| 组件 | 路径 |
|------|------|
| HTTP/SSE 层 | `backend/routers/ai/recommendation.py` |
| 服务入口与 Gate 分流 | `backend/ai_func/recommendation/__init__.py` |
| Legacy 对话 Agent | `backend/ai_func/recommendation/agent.py` |
| 工具层（现状） | `backend/ai_func/recommendation/tools.py` |
| Prompt 模板 | `backend/ai_func/recommendation/prompts.py` |
| Typed 模型 / Validator / 常量 | `backend/ai_func/recommendation/orchestration/models.py` |
| Gate | `orchestration/gate.py` |
| Intent Rewrite | `orchestration/intent.py` |
| Planner | `orchestration/planner.py` |
| Workers / WorkerRegistry | `orchestration/workers.py` |
| Executor | `orchestration/executor.py` |
| Vision 契约 | `orchestration/vision.py` |
| Synthesizer | `orchestration/synthesizer.py` |
| Orchestrator | `orchestration/orchestrator.py` |
| Episodic Memory | `orchestration/memory.py` |
| Rerank R1 / R2 | `orchestration/rerank.py`、`model_reranker.py` |
| Vision provider | `backend/ai_func/image_analysis.py` |
| Embedding | `backend/ai_func/generate_vector.py` |
| 检索数据面 | `backend/db_func/repositories/search.py`（unified_search） |
| 记忆/会话/审计持久化 | `repositories/agent_episodes.py`、`conversation_messages.py`、`request_sessions.py` |
| 数据库 DDL | `backend/db_func/core/database.py` |
| 配置 | `backend/config/config.py`（AppConfig：MODEL_PATH / CHAT_MODEL / VISION_MODEL 等） |
| 测试与 benchmark | `backend/tests/agent_team/` |

## 附录 B 术语对照

| 术语 | 含义（本文语境） |
|------|------------------|
| Worker | 现有按 TaskType 注册的原子执行单元（run(task, context)） |
| Specialist Agent | 带 Agent Card 的角色化 Worker（Retrieval / Organizer / Vision） |
| Supervisor Agent | 拥有“规划 + 派发 + 复盘决策”的执行编排者（复用 Planner / Validator / Executor） |
| AgentMessage | Supervisor ↔ Specialist 间 typed 消息信封 |
| RunState | 单次 run 的状态壳（含 ExecutionContext + trace + budgets） |
| ToolRegistry | 工具单点注册 / 校验 / 计量 / 生成 schema 的统一入口（MCP 风格） |
| Agent Card | 角色元数据：agent_id / task_types / tools / output_schema / cost / enabled |










