# SMARTIMAGER_PROJECT_LEARNING_GUIDE

> 目的：帮助项目作者真正理解「现在的 SmartImager 是怎么工作的、为什么这样设计、面试怎么讲」。
> 一切以当前实际代码为准（已扫描 `backend/ai_func/recommendation/orchestration/*`、`routers`、`db_func`、`config`、`frontend/src`、`backend/tests/agent_team` 及审计/设计文档）。
> 已做「真实实现 / 测试实现 / Mock·Fake / 设计目标 / 当前限制」区分。README 与代码不一致处以代码为准，并已标注。
> **2026-09-10 更新（重要）**：数据层已由 SQLite / sqlite-vec 迁移至 **PostgreSQL + pgvector（HNSW / cosine）**，图库由 66 张扩至约 **1000 张 VOC2010**；`backend/tests/agent_team` 门禁 **463 passed**。本文据此更新了 §2 / §3 / §10 / §16 / §17 / §18 / §20 的技术栈描述。

---

## 1. 项目定位

**一句话**：一个"图片库管理 + 语义检索 + AI 对话式推荐"的一体化 Web 应用（FastAPI + React），其中复杂检索请求由一个 **Typed Agent Team** 编排执行。

**它解决什么问题**：
- 普通检索：文本 / 以图搜图 / 向量检索 / 模糊搜索（`SearchRepository.unified_search`，CLIP + PostgreSQL/pgvector）。
- 复杂对话式请求：例如"找旅行照片，按地点分组，再推荐最合适的几张"——不是一条 SQL 能表达的，需要 **先检索 → 再分组 → 再合成回答** 的多步任务。
- 像素级核验：CLIP 只能做"语义相似"，无法回答"画面里到底有没有生日蛋糕 / 构图是否对称"——这是 Vision 链的存在理由。

**用户可以做什么**（README + 代码交叉验证）：
- 图片 CRUD / 标签 / 元数据编辑 / 批量上传与 AI 自动分析（独立于 Agent Team 的功能域）。
- 多轮对话式推荐（`/recommend/chat/stream`），SSE 流式输出。
- 会话管理（创建 / 列出 / 删除会话）。

**技术栈**（以 `config.yaml` + 代码为准）：
- 后端 FastAPI + **PostgreSQL + pgvector**（三张 `vector(512)` 表 + `HNSW (vector_cosine_ops)` 索引；`docker-compose.yml` 起 `pgvector/pgvector:pg16`）；向量缓存 diskcache。
- 编码模型：**CLIP——文本 `clip-ViT-B-32-multilingual-v1`（512 维）+ 图像 `openai/clip-vit-base-patch32`（512 维）**（README 于 2026-09-10 已由"Jina v4 2048 维"更正，见 §10）。
- 文本 LLM：**DeepSeek `deepseek-chat`**（`OPENAI_API_BASE=https://api.deepseek.com`）。
- Vision LLM：**默认关闭**（`VISION_MODEL="null"`）；E2E 用 runtime override。
- 前端 React 18 + TypeScript + AntD 5 + Vite；Axios 调用 API，fetch 消费 SSE。

**当前最终架构类型**（§9 详述）：**Typed Agent Team / Agentic Workflow**——不是 Fully Autonomous Multi-Agent，不是 Agent Infra Platform。

---

## 2. 技术栈速览

| 层 | 技术 | 代码证据 |
|---|---|---|
| API | FastAPI Router | `backend/routers/ai/recommendation.py` |
| 编排 | **TEAM 默认控制层 = `SupervisorAgent(enhanced=True, backend="langgraph")`** → 自研 `DAGExecutor`（串行有界 DAG）；显式关闭 flag 时回退 `AdaptiveTeamOrchestrator` | `orchestration/supervisor.py`, `supervisor_graph/`, `orchestrator.py`, `executor.py` |
| 规划 | LLM Planner，强制 function-call schema | `orchestration/planner.py` |
| 校验 | Pydantic + 确定性 DAG 规则 | `orchestration/models.py`（PlanValidator） |
| 检索 | CLIP（512d）+ pgvector KNN（SQL `<=>` + HNSW(cosine)，元数据同库过滤） | `ai_func/generate_vector.py`, `db_func/repositories/search.py`, `db_func/core/database.py` |
| 视觉 | OpenAI-compatible Vision API（per-image） | `ai_func/image_analysis.py`, `workers.py`（VisionWorker） |
| 合成 | LLM Synthesizer + deterministic fallback | `orchestration/synthesizer.py` |
| Memory | PostgreSQL `agent_episodes`（episodic）+ `ExecutionContext`（working） | `db_func/repositories/agent_episodes.py`, `orchestration/memory.py` |
| SSE | `text/event-stream` | router + orchestrator |
| 前端 | React 18 / AntD 5 / fetch reader | `frontend/src/services/chatService.ts` |

---

## 3. 总体架构（当前真实实现）

```text
                    FastAPI Router (/api/v1/ai/recommend/chat/stream)
                                  │ conversation_id + messages(≤40)
                                  ▼
                    _RecommendationService.stream_core
                                  │
                    ComplexityGate.decide(messages)   ← 正则/启发式，不调 LLM
                    ┌──────────────┴───────────────┐
                 LEGACY                            TEAM
                    ▼                               ▼
            RecommendAgent                AdaptiveTeamOrchestrator.run_stream
            (Legacy Fast Path)             0) Memory.recall (若有 conversation_id)
            2-stage function-call          1) PlannerAgent (LLM×1, submit_task_plan)
            LLM×2 (search→choose)          2) PlanValidator (确定性)
                                           3) DAGExecutor (串行) → ExecutionContext
                                              ├ RetrievalWorker (CLIP + pgvector)
                                              ├ MetadataWorker (确定性过滤)
                                              ├ GroupImagesWorker (确定性分组)
                                              ├ VisionWorker (per-image LLM, 若启用)
                                              └ SynthesizerAgent (LLM×1, 失败→deterministic)
                                           4) Memory.write (仅成功 Team run)
                                            plan.failed → 回退 Legacy
                    └──────────────┴───────────────┘
                                  ▼
                    SSE: rewrite_start → (plan.*/task.* 透传) → assistant_delta → complete
```

关键点（代码事实）：
- **Gate 是确定性路由**：简单查询 → Legacy；高置信复合特征 → Team（`gate.py`，正则匹配，默认保守 Legacy）。
- **Executor 是串行的**：`ready.pop(0)`（`executor.py:80`），不是并行多 Agent。
- **真正的 LLM 节点只有 3 类**：Planner（1 次）、Synthesizer（1 次）、Vision（逐图）。Legacy Agent 是 2 次 function-call 的独立旧路径。
- **Workers 之间不通信**，只通过 `ExecutionContext.results` 读写（`workers.py:9-10`）。
- **检索底座是共享的**：Legacy 与 Team 共用 `tool_search_images → SearchRepository`。
- **TEAM 默认控制层是 Supervisor（默认开启，别答成"默认关闭"）**：`recommendation/__init__.py:38-45` — `ENABLE_SUPERVISOR_RUNTIME` 未设置 → True；走 `SupervisorAgent(enhanced=True, backend="langgraph")`。LangGraph 只承载 Supervisor 状态机（`execute → review → review_router → revise/finalize`），节点内部复用同一套 `DAGExecutor._execute` / `review_execution` / `revise_plan` / `finalize_output`；依赖缺失或显式关 flag 时回退 `AdaptiveTeamOrchestrator`（`enhanced=False` 为基线等价模式）。

---

## 4. 完整请求链路（从用户输入到前端）

1. **前端** `useChatRecommendation.start(query)` → `chatService.startChatStream` → `fetch POST /api/v1/ai/recommend/chat/stream`，携带 `query`、`conversation_id`（无则前端自生成 `conv_*`），用 `ReadableStream` 按 `\n\n` 切帧。
2. **Router** `ai_chat_recommend_stream`（`routers/ai/recommendation.py`）：
   - 用 `conversation_id` 拉最近 40 条 `conversation_messages` 拼成 history；
   - 写 `request_sessions` 审计、写本次 user 消息；
   - 组装 `final_messages = history + [user]`，调用 `recommendation_service.stream_core(messages, conversation_id=conv_id)`；
   - 产出 `rewrite_start`，把 Team 事件透传（`plan.created`/`task.*`），把 `selection` 转为 `complete`，写回 assistant 消息。
3. **Service** `stream_core`（`recommendation/__init__.py`）：`ComplexityGate.decide(messages)`。
   - TEAM → `AdaptiveTeamOrchestrator.run_stream(messages, conversation_id, memory_service)`；若 `last_plan_ok=False`（planner/validator 失败）→ 回退 `RecommendAgent.run_stream`（Legacy）。
   - LEGACY → 直接 `RecommendAgent.run_stream`（2 次 LLM：先 `search_images` tool call，再 `choose_images` tool call；64K 字符窗口裁剪历史）。
4. **Orchestrator**（`orchestrator.py`）：
   - **0)** 若有 `conversation_id`+memory_service：`recall_episodes(conversation_id, goal)` → `render_memory_hints`（≤3 条，≤800 字符）→ `memory_hints`。
   - **1)** `PlannerAgent.create_plan(goal, recent_messages=messages, memory_hints=...)` → 一次 LLM 调用 → `TaskPlanDraft`。
   - **2)** `PlanValidator.validate(draft)` → 校验失败 `plan.failed`（触发 Legacy 回退），成功 → 不可变 `TaskPlan`。
   - **3)** `DAGExecutor.execute_stream(plan)`：串行按依赖执行；产出 `task.started/completed/failed/skipped` 事件；把 WorkerResult 写入 `ExecutionContext.results`。
   - **4)** `_build_final(plan, context)`：取 synthesis 输出 → `assistant_delta` + `selection`；随后 `Memory.write`（仅成功 Team run）。
5. **前端** `chatService.ts` 只处理 `rewrite_start / assistant_delta / complete / error`：把 delta 追加到草稿，`complete` 时替换为正式消息并附带 `images_brief`/`image_ids` 展示图片。**`plan.*` / `task.*` 事件被静默忽略**（§15）。

**上下游数据流**：Router 传 `conversation_id`+messages → Service 传 goal+history → Orchestrator 注入 memory hints → Planner 输出 draft JSON → Validator 产出 typed `TaskPlan` → Executor 通过 `ExecutionContext` 传递 `WorkerResult`（结构化 Pydantic 输出）→ Synthesizer 读取 candidates/groups/vision → `validate_selection` 产出最终 `image_ids` → Router 聚合为 `complete`。



---

## 5. Planner（`orchestration/planner.py`）

| 问题 | 实际答案（代码） |
|---|---|
| 输入 | `goal`（最后一条 user query）+ `recent_messages`（≤4 条×200 字符）+ `memory_hints`（可选，≤3 条）+ capability snapshot |
| 输出 | `TaskPlanDraft`（`goal` + `tasks[]` + `synthesis_task_id`），JSON 经 Pydantic 解析 |
| 为什么需要 | 复杂请求需要「多步任务分解」，不能由单一 prompt 直接出最终结果；Planner 产出的是**执行蓝图** |
| 如何约束输出 | 强制 `submit_task_plan` function-call + JSON Schema（`task_type` enum、task 数量 2..6、input 契约描述、additionalProperties:false）+ Pydantic + 后续 Validator |
| 是否自由决定任务 | **否**：只能使用 `enabled_task_types`（capability-aware），且 input 必须符合每类型契约 |
| 支持哪些 TaskType | `retrieve_images` / `filter_metadata` / `group_images` / `analyze_visual`（仅 vision 可用时）/ `synthesize` |
| 能力信息从哪来 | `registered_task_types()`（`models.py`）→ `_default_capabilities()` → 注入 system prompt 的 `{enabled}` / catalog / routing policy |
| 如何决定是否用 Vision | 通过 `TASK_CAPABILITY_CATALOG` 的 use_when/avoid_when + `_CAPABILITY_ROUTING_POLICY`（"颜色/物体/场景/构图等可见事实必须 analyze_visual；不要因为出现'视觉'一词就无条件加入"），而非关键词匹配 |

**Planner 调用细节**：temperature=0、stream=False、`tools=[submit_task_plan]`、`tool_choice` 强制（`planner.py:210-230`）；每次规划**恰好 1 次 LLM 调用**。`memory_hints=None` 时 prompt 与旧版逐字符一致（有回归测试）。

## 6. Typed Task / Task Plan（`orchestration/models.py`）

- **TaskDraft**：Planner 原始输出，`input` 是 raw dict（`extra=forbid`，task_id 正则 `^[a-z][a-z0-9_]{0,31}$`）。
- **Task / TaskPlan**：Validator 校验后产物；`input` 已被类型化为 `RetrieveImagesInput / FilterMetadataInput / GroupImagesInput / AnalyzeVisualInput / SynthesizeInput`（`INPUT_FOR_TASK_TYPE` 映射）；`plan_id` 由服务端生成（不信任 LLM）。
- **依赖表达**：`Task.depends_on: List[str]` 引用本计划内 `task_id` → 构成 DAG（边 = "产出方 → 消费方"）。
- **为什么不用自由文本计划**：自由文本无法做结构校验、无法保证 input 契约、无法静态检测环/孤立节点、无法安全 dispatch。Typed 化后：LLM 只能从有限枚举选类型、input 字段受限（`extra=forbid`）、依赖关系可做图算法校验。

---

## 7. Plan Validator（`models.py` PlanValidator）

**为什么需要**：LLM 一次生成可能结构合法但语义错误（错类型、漏检索、环、孤儿、越界 input）。Validator 在**任何执行之前**确定性拦截。

**实际检查（代码）**：
1. 空计划 / task 数量 <2 或 >6；
2. task_id 重复；
3. task_type 未注册（capability-aware：vision 关闭时 `analyze_visual` 被拒 → `WORKER_UNAVAILABLE`/`VISION_UNAVAILABLE` 语义）；
4. task_id 格式（模型层）；
5. **循环依赖**（图检查）与依赖链深度 ≤ `MAX_PLAN_DEPTH(4)`；
6. 必须至少一个 `retrieve_images`；
7. **恰好一个 synthesize 且是终点**（`SYNTHESIZE_COUNT` / `SYNTHESIS_TARGET` / `SYNTHESIZE_NOT_TERMINAL`）；
8. 无孤儿（每个非 synthesize task 必须能到达 synthesize）；
9. **候选消费型 task（filter/group/analyze_visual）必须（传递）依赖候选生产者**（`NO_PRODUCER`）；
10. input raw dict → Typed Input 强制转换（`INVALID_TASK_INPUT` 拒绝枚举外字段/越界值）。

**能提前阻止的典型错误**：幻觉 task type、无检索兜底、synthesize 不是终点、循环 DAG、input 字段越界（如 `filter_metadata.include_fields` 用枚举外值）——**Validation failed → plan.failed → Legacy fallback，绝不部分执行**。Validator 与 Planner 的边界：Planner 负责"生成"，Validator 负责"保证可安全执行"。

## 8. DAG Executor（`executor.py`）

- **为什么 DAG**：任务间有真实依赖（分组必须等检索、视觉必须等候选、合成必须等全部），DAG 表达"必须先发生什么"。
- **节点/边**：节点 = Task；边 = `depends_on` 引用的生产→消费关系。
- **依赖处理（Kahn 式）**：入度 0 的进 `ready` 队列；执行完解除后继入度。
- **是否并行**：**否——严格串行**（`ready.pop(0)`，单循环逐个执行）。面试必须如实说"串行 DAG"。
- **失败语义**：
  - 依赖全部失败/跳过 → 当前任务 `SKIPPED(DEPENDENCY_UNAVAILABLE)`；
  - **至少一个依赖 SUCCEEDED → 继续执行**（Partial Failure Isolation）；
  - Worker 崩溃 → `FAILED(WORKER_CRASH)`；未注册 Worker → `FAILED(WORKER_UNAVAILABLE)`；
  - **VisionWorker 恒 SUCCEEDED**（失败收敛进 `VisionTaskOutput`），绝不阻断下游 Synthesizer。
- **执行边界**：`MAX_EXECUTED_TASKS=16` 防御上限；`ExecutionContext` 是唯一数据通道（`results: task_id→WorkerResult`）。

---

## 9. Worker / Agent 边界（重要）

| 模块 | 用 LLM? | 自主决策? | 确定性逻辑? | 输入 | 输出 | 通信 |
|---|---|---|---|---|---|---|
| **PlannerAgent** | ✅ 1 次 | 仅选任务类型/依赖（被 schema+validator 约束） | 否 | goal+history+hints+capabilities | TaskPlanDraft | → Validator |
| **RetrievalWorker** | ❌ | ❌ | ✅ 调 CLIP + pgvector 检索 | RetrieveImagesInput | RetrievalOutput(candidates) | 产出进 results |
| **MetadataWorker** | ❌ | ❌ | ✅ DB 元数据过滤 | FilterMetadataInput | MetadataOutput | 消费候选 |
| **GroupImagesWorker** | ❌ | ❌ | ✅ 枚举字段分组 | GroupImagesInput | GroupOutput | 消费候选 |
| **VisionWorker** | ✅ 逐图 | ❌（不 retry/不换模型/不改 DAG） | 失败隔离逻辑 | AnalyzeVisualInput + 候选 | VisionTaskOutput | 消费候选，按 image_id 注入 evidence |
| **SynthesizerAgent** | ✅ 1 次 | ❌（输出被 validate_selection 约束） | 有 deterministic fallback | goal+plan+candidates+groups+vision+failures | SynthesisResult→SynthesisOutput | 消费全部 results |
| **DeterministicSynthesisWorker** | ❌ | ❌ | ✅ score 排序取 top-N | SynthesizeInput | SynthesisOutput | fallback |

**明确结论**：本项目应称为 **Typed Agent Team（Agentic Workflow）**，最准确的通俗说法是 **"LLM 驱动的结构化多步工作流 + 确定性能力模块"**。

理由：只有 Planner/Synthesizer/Vision 三个 LLM 节点，且都是**一次性调用、无 loop、无自主 tool 选择、无 replan**；Retrieval/Metadata/Group 是完全确定性的能力模块；编排是固定流水线（Gate→Plan→Validate→Execute→Synthesize）；Worker 之间不自由通信。**不要对外宣称 "Fully Autonomous Multi-Agent" 或 "Multi-Agent Free Collaboration"**——代码不支持。



---

## 10. CLIP + PostgreSQL/pgvector 检索链路

**向量何时生成**：图片上传/重建时，`encode_image`（`ai_func/generate_vector.py`）用 **`openai/clip-vit-base-patch32`** 提取图像特征（512 维，L2 归一化）；图片的 title/description 文本向量用 **SentenceTransformer(`clip-ViT-B-32-multilingual-v1`)** 生成；写入 `{title,description,image}_vectors` 三张 `vector(512)` 表（`HNSW (vector_cosine_ops)` 索引）。diskcache 缓存文本/图像向量。

**文本 query 检索**：`RetrievalWorker.run` → `tool_search_images` → `SearchRepository.unified_search("text", ...)` → `encode_text(query)`（512 维）→ `_multi_vector_search`：对每个 target 在对应表执行 `SELECT image_id, (embedding <=> %s::vector) AS distance ... ORDER BY embedding <=> %s::vector LIMIT %s`（pgvector KNN，按 cosine distance 升序），`score = 1 - distance`；带 filters 时走 `JOIN images` 在**库侧**同一条 SQL 里过滤（时间/标签/宽高/关键字）；多 target 时按权重加权平均再取 Top-K。

**⚠️ 重要不一致（必须如实）**：
1. **README 旧版曾声称 "Jina Embeddings v4 / 2048 维"**，与代码不符（`config.yaml` 是 `MODEL_PATH: clip-ViT-B-32-multilingual-v1`、`EMBEDDING_DIMENSION: 512`，代码也是 512 维）——**README 已于 2026-09-10 更正**；`models/jinaai/...` 是残留目录，生产代码未使用。
2. **文本编码器与图片编码器不是同一个模型**：文本用 multilingual CLIP，图片用 `openai/clip-vit-base-patch32`。两者同为 512 维但**权重不同**，严格来说"文本-图片跨模态相似度"不在同一语义空间（CLIP 对齐需要同一模型双塔）。这是一个**真实的历史遗留风险**：跨模态检索质量依赖两个模型各自的语义对齐，且没有专用 reranker（当前检索排序就是 CLIP cosine；重排（R1/R2 与 GLM-4V）是可选后处理，非默认链路）。面试被问到"文本/图片是否同一语义空间"时应如实说明此限制。
3. **检索性能（迁库后实测）**：1000 张图、limit 40 下检索均值 **65.0ms**（`r1_eval_pg1000.json`）；迁移前 sqlite-vec / 66 图约 37ms，两个数字口径不同不可混用。

## 11. Vision（`workers.py` VisionWorker + `image_analysis.py` + `vision.py`）

**为什么存在**：CLIP 检索 = 粗粒度语义相似；Vision = **像素级核验**（画面里有没有生日蛋糕、构图是否对称、天空是否阴沉）。两条链职责互补。

**调用链（代码确认）**：候选 → `VisionWorker._collect_valid_candidates`（复用 `collect_candidates`，去重、过滤非法 id）→ 截断 ≤8（默认 4）→ `build_vision_prompt(criteria, goal)` → 串行逐图 `ImageAnalysis.analyze_image`（OpenAI-compatible，per-item timeout 30s，`raise_on_error=True`）→ `parse_vision_response` → `VisionItemResult` → `build_vision_output` 聚合 `VisionTaskOutput` → `collect_candidates/_attach_vision_evidence` 按 image_id 合并进 `ImageCandidate.vision` → `SynthesizerAgent` payload 的 `candidates[].vision` → `validate_selection`。

**输入/输出**：输入 = `AnalyzeVisualInput(criteria/max_candidates/detail)` + 上游候选；图片来源 = 本地 `filepath`（存在时）或 `public_url`，无 base64 持久化。Prompt 只描述图片可见内容、禁止猜 EXIF/地点/身份。输出 = `VisionItemResult`（image_id/title≤60/description≤200/tags≤8/evidence≤300/confidence∈[0,1]，`extra=forbid`）；聚合 `VisionTaskOutput(total/success_count/failed_count/systemic_failed)`；evidence 总预算 `cap_evidence_total` 4000 字符。

**数据边界**：SSE 的 `task.completed.output_summary.vision` **只含计数**（candidate/success/failed/systemic）；`ImageCandidate.filepath` 禁止进 SSE/LLM 输出；Memory 只存 `vision_used` 类统计。

**失败处理**：单图失败 → `ITEM_FAILED`（timeout/provider/parse/unsupported_mime），该图**不注入证据**；全部失败 → `success_count=0`、非 systemic，Synthesizer 仍执行；**systemic 失败**（client 不可用）→ `systemic_failed=True`，同样不阻断下游；**VisionWorker 恒返回 SUCCEEDED**（失败收敛进输出），因此 Synthesizer 永远能运行。

## 12. Capability Routing

- **能力来源**：`registered_task_types()` = 基础 4 类 + `analyze_visual`（仅当 `vision_available()` 为真：`VISION_MODEL` 非空/非 "null" 且 OpenAI client 存在）。Planner 的 `enabled_task_types`、Validator 的 registered 集合、`default_workers()` 三处同源，杜绝 capability drift。
- **TASK_CAPABILITY_CATALOG**：每类型 `purpose / evidence_source / use_when / avoid_when / dependency_rule / cost / input_contract`。同一份数据同时渲染进 **system prompt**（能力语义/路由策略/DAG 模式/few-shot）与 **tool schema 的 input 描述**，避免双份语义漂移。
- **Vision disabled**：`analyze_visual` 从 Planner schema enum 移除、Validator 拒绝含它的计划（`WORKER_UNAVAILABLE`/`VISION_UNAVAILABLE` → plan.failed → Legacy 回退）、`default_workers()` 不注册 VisionWorker——三层一致关闭。
- **Vision enabled**：enum 含 `analyze_visual`、workers 注册；Planner 依 catalog 的 use_when/avoid_when 决定是否使用。
- **为什么不能"看到 visual 关键词就强制 Vision"**：routing policy 明确"不要因为出现'视觉/颜色/图片'一词就无条件加入；能用便宜的检索/元数据满足就不用昂贵能力；颜色/物体/场景/构图等**可见事实**才必须 analyze_visual"。benchmark 证明：12 条 vision-positive 全选、16 条 vision-negative 全不选（recall/precision 100%）。
- **当前限制（不隐藏）**：① **Gate under-routing**——9/12 条 vision-positive query 被 `ComplexityGate` 判为 legacy 而**根本进不了 Planner**（Gate FN=9，详见 §18）；② **Planner valid plan rate 87.5% < 98%**——7/56 无效计划的**任务类型全部正确**，失败集中在 `filter_metadata.include_fields` 枚举外值（input 契约忠实度）+ 1 例 synthesize 自引用漂移（provider 随机性）。

## 13. Synthesizer（`orchestration/synthesizer.py`）

- **职责**：把 DAG 各任务的结构化结果（candidates / groups / vision evidence / failures）汇总为最终自然语言回答 + 推荐图片列表；是整个 Team 的**终点消费者**。
- **输入**（`_user_prompt` JSON payload）：`goal` + `plan_summary`（各任务状态）+ `failures` + `groups` + `candidates`（id/score/title/tags，带 vision 时含 title/description/tags/evidence/confidence）+ `recommendation_limit`。**不含** history / filepath / raw provider response。
- **selection 与校验**：LLM 输出 `SynthesisResult(answer, image_ids, rationale)` → `validate_selection(image_ids, candidates, limit)`：剔除不在候选集内的非法 id、去重、保持 LLM 顺序、截断到 limit——**LLM 不能凭空推荐不存在的图片**。

- **fallback**：LLM 失败（超时/非法 JSON/空合法 selection）→ `_degraded` 复用 `DeterministicSynthesisWorker`（按 score 降序取 top-N，`synthesis_degraded=True`）；空候选时**不浪费 LLM 调用**直接走 deterministic。Orchestrator 的 `_build_final` 也兜底：无 LLM answer 时输出确定性摘要文本。

---

## 14. Reliability / Agent Engineering（"为什么不能只让 LLM 自由输出"）

| 机制 | 原本可能出什么问题 | 当前设计如何解决（代码） |
|---|---|---|
| Typed output | LLM 返回自由文本/错误字段 | 每个节点 Pydantic `extra=forbid` + 字段上限（`models.py`/`vision.py`/`synthesizer.py`） |
| Plan validation | 非法/危险计划被执行 | `PlanValidator` 在执行前做 10 类确定性检查；失败 → plan.failed → Legacy |
| Capability constraints | Planner 用未实现/未启用能力 | `registered_task_types()` 同源收窄 schema enum + validator + workers |
| Selection validation | LLM 推荐不存在的图片 | `validate_selection` 只保留候选集内 id |
| Partial failure isolation | 一个 Worker 失败拖垮整链 | 依赖至少一个 SUCCEEDED 即继续；SKIPPED/FAILED 语义化 |
| Deterministic fallback | LLM 超时/坏输出无结果 | Synthesizer → DeterministicSynthesisWorker；Orchestrator `_build_final` 摘要文本 |
| Execution bounds | 无限任务/深度/候选 | `MAX_TASKS=6`、`MAX_PLAN_DEPTH=4`、`MAX_EXECUTED_TASKS=16`、limit≤40、vision≤8、synthesis≤20 |
| Error handling | 未分类异常泄漏细节 | `ImageAnalysisError` 机器可读 code + safe_message；Vision 错误码收敛 |
| Event/SSE | 敏感信息进前端/日志 | `_output_summary` 只出计数；SSE 测试断言无 filepath/base64/key |
| Security sanitization | Memory 存密钥/路径/证据 | `_FORBIDDEN_PATTERNS` 清洗；不持久化 VisionEvidence 全文 |
| Legacy isolation | 改造破坏旧路径 | `agent.py` 无 orchestration 依赖；Team 失败仅 plan.failed 才回退 |

**核心结论**：LLM 只被允许做两件事——"选类型+排依赖"（Planner，且被 schema/validator 双向约束）和"选图片+写回答"（Synthesizer，且被 validate_selection 约束）。其余全部确定性代码兜底。这就是"Agent 工程"与"裸调 LLM"的区别。

---

## 15. SSE 与前端交互

**后端事件（`routers/ai/recommendation.py` + orchestrator/executor）**：
`rewrite_start`（开始）→ TEAM 时透传 `plan.created` / `plan.failed` / `task.started|completed|failed|skipped` → `assistant_delta`（可多次）→ `complete`（由 `selection` 聚合：image_ids / images_brief / assistant_text / total_found）→ `error`（异常）。事件在各自阶段产生：plan 校验通过/失败、每个 task 起止、最终合成。

**前端真实消费（`frontend/src/services/chatService.ts` + `useChatRecommendation.ts`）**：**只处理 `rewrite_start / assistant_delta / complete / error`**。`complete` 前以草稿方式实时拼接 delta；`complete` 时替换为正式消息并展示图片。

**⚠️ 前后端能力不一致（如实记录）**：后端发出的 `plan.created` / `task.*` 等 **Team 执行 trace 事件，前端没有对应 handler，被静默忽略**。即：用户在前端看不到"规划了哪些任务、哪个 Worker 在跑"——只能看到回答文本和最终图片。README 说"前端可基于事件类型实时渲染 AI 思考中/追加回答/展示结果"，这个只对 legacy 事件成立。**面试时若被问"前端是否展示 Agent 过程"，答案是：目前不展示**（后端已透传，前端未消费）。

---

## 16. Memory（`orchestration/memory.py` + `agent_episodes.py`）

### Working Memory
- **存在哪里**：`ExecutionContext`（`workers.py`）——一次请求内 `plan` + `goal` + `results{task_id→WorkerResult}` + `metadata{}`。
- **生命周期**：单次 `DAGExecutor.execute_stream` 创建，请求结束即丢弃；**不跨请求缓存**。

### Episodic Memory（agent_episodes）
- **存储**：PostgreSQL `agent_episodes` 表（conversation_id 主 scope + route/task_types/query_summary≤120/strategy_summary≤200/result_summary≤200/preference_hints/selected_count/success/plan_valid/created_at/expires_at）。
- **写**：仅**成功完成的 Team run**（Gate→TEAM + Planner/Validator 成功 + synthesis SUCCEEDED 且非空 selection + 无 FAILED worker）；经 `build_episode_from_context → should_write_episode → insert`。
- **不写**：plan.failed / worker fatal / synth 失败 / 空 selection / 无 conversation_id / Legacy 路径。
- **recall**：`recall_episodes(conversation_id, query, top_n=3, budget=800)`；排序 = keyword overlap + recency + success + route；硬限制 **Top-N≤3、渲染总字符≤800、单条≤280**。
- **TTL**：`expires_at = created_at + 30d`；查询自动排除过期；`purge_expired()` 清理；per-conversation **cap 20**。
- **去重**：相同 conversation + 相同 task_types + query_summary Dice 2-gram 相似度 ≥0.85 + 24h 窗口 → 跳过。
- **安全清洗**：`_FORBIDDEN_PATTERNS`（sk-/api_key/authorization/base64/data:image/filepath/chain_of_thought/raw_response 等）整体替换 `[REDACTED]`。
- **Planner 集成**：`memory_hints` 进入 user prompt 独立段落，且写明"当前用户请求优先级最高，冲突以当前目标为准，历史记忆不得覆盖 TaskType 选择"；`memory_hints=None` 时 prompt 与旧版逐字符一致（路由零回归）。

**为什么叫"轻量级 Episodic Memory"而不是"长期记忆系统"**：只有一张表、确定性摘要（无 LLM 摘要）、conversation 级 scope（无 user/全局）、TTL 固定 30 天、无向量/embedding、无 Memory Agent、无跨会话迁移学习、无偏好推理。它是"上一轮任务策略的蒸馏便签"，不是持久化人格/知识库。


---

## 17. Tests / E2E 真实验证情况

**测试总量**：`backend/tests/agent_team/` 共 **463 个用例**（2026-09-10 实测：`pytest backend/tests/agent_team -q` → **463 passed**，24.5s；PG 迁移前为 275）。⚠️ 注意：这些用例内部用**内存 SQLite / Fake**（如 `test_memory*.py`），**并不覆盖 PostgreSQL/pgvector 数据层本身**。分四类：

| 能力 | 测试级别 | 真实验证情况 | 可信程度 |
|---|---|---|---|
| ComplexityGate | 单元 + 集成 | Layer A 基准确定性（无 API）：FN=9/FP=0/team reachability 11/20 | 高（确定性） |
| Planner Capability Routing | 单元 16 + 集成 + benchmark | **真实 DeepSeek 56 次调用**（28×2），frozen 结果：Vision recall/precision 100%、FP=0、有效计划率 87.5% | 高（真实 LLM） |
| PlanValidator | 单元 + 集成 | 确定性；覆盖环/孤儿/越界/能力门控 | 高 |
| DAGExecutor | 单元 + 集成 | 串行语义、部分失败隔离、上限保护（Fake workers） | 高 |
| Retrieval (CLIP + pgvector) | 单元 + 集成 | **Stage 9 真实 E2E**（当时为 sqlite-vec / 66 图）：真实库 top-40 候选；迁库后 1000 图实测检索均值 65.0ms（`r1_eval_pg1000.json`） | 高 |
| Metadata / Group | 单元 + 集成 | 确定性；真实库 metadata 常空（⚠️） | 中 |
| Vision | 单元 8 + SSE + 集成 | **Stage 9 真实 DeepSeek Vision 4/4 成功**（supplement 计划）；默认 config 关闭 | 高（需 runtime VISION_MODEL） |
| Synthesizer | 单元 + 集成 | **真实 DeepSeek** 合成 + validate_selection + deterministic fallback | 高 |
| SSE 契约 | 集成（FastAPI TestClient） | **全部 Fake**；断言事件序、terminal 配对、无泄露 | 中（契约层） |
| Memory | 单元 23 + 集成 6 | **Fake Planner + 真实 repository/service**；`demo_memory_e2e.py` 多轮 demo 亦为 Fake Planner | 中（无真实 LLM 多轮） |
| Browser E2E | — | **无 documented 浏览器 E2E**；Stage 9 明确 Frontend NOT REQUIRED | — |

**必须如实区分的"真实 vs Fake"**：真实 DeepSeek 的证据集中在 Planner benchmark（56 调用）与 Stage 9 Vision/Synthesizer E2E；**Memory 多轮 demo 是 Fake Planner**；SSE 测试是 Fake stream；浏览器级 E2E 不存在。

---

## 18. Current Limitations（当前项目边界）

1. **DAG 串行**：`ready.pop(0)` 逐个执行；Vision 逐图串行（每图 2.5–4s），大候选集会慢。**非并行多 Agent**。
2. **Planner 可能产生非法计划**：真实 benchmark 有效计划率 **87.5% < 98%**——7/56 无效（6 例 `filter_metadata.include_fields` 枚举外值 + 1 例 synthesize 自引用漂移）。类型选择全对，属 input 契约忠实度问题；Validator 正确拦截并触发 Legacy 回退。
3. **ComplexityGate under-routing**：9/12 条 vision-positive query 被 Gate 判为 legacy，**根本进不了 Planner**（生产路径上 Vision 不可达）。Gate 是当前视觉生产路径的最大瓶颈（已解耦，未修复）。
4. **Vision 默认关闭**：`VISION_MODEL="null"` → `analyze_visual` 未注册，生产环境默认无像素核验能力；需显式配置/运行时 override 才启用。
5. **前端不消费 Team SSE trace**：`plan.*`/`task.*` 事件被静默忽略；用户看不到 Agent 执行过程。
6. **无 resume / checkpoint**：`model_migrations.last_image_id` 仅向量重建断点，Agent 执行无持久化恢复。
7. **无多租户 / 无 auth**：conversation_id 是唯一 scope；无用户表、无权限；`user_id` 可选且不被 Agent 读取。
8. **Memory 局限**：仅 conversation 级、仅成功 Team run 写入、TTL 固定 30 天、keyword/recency 相关性（无语义 recall）、Legacy 不写。
9. **文本/图片编码器不一致**：multilingual CLIP 文本 vs `openai/clip-vit-base-patch32` 图像——严格语义空间不对齐，且无专用 reranker。
10. **数据层已迁 PG，但仍有 SQLite 残留**：`routers/search/similar_search.py` 仍用 `?` 占位符（PG 上该接口会返回错误，被 `search_handler` 吞掉）；部分模块仍 `import sqlite3` / `except sqlite3.Error`（异常类型不匹配）；`base.py` 靠 `adapt_sql()` 做 `?`→`%s` 改写。**测试套件不覆盖 PG 层**，因此这类问题不会被 pytest 发现。
11. **安全（Medium）**：`config.yaml` 存有明文格式 API key（审计已记录，封版前建议轮换并确认 git history）。
12. **真实库元数据分布**：VOC2010 千张图库的 `tags` 来自官方 Annotation（可用，Metadata/Group 可依赖），但 `description` / 结构化 location 仍为空 → "按地点分组" 类需求真实命中有限。


---

## 19. Interview Learning Map

### Level 1 — 一定要会（15 题）
1. 这个项目是什么？一句话 + 它解决什么问题。
2. 你的系统为什么算 Agent Team？—— 只有 Planner/Synthesizer/Vision 用 LLM，Retrieval/Metadata/Group 是确定性 Worker；固定流水线。
3. Agent 和普通 Function Calling 的区别？—— Legacy `RecommendAgent` 就是 2 次 function-call；Agent Team 把"任务分解+执行+校验"结构化，LLM 只参与规划与合成。
4. 为什么需要 Planner？—— 复杂请求需要多步任务分解成可执行 DAG。
5. 为什么需要 Validator？—— LLM 一次生成不可信：错类型/漏检索/环/越界 input；执行前确定性拦截。
6. 为什么需要 DAG？—— 任务间真实依赖（先检索再分组再合成）。
7. Planner 和 Worker 有什么区别？—— Planner 选"做什么+依赖"，Worker 做"怎么执行"；Worker 无自主决策。
8. 为什么 CLIP 后面还要 Vision？—— CLIP 是语义相似，Vision 是像素核验（"画面里有没有生日蛋糕"）。
9. 为什么不是直接 Vision 全部图片？—— 昂贵（串行每图 2.5–4s、上限 8 张）；先便宜检索/元数据缩小候选。
10. 为什么需要 SSE？—— 长耗时多步执行需要逐步反馈；前端按 delta 实时渲染。
11. Memory 在这里解决什么问题？—— 跨轮复用成功策略（上次用了分组/视觉），但 current query 永远优先。
12. 你的系统里哪些是"真正 Agent"哪些是"确定性模块"？—— Planner/Synthesizer = LLM Agent；Gate/Validator/Executor/Retrieval/Metadata/Group/Memory = deterministic。
13. Vision 部分失败会发生什么？—— 单图失败只丢该图证据；systemic 失败置标志；Synthesizer 永远能跑（确定性兜底）。
14. 为什么不能"看到 visual 关键词就强制 Vision"？—— 最小必要原则；颜色/物体/场景等可见事实才需要，其余用检索/元数据。
15. 你验证过 Planner 吗？—— 28 条 gold × 2 次真实 DeepSeek 调用，Vision recall/precision 100%。

### Level 2 — 能够展开（8 题）
1. Typed Task 为什么比自由文本可靠？—— `extra=forbid` + 类型化 input + 依赖图校验 + 安全 dispatch。
2. DAG 如何处理 dependency？—— Kahn 式入度队列；串行 pop(0)。
3. 一个 Worker 失败以后会发生什么？—— 依赖全失败→SKIPPED；至少一个成功→继续（partial isolation）；崩溃→FAILED 隔离。
4. Vision 部分失败如何保证最终结果？—— VisionWorker 恒 SUCCEEDED，失败收敛进 VisionTaskOutput；无证据图不注入候选。
5. LLM 输出如何做 deterministic validation？—— Planner→Pydantic+PlanValidator；Synthesizer→SynthesisResult+validate_selection。
6. Capability routing 为什么重要？—— 防止 Planner 用未启用能力、防止"视觉词→强制 Vision"、杜绝三层 capability drift。
7. Memory 如何避免覆盖当前 query？—— 硬限制（≤3/≤800 字符）+ prompt 明确优先级 + 不自动改 plan。
8. 如何限制 LLM 对执行流程的影响？—— schema enum 收窄、input 契约、图校验、selection 白名单、deterministic fallback。

### Level 3 — 代码追问（8 题）
1. Planner 输出 schema 怎么约束？—— `_build_task_plan_schema(enabled)` 按 enabled 收窄 task_type enum + input 契约描述；tool_choice 强制。
2. Validator 如何检测循环依赖？—— 图遍历/入度消解（`_validate_acyclic`），深度 >4 拒绝。
3. Retrieval 和 Vision 如何通过 ExecutionContext 传递数据？—— WorkerResult 写入 results，`collect_candidates`/`_attach_vision_evidence` 按 image_id 合并。
4. VisionEvidence 为什么限制字段？—— 白名单 + 长度/数量上限（title≤60/desc≤200/tags≤8/evidence≤300）+ `extra=forbid`；防幻觉字段污染。
5. Synthesizer 如何保证不选不存在的图片？—— `validate_selection` 只保留 `{c.id for c in candidates}` 内 id。
6. SSE 如何保证 terminal event？—— 每个 `task.started` 必有对应 completed/failed/skipped；`complete` 唯一（有测试断言）。
7. Memory 如何做 dedup？—— 相同 task_types + Dice 2-gram 相似度 ≥0.85 + 24h 窗口。
8. 为什么当前 DAG 不做并行？—— 设计决策：串行确定性优先、避免共享状态竞争；`MAX_CONCURRENCY` 常量已预留但未使用。

---

## 20. 一页式项目总览（面试前 60 秒复习）

```
SmartImager = 图片库 + CLIP 语义检索 + LLM 对话式推荐（FastAPI + React + PostgreSQL/pgvector + DeepSeek）

一条主线：POST /recommend/chat/stream
  Router(conv_id) → stream_core
    → ComplexityGate（正则，不调 LLM）：简单→Legacy(2次function-call)；复杂→TEAM
    → AdaptiveTeamOrchestrator：
        0) Memory.recall（若有 conv_id）→ hints(≤3/≤800字符)
        1) Planner（DeepSeek×1，强制 submit_task_plan schema）→ TaskPlanDraft
        2) PlanValidator（10类确定性检查；失败→plan.failed→Legacy 回退）
        3) DAGExecutor（串行 Kahn 式）→ ExecutionContext
             Retrieval(CLIP + pgvector KNN: `<=>`, score=1-distance；元数据同库过滤)
             → Metadata(确定性过滤) / Group(确定性分组)
             → Vision(per-image LLM，仅 vision_available 时注册；失败不阻断)
             → Synthesize(DeepSeek×1；validate_selection 白名单；失败→确定性 top-N)
        4) Memory.write（仅成功 Team run）
    → SSE: rewrite_start → plan.*/task.*(透传但前端未消费) → assistant_delta → complete

核心工程点：
  · LLM 只干两件事：Planner(选类型+依赖，被 schema+validator 双约束)、Synthesizer(选图+写答，被 selection 白名单约束)
  · 其余全部确定性：Gate/Validator/Executor/Retrieval/Metadata/Group/Memory
  · 每层都有兜底：plan.failed→Legacy；synth 失败→deterministic；vision 失败→不阻断
  · Capability 同源：registered_task_types() 同时驱动 Planner schema / Validator / workers
  · Vision 默认关（VISION_MODEL="null"）；开启后 recall/precision=100%（56 次真实 benchmark）
  · Memory = ExecutionContext(working) + agent_episodes(episodic, TTL30d, cap20, dedup24h)

如实说边界：串行 DAG、Gate under-routing(9/12 vision query 被拦)、有效计划率 87.5%、
前端不展示 Team trace、无多租户/权限/resume、README 模型描述过时、config 明文 key。

一句话定位：Typed Agent Team / Agentic Workflow —— 不是 Autonomous Multi-Agent，
不是 Agent Infra Platform，而是一个"LLM 规划 + 确定性执行 + 强校验兜底"的多模态图片推荐应用。
```

---

*本指南基于当前代码（2026-09-01）与 `FINAL_SMARTIMAGER_AUDIT.md`、`AGENT_MEMORY_IMPLEMENTATION_REPORT.md`、`PLANNER_CAPABILITY_ROUTING_BENCHMARK_REPORT.md` 交叉验证。所有"真实/测试/Fake/限制"均已区分。*

