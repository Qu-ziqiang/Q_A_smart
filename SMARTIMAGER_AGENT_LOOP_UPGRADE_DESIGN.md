# SmartImager Agent 闭环升级设计

> 状态：设计稿，**本轮未改代码**  
> 日期：2026-09-12  
> 目标：`Plan → Execute → Observe → Replan → Execute`（最多 2 轮）  
> 不做：Agent 互聊、Specialist 独立 LLM loop、REVIEW 全面 LLM 化、默认 MCP

---

# 1. Current Architecture

默认 TEAM 路径（`ENABLE_SUPERVISOR_RUNTIME` 空值视为开）：

```text
IntentRewrite → ComplexityGate
  ├ LEGACY：RecommendAgent 直达检索
  └ TEAM：SupervisorAgent(enhanced=True)
        ├ Memory recall（有 conversation_id）
        ├ Planner.create_plan（1 次 deepseek-chat，强制 submit_task_plan）
        ├ PlanValidator.validate
        ├ DAGExecutor 串行 → Retrieval / Organizer / Vision
        └ review_execution → continue | revise | replan | abort
```

LangGraph（`supervisor_graph/`）只替换状态转移，节点复用同一套 `_execute` / `review_execution` / `revise_plan`。

角色分工（保持）：

| 模块 | 决策还是执行 | LLM |
|---|---|---|
| Planner | 决策：出任务图 | 是，每轮规划 1 次 |
| Validator | 门禁：图能不能跑 | 否 |
| Specialist | 执行：按 task_type 接单 | 否 |
| Supervisor | 编排：观察 + 是否发起再规划 | REVIEW 否 |
| Synthesizer | 收尾：推荐文本与 id | 可 |

通信现状：Worker 写 `ExecutionContext` / `RunState`；replan 时只把 `decision.reason` 拼进伪 user 消息 `[supervisor 重新规划] {note}`。没有结构化 Observation。

---

# 2. Current Limitation

`review_execution`（`supervisor.py`）优先级：

1. FAILED 或 retrieve 成功但候选数 = 0 → **revise**（`revise_plan` 只把 retrieve `limit+10`，cap 40）
2. synthesis 无 `image_ids` → **replan**
3. 总候选 < `min_candidates`（默认 1）→ **replan**
4. 否则 continue

缺口：

- **Empty 与 Low Relevance 未分开。** 有 20 张低分图会 continue。
- **revise 不改变策略。** 同一 query 多拿几张，不是改写，也不是补视觉。
- **replan 观察太弱。** Planner 看不到 top1/topk、tags、原 retrieve.query。
- **失败与空检索走同一条 revise。** 工具崩溃和「库里没有」被同样加 limit。
- 角色执行本身没有问题；缺的是 **执行结果如何改变下一轮计划**。

角色分工要优化的是「谁根据观察做结构化决策」，不是给 Specialist 加嘴。

---

# 3. Proposed Plan-Execute-Observe-Replan Architecture

```text
第 1 轮（行为与现在首次执行一致）
  Planner → Validator → Execute → build Observation → classify Trigger
       ├ NONE → finalize（DONE）
       ├ ABORT 条件（已是第 2 轮）→ finalize（ABORT）
       └ 需纠正 → 把 Observation 交给 Planner

第 2 轮（仅当 Trigger ≠ NONE）
  Planner(goal + Observation + 建议 action) → Validator → Execute → Review
       ├ 仍触发 → ABORT（到顶）
       └ 否则 DONE
```

硬规则：

- 第 1 轮必须先完整走完现有执行（含现有 R1/可选 R2），不在执行中途插 LLM Supervisor。
- 第 2 轮禁止再走「只 +limit」的 `revise_plan` 作为 Empty/LowRelevance 的主路径。
- 新计划必须 `PlanValidator.validate`，失败记 `plan.failed` 并 ABORT（或按现网回退 Legacy，见 §12）。
- Specialist 仍无独立 loop；Vision 只有新计划里出现 `analyze_visual` 才执行。

---

# 4. Observation Schema

落地建议：新模块 `orchestration/observation.py`，Pydantic 模型。字段对齐现有 `ImageCandidate.score`（`search.py`：`score = 1 - cosine_distance`，约在 0–1）、`tags`、`RetrievalOutput`、`WorkerResult.status`。

```text
RoundObservation
  request_id: str
  iteration: int                    # 刚结束的轮次，首次为 1
  original_goal: str                # Supervisor 用的 goal
  retrieve_query: Optional[str]     # 本轮 retrieve_images.input.query
  trigger: TriggerType              # NONE|EMPTY_RETRIEVAL|LOW_RELEVANCE|EXECUTION_FAILURE|SYNTHESIS_FAILURE
  suggested_action: ReplanAction    # 给 Planner 的受限建议，非强制执行指令

  retrieval_empty: bool
  candidate_count: int              # 本轮 retrieve 成功候选数之和
  top1_score: Optional[float]       # 最高 CLIP 相似度
  topk_mean_score: Optional[float]  # top-min(k,n) 均值，k 默认 5
  tag_overlap: float                # 0~1，query/goal 词与候选 tags/title 的重合
  tag_hit: bool                     # overlap >= TAG_HIT_FLOOR（实验常数，非简历数字）

  failed_task_ids: List[str]
  empty_retrieve_task_ids: List[str]
  error_codes: List[str]            # RETRIEVAL_ERROR / WORKER_CRASH / INVALID_TASK_INPUT 等
  vision_available: bool
  plan_had_vision: bool
  synthesis_has_selection: bool
  plan_task_types: List[str]        # 有序，如 retrieve_images→filter_metadata→synthesize

  # 有界摘要，禁止塞整池候选
  top_titles: List[str]             # ≤3，各 ≤40 字
  render_for_planner() -> str       # ≤600 字，注入 create_plan
```

从哪取：

| 字段 | 来源 |
|---|---|
| candidate / score / tags | `results[retrieve].output.candidates`（`ImageCandidate`） |
| retrieve_query | `plan.tasks` 里 `RetrieveImagesInput.query` |
| failed / error | `WorkerResult.status` + `error` |
| synthesis | `results[synthesis_task_id].output.image_ids` |
| vision_available | 现有 `vision_available()` |

`tag_overlap` 复用 `memory._keyword_overlap` 同类逻辑：query 分词 vs `tags + title`，不要新 embedding。

Observation 进 Planner 的方式：扩展 `create_plan(..., observation: Optional[RoundObservation]=None)`。`None` 时第 1 轮字面行为不变。禁止只靠拼接松散 reason 字符串。

---

# 5. Trigger Classification

`review_execution` 改为两段：先 `build_observation` + `classify_trigger`，再映射 action。不要一个 `if failed`。

优先级（高 → 低），与现网「失败先于空」对齐：

### A. EXECUTION_FAILURE

任一 task `FAILED`（`RETRIEVAL_ERROR`、工具异常、Worker crash）。  
**不等于**空结果。同一 query 重试一次可接受（见 action `RETRY_SAME`），因为问题是执行，不是相关性。

### B. EMPTY_RETRIEVAL

retrieve `SUCCEEDED` 且候选数 = 0。  
现网走 revise+limit；升级后走 **REWRITE 再规划**。

### C. LOW_RELEVANCE

同时满足（交叉，减少误杀）：

1. `candidate_count > 0`
2. `top1_score is not None` 且 `top1_score < SCORE_THRESHOLD`（exploratory，起点 0.4）
3. `tag_hit is False`

缺 score 时 **不判** LOW_RELEVANCE（避免 Fake 测试全误触发）。仅 tag 差或仅分低，不单独触发。

### D. SYNTHESIS_FAILURE

无 FAILED/EMPTY/LOW 之后，synthesis 无有效 `image_ids`，或总候选 < `min_candidates`。  
与现网 2/3 条对齐，仍触发再规划。

### E. NONE

否则 continue。第 1 轮成功则不再进第 2 轮。

`attempt >= max_iterations` 时任何非 NONE → ABORT。

---

# 6. Replan Action Space

Supervisor **只选出建议 action**，不直接改 Task.input（Empty/LowRelevance 不再调用 `revise_plan`）。

| Action | 何时建议 | Planner 必须产生的结构化差异 |
|---|---|---|
| RETRY_REWRITE | EMPTY 或 LOW_RELEVANCE | 至少一个 retrieve 的 `query` ≠ 上轮 `retrieve_query`（允许改 filters，禁止只改 limit） |
| ADD_VISION | LOW_RELEVANCE 且 `vision_available` 且上轮无 `analyze_visual`，用户 goal 含核验语义时优先 | 新计划含 `analyze_visual`，且依赖 retrieve |
| RETRY_SAME | 仅 EXECUTION_FAILURE | 可同结构；允许 limit 微调。这是唯一仍接近旧 revise 的路径 |
| REPLAN_STRUCT | SYNTHESIS_FAILURE 或 REWRITE 后仍建议换拓扑 | 任务类型集合或依赖边与上轮不同 |
| ABORT | 第 2 轮仍触发 / Planner 失败 / 校验失败 | 不执行新图 |

「与第一轮不同的行动」校验（确定性，无 LLM）：

- `RETRY_REWRITE`：新计划所有 retrieve.query 与 observation.retrieve_query 规范化后全相同 → **Validator 或 Supervisor 后置检查失败**，当 invalid replan，ABORT 或回退展示第 1 轮结果（实现时二选一，测试锁死）。
- `ADD_VISION`：新计划无 `analyze_visual` → invalid replan。
- 视觉不可用时不得建议 ADD_VISION，改 RETRY_REWRITE。

Planner prompt 第 2 轮增加 bounded 段：Observation 摘要 + 建议 action + 「禁止只增加 limit」。仍强制 `submit_task_plan`，类型集合不变。

---

# 7. Validator Integration

第 2 轮与第 1 轮同一入口：

```text
draft = planner.create_plan(goal, messages, memory_hints, observation=obs)
vres = validator.validate(draft)
if not vres.valid: ABORT / plan.failed
```

禁止 Supervisor 写 `Task.input.query` 后直接 `_execute`。  
`revise_plan` 降级为仅 `RETRY_SAME` 可选实现；Empty/LowRelevance **禁止**走它。

LangGraph `graph.py` 的 `_replan_events` 必须改成传入同一 Observation，而不是只拼 reason。

---

# 8. Supervisor / Planner Boundary

```text
Supervisor
  建 Observation、分类 Trigger、选 suggested_action
  不改写自然语言 query
  不新增 task_type
  不当 LLM Judge

Planner
  读 Observation
  输出新 TaskPlanDraft（改 query / 加视觉 / 改依赖）

Validator
  静态可执行性 +（可选）与 Observation 的差异断言

Executor / Specialist
  只执行新图，不读 Observation 做路由
```

这样「角色分工」的优化点是：

- **检索角色**：第 2 轮用新 query 再搜，仍是确定性 Worker。  
- **整理角色**：只有新计划还包含 filter/group 才跑。  
- **视觉角色**：只在 ADD_VISION 计划里出现。  
- **Supervisor**：从「加 limit 的修补工」变成「观察与是否再规划的闸门」。  
- **Planner**：从「只看用户话」变成「看用户话 + 上一轮证据」。

智能加在规划第二次调用，不加在三个 Worker 互评。

---

# 9. Agent Communication Model

采用共享状态，不采用邮箱：

```text
Specialist.execute → WorkerResult
        ↓
ExecutionContext / RunState.execution_results
        ↓
RoundObservation（结构化，有界）
        ↓
Supervisor.classify
        ↓
Planner.create_plan(observation=...)
```

禁止：

- RetrievalAgent 调 VisionAgent  
- Organizer 与 Retrieval 对话  
- AgentMessage 变成异步 mailbox（现有 task/report 信封可继续当日志）

SSE：第 2 轮仍 `plan.created`；建议新增只增不改字段，例如 `observation.trigger` / `suggested_action` 挂在 `agent.review` 上，便于证明发生了真 replan。

---

# 10. Threshold Calibration

`candidate.score = 1 - (embedding <=> query)`（pgvector cosine distance）。若向量已归一化，score ≈ cosine similarity。

**0.4 是 exploratory threshold，不是生产结论。** 落地前用当前千级库跑标定脚本（新脚本，不改线上默认也可先 dry-run）：

建议构造：

- 相关：现有 `gold_voc2010_gallery.json` 的类目 query（「公交车的照片」等）  
- 不相关：同类目 query 对 **其他类** 图的 top 分，或随机错配 query（「核磁共振设备」对 VOC 图库）

报告：

- 相关 query 的 top1 / top5 mean 分布  
- 错配 query 的 top1 / top5 mean  
- tag_overlap 在「类名出现在 tags」vs「不出现」上的分布  
- 候选数（limit=20）分布  

选阈值原则：相关样本 top1 大部分 **高于** 阈值，错配大部分 **低于**，且与 `tag_hit` 联合后误触发率可接受。若两分布重叠大，保持交叉（分低 **且** 无 tag）并在报告写「exploratory」。

未跑标定前：

- 代码用命名常量 `EXPLORATORY_CLIP_TOP1_THRESHOLD = 0.4`  
- 环境变量可覆盖  
- 简历不写 0.4  

本设计阶段 **没有** 现成千级 CLIP raw-score 分布表；`r1/r2_eval_pg1000.json` 是 Hit/MRR，不能当阈值依据。

---

# 11. Evaluation Plan

拆开，禁止合成一个「准确率」。

### Agent 闭环（新，必须）

对照 **同一批** 构造用例（空库/错 query/低分错配/工具注入失败）：

| 指标 | Before | After 期望 |
|---|---|---|
| Empty → 第 2 轮 retrieve.query 变化率 | ~0（只改 limit） | → 1 |
| Empty → 最终有候选的恢复率 | 测基线 | 报告 delta |
| Low relevance 触发率 / 恢复率 | 现网约 0 触发 | 有触发且动作变化 |
| Replan rate / abort@2 | 现网 | 不爆炸 |
| invalid replan（只改 limit / 未过 Validator） | — | 0 |

Trace 断言：存在第二次 `plan.created`，且 retrieve.query 或 task_types 与第一轮不同。

### Planner

第 2 轮：建议 action 与计划差异一致；Valid 率；TaskType 不因 Observation 乱加类型（除非 ADD_VISION）。

### Retrieval

同一 gold：第 1 轮 vs 最终轮 Hit@1 / 空结果率。闭环评测与检索 gold **分表**，Fake Runtime 的 200-case 只测控制流。

200-case Judge **不要**当本改造成功标准（Fake 无真 score）。另加确定性用例：空检索、低分双差、失败、synthesis 空。

---

# 12. Code Impact Map

| 文件 | 目的 |
|---|---|
| **新建** `orchestration/observation.py` | Schema、build、render、classify、差异断言 |
| `orchestration/supervisor.py` | review 接 Observation；Empty/LowRelevance 不再 `revise_plan`；replan 传 observation |
| `orchestration/planner.py` + prompts | `create_plan(observation=)`；第 2 轮 user prompt |
| `orchestration/supervisor_graph/graph.py` | `_replan_events` 传入 observation |
| `orchestration/supervisor_graph/state.py` | state 可挂 observation dict |
| `orchestration/supervisor_graph/runtime.py` | 首轮后写入 observation |
| **可选** `scripts/calibrate_clip_threshold.py` | 分数分布，不改线上 |
| `backend/tests/agent_team/test_supervisor_*.py` | 触发分类、query 必变、Validator 再入、2 轮封顶 |
| **新建** `test_observation_replan.py` | Fake 检索空/低分/失败 |

### 必须保持

- ComplexityGate + Legacy fallback  
- 第 1 轮无 observation 时 Planner 字面行为  
- Specialist 无 LLM loop、无互聊  
- `max_iterations=2`  
- SSE 旧字段只增不改  
- MCP 默认不进  
- Memory 接线不变  
- `enhanced=False` 仍全部 continue  

### 建议新增测试

1. Empty → 第 2 轮 query ≠ 第 1 轮  
2. 低分 + 无 tag → LOW_RELEVANCE → REWRITE 或 ADD_VISION  
3. 仅低分或仅无 tag → 不触发  
4. FAILED → 不走「改 query」主路径  
5. 第 2 轮仍空 → abort  
6. 第 2 轮 draft 未过 Validator → 不执行  
7. 第 2 轮只改 limit → 视为 invalid  
8. Legacy / Gate 简单请求不进闭环  
9. LangGraph 与 legacy 对上述决策一致  

### 建议重跑的 benchmark

- 闭环小集 before/after（新，必跑）  
- Planner v2：**不必**全量重跑，除非改了无 observation 的 system prompt  
- 200-case：控制流回归即可，不作为质量分数  
- `r1_eval_pg1000`：仅当改了默认检索公式才重跑；本设计默认不改检索公式  

---

# 13. Risks

1. 0.4 误杀相关结果 → 必须交叉 tag，阈值可配。  
2. 第 2 轮再 2.5s 规划 + 再检索 → 最坏约双倍规划成本；用 2 轮硬顶。  
3. 改写 query 漂移（更泛/更歪）→ prompt 约束 + 差异断言只要求「不同且可执行」，用评测看恢复率。  
4. 视觉默认关 → ADD_VISION 常不可用，Empty/LowRelevance 应能单独靠 REWRITE 闭环。  
5. Fake 测试无真实 CLIP 分 → 单测注入 score；标定脚本走真库。  
6. 旧测试假定空检索走 revise+limit → 要改期望，避免「只加测试不加改断言」。  

---

# 14. Success Criteria

1. Supervisor 能区分 EMPTY vs LOW_RELEVANCE vs EXECUTION_FAILURE vs SYNTHESIS_FAILURE  
2. Observation 结构化回灌 Planner（不是一句 reason）  
3. 第 2 轮计划相对第 1 轮有可检测的结构化差异（query 或 task_types）  
4. 新计划必经 Validator  
5. 最多 2 轮  
6. Legacy / Gate 不被破坏  
7. 自动化测试覆盖上述路径  
8. 至少一组 before/after 对照  
9. runtime / SSE 能证明第二次 `plan.created`  
10. 无 Agent 互聊、无 Specialist LLM loop  

辅线（标签补全、hybrid、节点级 schema 检查）**不在本次成功标准内**，避免和 Agent 闭环抢范围。

---

# 15. 落地顺序（实现阶段，非本轮）

1. Observation 模型 + 单测（Fake 候选）  
2. `classify_trigger` 替换 `review_execution` 前半  
3. Planner 接收 observation；关掉 Empty 的 `revise_plan`  
4. LangGraph 对齐  
5. 回归 agent_team  
6. 真库阈值标定脚本 + 闭环 before/after  

本轮停止。未修改任何运行时代码。
