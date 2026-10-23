# SmartImager Task 能力扩充实现报告

本次目标不是增加 Specialist 数量，也不是让 Specialist 自带 LLM 或互聊。  
真正完成的是：

> **在现有 3 个接单 Specialist 上，抽出/补齐边界清晰的 Task，让 Planner 能组合更丰富的 Typed DAG。**

```text
Planner 决定任务组合
        ↓
    Typed DAG
        ↓
    Validator 整单拒绝或整单通过
        ↓
 RetrievalAgent     OrganizerAgent      VisionAgent
  retrieve/rerank    filter/dedup/       analyze/verify
                     group/sort/select
        ↓
    Supervisor Observe / Replan
```

没有为了凑满 10 个 Task 再硬拆相同逻辑。  
也没有再往上堆没有代码基础的 Task（见 §12）。

---

# 1. Existing Task Audit

只读审计结论（改代码前）：

| 审计项 | 现状 | 处理 |
| --- | --- | --- |
| `group_images` | `GroupImagesWorker`：`location/year/month/orientation/file_type/tag`；空字段进 `未分类`；输出 `GroupOutput{group_by, groups[{key,image_ids,candidates}]}` | **保留并作为 Organizer 核心能力** |
| rerank / R2 | `rerank.py`（R1 确定性加权）+ `model_reranker.py`（R2 GLM-4V，失败回退 R1）；原先嵌在 `RetrievalWorker` 内部 | **抽出 `rerank_candidates` Task**；retrieve 内部 rerank **保留**，避免破坏 baseline |
| candidate 传递 | `collect_candidates()` 读 Retrieval / Metadata / Group / Vision 输出 | 新输出接入 `CandidateListOutput` / `VerifyTaskOutput` |
| filter / group / synthesize schema | `FilterMetadataInput` / `GroupImagesInput` / `SynthesizeInput` 已结构化 | 保留 |
| dedup | 仅有 `dedup_candidates()` **按 image id**；无 hash / pHash / 近重复 | **抽出 Task，禁止声称近重复** |
| sort | 候选已有 `score/created_at/width/height/title` | **最小 `sort_images`** |
| select | synthesizer 的 `recommendation_limit` 是终点截取，不是独立 Task | **独立 `select_images`（top_n / per_group）** |
| verify | `analyze_visual` 是开放描述；parser 拒 unknown 字段（含 `passed`） | **独立 `verify_visual_condition` + 自有 parser** |
| 注册面 | Card / `default_workers` / `registered_task_types` / Planner catalog / Validator `INPUT_FOR_TASK_TYPE` / `CANDIDATE_CONSUMING_TYPES` / Registry | 全部扩到新 Task |
| Planner / Validator | `TaskType` 枚举 + catalog 限制可生成类型 | 同步扩 |
| DAGExecutor | `workers[task_type].run`，串行 ready-queue | 不改并行；新 Worker 入表即可 |
| 旧测试 | retrieve / filter / group / analyze / synthesize | 保留；新增组合测试 |

`WorkerResult` 已能表达 `task_id / task_type / status / output / error`。  
没有另造一套 `TaskResult`。

---

# 2. New Task Design

最终落地 **10 个 TaskType / 3 个 Specialist**：

```text
retrieve_images
rerank_candidates
filter_metadata
deduplicate_images
group_images
sort_images
select_images
analyze_visual
verify_visual_condition
synthesize
```

| Task | 来源 | 输入 | 输出要点 |
| --- | --- | --- | --- |
| `retrieve_images` | 已有 | query / vector_targets / filters / limit | 候选召回；不改 query、不自触发下一 Task |
| `rerank_candidates` | **抽出** retrieve 内部 R1/R2 | query? / provider=`auto\|r1\|r2` | 同集合重排；`ordered_ids` / scores 仍在 candidate.score |
| `filter_metadata` | 已有 | predicates | 删除不满足谓词的候选 |
| `deduplicate_images` | **抽出** `dedup_candidates()` | `{}` | 按 id 去重；`removed_ids` / `removed_count` |
| `group_images` | 已有（强化被 select 消费） | group_by | `GroupOutput`；空字段 → 未分类 |
| `sort_images` | **新建最小实现** | sort_by / order | 改变顺序，不增删成员 |
| `select_images` | **新建**（区别于 synthesizer 截取） | strategy=`top_n\|per_group` / limit | 按数量策略取子集 |
| `analyze_visual` | 已有 | criteria / max_candidates | 开放视觉描述 |
| `verify_visual_condition` | **新建**（复用 ImageAnalysis 客户端） | condition / max_candidates | `passed` / evidence / confidence；下游只消费通过项 |
| `synthesize` | 已有控制面任务 | recommendation_limit | 终点汇总 |

R2（GLM-4V）是 `rerank_candidates` 的一种实现，**只重排，不增加 recall**。  
retrieve 内部默认 rerank 行为未删除。

---

# 3. Specialist Capability Matrix

```text
RetrievalAgent
├── retrieve_images
└── rerank_candidates

OrganizerAgent
├── filter_metadata
├── deduplicate_images
├── group_images
├── sort_images
└── select_images

VisionAgent
├── analyze_visual
└── verify_visual_condition

Control Plane（不是接单 Specialist）
├── Planner
├── Supervisor
├── Synthesizer
└── Validator
```

`AgentRegistry` 按 `base_workers` **过滤**后再 `model_copy` Card：  
测试只注入 `retrieve` 时，不会把 `rerank` 误指到 `RetrievalWorker`。  
Vision 启用条件改为：workers 中存在 `analyze_visual` **或** `verify_visual_condition`。

一个 Agent 承接多类 Task；**不是** 一个 Task = 一个 Agent。

---

# 4. Task Boundary Definition

| Task | 做什么 | 不做什么 |
| --- | --- | --- |
| `filter_metadata` | 删除不满足 metadata predicate 的候选 | 不按数量取子集；不读像素 |
| `group_images` | 按 group key 组织候选集合 | 不删成员（空字段进未分类）；不检索 |
| `deduplicate_images` | 按 image id 去掉重复候选 | **不是** 近重复 / pHash |
| `sort_images` | 按明确字段改顺序 | 不增删、不召回；不是相关性重排 |
| `select_images` | 按 top_n / per_group 确定最终子集 | 不是谓词过滤 |
| `rerank_candidates` | 对已有 candidate set 重排 | 不增加召回、不改 query、不决定下一步 |
| `analyze_visual` | 对图片做开放视觉分析 | 不输出结构化 pass/fail |
| `verify_visual_condition` | 对明确视觉条件做通过/不通过 | 不改 query、不召回、不规划下一步 |

`select_images` 与 `filter_metadata` **不是** 同一个函数换名：

```text
filter  = 删除不满足谓词的候选
select  = 从剩余候选按数量/策略取子集
```

`group → select(per_group)` 是真实消费关系：`SelectWorker` 读 `GroupOutput`；没有分组则 fallback `top_n`。

`verify` 的 `candidates` 只保留 **passed** 子集，因此可以被 `select` / `synthesize` 直接消费。  
全部核验明细留在 `items`。

---

# 5. Planner Changes

- `TaskType` 枚举扩到 10 个。
- `TASK_CAPABILITY_CATALOG` 为每个新类型补齐 purpose / evidence / use_when / avoid_when / dependency / cost / input_contract。
- per-call schema 的 `task_type.enum` 仍按 `registered_task_types()` 收窄；vision 关闭时不含 `analyze_visual` / `verify_visual_condition`。
- DAG 示例增加 retrieve→rerank、filter→dedup、group→select、filter→group→select、verify。
- few-shot 增加「分组后选择」「条件核验」。
- routing policy 明确：rerank ≠ recall；filter ≠ select；analyze ≠ verify；dedup ≠ 近重复。
- Planner 仍然只生成 Typed Task（`task_type / objective / input / depends_on`），不执行业务工具。

Specialist 中 **没有** 增加 LLM planner / router / 自主 query rewrite / 自主 task selection。

---

# 6. Validator Changes

新 Task 进入：

- `INPUT_FOR_TASK_TYPE`
- `REGISTERED_TASK_TYPES`（视觉任务仍 capability-gated）
- `CANDIDATE_CONSUMING_TYPES`
- `_fallback_input`
- `VISION_UNAVAILABLE`（覆盖 verify）

保留并继续生效：

- cycle / unknown dep / self dep / orphan
- producer ancestry
- task count / depth
- retrieve 必有
- synthesize 唯一且终点
- 非法计划 **整单拒绝**

为支撑真实组合 DAG（case 4 / case 6），上限调整：

| 常量 | 旧 | 新 | 原因 |
| --- | --- | --- | --- |
| `MAX_TASKS` | 6 | **8** | `retrieve→filter→group→select→synthesize` 及更长链 |
| `MAX_PLAN_DEPTH` | 4 | **7** | 线性整理 + 核验链 |
| `MAX_WORKERS` | （原注释口径） | **10** | 10 个 TaskType 各 1 个 Worker |

`test_too_many_tasks` / `test_too_deep` 改为按常量动态构造超限计划。

---

# 7. Executor Changes

`DAGExecutor` 仍是 **serial ready-queue**。  
没有改成并行。

分派链：

```text
Task
 → AgentRegistry.get_agent_for_task(task_type)
 → SpecialistAgent.execute
 → 对应 Worker.run
 → WorkerResult（结构化 output）
```

`default_workers()` / `build_registry_backed_workers()` 注册了全部新 Worker。  
vision 关闭时不注册 analyze / verify。

`collect_candidates()` 识别 `CandidateListOutput` 与 `VerifyTaskOutput.candidates`，  
因此 rerank / dedup / sort / select / verify 的结果可以沿 DAG 传递。

---

# 8. Runtime Flow

```text
用户请求
  → ComplexityGate（简单 Legacy / 复合 TEAM）
  → IntentRewrite（规则清洗，不是二次规划改 query）
  → Planner 生成 Typed DAG
  → PlanValidator
  → DAGExecutor（串行）
       RetrievalAgent.retrieve / rerank
       OrganizerAgent.filter / dedup / group / sort / select
       VisionAgent.analyze / verify
  → Synthesizer
  → Supervisor Observe
       EMPTY_RETRIEVAL / LOW_RELEVANCE / EXECUTION_FAILURE / SYNTHESIS_FAILURE
  → 至多 1 次 Replan（仍整单 Validate）
```

未改：Legacy Fast Path、ComplexityGate、IntentRewrite、Supervisor 状态机、Memory、LangGraph Supervisor backend、MCP optional path、CLIP 三路检索底座、retrieve 内部 rerank fallback。

Rerank 失败仍回退 R1，请求不失败。

---

# 9. New DAG Examples

已在测试中真实跑通：

```text
Case 1  retrieve → rerank → synthesize
Case 2  retrieve → filter → dedup → synthesize
Case 3  retrieve → group → select → synthesize
Case 4  retrieve → filter → group → select → synthesize
Case 5  retrieve → verify_visual_condition → synthesize
Case 6  retrieve → dedup → group → verify → select → synthesize
```

业务意义：

- Case 1：召回后按相关性重排，不扩大候选池。
- Case 2：先按元数据删；`filter` 内部已做 id 去重，其后的 `dedup` 是幂等收口。重复 id 的真实删除在 Case 6。
- Case 3：分组后每组取 N 张。
- Case 4：先过滤再分组再选，Organizer 主链。
- Case 5：对明确视觉条件做 pass/fail。
- Case 6：整理后再核验，再按数量收口。

Planner 现在也能合法生成分支图，例如：

```text
retrieve
 ├── rerank
 └── verify → select → synthesize
```

（两条消费链都（传递）依赖 retrieve，且都能到达 synthesize。）

---

# 10. Tests

新增 `backend/tests/agent_team/test_task_expansion.py`：

- 6 个组合 DAG
- filter ≠ select
- sort 只改序
- rerank 不增召回
- vision 关闭拒 verify
- 无 producer 拒 rerank
- 3 个 Specialist 接单映射
- 旧 `retrieve → synthesize` 回归
- `WorkerResult` 结构

同步修改：

- `test_plan_validation.py`：上限按 `MAX_TASKS` / `MAX_PLAN_DEPTH`
- `test_agent_cards.py`：三角色 `supported_tasks` 扩到新能力

未改 benchmark 查询、gold、阈值。

---

# 11. Regression Results

实现后在仓库根目录运行，结果：

```text
test_task_expansion.py              15 passed
test_plan_validation.py             20 passed
test_registration.py                19 passed
test_agent_cards.py                  6 passed
test_planner_routing_policy.py      16 passed
test_planner.py                     11 passed
test_workers.py                     12 passed
test_executor.py                     8 passed
test_specialist_agent.py            10 passed
test_agent_loop.py                  11 passed
test_agent_registry.py               7 passed
test_gate.py                        13 passed
test_langgraph_supervisor.py         6 passed
test_supervisor_state_machine.py     5 passed
```

全部 0 failed。未改 benchmark 数据或阈值。

评估口径（不是“Task 从 5 变成 10”）：

| 维度 | 口径 |
| --- | --- |
| Planner TaskType coverage | catalog 覆盖全部 `TaskType`；vision off 不含视觉任务 |
| Plan validity | 6 个组合 DAG 均通过 Validator |
| dependency validity | 消费型必须有 retrieve 祖先；verify 无视觉能力整单拒绝 |
| new Task execution success | Fake search / Fake analyzer 下 Worker 全部 SUCCEEDED |
| DAG composition cases | Case 1–6 |
| existing regression | retrieve→synth、filter/group、Legacy/Gate、Agent Loop |
| runtime trace | Executor 仍逐任务 `task.started/completed`；Observation 可读 verify 切片 |

---

# 12. Known Limitations

1. **`deduplicate_images` 只做 image id 去重。** 项目里没有 pHash / 感知哈希 / 字节级近重复。
2. **`sort_images` 是最小版本。** 可排序字段仅 `score / created_at / width / height / title`；`created_at` 按字符串比较。
3. **`select_images(per_group)` 依赖上游 `GroupOutput`。** 没有分组时 fallback `top_n`，不假装做了分组选择。
4. **`rerank_candidates` 与 retrieve 内部 rerank 可能叠两次。** 这是为了不破坏旧 baseline；Planner 若只要显式 rerank，retrieve 仍可能在 Worker 层再排一次。这是已知重叠，不是 recall 提升。
5. **`verify_visual_condition` 复用 `ImageAnalysis.analyze_image`。** 条件判断质量取决于模型是否遵守 verify prompt；契约上与 analyze 分离，实现上共用客户端。
6. **没有 Planner 在真实 LLM 上的新 DAG 覆盖率评测。** 本次用结构化测试证明“能生成且能执行”，没有为了好看改 Planner benchmark 数据。
7. **Observation 的 Organizer 切片仍以 filter/group 为主。** dedup/sort/select 的结果可通过 `plan_task_types` 看到，没有再扩一套 Observation schema。

---

## 关于“还能不能再加更多 Task”

审计后 **明确不再加**。下面这些名字看起来像能力，但当前没有独立、可组合、不重复的代码基础：

| 候选 | 为什么暂缓 |
| --- | --- |
| `compare_images` | 没有成对视觉比较实现 |
| `near_duplicate` / pHash | 只有 id 去重 |
| `caption` / `describe` | 与 `analyze_visual` 重叠 |
| `cluster_images` | 没有聚类算法 |
| `score_threshold` | 与 filter/select 边界不清 |
| `merge_candidate_sets` | 计划通常只有一个 retrieve |
| `export` / `thumbnail` | 不是检索整理 DAG 的 Specialist 职责 |
| `ocr` / `detect` / `crop` | 没有对应模型与 Worker |

原则仍然是：

> **宁可暂缓，也不要为了数量硬实现。**

---

## 必须明确回答的六个问题

### 1. 哪些新 Task 是真正实现的？

- `sort_images`：按白名单字段排序（最小实现）。
- `select_images`：`top_n` / `per_group` 取子集。
- `verify_visual_condition`：条件核验 + 自有 parser；下游只拿 passed 候选。

### 2. 哪些是已有能力抽象成 Task？

- `rerank_candidates`：抽出 R1/R2 rerank（含 fallback）。
- `deduplicate_images`：抽出 `dedup_candidates()` 的 id 去重。

### 3. 哪些 Task 暂时没有实现？

目标清单 10 个全部落地。  
**没有实现** 近重复检测、成对比较、聚类、OCR、caption 等“看起来还能再加”的类型。

### 4. 每个 Task 属于哪个 Specialist？

见 §3。`synthesize` 属于控制面 Synthesizer，不属于 3 个接单 Specialist。

### 5. 为什么这些 Task 不需要 Specialist 自带 LLM？

它们都是 **有界、确定性或已有工具封装** 的执行：

- 检索 / 重排：复用 CLIP 检索与 R1/R2 函数
- 过滤 / 去重 / 分组 / 排序 / 选择：纯数据变换
- 视觉分析 / 核验：复用已有 `ImageAnalysis` 客户端，Worker 不规划、不改 query、不选下一 Task

规划决策留在 Planner；重规划留在 Supervisor。

### 6. 怎样体现 Planner 的策略决策与 Specialist 的专业执行边界？

```text
Planner：选哪些 Task、如何依赖、用什么 input（top_n 还是 per_group，r1 还是 r2，核验哪句条件）
Specialist：只执行分配到的那一个 Task，回传 WorkerResult
Supervisor：看结构化观察，决定是否整单重规划
```

同一批候选可以走出不同图：

```text
retrieve → dedup → filter → group → select → verify → synthesize
retrieve → rerank → synthesize
retrieve ── rerank
       └── verify → select → synthesize
```

图不同，是因为 Planner 对目标做了不同分解；  
Specialist 不会自己把 `filter` 改成 `select`，也不会自己发起下一跳。

---

## 未做的禁止项（核对）

- 没有增加第 4 个 Specialist
- 没有给 Specialist 加 LLM
- 没有 Agent-to-Agent chat
- 没有用 Task 数量冒充 Multi-Agent
- 没有删除旧 Task / Legacy
- 没有改成并行 DAG
- 没有把 Rerank 说成 Recall 提升
- 没有改 benchmark 数据
