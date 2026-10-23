# PLANNER_CAPABILITY_ROUTING_BENCHMARK_REPORT

> 项目：SmartImager（多模态智能图库与对话推荐系统）
> 类型：**Evaluation only — 真实 Planner Capability Routing Benchmark**
> 日期：2026-09-01
> 依据：`PLANNER_CAPABILITY_ROUTING_AUDIT_AND_FIX_PLAN.md`（audit §10.5 / §11.3）+ 已冻结的 `routing_benchmark_28.py` gold dataset
> 本 benchmark **未修改** production code / prompt / schema / gate / validator / tests / database / config。

---

## 1. Benchmark Scope

只回答一个问题：**修复后的 Planner 能否根据任务语义正确选择 `ANALYZE_VISUAL`，同时避免无必要地调用 Vision。**

执行链路（每次 run）：

```text
Query
 ↓
PlannerAgent.create_plan（真实 DeepSeek 文本 LLM）
 ↓
TaskPlanDraft（Pydantic）
 ↓
PlanValidator.validate(vision_available_flag=True)
 ↓
score_plan（离线 scorer）
```

**禁止执行**：Retrieval / VisionWorker / Synthesizer / DAG Executor / 完整 E2E / Frontend。本 benchmark 不验证 CLIP、Vision 模型、Synthesizer、DAG 执行或 SSE——这些已有 Stage 8 / Stage 9 证据。

## 2. Environment

| 项 | 值 |
|---|---|
| 平台 | win32，`venv\Scripts\python.exe` |
| Provider | DeepSeek（真实 API） |
| Endpoint | `https://api.deepseek.com`（config `OPENAI_API_BASE`，读取未修改） |
| Planner 文本模型 | `deepseek-chat`（config `CHAT_MODEL`，读取未修改） |
| temperature / stream | 0 / false（planner.py 固定，未修改） |
| 调用形态 | 单次 `chat.completions.create`，`tools=[submit_task_plan]`，tool_choice 强制 |
| timeout | `PLANNER_TIMEOUT_SECONDS=20`（默认，未修改） |
| capability snapshot | **runtime patch `vision_available()=True`**（仅 benchmark 进程；`config.yaml` 的 `VISION_MODEL="null"` 保持原样） |
| scorer | `routing_benchmark_28.score_plan` / `aggregate_scores`（未修改，确定性） |
| 数据集 | `routing_benchmark_28.GOLD_QUERIES`（28 条，未修改） |
| 运行方式 | 28 queries × 2 runs = **56 次真实 Planner 调用**，同 model / 同 snapshot / 同 timeout / 同 prompt / 同 schema / 同 scorer；无 retry-select、无无效 run 丢弃、run1/run2 各自独立保存 |

## 3. Dataset

- 共 28 条 gold queries：A Retrieval-only（4）/ B Metadata-only（4）/ C Group-only（4）/ D Vision-needed（4）/ E Retrieval+Vision（4）/ F Retrieval+Metadata（4）/ G Retrieval+Metadata+Vision（4）。
- Vision-positive = **12**（D+E+G），Vision-negative = **16**（A+B+C+F）。
- 每条含 `expected_route`（legacy/team）与 `expected_task_types`（全部含 `retrieve_images` + `synthesize`）。
- **Benchmark Integrity Check：PASS** — 28 条完整；id 1–28 连续唯一；7 类别 × 4 结构正确；expected_task_types/expected_route 字段齐全；scorer 确定性；脚本仅 import planner/validator/gate；无 Vision/Retrieval/Synthesizer/DB/config 写入路径。**未发现 dataset 问题，gold 未修改。**

## 4. Pre-fix Baseline Availability

```text
REAL PRE-FIX BASELINE = NOT AVAILABLE
```

- 仓库扫描未发现任何已保存的真实 pre-fix benchmark 结果（`*baseline*` / `*benchmark_result*` / planner probe 结果文件均不存在）。
- audit 文档本身声明："真实 baseline 尚未保存，不能现在编造 X/Y"（§10.5）。
- **未 checkout、未回滚、未手工编造 baseline。**
- 因此 `baseline recall + 20pp` 与 `precision ≥ baseline` 两个相对指标 **无法计算** → 报告中记为 `N/A`；只判断绝对指标（§11 对照 Cursor acceptance）。

## 5. Post-fix Results（真实 56 runs）

- **LLM calls：56/56 成功**（每次 run 1 次真实 `deepseek-chat` 调用；无异常 run，`raw_plan_valid=True` 全部成立）。
- 平均 latency ≈ 2–4s/run（见 `planner_routing_benchmark_results.json` 各 run `planner_latency_ms`）。
- **任务类型选择在所有 56 个 run 上精确等于 gold**（`exact_task_types_accuracy = 100%`，含无效 run——无效仅发生在任务 input 负载，不影响类型集）。
- **Vision 选择：24/24 个 Vision-positive run 全部选中 `analyze_visual`；32/32 个 Vision-negative run 全部未选**。
- 依赖结构：Vision-positive 计划均为 `retrieve → analyze_visual → synthesize`（synthesize 依赖 [t1,t2]）；全部有效计划 `dependency_ok=True`。

## 6. Overall Metrics

| Metric | Result | Target | Status |
| -----: | -----: | -----: | ------ |
| Vision Recall | **100%**（24/24） | ≥ 91.7% | ✅ PASS |
| Vision Recall (vs baseline) | **N/A**（baseline unavailable） | ≥ baseline + 20pp | ⚪ N/A |
| Vision Precision | **100%**（24/24） | ≥ 90% | ✅ PASS |
| Vision Precision (vs baseline) | **N/A**（baseline unavailable） | ≥ baseline | ⚪ N/A |
| Unnecessary Vision (FP) | **0/56**（vision-negative runs 32/32 无 Vision） | ≤ 1/32 | ✅ PASS |
| Valid Plan Rate | **87.5%**（49/56） | ≥ 98% | ❌ FAIL |
| Average Tasks / plan | **3.06**（valid plans） | gold avg 3.14 + 0.25 = 3.39 | ✅ PASS |
| Retrieval-only / Metadata-only 增 Vision | **0/16 queries**（A+B+C+F 全未选 Vision） | 不增加 | ✅ PASS |
| Exact TaskType Accuracy | **100%**（56/56） | — | ✅ |
| Dependency Correctness | **100%**（49/49 valid） | retrieve→vision→synthesize 合法 | ✅ |

补充（非 acceptance 但按 §九记录）：
- `vision_false_negative = 0`；`vision_false_positive = 0`。
- 平均任务数未因追求 recall 而膨胀（3.06 ≤ 3.14），无"无条件加 Vision 任务"迹象。


## 7. Failure Cases

### Vision False Negative（Planner 未选 Vision）
```text
无（0/56）
```
12 条 Vision-positive query（q13–q20、q25–q28）× 2 runs 全部选中 `analyze_visual`。

### Vision False Positive（Planner 误选 Vision）
```text
无（0/56）
```
16 条 Vision-negative query（A+B+C+F = q1–q12、q21–q24）× 2 runs 全部未选 Vision。

### Invalid Plan（Validator 拒绝）— 7/56 runs
| run | category | 根因 |
|---|---|---|
| q8.1 / q8.2 | B | `FilterMetadataInput.include_fields[0]` 出现枚举外字段（合法值仅 `created_at, width, height, orientation, file_type, tags, location`） |
| q25.1 / q25.2 | G | 同上（`include_fields[1]`） |
| q27.1 / q27.2 | G | 同上（`include_fields[0]`） |
| q23.2 | F | `SYNTHESIS_TARGET: synthesis_task_id='t3' 与 synthesize task_id='t4' 不一致`（run1 正常，仅 run2） |

**共性**：7/7 无效 run 的**任务类型完全正确**（与 gold 精确一致），失败全部发生在 **task input 负载**层面：6 例为 `filter_metadata` 的 `include_fields` 字段名超出 schema 枚举（LLM 猜了 schema 外的元数据字段名）；1 例为 synthesize 自引用 `synthesis_task_id` 与自身 task_id 不一致（纯随机性，run1 同 query 通过）。**均非 capability misrouting。**

### Inconsistent Runs（同 query 两次 run 不一致）
```text
INCONSISTENT query_ids = [23]
```
- **任务类型选择稳定性：28/28 query 的 run1=run2（Vision 选择 100% 一致，无"一个 run 用 Vision 一个不用"）。**
- q23 的唯一不一致是 run1 valid / run2 invalid（synthesize 自引用 id 漂移）；任务类型集合两 run 相同。

## 8. Layer A Gate Results（静态，无 API，未修改 gate.py）

| Metric | Result |
|---|---|
| Gate False Negative（expected=team，实际=legacy） | **9** |
| Gate False Positive（expected=legacy，实际=team） | **0** |
| Team reachability | **11/20** |
| Vision-positive queries 被 Gate 阻断 | **9/12** |

被阻断的 Vision-positive queries：**q13–q16（D 全 4 条）、q18、q20（E 2 条）、q25、q26、q28（G 3 条）**。
放行的：C 4/4、F 4/4、E 2/4（q17/q19）、G 1/4（q27）。

**Gate 是当前生产路径上 Vision 能力的主瓶颈**（审计 F-002a 的"under-routing"被量化确认：12 条 Vision-positive 中有 9 条根本进不了 Planner）。

## 9. Layer C Production-path Results（gold → Gate → Planner → Validator）

| 路径结果 | 数量 | 说明 |
|---|---|---|
| expected Vision 但 **Gate BLOCKED**（Planner 未触达） | **9 queries** | q13–q16, q18, q20, q25, q26, q28 → legacy fallback，Vision 永远不可达 |
| Gate TEAM 且 **Planner MISROUTED**（类型错误） | **0 queries** | 11 条 team 可达 query 全部规划正确类型 |
| Gate TEAM 且 Planner 正确但 **Validator 拒绝** | 2 queries | q27（2 runs，filter_metadata input）、q23 run2（synthesize 自引用） |
| Gate TEAM 且完整 valid | 9 queries | q9–q12, q17, q19, q21, q22, q24 至少 1 个 valid run |

**结论**：生产路径失败 = **Gate 主导**（9 条被阻断）；Planner 在触达时 0 misroute。Validator 在 2 条 query 上拒绝（input 负载问题，非路由问题）。

## 10. Interpretation

| 归因 | 结论 |
|---|---|
| **Gate issue** | ✅ 主要瓶颈：9/12 Vision-positive query 被 `ComplexityGate` 阻断（expected=team 却判 legacy）；Gate FP=0。这是 under-routing（F-002a），本轮未修改，需后续单独迭代。 |
| **Planner issue** | 无能力路由问题：recall 100%、precision 100%、FP 0、0 misroute。存在 **task input 契约忠实度问题**（7 无效 run 中 6 例为 `filter_metadata.include_fields` 枚举外字段；属 prompt/schema 忠实度，非 capability 选择）——这是与本 benchmark 目标正交的独立缺陷。 |
| **Validator issue** | Validator 行为正确（拒绝非法 input 是设计内职责）；拒绝样本暴露的是 Planner input 负载问题而非 Validator 缺陷。 |
| **Provider stochasticity** | 唯一证据：q23.run2 的 `synthesis_task_id` 自引用不一致（run1 同 query 通过）。任务类型选择层面 0 随机性（28/28 稳定）。 |
| **Gold dataset issue** | 未发现（integrity PASS；全部 56 run 的任务类型都与 gold 精确一致，说明 gold 可判且被模型稳定满足）。 |

## 11. Acceptance

```text
总评：PARTIAL
```

| 维度 | 判定 |
|---|---|
| Vision routing acceptance（recall / precision / FP / avg tasks） | **PASS**（recall 100% ≥ 91.7%；precision 100% ≥ 90%；FP 0 ≤ 1/32；avg tasks 3.06 ≤ 3.39） |
| Valid Plan Rate | **FAIL**（87.5% < 98%；7/56 invalid，全部为 task input 负载问题） |
| 相对 baseline 指标 | 不可判定（baseline unavailable） |

**关键说明**：Valid Plan Rate 未达标与 capability routing 无关——7 个无效 run 的任务类型全部正确（无 FN/FP），失败集中在 `filter_metadata.include_fields` 枚举外字段（6 例）与一次 synthesize 自引用漂移（1 例）。capability-routing fix 本身达到全部路由类 acceptance。

## 12. Recommendation

```text
KEEP CURRENT FIX
```

- 对"Planner 能否正确选择 Vision"这一目标：**保留当前 `TASK_CAPABILITY_CATALOG` + routing policy + capability-aware prompt/schema 修复**，证据充分（recall 100% / precision 100% / FP 0 / 类型稳定）。
- 不因 benchmark 结果继续改 prompt / schema / gate / validator（§十二 HARD STOP，避免 benchmark 泄漏与过拟合）。
- `filter_metadata.include_fields` 契约忠实度问题（6/56）属于**独立的、与 routing 无关的**输入契约缺陷，应作为未来单独 workstream 评估——**本报告不实施任何修改**。
- Gate under-routing（9/12 Vision-positive 被阻断）确认 F-002a，需独立迭代，与本次 fix 解耦。

## 13. 最终回答

### Q1 真实 Planner 是否开始主动选择 Vision？
**是。** 修复后真实 `deepseek-chat` Planner 在 12/12 条 Vision-positive query × 2 runs 全部主动输出 `analyze_visual`（24/24），且仅在 Vision-positive query 上输出（FP=0）。对比修复前实证（Stage 9：真实 Planner 从不自发选择 Vision），本次修复产生了**定性反转**。

### Q2 Vision recall 是多少？
**100%**（24/24 runs；12/12 queries）。目标 91.7% ✅。

### Q3 Vision precision 是多少？
**100%**（24/24 runs）。目标 90% ✅。

### Q4 有没有明显的 unnecessary Vision？
**没有。** Vision FP = 0/56；A/B/C/F 16 条 Vision-negative query × 2 runs = 32 runs 无一条使用 Vision；avg tasks 3.06 未膨胀。

### Q5 有没有明显的 false negative？
**没有。** Vision FN = 0/56。所有需要像素级判断的 query（天空阴沉、构图对称、照片内容核对等）均正确进入 Vision。

### Q6 Gate 是否仍然是主要瓶颈？
**是。** Layer A：Gate FN=9，team reachability=11/20，**9/12 条 Vision-positive query 在生产路径上被 Gate 阻断**（q13–q16, q18, q20, q25, q26, q28）。Layer C 中"expected Vision 但 Gate BLOCKED"=9，"Gate TEAM 但 Planner MISROUTED"=0。→ **生产端瓶颈在 Gate，不在 Planner**（F-002a 量化确认）。

### Q7 同一个 query 两次运行是否稳定？
**任务类型选择：28/28 稳定**（run1 与 run2 类型集合完全一致，无"一个 run 用 Vision 一个不用"）。唯一不一致 q23 是 run1 valid / run2 invalid（synthesize 自引用 id 漂移，纯随机性，类型集仍相同）。

### Q8 当前 capability routing fix 是否达到 acceptance？
**路由部分：达到（PASS）**——recall/precision/FP/avg-tasks 全部达标；**整体：PARTIAL**——Valid Plan Rate 87.5% < 98%，但 7 个无效 run 全部是 task input 负载问题（`filter_metadata.include_fields` 枚举外字段 ×6 + synthesize 自引用 ×1），任务类型全部正确，**不构成对 capability routing fix 的否定**。

---

## 14. HARD STOP — 已停止

- ✅ 未修改 Planner / Prompt / Schema / Gate / Validator / 生产代码 / 测试 / 数据库 / 配置（`VISION_MODEL` 仍为 `"null"`）。
- ✅ 未进行"改一点→跑一点"调优循环；本报告为单次 Evaluation 结果（首轮 runner 因脚本 import bug 丢结果后以修复脚本重跑一次，未改变任何 benchmark 配置；最终数据来自该单次完整 56-run，无 retry-select / 无择优覆盖）。
- ✅ 未运行 Stage 10、未运行新 Vision E2E、未开发新功能。
- ✅ 回归：benchmark 后全量测试 **246/246 PASS**（与 benchmark 前一致）。
- 产物：本报告 + `planner_routing_benchmark_results.json`（56 runs 全量 + gate rows + metrics，UTF-8，无 API key / secret / authorization）+ runner 脚本 `benchmark_planner_runner.py`（可复现，属 benchmark 工具，非生产代码）。

### API 成本说明
按批准规模执行 **28 × 2 = 56 次真实 Planner 文本调用**（deepseek-chat，无 Vision API）。首轮尝试因 runner 脚本 import bug 在落盘前崩溃，损失 56 次调用；修复脚本后重跑 56 次并完整保存。两次均为同一配置、同一 prompt/schema/scorer/snapshot；最终报告仅采用第二次（完整保存）的数据。总真实调用 ≈112 次文本 LLM 调用，成本量级较低（deepseek-chat），未超出预算、未增加 run 数。

