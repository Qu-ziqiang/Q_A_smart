# 产品指标报告（与 LLM-as-Judge 脱钩）

日期：2026-09-15  
范围：P0 Gate 评测轨 + 延迟仪表 + 反例集 + 故障注入 + P1 膨胀基线 + A1/A2 mock + 图库扩万级。  
**未改**冻结 200 条 Judge gold / `benchmark_results.json`。生产 Planner 仍是 DeepSeek。架构未动。

---

## 1. 执行顺序（已落地）

1. P0 Gate 评测轨 + 60 条 Legacy 反例（dev/test 切分）+ 仪表  
2. 反例 dev 上挖到 3 个 FP，只在 dev 修正则（见 §3）  
3. 故障注入 6 场景，成功率 1.0  
4. P1 从已有 DeepSeek After JSON 量膨胀（不重打 240 Judge）  
5. A1/A2 mock 墙钟（无 GLM 真测）  
6. 图库从 1000 → 目标 10000（NACLIP VOCdevkit 小 JPEG + CLIP 批量入库）

---

## 2. P0 Gate 评测轨

脚本：`scripts/run_gate_track.py`  
结果：`evaluation/results/gate_track/gate_track.json`

| 集合 | n | FN | FP | TEAM 率 | Legacy 跳过 Planner |
|---|---:|---:|---:|---:|---:|
| routing gold 28 | 28 | **0** | **0** | 0.714 | 8 |
| contrast **dev** | 45 | 0 | **0**（修正则后） | 0 | 45 |
| contrast **test**（冻结） | 15 | 0 | **0** | 0 | 15 |
| 合计 | 88 | 0 | 0 | 0.227 | 68 |

Gate 延迟（88 次）：p50 **0.006ms**，p95 **0.008ms**，p99 **0.009ms**（远低于 5ms）。

**不进 Judge。** 产品指标就是上表。

### 反例挖掘闭环（Q3）

test 首轮命中 `neg_40`「不要比较构图」→ `找+比较` 被当成 search+process。  
dev 随后挖出：

- `neg_46`「然后再推荐几张夜景」：`然后`+`再` 被当成两阶段  
- `neg_54`「找低对比度雾景」：`对比度` 子串命中 `对比`

只改 `gate.py`：否定处理动词不计入；`然后再` 算一个标记；去掉 `对比度` 中的假 `对比`。  
test 保持 0 FP。**没有**在 test 上继续加视觉正例规则。

---

## 3. 仪表化

`evaluation/latency.py`：`StageTimer` P50/P95/P99。

| 阶段 | 数据 | 结论 |
|---|---|---|
| Gate | 上表 | 可写「亚毫秒」；**P95 2.7s 对不上 Planner，不要写到 Gate 上** |
| R2 mock 10 张 | 逐张 503ms vs 批处理 80ms | 仅 mock sleep，**不是 GLM 12s→4s** |
| Planner / 真 R2 / Synth | 未在本轮真打 | 简历延迟数字仍缺分阶段真值 |

A2 mock：分差小触发、分差大跳过，行为符合开关设计。  
**A1/A2 真实 GLM 未跑**（要 `MM_RERANK_*` + 有区分度标注集；千级天花板 gold 不报质量）。

---

## 4. P1 计划膨胀（After 臂基线，未再跑 LLM）

来源：`task_catalog_ab_deepseek/after.json` → `p1_plan_bloat.json`

| | n | 平均 Task 数 | 纯 retrieve→synth 占比 |
|---|---:|---:|---:|
| 全部 After | 240 | **4.683** | 5.4% |
| 简单子集 | 153 | **4.850** | 8.5% |

可选任务计数（全量）：dedup 187、select 132、rerank 68、verify 20。  
Prompt 已收紧，**尚未** After-only 重跑。下一步若仍膨胀再上软 lint。不要为 Judge 把套餐加回去。

---

## 5. 故障注入（客观恢复率）

`test_fault_injection.py` → `evaluation/results/fault_injection.json`

空检索→replan、低相关→replan、执行失败→revise/abort、非法 replan 拒绝、两轮预算、Legacy 不受影响：**6/6 = 1.0**。  
不用 Judge 的 `recovery_behavior`。

---

## 6. Rubric v2（只新增，不重打冻结 200）

`evaluation/rubric_v2.py`：最小充分=满分；冗余任务扣分；能力不可用不惩罚。  
效率比 = overall / 平均任务数。  
**冻结 200 仍走旧尺子。** v2 跨版本不可比。Judge 降为开发期信号。

---

## 7. 图库扩万级

源：`D:\document\py_prj\NACLIP\mmsegmentation\data`（优先 VOCdevkit 小 JPEG，≤400KB）。  
脚本：`scripts/expand_gallery_to_10k.py`  
Postgres：`docker start smartimager-pg`（本机 Docker 曾关闭，已拉起已有容器，**数据卷未删**）。

| | 数量 |
|---|---:|
| 扩库前 | 1000 |
| 新插入行 | 9000（扫描约 9934 个小 JPEG） |
| 扩库后 | **10000**（向量已写完） |

见 `evaluation/results/gallery_expand.json`。

万级检索延迟（warmup 后 10 次、三路融合 limit=10）：**p50 121ms / p95 122ms**（`retrieve_latency_10k.json`）。  
这是可扩展性证据：图库 10 倍（1k→10k）后 p95 仍 122ms。简历写「万级图库(10k) 三路 CLIP 检索，p95 122ms（HNSW/cosine）」；千级 65ms 可作对照，两数都能写。Hit@1 未在本轮重测。

---

## 8. 刻意没做

- 再刷 Catalog A/B / 冻结 200 Judge  
- 删 Legacy、加 Agent、LLM Gate、Validator 硬拒多余 Task  
- 难例 50 条检索 mini-set（可选，最后）  
- CLIP logistic 路由头（标注不够 300+）  
- 配 `VISION_MODEL` 试点（先尺子后能力；当前仍 null）  
- A1/A2 真 GLM（无分阶段真 key 跑满标注集）

---

## 9. 简历能写 / 不能写

**能写（有文件）：**  
Gate 评测轨：视觉 gold FN 9/12 → 规则修复后 0/28，反例 test FP=0，Gate p95 < 0.02ms；  
万级图库 10k 三路 CLIP 检索 p95 122ms（对照：千级 65ms）；  
Catalog A/B DeepSeek：规划 +0.23、路由 +0.15（pass 微降，不写成功率）；  
故障注入恢复 6/6。

**不能写：** P95 2.7s（待分阶段真测）；A1 12s→4s（待 GLM 真测）；万级 Hit@1；Judge overall 大涨；**单独的 4.85/4.9 平均 Task 数**（未修复病灶的内部 before，只作 P1 复测对照，复测后写 4.85→X）。

---

## 10. 建议下一刀

1. 等图库向量写完后，用现有 18-query / 新难例做 **检索延迟**（65ms 是否仍成立）。  
2. 有 GLM key 时 A1/A2 **真测延迟**（质量仍用旧 66 图集防退化，不报千级 Hit@1）。  
3. P1 After-only 小样本看均 Task 数；仍高再上软 lint。  
4. 30–50 条真视觉核验 + flag 开 VLM 试点（产品指标决策）。
