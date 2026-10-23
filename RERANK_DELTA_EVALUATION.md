# Rerank Delta Evaluation（Rank Delta 分析框架）

> ⚠️ **历史快照（迁移前口径）**：文中"图库 66 张、无 tags/location/description"描述的是旧 SQLite 图库。当前为 **PostgreSQL + pgvector** 且已摄入约 1000 张 VOC2010（`tags` 来自官方 Annotation，`description` 仍为空）。Rank Delta 框架本身与存储无关，结论仍适用。


> 阶段：框架实现 + 单元验证（无真实 Ground Truth、无 GLM 调用）
> 修改范围：仅 Evaluation/Benchmark 代码与文档。未修改 rerank.py / model_reranker.py /
> workers.py 生产逻辑 / Intent / Planner / Validator / Executor，未改任何排序算法或权重。

## 1. 定义与公式（1-based rank）
```
rank_delta = baseline_rank - rerank_rank
    > 0 → improved（排名提前）
    = 0 → unchanged
    < 0 → degraded（排名下降）
```
某侧不存在该 relevant image 时 rank 记 `None`（**禁止用 0 占位**），delta 为 `None`。

## 2. 数据结构与接口（新增 `backend/tests/agent_team/rerank_delta.py`）
- `compute_rank_delta(baseline_ids, rerank_ids, relevant_ids)`
  返回：`pool_consistent`、`pool_diff`、`duplicates`、`items[]`
  item = `{relevant_id, baseline_rank, rerank_rank, rank_delta, status}`
- `query_summary(query, baseline_ids, rerank_ids, relevant_ids, pair="")`
  Query 级：relevant_total / improved·unchanged·degraded / missing_* / average·median delta /
  max_rank_gain / max_rank_drop / First Relevant Rank（双侧 rank/RR/delta）
- `evaluate_pair(pair, query_results)`：把一组 query 标记为 `Baseline vs R1` / `Baseline vs R2` / `R1 vs R2`
- `global_summary(summaries, pair=None)`：全局统计（见 §4），pair 隔离
- 表渲染：`format_query_table` / `format_first_rr_table` / `movement_rows` / `format_movement_table`

status 取值：`improved | unchanged | degraded | missing_baseline | missing_rerank | missing_both`。

## 3. Query-level 统计（每 Query）
- relevant images 总数
- improved / unchanged / degraded 张数
- average_rank_delta、median_rank_delta、max_rank_gain、max_rank_drop（多张相关图全部参与，非只取第一张）
- First Relevant Rank：
  `baseline_first_relevant_rank = min(相关图 baseline rank)`（无则 None）
  `rerank_first_relevant_rank   = min(相关图 rerank rank)`（无则 None）
  `first_relevant_rank_delta    = baseline_first_relevant_rank - rerank_first_relevant_rank`（任一侧缺失→None）

## 4. Relevant-image-level / 全局统计
- total_relevant_images = pool 一致 query 的全部相关图（含 missing 项）
- improved / unchanged / degraded / missing_* 计数
- average·median rank delta（只统计两侧都有的标准 delta）
- `improvement_rate = improved_count / total_relevant_images`
- `degradation_rate = degraded_count / total_relevant_images`
- Query-level 单独报告：以 first_relevant_rank_delta 符号划分 improved/unchanged/degraded/unknown query；
  Query 数量与 relevant image 数量**不混用**（分别输出 image-level 与 query-level 两个 rate）。

## 5. First Relevant Rank 与 MRR 的关系
MRR 对单个 query 取 `1 / first_relevant_rank`。因此：
- `first_relevant_rank_delta > 0` 通常表示该 query reciprocal rank 提升；
- 报告展示 `Baseline First Rank | Rerank First Rank | Rank Delta | Baseline RR | Rerank RR`
  供人工解释 MRR 变化来源；本模块**不重复实现 MRR 聚合**。

## 6. Candidate Pool 一致性检查
每次计算前验证 `set(baseline_ids) == set(rerank_ids)`：
- 不一致 → 记录 `candidate_pool_mismatch`（含 query 与 pool_diff）到全局报告的 `queries_pool_mismatch`；
- 该 query 的 item **不进入** image-level 标准统计/rates（避免把异常当标准 rerank delta）。
R1/R2 契约本身要求候选集合不变，正常时应 0 mismatch。

## 7. 边界情况处理
空候选列表 / 空 relevant_ids / relevant 不在候选池（missing_*）/ 重复 ID（取第一次出现位置并在
`duplicates` 暴露）/ 候选集合变化（§6）/ 单候选 / 全部 unchanged / 全部 degraded / 多相关图 /
rank 在 Top-K 之外但仍完整 ranking 内 → 使用完整排名，不误判 missing。

## 8. 当前 Ground Truth 状态
**无可靠 semantic Ground Truth**（图库 66 张 VOC 样张：无 tags/location/description 语义；
title=文件名；仅尺寸/上传时间等属性）。因此：
- 未伪造 relevant IDs；
- 不用 CLIP/GLM/聚类结果当 Ground Truth；
- 本次仅交付计算框架与演示（toy ranking，零 API）。

## 9. 当前是否能够正式产生结果
**Evaluation framework ready, but no semantic ranking conclusion is claimed.**
待外部/人工独立 Gold Set 就绪、且经 `eval_rerank_glm.py --yes`（或 R1-only 对比）产出真实
Baseline/R1/R2 排名后，本模块才用于正式 Rank Delta 报告。

## 10. 测试结果
`test_rerank_delta.py` 18/18 PASS：
toy 定义样例（A/C: -1/+2、first delta 0）、+5/-3/unchanged、多相关图统计、first relevant rank、
missing 三类、空候选、空 relevant、重复 ID（首现位）、pool mismatch 隔离、Top-K 外完整排名、
全局 rates、pair 分离、确定性、单候选、表渲染。
全量回归：`backend/tests/agent_team` **351 passed**。

## 11. 演示
`venv\Scripts\python.exe -X utf8 backend/tests/agent_team/run_rerank_delta_demo.py`
（本地固定 ranking 演示 Baseline vs R1 / vs R2 / R1 vs R2，含 mismatch 展示，零 API、不宣称效果）。
