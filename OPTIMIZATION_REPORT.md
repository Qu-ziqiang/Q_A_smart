# 优化刀（A2 / A1 / B1 / C2）

默认全部关闭，旧路径保持可对照。未改冻结的 200 条 Judge 评测代码与 gold。

## 开关（环境变量）

| 变量 | 默认 | 作用 |
|---|---|---|
| `SM_RERANK_GAP_TRIGGER` | 关 | A2：CLIP top1−top5 分差 `< τ` 才走昂贵 R2 |
| `SM_RERANK_GAP_TAU` | `0.08` | A2 分差阈值 |
| `SM_RERANK_BATCH` | 关 | A1：多图单次重排；关则逐张旧路 |
| `SM_RERANK_BATCH_MAX` | `10` | 单次最多图数；更大则两轮淘汰 |
| `SM_DAG_PARALLEL` | 关 | B1：同层非视觉最多 2 路；Vision 仍串行 |
| `SM_PLAN_CACHE` | 关 | C2：Planner 计划缓存（与 Memory episode 不是一层） |

`MM_RERANK_*` 仍控制是否接入 GLM-4V。未配 R2 时 A2/A1 不生效。

## 做了什么

### A2 重排触发（第一刀）

`default_model_reranker` 在调用 R2 前看 CLIP/检索分差。flag 关 = 现在的「配了 R2 就逐张打」。flag 开且 gap ≥ τ 则保持 CLIP 原序，不打 VLM。

计数：`opt_flags.STATS` 的 `rerank_considered / triggered / skipped`，`rerank_trigger_rate()`。

### A1 重排批处理（第二刀）

`MultimodalReranker.rerank`：`SM_RERANK_BATCH=1` 时一次多图 + JSON `{"order":[1..K],"reason":"..."}`。解析失败由 provider 重试一次；仍失败或超时 → **R1/CLIP 序**，不回退逐图（避免再吃 12s）。K > `BATCH_MAX` 先全量排序再对 top-cap 精排（两轮）。

单测全部 mock VLM。实测延迟 / Hit@1 必须另跑标注集，不要动已冻结的 200 条。

迁库后千级库 Hit@1 已接近天花板；简历上站得住的是 **12s → 批处理后的墙钟**，不是再刷 Hit@1。

### B1 DAG 层内并行

`execute_stream` 仍是同步生成器。flag 开时按 ready 波次：非视觉 `ThreadPoolExecutor(max_workers=2)`，Vision 任务始终主线程串行。flag 关则原串行路径一行不改。多数计划是 retrieve → X → synthesize，墙钟收益小于 A1。

### C2 计划缓存

规范化 query 精确命中，否则相似度 ≥ 0.95（可注入 `embed_fn`；默认 token Jaccard）。TTL 1h。`registered_task_types` 指纹变更失效。Replan（带 `observation`）不读写缓存。命中不计 `planner_call_count`。无线上流量时命中率只是日志，不要写成简历「成功率」。

## 怎么测（mock）

```text
venv\Scripts\python.exe -X utf8 -m pytest backend/tests/agent_team/test_optimizations.py backend/tests/agent_team/test_model_reranker.py backend/tests/agent_team/test_executor.py -q
```

## 实测（另跑，不要进冻结 Judge）

建议单独脚本，输出到 `evaluation/results/opt_ab/`，记录：

1. **A2**：同一标注集，flag 关 vs 开。列触发率、平均延迟、Hit@1。简单 query 应少打 R2。
2. **A1**：同一标注集 + 真实 GLM-4V。列平均延迟（目标对照逐图 ~12s）、Hit@1（质量未验证，只报告不吹）。
3. **B1**：抽「retrieve 后挂 group + 非视觉」计划，墙钟前后对比。
4. **C2**：重复同一句 / 近义句，Planner 调用次数与延迟。无线上流量不要写命中率百分比。

## 没做（按建议推迟）

B2 embedding 路由、B3 LLM Supervisor review、C1 额外 SQL 抽取、Gate 视觉意图（9/12 FN）。
