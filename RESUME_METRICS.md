# 简历口径：能写 / 不能写

日期：2026-09-15。每个数字带出处。生产 Planner 仍是 DeepSeek；`config.yaml` 未改。

路径相对于 `backend/ai_func/recommendation/`。

---

## 能写

| 口径 | 出处 |
|---|---|
| Gate 视觉核验 FN 9/12→规则修复后 **0/28**；反例 test FP=**0**（现 test n=32）；Gate p95 **<0.02ms**（round2 合计 p95 0.011ms） | `evaluation/results/gate_track/gate_track.json`；加厚后 `evaluation/results/gate_track_round2/gate_track.json`；数据集 `evaluation/datasets/gate_contrast_legacy.json` |
| 万级图库(10k) 三路 CLIP 检索，p95 **122ms**（HNSW/cosine）；千级对照 65ms 可并写 | `evaluation/results/retrieve_latency_10k.json`（warmup 后 p95 121.879ms） |
| 故障注入恢复 **6/6** | `evaluation/results/fault_injection.json` |
| Catalog A/B DeepSeek：规划 **+0.23**、路由 **+0.15**（pass 微降，不写成功率） | `evaluation/results/task_catalog_ab_deepseek/comparison.json`（planning +0.234，routing +0.15） |
| P1 简单子集均 Task **4.85→3.261**；全量 plan_ok **100%**（T1 空计划回归已用硬底线修回） | before：`evaluation/results/task_catalog_ab_deepseek/p1_plan_bloat.json`；T1.5：`evaluation/results/p1_after_retest_t15/p1_after_summary.json` |
| 冻结 200 条 Judge：路由 **3.83**（五维最低，用来定位短板）；Catalog A/B 路由 **4.05→4.20**。禁止接成「3.8→4.2」 | 金标：`evaluation/results/benchmark_summary.md`；A/B：`evaluation/results/task_catalog_ab_deepseek/comparison.json` |
| 400 张标注子集 70 query：Hit@1 **76.8%→89.9%**，MRR **0.86→0.94**；逐张约 **1.2s/张**；τ=0.006 触发率 **31%** | `evaluation/results/a2_gap_400/a2_eval.json`（不是万级 Hit@1） |
| VLM 试点 36 条：Gate TEAM **36/36**，结构化断言 **36/36**（进程内 `deepseek-v4-pro`，yaml 仍 null） | `evaluation/results/vlm_pilot/vlm_pilot.json` |

---

## 不能写

| 口径 | 原因 / 出处 |
|---|---|
| 端到端 P95 **2.7s** | 否。T1 Planner 真打 p95 **3029ms**（`p1_after_retest/p1_after_summary.json` stage_latency.planner）。T2 GLM/R2/Synth 分阶段未跑（无 `MM_RERANK_*`：`evaluation/results/a1_a2_real/blocked.json`）。mock 80ms/503ms 继续不写。 |
| A1 **12s→4s** / 已批处理 | 否。`glm-4v-flash` 一次 10 张报 1210，评测仍逐张 p50 12.1s。 |
| 路由 **3.8→4.2** 同一评测前后 | 否。3.83 是冻结 200 条；4.20 是 240 条 A/B After。简历写「短板 3.8，A/B 4.05→4.20」。 |
| 万级 Hit@1 | 否。质量只在 400 张子集上测过。 |
| Judge overall 大涨 | 否。Catalog overall 仅 +0.093，pass 63.8%→62.9%。 |
| 单独的 **4.85 / 4.9 Task** | 否。那是未修复 before。只允许 **4.85→3.261**。 |
| 「简单均 Task ≤3.2」 | 否。T1.5 为 **3.261**（`p1_after_retest_t15/p1_after_summary.json`）。plan_ok 100% 可写。 |
| VLM「已很快 / 已上生产」 | 否。试点 p50 24.8s / p95 67s；生产 `VISION_MODEL` 仍 `"null"`。断言成功只说明结构化输出可用，不说明条件判定质量。 |
| A2 触发率 20–40%（默认 τ=0.08） | 否。默认 0.08 在 400 集上 100% 触发；**31%** 是评测 τ=0.006。 |

---

## 本轮任务文件

- T1：`evaluation/results/p1_after_retest/`（空计划回归基线）
- T1.5：`evaluation/results/p1_after_retest_t15/`
- T2：`evaluation/results/a1_a2_real/blocked.json`（旧阻塞记录）；真 GLM 见 `a2_gap_400/`
- T3：`evaluation/results/gate_track_round2/`（旧 `gate_track/` 未覆盖）
- T4：`evaluation/results/vlm_pilot/`
- A2 400 子集：`evaluation/results/a2_gap_400/`
