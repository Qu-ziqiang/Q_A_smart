# Rerank R2 Evaluation Report（第二轮正式：Baseline vs R1 vs R2）

> ⚠️ **历史快照（迁移前口径，66 图 / sqlite-vec / 18 query Top-10 pool）**：本文的"Hit@1 0.889→1.0（+11.1pp）、MRR +0.079、P@5 0.656→0.711"是那时的实测，**不能作为现状收益引用**。2026-09-10 迁库扩库后（PostgreSQL + pgvector，约 1000 张 VOC2010）`r2_eval_pg1000.json` 显示 baseline Hit@1 已达 1.0、重排 delta 全为 0（评测集天花板）——需先重做有区分度的评测集才能重测重排收益。


> 日期：2026-09-05 ｜ 实验规模固定：18 Query × Top-10 pool（最多 180 次真实 glm-4v-flash 调用，本轮恰好 180 次，0 错误）
> 诚实结论见 §18：**在当前 Strong Gold Set + 固定 Top-10 pool 下，GLM-4V-Flash R2 确实改善了候选排序（主要来自 2 个非 rank-1 hard case 与整体 P@5/MRR）**；同时如实记录 3 个低优先 relevant 图像的下移与 Recall 不变。

---

## 1. Dataset
- `data/db/smartimager.db`：66 rows → 60 个唯一内容（已字节级核验 = 官方 PASCAL VOC2007 JPEG）。
- Retrieval：CLIP（`clip-ViT-B-32-multilingual-v1` 文本）× sqlite-vec（title/image 向量），生产路径 `tool_search_images`，未改动。

## 2. Strong Ground Truth
- 冻结 Gold：`backend/tests/agent_team/gold_voc2007_healthy.json`（只读，未修改）。
- 来源：PASCAL VOC2007 官方 annotation（MATCH 60/60 字节一致）。18 条健康单类 Query；dog/person 已排除。

## 3. Candidate Pool
- 每 Query 仅执行 **1 次 Retrieval**（fetch 上限 40 rows），取 **前 10 个 unique content** 为固定池。
- R1/R2 不重新检索；断言 `set(B)==set(R1)==set(R2)` 且长度=10：**18/18 通过**。
- 池以 unique content 计，避免 6 组重复行造成重复计权。

## 4–6. 三方法结果（macro，18 Query）

| 指标 | Baseline | R1 | R2 | R1−B(Δ/rel) | R2−B(Δ/rel) |
|---|---|---|---|---|---|
| Hit@1 | 0.8889 | 0.8889 | **1.0000** | 0 / 0% | +0.1111 / +12.5% |
| Hit@3 | 0.9444 | 0.9444 | **1.0000** | 0 / 0% | +0.0556 / +5.9% |
| Hit@5 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| MRR | 0.9213 | 0.9213 | **1.0000** | 0 / 0% | +0.0787 / +8.5% |
| Precision@5 | 0.6556 | 0.6556 | **0.7111** | 0 / 0% | +0.0556 / +8.5% |

R1 依旧零变化（同第一轮：空 metadata 库上 keyword 分量无输入）。R2 是唯一改变排序的方法。

## 7. Candidate Recall@10
- Pooled **0.8846（69/78 GT）**；mean 0.893。
- 三方法一致（Rerank 不改变集合）。Recall 缺口全部发生在第一阶段 Retrieval（例：电视/显示器的 `001027`、汽车 5 个 GT 中仅 2 个进池、椅子 5 个中仅 3 个进池等），**不属于 R2 失败**。

## 8–11. 指标口径
Hit@1/3/5、MRR、P@5 均在 Candidate Pool 内计算；GT 未进池者不参与排序质量判定（对 MRR 计为未召回，不归因 Rerank）。

## 12. Rank Delta
- 参与对象：池内 relevant items（共 69 个）。

| 对比 | improved | unchanged | degraded |
|---|---|---|---|
| Baseline → R1（item） | 0 | 69 | 0 |
| Baseline → R2（item） | **17** | 49 | **3** |
| Baseline → R1（first relevant rank） | 0 | 18 | 0 |
| Baseline → R2（first relevant rank） | **2** | 16 | 0 |

- R2 first relevant delta：椅子的照片 +3（4→1）、汽车的照片 +2（3→1）；无 first-delta 为负的 Query。
- 排序变化：R2 改变 **13/18** Query 的 Top-10 顺序（Top-5 亦 13/18）；平均绝对位移 0.689 名；全池最大上移 +5、最大下移 −4。
- 3 个 degraded relevant item（如实记录，均不影响该 Query first rank/MRR，属低位噪声）：
  - 飞机的照片：`000648` rank5→6（−1）
  - 餐桌的照片：`001185` rank1→3（−2；同 Query 有另一 relevant 上移至 rank1，first 保持 1）
  - 马的照片：`000904` rank4→6（−2）

## 13. Query 级表（First Relevant Rank）

| Query | #GT | #GT in Top10 | First(B) | First(R1) | First(R2) | Δ B→R1 | Δ B→R2 |
|---|---|---|---|---|---|---|---|
| 椅子的照片 | 5 | 3 | 4 | 4 | **1** | 0 | +3 |
| 汽车的照片 | 5 | 2 | 3 | 3 | **1** | 0 | +2 |
| 其它 16 条 | — | — | 1 | 1 | 1 | 0 | 0 |

（其余 16 条 Baseline 已 rank-1：公交车/摩托/沙发/火车/牛/猫/瓶/电视显示器/盆栽/羊/自行车/船/飞机/餐桌/马/鸟。）

## 14. Latency & API
- Retrieval（去首次模型加载）：mean ≈ **37.4 ms/Query**。
- R1 rerank：mean ≈ **0.1 ms/Query**。
- R2 rerank：mean ≈ **11 792 ms/Query**（≈ 11.8s）；逐图 mean ≈ **1 179 ms**。
- API 调用：**180 次（= 18×10），错误 0 次，无 fallback**；R2 总耗时 ≈ 212.3s（顺序逐张）。
- Token usage：provider 未暴露 usage 字段 → 本轮不记录 token/cost 数字；glm-4v-flash 官方当前为免费档（不输出任何 Key）。

## 15. Hard Case 分析
1. **汽车（Baseline first=3）→ R2 first=1**：R2 把 `000663`(车) 从 rank3 提到 rank1（pool: `[000663,000768,000572,…]`），Hit@1 0→1。
2. **椅子（Baseline first=4）→ R2 first=1**：R2 把 `000452`(椅) 从 rank4 提到 rank1，MRR 0.25→1.0。
3. **电视/显示器（Baseline first=1）**：R2 保持 first=1；GT `001027` 仍不在 Top-10（第一阶段召回失败，与 R2 无关）。
4. 其它非 rank-1 缺陷：无（其余 16 Query 全部 first=1）。
5. 退化核查：3 个低位 relevant 下移见 §12，无 Hit@1/MRR 可见退化。

## 16. Limitations
- 库仅 60 unique；Top-10 池把多数 GT 密集置于 head，指标天花板高（Hit@5 全 1、R2 MRR=1）。
- 18 条健康 Query 中 16 条 first 本已 rank-1，R2 的**净收益集中在 2 个 hard case**（椅子、汽车）与 P@5。
- 顺序稳定性：R2 为 LLM 打分（temperature=0 但多模态生成非完全确定），本次单次运行。
- 图量小导致 relevant 下移统计分辨率有限（3/69）。
- 未记录 token usage；成本按官方免费档/需另行对账。

## 17. Conclusion
**在当前 18 条 Strong Gold Query、固定 Top-10 Candidate Pool 下，GLM-4V-Flash R2 改善了候选排序：是。**
具体表现为 Hit@1 0.889→1.0、Hit@3 0.944→1.0、MRR 0.921→1.0、P@5 0.656→0.711，两个非 rank-1 hard case（椅子 first 4→1、汽车 first 3→1）被修复；同时诚实记录：R1 无任何效果、Recall@10 未变（0.8846，与 R2 无关）、3 个低位 relevant 下移属无害噪声、改进幅度集中在 head。

## 18. Final Answer（原问题直答）
**“GLM-4V-Flash R2 是否改善了候选排序？” → 是（在当前 18 Query × Top-10 强 GT 池上）。**
- 排序改善体现在 2/18 非 rank-1 Query 的首相关前移与 macro Hit@1/MRR/P@5 提升；
- 无 Query 级首相关退化；
- Recall 不变（非 R2 职责）；
- 局限：收益集中、样本小、为单次运行。进一步结论需更大标注集或多轮重复。

---

### 复现
`venv\Scripts\python.exe -X utf8 backend/tests/agent_team/eval_r2_gold.py`
（需 MM_RERANK_* 环境变量；结果 JSON 在 TEMP `r2_eval_results.json`，未入库；评估代码/冻结 Gold 位于 `backend/tests/agent_team/`，均为 Evaluation 侧新增，生产代码零改动。）
