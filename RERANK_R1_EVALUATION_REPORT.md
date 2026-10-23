# Rerank R1 Evaluation Report（第一轮正式：Baseline vs R1）

> ⚠️ **历史快照（迁移前口径，66 图 / sqlite-vec）**：本文结论"R1 在本 Gold Set 上对排序零影响"是那时的实测。2026-09-10 迁库扩库后（PostgreSQL + pgvector，约 1000 张 VOC2010）见 `r1_eval_pg1000.json`：baseline Hit@1/MRR 已为 1.0、P@5 0.944、检索均值 65.0ms、Candidate Recall pooled 0.569。**两个口径不可混用**。


> 日期：2026-09-05 ｜ R2/GLM：**NOT RUN**（0 付费 API 调用）
> 结论（诚实）：在本 Strong Gold Set 上，**R1（deterministic rerank，默认参数）对排序零影响**——
> Baseline 与 R1 的 18/18 条 Query 排序完全相同：0 improved / 0 degraded。不做任何“R1 更优”宣称。

---

## 1. Dataset
- SmartImager `data/db/smartimager.db`：66 rows → **60 个唯一内容**（6 组字节重复）。
- 全部经 VOC2007 官方核验：**60/60 与官方文件 SHA-256 一致**（见 `VOC_GROUND_TRUTH_VALIDATION.md`）。
- 检索管线：CLIP 文本编码（`clip-ViT-B-32-multilingual-v1`）+ sqlite-vec（title/image 向量）→ `tool_search_images`（生产路径，未改动）。

## 2. Strong Ground Truth 来源
- 仅使用 PASCAL VOC2007 **官方 annotation XML**（Oxford 官方源，字节级 MATCH 后解析）。
- 标签冻结于：`backend/tests/agent_team/gold_voc2007_healthy.json`（本次评估只读该文件，不重新生成标签）。
- 排除：`dog`(n=1)、`person`(n=26)。未使用 CLIP/R1/GLM/聚类产生的任何标签。

## 3. Query 列表（18 条健康单类）
飞机 6、马 6、自行车 5、汽车 5、公交车 5、牛 5、瓶子 5、椅子 5、沙发 7、猫 4、船 4、
电视/显示器 4、餐桌 4、火车 3、摩托车 3、盆栽 3、鸟 2、羊 2（数字=GT 唯一内容数）。

## 4. Experimental Setup（固定参数）
| 参数 | 值 |
|---|---|
| Retrieval | `tool_search_images(query, targets=[title,description,image], filters={}, limit=40)` |
| 排序基线 | 检索返回行顺序 → 去重为 unique content 序 |
| R1 | `rerank_candidates(query, pool_candidates)`（默认权重；未改 weights/tokenizer） |
| 评估单位 | **unique image content**（6 位 VOC id；与 Gold Set 一致，避免重复行计权） |
| 候选池 | 每次仅检索一次；R1 复用同一对象池；断言 `set==` 且长度相等（全部通过） |
| 指标域 | Candidate Pool 内（说明：不把“未召回”当成 Rerank 失败） |
| 环境 | venv python 3.12；单机；模型已本地缓存 |

实测 pool_size 为 35–39（检索在 66 张库上返回 35–40 行；同一池去重后）。

## 5. Candidate Pool Strategy
- 每个 Query：只做 1 次检索（含召回），R1 不做二次召回，仅重排同一池。
- pool 一致校验：18/18 Query `set(baseline)==set(r1)` 且 `len` 相等 —— 通过。

## 6–7. Baseline 与 R1 结果（macro 均值，18 Query）
| 指标 | Baseline | R1 | Abs Δ | Rel Δ |
|---|---|---|---|---|
| Hit@1 | 0.8889 | 0.8889 | 0.0 | 0% |
| Hit@3 | 0.9444 | 0.9444 | 0.0 | 0% |
| Hit@5 | 1.0000 | 1.0000 | 0.0 | 0% |
| MRR | 0.9213 | 0.9213 | 0.0 | 0% |
| Precision@5 | 0.6556 | 0.6556 | 0.0 | 0% |

Query 级（B=R1，均相等）：

| Query | #GT | #GT in Pool | Pool | Baseline/R1 Hit@5 | First Rank (B=R1) | First Δ |
|---|---|---|---|---|---|---|
| 公交车的照片 | 5 | 5 | 36 | 1 | 1 | 0 |
| 摩托车的照片 | 3 | 3 | 36 | 1 | 1 | 0 |
| 椅子的照片 | 5 | 5 | 37 | 1 | 4 | 0 |
| 汽车的照片 | 5 | 5 | 35 | 1 | 3 | 0 |
| 沙发的照片 | 7 | 7 | 35 | 1 | 1 | 0 |
| 火车的照片 | 3 | 3 | 36 | 1 | 1 | 0 |
| 牛的照片 | 5 | 5 | 37 | 1 | 1 | 0 |
| 猫的照片 | 4 | 4 | 35 | 1 | 1 | 0 |
| 瓶子的照片 | 5 | 5 | 36 | 1 | 1 | 0 |
| 电视或显示器的照片 | 4 | 3 | 37 | 1 | 1 | 0 |
| 盆栽的照片 | 3 | 3 | 35 | 1 | 1 | 0 |
| 羊的照片 | 2 | 2 | 35 | 1 | 1 | 0 |
| 自行车的照片 | 5 | 5 | 36 | 1 | 1 | 0 |
| 船的照片 | 4 | 4 | 38 | 1 | 1 | 0 |
| 飞机的照片 | 6 | 6 | 39 | 1 | 1 | 0 |
| 餐桌的照片 | 4 | 4 | 35 | 1 | 1 | 0 |
| 马的照片 | 6 | 6 | 35 | 1 | 1 | 0 |
| 鸟的照片 | 2 | 2 | 39 | 1 | 1 | 0 |

## 8. Metric Delta
- 绝对 delta：全部 0.0；相对 delta：全部 0%。
- MRR 差异来源不存在（first relevant rank 18/18 相同）。

## 9. Rank Delta Analysis（pool 内 77 个 relevant 项）
- improved：**0**
- unchanged：**77**
- degraded：**0**
- average rank delta：0.0（每 Query 0.0）
- median rank delta：0.0
- max gain：0；max drop：0
- First Relevant Rank Delta：18/18 = 0
- Top 10 Improved / Top 10 Degraded：**无**（空）

原因：库内 description/tags/metadata 为空、title=文件名，keyword/metadata 分量对排序无输入；
R1 默认权重在纯向量池上等价于保持原序。属数据集特性的诚实结果，非实现缺陷。

## 10. Candidate Recall
- Pooled：**0.9872（77/78）**；mean：0.9861。
- Baseline 与 R1 相同（契约：R1 不改变集合）；出现差异即视为评估 bug —— 未出现。
- 唯一召回缺口：`电视或显示器的照片` 的 GT 之一 `001027` 未进入 pool（检索阶段），与 Rerank 无关。

## 11. Latency
- Retrieval（含首次模型加载，第 1 Query 14.9s 冷启动）：mean ≈ **844 ms/Query**；冷启动后稳态 ≈ **18 ms/Query**。
- R1 rerank：mean ≈ **0.21 ms/Query**。
- Total R1：retrieval + 0.21 ms（rerank 成本可忽略）。
- Delta latency：+0.21 ms/Query。

## 12. Improved / Unchanged / Degraded
- Query-level（first delta 符号）：improved 0 / unchanged 18 / degraded 0。
- Image-level（pool 内 relevant item）：improved 0 / unchanged 77 / degraded 0。

## 13. Failure Cases
1. **电视或显示器的照片 → 001027 召回缺失**（#GT 4 中 pool 内仅 3）：第一阶段向量召回问题，R1 无权改变集合，非 Rerank 退化。
2. **椅子的照片 / 汽车的照片 MRR 低（0.25 / 0.333）**：first relevant 在 B=R1 的 rank 4/3，两侧完全一致 → Baseline 排序局限，非 R1 引入。
3. 无其它异常；无 pool mismatch；无 GT/Query 变更。

## 14. Limitations
- 库体量小（60 unique），类别 Query 每类 2–7 相关图，统计分辨率有限。
- 元数据空导致 R1 的 keyword/metadata 维度在本库**无输入**——结论只适用于“空 metadata 库 + 默认 R1”。
- Hit@5 全部为 1 属天花板效应；MRR/P@5 分辨率稍高但仍受库小限制。
- 一次运行（无重采样）；latency 受冷启动影响已单独标注。

## 15. Conclusion
- **R1 在本 Strong Gold Set 上：既无提升也无退化（全部 0 delta），不能宣称任何排序收益。**
- Rerank 之前，检索已把 16/18 Query 的 first relevant 放在 rank 1，MRR 0.92 主要由 Baseline 贡献。
- 因 R1 不使用任何视觉/语义二次信号，在空 metadata 数据上的空操作结果符合预期。

### 是否建议继续跑 R2
- **可选、不强制**：R2（GLM 多模态打分）是唯一能引入“图像内容”信号的 rerank；在 001027 这类“向量召回成功但排位不高”的 case 上有理论改进空间，但需：
  1. 显式成本批准（40 候选规模估算见 `GLM_R2_REAL_PROVIDER_INTEGRATION_REPORT.md`）；
  2. 复用本 Gold Set 与同一候选池；
  3. 仅作受控实验，先出结果再谈结论（本报告不预先宣称任何效果）。

---

### 复现
`venv\Scripts\python.exe -X utf8 backend/tests/agent_team/eval_r1_gold.py`
结果 JSON 在系统 TEMP `r1_eval_results.json`（未入库）。评估代码与 Gold Set 文件均位于 `backend/tests/agent_team/`（仅 Evaluation 侧新增）。
