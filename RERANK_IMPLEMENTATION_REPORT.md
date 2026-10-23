# RERANK_IMPLEMENTATION_REPORT

> ⚠️ **历史快照（迁移前口径）**：本文的检索链路描述为 `sqlite-vec`（图库 66 张）。当前为 **PostgreSQL + pgvector（SQL `<=>` + HNSW/cosine，元数据同库过滤）**，图库约 1000 张。重排算法与代码结论不变。


> 项目：SmartImager — Rerank R1（Deterministic Second-Stage Reordering）
> 日期：2026-09-01
> 范围：仅实施 Rerank R1。**未修改 Intent Rewrite**；未引入 LangGraph / 新 Agent / 新 TaskType /
>      新 LLM 调用 / 新 Reranker 模型 / Cross-Encoder；未改 Planner / PlanValidator / DAGExecutor 核心逻辑。
> 回归基线：改造前 298/298 → 改造后 **316/316 PASS**

---

## 1. 修改文件

| 文件 | 动作 | 内容 |
|---|---|---|
| `backend/ai_func/recommendation/orchestration/rerank.py` | **新增** | 纯函数式确定性 Rerank（`rerank_candidates`） |
| `backend/ai_func/recommendation/orchestration/workers.py` | 最小改 | `RetrievalWorker.__init__` 增加 `rerank_enabled=True` + `reranker=None`；`run()` 在 `_to_candidate` 后、构造 `RetrievalOutput` 前插入 rerank（异常回退原序） |
| `backend/tests/agent_team/test_rerank.py` | **新增** | 18 项单元 + RetrievalWorker 集成测试 |

Intent Rewrite（`orchestration/intent.py`、`__init__.py`、`orchestrator.py` goal_override 接线）**零改动**。

## 2. Rerank 真实数据流

```
Query
 → CLIP text embedding（encode_text，512d）
 → sqlite-vec cosine retrieval（vec0 MATCH k=，多目标加权融合）
 → RetrievalWorker._search_tool 返回 rows → _to_candidate → candidates（ImageCandidate 列表）
 ↓ [R1 插入点，workers.py RetrievalWorker.run]
 → rerank_candidates(query=task.input.query, candidates, intent=None)
      vector_score（复用 candidate.score = 1 - distance，池内 min-max 归一化）
    + keyword_overlap（复用 memory._keyword_overlap：CJK bigram + latin 词，title/desc/tags）
    + metadata_match（intent.metadata_constraints 存在时按 MetadataWorker 语义计数；当前未接线 → 0）
 → 稳定降序（同分保持原序）
 → RetrievalOutput(candidates=重排后同集合, query, total)
 → Synthesizer → validate_selection → SSE
```

候选集合不变：`set(before_ids) == set(after_ids)`；仅排序变化。

## 3. Scoring 公式

```
rerank_score = w_vec * vector_score  +  w_kw * keyword_overlap  +  w_meta * metadata_match
```
- `vector_score` = `(score_i - min)/(max - min)` 池内归一化 ∈ [0,1]（所有 score 相等 → 全 0；不重算 embedding）。
- `keyword_overlap` = `|tokens(query) ∩ tokens(title+desc+tags)| / |tokens(query)|` ∈ [0,1]，复用 memory.py 的 `_tokens`/`_keyword_overlap`。
- `metadata_match` = 命中的 metadata constraint 比例 ∈ [0,1]；**无 constraints（普通 query）时恒为 0（中性）**；约束匹配复用 `MetadataWorker._field_value/_apply_op`（延迟导入，避免 workers↔rerank 循环）。

## 4. 权重（初始默认值）

| 权重 | 默认值 | 理由 |
|---|---|---|
| `w_vec` | **1.0** | vector 分是主信号；归一化后 [0,1] 且与检索序单调一致，保证无其他信号时退化为原排序 |
| `w_kw` | **0.35** | 次信号：词法精确命中最多在 vector 归一化差距 ≤0.35 内抬升候选；无法推翻"强 vector 差距"，只在近邻序中做词法修正 |
| `w_meta` | **0.20** | 三信号：仅在显式 metadata constraints 传入时生效（当前 Team 链路未把 Intent 接入 Retrieval，因此为 0/中性，不把硬过滤软化成排名） |

三种分量均已归一化到相同量纲 [0,1]，避免某一项因量纲压制其它项。测试中可显式传入 `weights` 覆盖以验证单信号效应。

## 5. 为什么选择这些信号

- **全部来自当前系统已有数据/代码**：`candidate.score`（CLIP cosine 分，检索层已有）、`title/description/tags`（库内文本字段）、metadata 字段（`MetadataWorker` 语义）、tokenizer（`memory._keyword_overlap`，已被 memory 测试验证）。
- 不引入任何新依赖/模型；全部确定性、可解释、易测。
- keyword/metadata 补偿的是 CLIP 语义召回"词面不敏感"的短板（例如 exact tag/标题词命中）；metadata 项默认中性，避免误把未来硬过滤变软。

## 6. 插入点

`RetrievalWorker.run()`（workers.py:265+）：`rows → _to_candidate → candidates` 之后、`RetrievalOutput(...)` 之前。
- **不是** DAG 新节点；**不是** Planner 的 TaskType；不改变 Executor 拓扑。
- 默认 `rerank_enabled=True`；`reranker=` 可注入（测试/未来换实现）；任何异常 print 后**回退原候选顺序**，绝不让 Agent 请求失败。

## 7. Candidate ID 集合是否不变

**是（硬约束 + 守卫）**：
- `rerank_candidates` 只对同一批对象排序并返回新列表，不做增删改。
- 守卫：空/单候选、全部权重为 0、任一候选缺数值 `score`（真实链路恒有）→ 直接返回原顺序。
- 稳定排序保证同分不产生随机顺序。
- 测试 `test_pool_preserved_across_reorders` 等显式断言 `set(before_ids)==set(after_ids)` 且长度不变。

## 8. 测试数量与结果

`test_rerank.py` **18/18 PASS**：
vector 单独排序 / keyword 改变排序 / 中英混合 tokenizer 复用 / metadata 改变排序 / 无 intent 中性 / 多信号组合 / 同分稳定 / 空候选 / 单候选 / 缺字段不崩溃 / 缺分数保持原序 / ID 集合不变 / 数量不变 / 确定性多次一致 / 不修改输入对象 / 权重全零回退原序 / RetrievalWorker 集成（开启重排同集合、关闭保持行序、reranker 异常回退原序）。

## 9. Agent Team 全量测试结果

**316/316 PASS**（= 改造前 298 + 新增 rerank 18）。覆盖：workers/dag/integration/memory/intent/routing benchmark 等全部既有用例在默认开启 rerank 后仍通过（既有 retrieval 测试为 1/0 候选，不受排序影响；其它多候选 fixture 无 score 或缺词法信号 → rerank 守卫走原序）。

## 10. 是否已可进入正式 Retrieval Evaluation

**功能已就绪，可以进行评测——但本报告不宣称任何效果提升**（未运行正式评测）。

可评测性保证：
- Baseline（关闭 rerank）：`RetrievalWorker(search_tool=..., rerank_enabled=False)` 或 `rerank_candidates(..., weights 全 0)` → 与原 CLIP 排序逐位一致，已测。
- Rerank（开启）：同一候选池新顺序，已测。
- 因此可以设计离线评测：固定一组 query + 相关度标注 → 分别跑 Baseline 与 Rerank 输出 top-K → 比较 Recall@K / MRR 等。
- 诚实边界：目前 metadata 项在 Team 链路为中性（Intent 未接入 Retrieval），当前 Rerank 实际生效的是 vector+keyword 两项；keyword 只有在标题/描述/标签与检索词有词面重叠时才改变排序。正式评测前建议先确认候选库中 title/desc/tags 的覆盖率与质量。

---

*本实施未修改 Intent Rewrite / Planner / Validator / Executor / SearchRepository / Synthesizer / SSE / 前端 / 配置；未虚构 Recall/Precision 提升。Rerank Evaluation 需另行设计并在真实/标注数据上运行后才可下结论。*
