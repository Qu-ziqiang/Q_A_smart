# MULTIMODAL RERANKER MODEL SURVEY（R2 候选调研）

> 项目：SmartImager — R2：模型型 Multimodal Reranker（只读调研，未修改任何代码）
> 日期：2026-09-01
> 结论速览：A=SigLIP2 · B=Chinese-CLIP(中文强项) · C 见文末（含 jina-clip-v2 因 **CC-BY-NC-4.0** 不适用商用等）

---

## 0. 当前检索栈的约束（决定选型的事实）

| 项 | 现状 |
|---|---|
| 检索文本编码 | `SentenceTransformer(clip-ViT-B-32-multilingual-v1)` 512d（多语言 CLIP 文本塔） |
| 检索图片编码 | `openai/clip-vit-base-patch32`（**另一套权重**）——文本塔与图片塔**不是同一模型**，跨模态相似度严格来说不在同一语义空间 |
| 分辨率/规模 | ViT-B/32 + 224px，粗粒度 |
| 查询语言 | **以中文为主**（28 条 gold 全中文） |
| 运行环境 | 本地 torch（cuda 可选）；需 **12GB VRAM 可跑**、HF 可下载、可离线评测、无训练 |
| R1 | 确定性重排（vector+keyword+metadata），已实现；R2 作为第二种方案评估 |

结论：当前 CLIP 语义空间已不一致 + 中文查询 + 需要比"粗召回"更精的 relevance，这三点决定了：**应优先找一个"text/image 同塔对齐、能输出 query-image 相似度、中文可用、Apache/BSD 许可"的编码器模型**；精确匹配头（ITM/VLM）是另一族，成本/许可权衡更重。

---

## 1. 候选逐项分析

### 1.1 Google SigLIP2（最推荐）— `google/siglip2-base-patch16-384`

| 项 | 事实 |
|---|---|
| 官方来源 | HF `google/siglip2-*`；arXiv 2502.14786（2025） |
| 模型家族 | ViT-B(86M)/L(303M)/So400m(400M)/g(1B)；base-384 权重总参 **≈375M**（F32 1.5GB，fp16 ≈0.75GB） |
| 输入格式 | `text`（句子）+ `image`（多分辨率/aspect-ratio 保持，384 输入） |
| 输出 | image-text 对比相似度 / zero-shot 分类分（sigmoid 目标训练，可直接排序） |
| image-text matching | ✅ 是（同一模型双塔，天然对齐） |
| 多语言 | 论文称多语言数据去偏提升；**以英文为主、中文为覆盖语种之一**——中文质量需实测 |
| 显存 | base-384 fp16 权重 <1GB，激活 384 分辨率小 batch 后总量 ≪12GB；so400m 也可（~1.6GB 权重） |
| 成本 | N≤40 图/query：GPU 下约秒级（批量 ~100ms/图级） |
| 适合作 rerank | ✅ Top-N→Top-K；直接输出可排序相似度 |
| License | HF 标 **apache-2.0**（Google 权重条款建议顺带复核） |
| 当前项目适配 | ✅ transformers 原生、离线缓存支持（`HF_HUB_OFFLINE=1`）、12GB 充裕；**同时可修复"文本/图片异塔不匹配"**

### 1.2 OFA-Sys Chinese-CLIP（中文强项）— `chinese-clip-vit-large-patch14`

| 项 | 事实 |
|---|---|
| 官方来源 | OFA-Sys/Chinese-CLIP（arXiv 2211.01335）；HF 同名 |
| 结构 | 图片 ViT-L/14（≈300M）+ 文本 RoBERTa-wwm-base（≈110M）；约 2 亿中文图文对预训练 |
| 输入/输出 | `ChineseCLIPModel`：`(texts, images)` → `logits_per_image`（image-text 相似度） |
| image-text matching | ✅ text→image 专用（零样本中文基准） |
| 零样本中文 | MUGE T2I R@1 **63.0**；Flickr30K-CN **71.2**；COCO-CN **69.2**（卡片公开值） |
| 显存/成本 | large fp16 权重 <1GB；base 更小；N≤40 图 batch 快 |
| License | HF 卡片**未标注 license 元数据**；官方 GitHub 开放源码——**商用前需人工核实权重许可** |
| 适配 | ✅ 中文 query 最对口；transformers `ChineseCLIPModel` 原生；缺点：2022 年架构/数据较旧、英文弱、分辨率低（224） |

### 1.3 Salesforce BLIP-ITM（显式匹配头，英文）— `blip-itm-large-coco`

| 项 | 事实 |
|---|---|
| 结构 | ViT-Large + 文本编码 + **ITM（Image-Text Matching）分类头** + 可关闭头用 cosine（`BlipForImageTextRetrieval`） |
| 输出 | `itm_scores`（匹配概率）或 `use_itm_head=False` 的 cosine |
| License | **bsd-3-clause**（商用友好） |
| 关键短板 | COCO **英文** caption 训练（2021）；中文 query 需翻译且语义弱；老模型，视觉理解一般；约 466M 参数、显存 ~1.9GB fp32 |
| 适配 | 仅当未来 query 转英文 / 作为"伪标签-匹配"对照实验；中文照片库不推荐为主方案 |

### 1.4 其余候选（摘要）

| 模型 | 为什么不作为主推 |
|---|---|
| `jinaai/jina-clip-v2` | **License=CC-BY-NC-4.0（非商用）**——商用硬伤；且是双塔检索嵌入模型，非显式 rerank 头 |
| `BAAI/bge-visualized(-m3)` | 官方 FAQ 明示 **text→image 跨模态非推荐场景**；定位"混合/组合检索"（text↔image+text 或 image+text↔image） |
| `microsoft/Florence-2(-base 0.23B)` | MIT、轻、但**无相关性打分头**（caption/grounding/检测类生成任务）；要排名只能靠生成式 prompt 评分，不稳定 |
| `apple/DFN5B-CLIP-ViT-H-14-378` | 强 zero-shot 但**仅英文**；License=apple-amlr（非 OSI 标准、商用需评估）；ViT-H 显存/吞吐高；rerank 收益不显著高于 SigLIP2 |
| 大型 VLM（Qwen2.5-VL / InternVL / LLaVA / SmolVLM…7B+ 或 <1B 生成式） | 12GB 仅 4bit 勉强；逐候选 generation 打分慢、结果不可复现/不可比、离线批量弱；它们是"视觉理解/像素核验"，与本项目 Vision(analyze_visual) 职责重叠——**不应用作相关性 reranker** |
| 维持现状 `clip-ViT-B-32-multilingual-v1` + `clip-vit-base-patch32` | 已证实**双塔不匹配**（跨模型相似度）、粗粒度；仅作 baseline，不作为新方案 |

---

## 2. 是否比当前 CLIP 更适合 image-text relevance？

**是。** 三个理由：
1. 当前图片编码与文本编码是**两套不同权重**（`openai/clip-vit-base-patch32` vs `clip-ViT-B-32-multilingual-v1`），相似度在严格意义上跨语义空间；
2. ViT-B/32 + 224 分辨率表达能力有限，且没有针对"精细相关度/排序"的训练；
3. SigLIP2/Chinese-CLIP 都是**单族双塔联合训练**（中文或英文各自原生对齐），作为 relevance 分天然更一致；BLIP-ITM 更进一步提供显式匹配分类头。
注意：以上均为对比式相似度；**不替代** analyze_visual 的"像素事实核验"。

## 3. 推荐结论

### A. 最推荐：Google SigLIP2（`google/siglip2-base-patch16-384` 起步；显存允许可上 `so400m`）
许可证 Apache、transformers 原生、零样本 image-text 检索强、双塔同模型对齐、12GB 充裕、离线易行。风险点：中文为"覆盖语种"而非母语级——**上线前先跑 20~50 条中文样例做 sanity**。

### B. 次推荐：OFA-Sys Chinese-CLIP（`chinese-clip-vit-large-patch14`；小库用 base-patch16）
中文 query 场景最对口（零样本中文 T2I 中文基准领先），transformers 原生。前提：**先核实权重商用许可**；若中文是唯一语言且准确率实测高于 SigLIP2，可优先 B。

### C. 不推荐（及原因）
- **jina-clip-v2**：CC-BY-NC-4.0 非商用（license 硬伤），且为双塔检索嵌入。
- **bge-visualized(-m3)**：官方明确 text→image 非推荐场景。
- **Florence-2**：无相关性打分头。
- **DFN5B-CLIP(apple)**：英文-only + apple-amlr 许可 + 重。
- **大型 VLM（Qwen2.5-VL 等）**：12GB 勉强、生成式打分不稳定、职责与 Vision 重叠。
- **BLIP-ITM**：BSD-3 商用 OK 且是"精确匹配头"族，但英文 COCO、较老 → 中文照片库不推荐为主方案（可作对照实验）。

## 4. 接入建议（Top-N → Top-K，不破坏现有 contract）

```
CLIP Retrieval (现状)
 → Top-N candidates (RetrievalWorker，limit≤40；同池不变)
 → R2 Multimodal Reranker（新模块，可选配置）
      score(q, img) = model similarity (SigLIP2 或 Chinese-CLIP)
      对候选池逐图打分 → 同池新序（仍保持 set 不变 / 数量不变）
 → Top-K（由下游 Synthesize recommendation_limit≤20 截断；离线评测另取 K）
```

- **插入点**：复用 `RetrievalWorker` 已预留的 `reranker=` 注入位（R1 的确定性 rerank 为默认；R2 传一个 `model_reranker(query, candidates)` 可调用对象即切换）。异常→既有 try/except 回退原序，绝不失败。
- **新模块**（未来实施）：`orchestration/model_reranker.py`：
  - `SigLIP2Reranker(model_id="google/siglip2-base-patch16-384", device=..., dtype=float16, batch=8)`，加载本地路径/缓存、`HF_HUB_OFFLINE=1` 支持；
  - 输入：`query`（中文原句）+ candidates（图片源沿用 `_image_source`：优先本地 `filepath`，否则 `public_url`，无 base64 持久化）；
  - 输出：同池重排（同对象列表），确定性（dropout 关闭、eval 模式、固定顺序）；
  - 计分：image-text similarity；`Top-K` 只在**离线评测脚本**内切片，worker 内不改变候选集合。
- **contract**：RetrievalOutput 字段/数量/ID 集不变（与 R1 相同约束）；Synthesizer/`validate_selection`/SSE 零改动。
- **评测（offline，单独设计）**：固定中文 query+人工标注相关集 → Baseline（当前 CLIP 检索序）vs R2（同池模型重排后 Top-K）→ Recall@K/MRR。**评测未跑之前不宣称任何提升**。
- **注意**：R2 只解决"阶段二排序"。若未来要让"召回"也用同族模型，需要**重新编码整个图片库**（离线迁移，区别于 rerank 的逐 query 编码）。
