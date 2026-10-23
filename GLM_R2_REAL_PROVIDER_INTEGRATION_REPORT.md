# GLM Multimodal Reranker 真实 Provider 接入报告（R2）

> ⚠️ **历史快照（迁移前口径）**：文中 smoke 用的"66 张本地真实图库 / sqlite-vec"为当时环境。当前为 **PostgreSQL + pgvector**、图库约 1000 张 VOC2010；本文结论（GLM-4V-Flash 可接入、单 query 约 11.8s、**未宣称效果提升**）不变。


> 日期：2026-09-05 ｜ 范围：真实智谱 GLM API ↔ 既有 `openai_chat` adapter 接入验证
> 纪律：R1 保留未动；Intent/Planner/PlanValidator/DAGExecutor/RetrievalRepository 均未修改；
> 无 LangGraph、无本地模型下载、无新增本地视觉依赖；API Key 只经环境变量读取，未写入任何代码/文档/测试/配置；
> 本轮已执行真实 API smoke（flash 免费档），**未宣称任何效果提升**。

## 1. 实际使用的模型与官方 endpoint

| 项 | 值 |
|---|---|
| Provider | 智谱 AI 开放平台（bigmodel.cn，OpenAI-compatible） |
| Base URL | `https://open.bigmodel.cn/api/paas/v4` |
| 模型（默认） | `glm-4v-flash`（多模态视觉，免费档，已验证真实接受图片输入） |
| 可替换模型 | `glm-4v-plus` / `glm-4v-plus-0111`（更强，付费）；`glm-4.1v-thinking-flash` 等，同一 endpoint 下改 `MM_RERANK_MODEL` 即可 |

接入前已按“不要凭名称猜”的原则用真实图片调用确认：该模型支持 text+image 联合输入，非视觉模型误配时会抛 `RelevanceError` 自动回退 R1，不会把请求打崩。

## 2. 环境变量配置方式（接入真实 provider 的全部配置）

```
MM_RERANK_PROVIDER=openai_chat
MM_RERANK_API_KEY=<你的智谱 API Key>          # 只放环境变量/.env，禁止进代码
MM_RERANK_BASE_URL=https://open.bigmodel.cn/api/paas/v4
MM_RERANK_MODEL=glm-4v-flash
MM_RERANK_TIMEOUT_SECONDS=45                   # 可选，默认 30
```

- 未配置 → `RetrievalWorker` 默认走 R1，行为与接入前完全一致（单元测试覆盖）。
- 已配置 → 模型型 R2 dispatcher 生效；缺 Key/模型不可用/超时等自动回退 R1。

## 3. 请求 payload 结构（复用的既有 adapter，未重写）

`POST {base_url}/chat/completions`，`temperature=0`：

```json
{
  "model": "glm-4v-flash",
  "temperature": 0,
  "messages": [
    {"role": "system", "content": "你是多模态图片相关性评分器。只判断图片内容与检索意图是否相关。只输出 JSON：{\"relevance\": 0到1的小数}，0=完全不相关，1=完全相关，不要输出其它文字。"},
    {"role": "user", "content": [
      {"type": "text", "text": "检索意图：海边日落的照片"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...."}}
    ]}
  ]
}
```

图片输入：`ImageCandidate.filepath` 存在 → 本地读文件转 `data:` base64 URL（优先）；
否则用 `public_url`。逐张顺序调用，无并发/batch（本轮不加优化）。

## 4. relevance 解析与校验

- 模型只回 JSON（实测原样：`{"relevance": 0}`、`{ "relevance": 1 }`）。
- adapter 提取首个 `{...}` → `json.loads` → `float(data["relevance"])`；缺字段/非数字/NaN/Inf → `RelevanceError`。
- `MultimodalReranker.score` 收敛并 clamp 到 `[0,1]`（越界 clamp；非有限报错 → fallback）。
- 模型输出整数 `0/1` 或小数均可被 float 化，全部落在 `[0,1]`。

## 5. fallback 机制（双层，全部保留）

```
R2 provider 任意失败（timeout / HTTP / auth / rate limit / model error /
invalid JSON / relevance 缺失或非数字或越界 / 某一张图调用失败）
   ↓ 丢弃本次部分结果（40 张里第 17 张失败 → 不采用前 16 张）
   ↓ 整轮回退
R1 rerank_candidates（确定性重排）→ 正常返回，Agent 请求不失败
```

单测覆盖：provider 全失败/部分失败/未知异常 → 结果=R1；候选数量、ID 集合、对象不变；排序稳定。

## 6. 真实 smoke 结果（glm-4v-flash，66 张本地真实图库）

脚本：`backend/tests/agent_team/smoke_glm_rerank.py`（env 不完整自动 SKIP，绝不进 CI、绝不打印 Key）。
Query「海边日落的照片」+ 其它对照 query，逐张顺序真实调用：

| 项目 | 结果 |
|---|---|
| HTTP/鉴权 | 成功（401 场景在未配 Key 时被正确转为 fallback） |
| 模型接受图片 | 是（真实 data-URL 图片输入） |
| relevance 解析 | 成功；全部 ∈ [0,1] |
| query=海边日落 | 3 张图全部 0.0（图库无海边/日落内容，低分可解释） |
| 对照 query=“一只猫或狗的照片” | 同一批图得分 1.0 / 0.0 / 0.0 / 0.0 / 0.0 → 相关/无关可明确区分 |
| Rerank contract | count=3、id 集合不变、同对象、稳定排序均通过 |
| 单图 latency | 约 0.88–1.59s；本次平均 **~1.15s/张**（顺序调用） |

真实调用共约 16 次（flash 免费档；1 次 query×3 + 1 轮 rerank 复验 + 2 个对照 query×5）。

## 7. 40-candidate 单 query 成本/耗时预估

- 调用次数：`N_queries × N_candidates`。40 候选 = 40 次/query。
- 顺序耗时（无并发）：40 × 1.2s ≈ **48s/query**（以本次实测为准）。
- 费用：`glm-4v-flash` 免费档即可支撑评估；换 `glm-4v-plus` 等按智谱当前 token 单价计价，正式评估前以控制台为准。
- 正式评估建议先 `--queries 3 --k 6`（18 次 ≈ 25s）小规模验证，再放大。

## 8. R2 Evaluation Runner（同一候选池）

`backend/tests/agent_team/eval_rerank_glm.py`（默认只打印计划与成本，`--yes` 才真实调用）：

- Baseline = CLIP Vector Retrieval（`tool_search_images` 原序）
- R1 = Baseline + deterministic rerank
- R2 = Baseline + GLM multimodal rerank
- 三条策略复用同一个候选池/Query/DB/Retrieval 参数，唯一变化是 rerank 策略。
- 示例成本输出：`3 × 6 = 18 次 ≈ 25s`；本报告发布时未执行 `--yes`（等人工确认成本）。

## 9. 测试与回归

- `test_model_reranker.py`（R2 接口/fallback/contract/worker 集成）：17/17 PASS
- `backend/tests/agent_team` 全量：**333 passed**
- smoke/eval 为 opt-in 脚本，pytest 不收集、CI 不调付费 API

## 10. 是否具备正式 R2 Evaluation 条件

**具备链路条件**：真实 provider 已验证、smoke 通过、runner 就绪、成本可预估。
**尚不具备“结论”条件**：当前图库无人工标注 ground-truth；正式评估应在同意成本后执行
`eval_rerank_glm.py --yes`，并在有标注集后再谈任何质量结论。禁止在无评测结果前宣称“R2 更优 / 提升 Hit@K”。
