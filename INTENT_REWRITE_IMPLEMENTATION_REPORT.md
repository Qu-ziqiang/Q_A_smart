# INTENT_REWRITE_IMPLEMENTATION_REPORT

> ⚠️ **历史快照（迁移前口径）**：文中"改造前 275/275 → 改造后 298/298"为当时的自研 runner/pytest 口径。当前门禁为 **463 passed**（2026-09-10 复测，PG 迁移后）。


> 项目：SmartImager — Lightweight Intent Rewrite（Phase B，确定性规则方案）
> 日期：2026-09-01
> 范围：仅实施 Intent Rewrite；**不实施 Rerank**（Rerank 仍为 NOT IMPLEMENTED，见 §8）
> 基线回归：改造前 agent_team 全量 275/275 → 改造后 298/298

---

## 1. 修改了哪些文件

| 类型 | 文件 | 动作 |
|---|---|---|
| 新增 | `backend/ai_func/recommendation/orchestration/intent.py` | `IntentRewrite` + `NormalizedIntent` + 确定性规则解析 |
| 新增 | `backend/tests/agent_team/test_intent_rewrite.py` | 23 项单元测试 |
| 最小改 | `backend/ai_func/recommendation/orchestration/orchestrator.py` | `AdaptiveTeamOrchestrator.run_stream` 增加可选 `goal_override=None` |
| 最小改 | `backend/ai_func/recommendation/__init__.py` | `_RecommendationService.stream_core` 接线 Intent → Gate → Planner |

## 2. 每个文件改了什么

### `orchestration/intent.py`（新增，核心）
- **`NormalizedIntent`**（Pydantic，`extra=forbid`）：
  `raw_query`（保留原文）· `normalized_query`（规范化 goal，可直接作为 Planner 输入）· `needs_images` · `retrieval_query`（首个检索分句）· `metadata_constraints: List[MetadataPredicate]`（**直接复用 models.py 的 MetadataPredicate**，field/operator 由 Pydantic 强制在既有枚举内）· `group_by`（**复用 `_ALLOWED_GROUP_BY` 词表**，model_validator 校验）· `needs_visual_verification` · `visual_criteria`（≤3 条、每条 ≤28 字符，仅提示）· `needs_recommendation` · `preference_hints`（预留）。
- **`IntentRewrite`**：`rewrite(query) -> NormalizedIntent`（纯函数、无状态、幂等）+ `rewrite_messages(messages)`（取最后一条 user）。
- 规则解析（模块级确定性函数，全部复用既有词表/句式）：
  - `_detect_group`：`按…分组/归组/分类/整理/归类` → `location/year/month/orientation/file_type/tag`（与 `_ALLOWED_GROUP_BY` 一致）。
  - `_detect_metadata`：年份→`created_at contains "2025"`；横向/竖向→`orientation eq landscape/portrait`；JPG→`file_type eq jpg`；宽度/高度→`width/height gte N`；标签→`tags contains X`；**跳过『按…分组』短语区**，避免把分组键误判为过滤条件（C12 类）。
  - `_has_vision_signal` / `_extract_visual_criteria`：命中 catalog 中 ANALYZE_VISUAL 的 use_when 句式（确认/判断/识别/核实/比较/对比/看见/画面中/构图/是否…）。
  - `_looks_greeting_or_irrelevant`：greeting/无关词 + 无检索信号 → `needs_images=False`，避免误触发 Team。
  - `_normalize`：仅去首尾客气词/语气词 + 规整空白，**不做语义改写**（保证 Gate 正则不受影响）。
- **不负责**：DAG 生成 / task decomposition / dependency 决策 / execution（intent 只是结构化描述 + 规范化文本）。

### `orchestration/orchestrator.py`（最小改）
`run_stream(messages, conversation_id, memory_service, goal_override=None)`：新增第 4 个可选参数；`goal = goal_override if (非空) else _last_user_query(messages)`。**默认 None → 行为与旧版完全一致**；Planner/Validator/Executor 核心逻辑零改动。

### `recommendation/__init__.py`（最小改）
`_RecommendationService` 增加 `self._intent_rewrite=None` + 惰性 `_get_intent_rewrite()` + `_last_user_text()` 静态助手。`stream_core` 中：
1. 取最后一条 user 文本 → `intent = IntentRewrite().rewrite(text)`（任何异常 → 回退 raw，不影响主链路）；
2. `ComplexityGate.decide(messages, query=intent.normalized_query)`（优先走已有 `query=` 接口；greeting/解析失败时 query 仍为原文）；
3. TEAM 分支：`orch.run_stream(..., goal_override=intent.normalized_query if intent.needs_images and normalized_query else None)` —— Intent 规范化 goal 仅喂给 Planner；Legacy 路径保持 raw messages 不变。

## 3. IntentRewrite 的真实输入输出

- **输入**：用户最后一轮自然语言文本（如 "帮我找 2025 年的横向照片，再按地点分组，最后确认画面中是否有人骑马"）。
- **输出**：一个 `NormalizedIntent` 结构化对象（示例见 §5）。
- **接线后的流向**：`User Query → IntentRewrite → NormalizedIntent → ComplexityGate(query=normalized) → Planner(goal_override=normalized) → Validator → Executor`。Planner/Validator/Executor 的**算法逻辑未改**，仅 goal 文本被规范化为 cleaner 版本。

## 4. 真实 Query 处理示例

输入：`帮我找 2025 年的横向照片，再按地点分组，最后确认画面中是否有人骑马`

```json
{
  "raw_query": "帮我找 2025 年的横向照片，再按地点分组，最后确认画面中是否有人骑马",
  "normalized_query": "找 2025 年的横向照片，再按地点分组，最后确认画面中是否有人骑马",
  "needs_images": true,
  "retrieval_query": "找 2025 年的横向照片",
  "metadata_constraints": [
    {"field": "created_at", "operator": "contains", "value": "2025"},
    {"field": "orientation", "operator": "eq", "value": "landscape"}
  ],
  "group_by": "location",
  "needs_visual_verification": true,
  "visual_criteria": ["最后确认画面中是否有人骑马"],
  "needs_recommendation": true,
  "preference_hints": []
}
```

对照：该 query 若进入 Planner，期望 task types 应为 `retrieve → filter_metadata → group_images → analyze_visual → synthesize`；Intent 的能力标记（metadata×2 + group + vision）与之一致。greeting 示例：`你好` → `needs_images=False`、无任何能力标记，Gate 继续走 Legacy。

## 5. 测试结果

| 测试组 | 结果 |
|---|---|
| `test_intent_rewrite.py`（新增 23 项） | ✅ 23/23 PASS |
| `test_routing_benchmark.py`（28 条 gold 离线评分器） | ✅ 8/8 PASS（在 31 passed 中） |
| 既有 Agent Team 全量 | ✅ **298/298 PASS**（= 改造前 275 + 新增 23） |

覆盖点（按需求 9）：metadata 条件（年份/orientation+file_type/宽度/标签，且经 MetadataPredicate 复用既有枚举）、group_by（location/year/month/file_type/orientation + 词表校验 + 分组短语不误判为过滤）、visual verification（12 条 vision query 全命中且 criteria 非空；5 条 metadata/group query 不误触）、普通检索（A 类无额外能力标记）、greeting/无关 query（`needs_images=False`）、同一 query 两次结果一致 + normalized 幂等、**28 条 routing gold 能力标记对齐（metadata⟺filter_metadata、group⟺group_images、vision⟺analyze_visual，0 偏差）**。

## 6. 是否改变了原有 Planner / Validator / Executor 行为

- **Planner**：未改。仅 `run_stream` 可能传入 `goal_override`；28 条 gold 无一条以"帮我/请…"开头，normalized == raw，**goal 文本与旧版一致** → capability routing 行为不变。
- **Validator**：未改（零改动）。
- **Executor**：未改（零改动）。
- **Gate**：逻辑未改；只是 `decide` 现在优先收到 `normalized_query`（内容与原文几乎一致），`query=` 是既有接口（gate.py:56-60）。
- **Legacy 路径**：仍使用原始 messages，不受 Intent 影响。
- **Memory**：不改；orchestrator 的 memory recall 用 goal（=normalized）打分，对无礼貌前缀的查询无差别。
- **SSE / 前端**：无改动。

## 7. Known Limitations

- 规则层是词法判定：近义表达（如"归类"之外的说法）可能漏检；当前 28 条 gold 与常见 catalog 句式覆盖已验证。
- `visual_criteria` 是"含触发词的分句截断"（≤28 字符），非权威对象抽取；仅供 Planner 提示用，不承诺精确核验对象。
- Intent 目前只把结构化信号存于 `NormalizedIntent` 并在服务层驱动 Gate query 与 Planner goal；**尚未把能力标记注入 Planner prompt**（保持"最小改动 + 零 prompt 回归"，未来可平滑扩展）。
- 简单请求行为不变策略：greeting/无关 query `needs_images=False` 时不下发 goal_override；Gate 仍按原文本判 Legacy。

## 8. Rerank 状态（明确）

**未实施、仍不存在**。本次范围仅 Intent Rewrite；Rerank 的任何改动（含 workers/search 层）均未触碰。检索链路仍为 `RetrievalWorker → tool_search_images → SearchRepository`（CLIP + sqlite-vec cosine，无第二阶段 rerank）。
