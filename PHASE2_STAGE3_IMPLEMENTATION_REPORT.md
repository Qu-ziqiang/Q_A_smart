# PHASE2 Stage 3 — Implement VisionWorker 实现报告

> 阶段：Phase 2 — Vision Capability Enablement / **Stage 3 — VisionWorker**
> 结论：**PASS**（158/158 测试全绿；ANALYZE_VISUAL 仍未注册；未进入 Stage 4）

---

## 1. Implemented

- 新增 **`VisionWorker`**（`orchestration/workers.py`）：消费有限候选 → dedup → 候选上限（默认 4 / 绝对 8）→ criteria-aware prompt → 串行逐图调用 `ImageAnalysis`（strict 边界 + 明确 per-item timeout）→ 逐图失败隔离 → 聚合为 Stage 1 定义的 `VisionTaskOutput`。
- `orchestration/vision.py`：新增 2 个 `VisionErrorCode` 成员（`UNSUPPORTED_MIME`、`CLIENT_UNAVAILABLE`，加法扩展不破坏 Stage 1 schema）；新增 `cap_evidence_total()`（evidence 聚合预算）；常量 `DEFAULT_VISION_CANDIDATES` / `VISION_ITEM_TIMEOUT_SECONDS` 改由 `models.py` 单一来源（消除 magic number，避免 capability drift）。
- `orchestration/models.py`（最小、有理由的修改，见 §16）：新增 `DEFAULT_VISION_CANDIDATES=4`、`VISION_ITEM_TIMEOUT_SECONDS=30` 常量；`AnalyzeVisualInput.max_candidates` 默认 8→4（PHASE2 计划 §626-628 明确建议，验收标准"默认建议 ≤4"）。
- `workers.py` `status_summary()`：新增 `VisionTaskOutput` 分支（无敏感数据的安全摘要）。

## 2. Files Changed

| 文件 | 动作 | 原因 |
|---|---|---|
| `backend/ai_func/recommendation/orchestration/models.py` | 修改 | 常量单一来源 + `AnalyzeVisualInput` 默认候选数 8→4（计划明确建议） |
| `backend/ai_func/recommendation/orchestration/vision.py` | 修改 | 2 个新增错误码、`cap_evidence_total`、常量改由 models.py 导入 |
| `backend/ai_func/recommendation/orchestration/workers.py` | 修改 | `VisionWorker` + `build_vision_prompt` + 错误映射 + `status_summary` 分支 |
| `backend/tests/agent_team/test_vision.py` | 修改 | 新增 29 项 VisionWorker 测试（Fake ImageAnalysis，零真实 API） |

## 3. VisionWorker Architecture

```
VisionWorker.run(task, context)
 ├─ 校验 input 为 AnalyzeVisualInput
 ├─ 解析客户端（不可用 → systemic_failed=CLIENT_UNAVAILABLE）
 ├─ _collect_valid_candidates：collect_candidates → dedup → 过滤非法 ID(≤0/None)
 ├─ _resolve_limit：min(requested, 8)，非法/缺失 → 4
 ├─ 空候选 → build_vision_output([])（不调用 provider）
 ├─ build_vision_prompt(criteria, goal) → criteria-aware bounded prompt
 ├─ 串行 for cand: _analyze_one()（每图 timeout + raise_on_error=True）
 │    └─ ImageAnalysis(strict) → dict → parse_vision_response(json) → VisionItemResult
 ├─ cap_evidence_total(items) → 聚合预算
 └─ build_vision_output(...) → VisionTaskOutput → WorkerResult(SUCCEEDED)
```

## 4. Candidate Collection
- 复用 `collect_candidates(task, context.results)`（RetrievalOutput / MetadataOutput / GroupOutput 均支持），不自行查库。
- 图片来源：`_image_source()` 优先合法本地 `filepath`（存在才用），否则 `public_url`（image_url）；两者皆无 → item_failed `INVALID_IMAGE_ID`（不调用 provider）。

## 5. Dedup / Limit
- Dedup 按 `image_id` 去重、保留首次出现顺序（测试：`[5,2,5,8,2]→[5,2,8]`）。
- `actual_limit = min(requested, MAX_VISUAL_CANDIDATES=8)`；非法/缺失 → `DEFAULT_VISION_CANDIDATES=4`。
- 非法 ID（None / 0 / 负数）静默跳过，不影响 Worker 成功。

## 6. Prompt Boundary
- `build_vision_prompt(criteria, goal)`：只允许描述图片中实际可见内容；禁止猜测地点/人物身份/拍摄时间/EXIF/metadata；未知写 unknown；输出 JSON 字段严格为 title/description/tags/evidence/confidence。
- 不含 conversation history / tool registry / 其他图片信息（测试断言）。

## 7. ImageAnalysis Integration
- 唯一 provider 调用路径 = `analyzer.analyze_image(detail, prompt, timeout, raise_on_error=True[, model, local_image_path|image_url])`。
- 复用 Stage 2 的 timeout / strict error / MIME 归一化边界；不绕过 ImageAnalysis 自己调 OpenAI client。
- 输出先经 Stage 1 `parse_vision_response()`（统一字段长度 / confidence 钳制 / unknown-field 拒绝）。


## 8. Timeout
- 默认 `VISION_ITEM_TIMEOUT_SECONDS = 30`（常量集中在 models.py）；`VisionWorker(item_timeout=...)` 可注入。
- 每次调用显式传 `timeout=<值>`（测试断言：默认=30、注入=12.5）。
- **无自动 retry**（测试断言每张恰好一次调用）、无模型/provider 切换。

## 9. Failure Semantics
- item 级失败（timeout / provider / parse / invalid image / unsupported MIME / unknown）→ `VisionItemResult(status=item_failed, error_code, failure_reason)`，单张失败不阻塞其余候选。
- 失败信息只含安全消息（`ImageAnalysisError.safe_message`），不外泄 raw response / API key / 绝对路径 / base64。

## 10. Partial Failure
- 全部成功：`success_count=N, failed_count=0, systemic_failed=False`。
- 部分成功：`success_count>0, failed_count>0, systemic_failed=False`（测试：2 成功 1 失败）。
- 全部 item 失败但 provider 正常：`success_count=0, failed_count=N, systemic_failed=False`（**不**升级为 systemic）。
- Worker 本身始终 `SUCCEEDED`（失败都收敛进 `VisionTaskOutput`），保证 Stage 5 的"vision 失败 → synthesize 仍执行"语义。

## 11. Systemic Failure
- 触发条件：视觉客户端初始化失败 / `client` 缺失（未配置 OpenAI 兼容客户端）。
- 结果：`systemic_failed=True, systemic_error_code=CLIENT_UNAVAILABLE, systemic_failure_reason=<safe>`，items 为空，不调用 provider。

## 12. Output Contract
- 严格返回 Stage 1 的 `VisionTaskOutput`（`extra=forbid`）；成功 item 为 `VisionItemResult(status=success, title/description/tags/evidence/confidence)`，失败 item 为 `VisionItemResult(status=item_failed, error_code, failure_reason)`。
- evidence / 字段长度由 parser 限制（title≤60、desc≤200、evidence≤300、tags≤8）+ `cap_evidence_total`（聚合≤4000）。

## 13. Tests（Stage 3 新增 29 项，全部 PASS，Fake ImageAnalysis 零真实 API）

| 类别 | 测试 | 数量 |
|---|---|---|
| Candidate | 正常 4 候选 / 上限截断 8 / 默认 4 / dedup 保序 / 非法 ID 跳过 / 空候选不调用 provider | 6 |
| Success | 单 item 字段完整校验 | 1 |
| Partial Failure | success+timeout / success+provider / success+parse | 3 |
| All Failed | 全失败不升级 systemic | 1 |
| Systemic | client unavailable → systemic_failed | 1 |
| Timeout | 显式 timeout 传入 / 默认=常量 30 | 2 |
| Strict | `raise_on_error=True` 传入 | 1 |
| Prompt | criteria 入 prompt / image-grounded 无 history / 空 criteria | 3 |
| Safety | 无 retry / 无 filepath/base64/public_url 泄露 / 单模型不切换 | 3 |
| Source | 优先本地 filepath / 缺失回退 URL / 无源 item_failed | 3 |
| Error Mapping | unsupported_mime → UNSUPPORTED_MIME / 未知异常 → UNKNOWN_ERROR | 2 |
| Evidence | cap_evidence_total 预算 / 常量预算约束 | 2 |
| status_summary | Vision 摘要 | 1 |

## 14. Regression

```
Phase 1-A/B/C   82/82 PASS（gate 13 / plan_validation 20 / planner 11 / workers 12 / executor 8 / orchestrator 6 / synthesizer 12）
Stage 1 Vision  20/20 PASS
Stage 2         27/27 PASS
Stage 3         29/29 PASS
合计           158/158 PASS
```

## 15. Security / Data Boundary
- 输出契约不含 filepath / public_url / base64 / raw provider response / API key（测试断言）。
- `filepath` 仅作为 ImageAnalysis 的本地输入；`public_url` 仅在无本地文件时使用。
- 不写数据库、不写 metadata、不改 embedding、不改 Legacy 路径。
- 单测全部 Fake/Mock；未调用真实 DeepSeek API；未打印/写入凭据。

## 16. Not Implemented
- `ANALYZE_VISUAL` registration：**NOT DONE**（`REGISTERED_TASK_TYPES` 与 `default_workers()` 均不含 ANALYZE_VISUAL，已断言验证）
- Capability enablement：**NOT DONE**（`vision_available()` 仍返回 False，VISION_MODEL 仍为 "null"）
- Planner modification：**NOT DONE**
- Validator modification：**NOT DONE**
- DAG scheduling modification：**NOT DONE**
- Executor / Synthesizer / Router / SSE / Frontend 修改：**NOT DONE**
- Parallel DAG / Memory / Context Compiler / Critic / Replanning：**NOT DONE**
- Stage 4：**NOT STARTED**

**额外说明（models.py 修改理由）**：`AnalyzeVisualInput.max_candidates` 默认 8→4 与 `DEFAULT_VISION_CANDIDATES=4`、`VISION_ITEM_TIMEOUT_SECONDS=30` 常量添加，来自 PHASE2 计划 §626-628 的明确建议；这是唯一一处对"允许修改清单"之外文件的最小、有据修改。Stage 1 schema（模型字段/枚举）未重新设计，仅加法扩展。

## 17. Known Limitations
1. `VisionWorker` 尚不可由 Executor 调用（未注册）；Stage 3 仅验证 Worker 单元行为。
2. `_image_source` 以 `os.path.exists(filepath)` 判断本地可用性；不存在的 filepath 且无 URL 时按 item_failed 处理（确定性）。
3. MIME 校验依赖 ImageAnalysis 的扩展名白名单（Stage 2 已实现）；本层不做二次 sniff。
4. 模型 override 通过 `VisionWorker(model=...)` 注入；真实 E2E 前需在 Stage 9 确认 `VISION_MODEL` 配置。
5. `_input_type_error` 防御路径在 typed `Task` 模型下不可达（Task 构造时强制 input 类型匹配），未单测（文档记录）。

## 18. Acceptance Checklist

```
[x] VisionWorker 实现完成
[x] candidate limit 默认 4（AnalyzeVisualInput 默认 + _resolve_limit 兜底）
[x] absolute candidate limit 8（_resolve_limit min(requested, 8)）
[x] candidate dedup（保序）
[x] invalid candidate safe skip（≤0/None 跳过）
[x] empty candidates 不调用 provider
[x] criteria-aware bounded prompt
[x] 串行逐图调用
[x] explicit per-item timeout（默认 30s，可注入）
[x] ImageAnalysis strict boundary（raise_on_error=True）
[x] no automatic retry（每张恰好 1 次调用）
[x] no model/provider fallback（单模型，无切换）
[x] item-level failure isolation
[x] systemic failure semantics（CLIENT_UNAVAILABLE）
[x] partial failure semantics（不升级 systemic）
[x] VisionTaskOutput contract（Stage 1 单一 schema）
[x] evidence / field length bounded（parser 上限 + cap_evidence_total 4000）
[x] no raw response/path/base64/key leakage
[x] Fake VisionWorker tests complete（29/29）
[x] 129 existing tests 全部通过（实际 158/158 全绿）
[x] ANALYZE_VISUAL 仍未注册（已断言）
[x] Planner/Validator/DAG/SSE/Frontend 未提前修改
[x] PHASE2_STAGE3_IMPLEMENTATION_REPORT.md 已生成
```

---

**Stage 3 Result: PASS**
**ANALYZE_VISUAL registration: NOT DONE**
**Capability enablement: NOT DONE**
**Planner modification: NOT DONE**
**Validator modification: NOT DONE**
**DAG scheduling modification: NOT DONE**
**Stage 4: NOT STARTED**（HARD STOP，等待指令）

