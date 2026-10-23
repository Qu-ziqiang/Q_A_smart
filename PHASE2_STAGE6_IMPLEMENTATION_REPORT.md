# PHASE2 Stage 6 — Synthesizer Contract / Vision Evidence Controlled Input 实现报告

> 阶段：Phase 2 — Vision Capability Enablement / **Stage 6 — Synthesizer Contract**
> 结论：**PASS**（198/198 测试全绿；未重写 Synthesizer；未进入 Stage 7）

---

## 1. Audit Result（先审计）
| 审计项 | 现状 |
|---|---|
| Synthesizer 接收 ImageCandidate | `_run_inner` → `dedup_candidates(collect_candidates(...))`（Stage 5 已把 Vision 证据合并进 `candidate.vision`） |
| prompt 构造 | `_system_prompt`（固定模板）+ `_user_prompt`（payload：goal/plan_summary/failures/groups/candidates/recommendation_limit） |
| `validate_selection()` | 存在且保留：按真实候选 id 过滤非法/重复、截断 limit |
| fallback | `DeterministicSynthesisWorker`（同样经 collect_candidates，天然携带 vision 字段） |
| 泄露风险 | `_user_prompt` 候选 dict **仍含 `public_url`**（旧行为）；`ImageCandidate.vision`/`filepath` 仅内部；SSE 的 `_public_images`/`_output_summary` 均白名单（不受影响） |
| 系统提示 | 无 vision evidence 使用指引；unknown≠fact 需显式写入 |

## 2. Files Changed
| 文件 | 动作 | 原因 |
|---|---|---|
| `backend/ai_func/recommendation/prompts.py` | 修改 | `SYNTHESIZER_SYSTEM_PROMPT_TEMPLATE` 新增规则 9：vision 证据受控使用 + unknown≠fact |
| `backend/ai_func/recommendation/orchestration/synthesizer.py` | 修改 | `_user_prompt` 候选 dict **移除 `public_url`**（Stage 6 §B 数据边界） |
| `backend/tests/agent_team/test_synthesizer_contract.py` | 新增 | 7 项 Stage 6 测试 |
| `PHASE2_STAGE6_IMPLEMENTATION_REPORT.md` | 新增 | 本报告 |

## 3. Architecture Impact
- **无架构改动**：Synthesizer 仍是"LLM 一次调用 + validate_selection + 确定性 fallback"。
- vision evidence 通过既有 `ImageCandidate.vision` → `_user_prompt` 白名单字段进入 prompt（Stage 5 数据通路，Stage 6 加固边界）。
- 系统提示新增规则 9：可引用 vision 中实际观察到的内容；vision 未提供/失败 → 一律"未知"。
- **数据边界收紧**：LLM payload 不再含 `public_url`（SSE `selection` 事件仍含 public_url，为前端展示所需，不受影响）。

## 4. Tests（Stage 6 新增 7 项，全部 PASS，零真实 API）
1. `test_synth_prompt_contains_bounded_vision_evidence` — 成功证据 6 字段进入 payload；系统提示含 vision 规则
2. `test_synth_prompt_no_sensitive_fields` — payload 全文本无 public_url/filepath/base64/raw/api_key/authorization/sk-/data:image/；vision 子字典白名单精确
3. `test_synth_vision_partial_failure_still_succeeds` — 部分失败 → synth 仍成功；失败 item 无伪证据
4. `test_synth_systemic_vision_fallback` — systemic + 空候选 → deterministic fallback（不调 LLM）
5. `test_synth_systemic_vision_not_blocking` — systemic + 有候选 → LLM 路径仍成功（无证据注入）
6. `test_vision_evidence_extra_fields_rejected` — VisionEvidence extra=forbid 拒绝未知字段
7. `test_vision_evidence_whitelist_fields` — 精确 6 字段集

## 5. Full regression
```
Phase 1-A/B/C   82/82 PASS
Stage 1 Vision  20/20 PASS
Stage 2         27/27 PASS
Stage 3         29/29 PASS
Stage 4         19/19 PASS
Stage 5         14/14 PASS
Stage 6          7/7  PASS
合计           198/198 PASS
```
（test_synthesizer 原 12 项全过 → public_url 移除无回归）

## 6. Non-goals
```
❌ 未重写 Synthesizer / 未引入新 Agent / Memory / Context Compiler / Critic / Reflection / Replanning
❌ 未修改 Executor / DAG scheduling / Planner / Router / SSE / Frontend
❌ 未改数据库 schema / metadata 持久化 / embedding
❌ 未改 Legacy agent/tools/search
❌ 未改 config.yaml（VISION_MODEL 仍 "null"）；单测零真实 API；无凭据输出
✅ 全部未做
```

## 7. Known limitations
1. `score` 仍保留在 payload（非 §B 禁止项；排序依据，LLM 可用）。
2. SSE `selection` 事件继续含 `public_url`（前端展示需要，非 LLM prompt）。
3. vision 证据依赖 Stage 5 的 `collect_candidates` 直接依赖合并；跨层传播不在本 Stage。

---

**STAGE 6 = PASS**
Stage 7（SSE Observability）= **NOT STARTED**（HARD STOP，等待指令）
