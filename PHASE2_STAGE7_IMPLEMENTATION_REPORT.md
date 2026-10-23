# PHASE2 Stage 7 — SSE Observability 实现报告

> 阶段：Phase 2 — Vision Capability Enablement / **Stage 7 — SSE / event-level Observability**
> 结论：**PASS**（207/207 测试全绿；复用 task.* 事件，未新增 vision.*；F-004 关闭）
> 状态：`STAGE 7 = PASS` · `F-004 = FIXED` · `Stage 8 = NOT STARTED`

---

## 1. Stage 7 Result
**PASS** — Vision task 通过已有 `task.completed.output_summary` 输出安全统计（candidate/success/failed/systemic），Vision 内容不进 SSE；router/SSE 事件契约由 FastAPI TestClient 集成测试锁定。

## 2. Pre-change audit
| 审计项 | 现状 |
|---|---|
| Event contract（router） | `rewrite_start` → Team 事件原样透传（`plan.created/plan.failed/task.started/task.completed/task.failed/task.skipped`）→ `assistant_delta` → 内部 `selection` 转 `complete`；异常 → `error` |
| `event_generator` | `for raw in stream_core(...)` 逐帧 JSON 解析；未知 event 静默跳过；解析失败 `continue` |
| `TEAM_EVENT_NAMES` | 6 个 Team 事件名，透传集 |
| `_output_summary`（executor） | 覆盖 Retrieval/Metadata/Group/Synthesis；**缺 VisionTaskOutput 分支** ← Stage 7 唯一生产改动 |
| `_public_images`（orchestrator） | 白名单（id/score/title/tags/public_url），`selection`/`complete` 不含 filepath/vision |
| Frontend | `chatService.ts` 只处理 `rewrite_start/assistant_delta/complete/error`；未知事件（task.*/plan.*/selection 内部）静默忽略；JSON 解析异常捕获 → **无需修改** |
| F-004 | `backend/tests/agent_team/` 原先无 router/SSE 集成测试 ← 本 Stage 补上 |

## 3. Event contract（保持 task.*，不新增 vision.*）
```
rewrite_start
→ plan.created / plan.failed
→ task.started → task.completed / task.failed / task.skipped（每个 started 有 terminal）
→ assistant_delta
→ complete（由内部 selection 生成，唯一）
→ error（异常时）
```

## 4. Vision summary（`task.completed.output_summary`）
新增 `_output_summary` 分支（executor.py）：
```json
{ "vision": { "candidate_count": 4, "success_count": 3, "failed_count": 1, "systemic_failed": false } }
```
- 仅统计/状态；禁止 title/description/tags/evidence/confidence/filepath/public_url/base64/raw/prompt/API key
- systemic 输出 → `{candidate_count:0, success_count:0, failed_count:0, systemic_failed:true}`

## 5. Router integration
- Team 事件仍经 `TEAM_EVENT_NAMES` 原样透传；`assistant_delta`/`selection→complete` 语义不变。
- 无需修改 router 源码（透传机制已足够表达 Vision 统计）。

## 6. Frontend compatibility
- 实测 `chatService.ts`/`useChatRecommendation.ts`：未知事件静默忽略、`complete`/`assistant_delta` 解析正常。
- **前端零修改**。

## 7. Security / data boundary
- `task.started`：仅 task_id/task_type/status
- `task.completed`：status/duration_ms/safe output_summary（vision 仅统计）
- `task.failed`：error.code + error.message[:160]（executor 既有截断）；无 traceback/secret
- `task.skipped`：reason
- 全帧扫描测试断言无 filepath/base64/authorization/api_key/sk-/data:image/raw_response/provider_secret/traceback

## 8. Integration tests（Stage 7 新增 9 项，全部 PASS，零真实 API）
| # | 测试 | 验证 |
|---|---|---|
| 1 | `test_sse_legacy_flow` | rewrite_start → assistant_delta → complete；无 task 事件 |
| 2 | `test_sse_team_flow_and_vision_summary` | plan.created → 3×task.* → assistant_delta → complete；vision 统计帧不含 title/desc/tags/evidence |
| 3 | `test_sse_task_failed_safe` | 安全 code + message；无 traceback/api_key/sk- |
| 4 | `test_sse_task_skipped_reason` | reason=DEPENDENCY_UNAVAILABLE |
| 5 | `test_sse_no_leakage` | 全帧扫描 11 个敏感标记；vision 键集合精确 |
| 6 | `test_sse_complete_unique_and_terminal_pairing` | complete 唯一；started==terminal(3)；id 集合相等 |
| 7 | `test_sse_event_ordering` | 事件顺序；task.* 均在 assistant_delta 之前 |
| 8 | `test_executor_output_summary_vision_counts` | 4 候选/3 成功/1 失败/非 systemic；无视觉内容 |
| 9 | `test_executor_output_summary_vision_systemic` | systemic 全零统计 + systemic_failed=true |

## 9. Regression
```
Phase 1-A/B/C   82/82 PASS
Stage 1 Vision  20/20 PASS
Stage 2         27/27 PASS
Stage 3         29/29 PASS
Stage 4         19/19 PASS
Stage 5         14/14 PASS
Stage 6          7/7  PASS
Stage 7          9/9  PASS
合计           207/207 PASS
```

## 10. F-004 treatment
**F-004 = FIXED**：`PHASE1_ABC_IMPLEMENTATION_AUDIT.md` §F-004（无 router/SSE 集成测试）已由 `test_sse_contract.py`（FastAPI TestClient + Fake service/repos）关闭。F-001/F-002/F-003 仍保持 defer（未处理）。

## 11. Files changed
| 文件 | 动作 | 原因 |
|---|---|---|
| `orchestration/executor.py` | 修改 | `_output_summary` 新增 VisionTaskOutput 安全统计分支（+import） |
| `tests/agent_team/test_sse_contract.py` | 新增 | 9 项 router/SSE 集成 + _output_summary 单元测试 |
| `PHASE2_STAGE7_IMPLEMENTATION_REPORT.md` | 新增 | 本报告 |

未修改：router（透传机制已够）、前端、VisionWorker、ImageAnalysis、Planner、Validator、Synthesizer、Legacy、DB、embedding、config.yaml。

## 12. Non-goals
```
❌ 未新增 vision.* 事件 / EventFramework / EventBus / EventRegistry / StreamingFramework
❌ 未修改 VisionWorker / ImageAnalysis / Planner / Validator / DAG scheduling / Synthesizer prompt / Selection Validation
❌ 未实现并行 / Memory / Context Compiler / Critic / Reflection / Checkpoint / MCP / A2A
❌ 未改 Router / SSE（源码）/ Frontend / 数据库 schema / metadata 持久化 / embedding / Legacy
✅ 全部未做
```

## 13. Known limitations
1. `task.failed` 的 message 为 executor 既有 `str(e)[:200]` 截断（非本 Stage 新增）；traceback 仅写服务端日志。
2. `complete` 事件无 `data:` 前缀（既有行为，前端 parser 兼容）；测试 parser 已按前端逻辑处理。
3. router 级测试使用 Fake `stream_core`（事件契约层）；真实 Gate/Legacy/Planner 行为由 test_gate/test_orchestrator 覆盖，真实 HTTP E2E 属 F-008/Stage 9。

## 14. Acceptance checklist
```
[x] task.started/completed/failed/skipped 正常（帧级测试）
[x] Vision task.completed 含安全统计；无 VisionEvidence；无 filepath/public_url/base64/raw；无 credential；无 traceback
[x] assistant_delta / selection(→complete) / complete 兼容；complete 唯一
[x] Legacy SSE 回归 PASS（无 task 事件）
[x] Team SSE 集成 PASS（plan.created + task.* 透传 + complete）
[x] Vision summary 集成 PASS
[x] failure / skipped event 集成 PASS
[x] event ordering（started↔terminal 配对）PASS
[x] F-004 = FIXED
[x] 198 个既有测试全过（实际 207/207）
[x] Stage 7 新增 9 项测试全过
```

---

**STAGE 7 = PASS** · **F-004 = FIXED**
Stage 8（Integration and Regression）= **NOT STARTED**（HARD STOP，等待指令）
