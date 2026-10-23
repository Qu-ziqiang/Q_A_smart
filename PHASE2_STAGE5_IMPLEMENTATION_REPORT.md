# PHASE2 Stage 5 — DAG & Candidate Propagation 实现报告

> 阶段：Phase 2 — Vision Capability Enablement / **Stage 5 — DAG & Candidate Propagation**
> 结论：**PASS**（191/191 测试全绿；未修改 Executor scheduling；未进入 Stage 6）

---

## 1. Stage 5 Verdict
**PASS** — VisionWorker 已正式接入 Team DAG，成为候选消费型 Worker；Vision 证据按 `image_id` 合并进候选并送达 Synthesizer；Vision 各种失败（部分/全部/systemic）均不阻断 synthesize。

## 2. Pre-change audit
| 审计项 | 修改前 | 缺什么 |
|---|---|---|
| `collect_candidates()` | 只收集 Retrieval/Metadata/Group 输出 | **不消费 VisionTaskOutput** → 无法传播 Vision 证据 |
| `VisionTaskOutput` | 只含 `VisionItemResult`（image_id + 视觉字段），无 ImageCandidate 引用 | 需要按 image_id 合并到既有候选 |
| Synthesizer payload | 候选 dict 无 vision 字段 | 视觉证据无法到达 LLM |
| Executor dependency | `deps_ok = any(dep SUCCEEDED)`（至少一个依赖成功即执行） | 无冲突：VisionWorker 恒 SUCCEEDED，synth 不会被误 SKIP |
| 泄露面 | `filepath/public_url/base64` 不进入 Vision 输出 | 需保持 |

**结论**：`collect_candidates()` 不能直接消费 VisionOutput；缺的是「按 image_id 把成功 Vision item 合并为 bounded 证据」的 propagation adapter。不需要修改 Executor。

## 3. Candidate propagation before/after
```
Before: retrieve → {group} → synth        （vision 结果不参与合成）
After:  retrieve → {group, vision} → synth（成功 vision 证据合并进候选 → Synthesizer）
```
- **`collect_candidates()`**：新增 `VisionTaskOutput` 分支 → 收集 vision 输出 → `_attach_vision_evidence()` 合并。
- **`_attach_vision_evidence()`**：按 `image_id` 把成功 item 转为 `VisionEvidence` 挂到候选；失败/systemic 不注入；vision-only id 不创造候选；已带 vision 不覆盖。
- **`ImageCandidate.vision: Optional[VisionEvidence]`**：新增字段（仅 Synthesizer 消费，不进 SSE）。

## 4. DAG structure
推荐结构验证合法（Validator + Executor 实测）：
```
retrieve_images
   ↓
 ┌─┴─────────────┐
 ↓               ↓
group_images  analyze_visual
 └───────┬───────┘
         ↓
     synthesize
```
- Vision 依赖 producer（`NO_PRODUCER` 拒绝）；Vision 必须到达 synthesize（`ORPHAN_TASK` 拒绝）。
- **Executor 未修改**：仍为串行；Vision 与 Group 逻辑同层、实际先后执行（`_VisionOn` 之外的 run 顺序由 ready 队列决定）。


## 9. Tests added（Stage 5 新增 14 项，全部 PASS，零真实 API）

| 组 | 测试 |
|---|---|
| collect_candidates 消费 VisionOutput | 成功合并 / 失败不注入 / systemic 不注入 / 无 vision 依赖不受影响 |
| VisionEvidence 白名单 | extra=forbid + 精确字段集 |
| DAG Case A | 全成功 → synth 含证据，payload 含 bounded vision |
| DAG Case B | 部分失败 → synth 仍执行，失败 item 无证据 |
| DAG Case C | 全 item 失败 → synth 仍执行（不升级 systemic） |
| DAG Case D | systemic → synth 仍执行（contract 保持） |
| 身份/重复/泄露 | 原始 title/score 保留 / 无重复候选 / payload 无 filepath/public_url/base64/raw |
| Validator DAG | 推荐结构合法 / NO_PRODUCER / ORPHAN_TASK |

## 10. Full regression

```
Phase 1-A/B/C   82/82 PASS
Stage 1 Vision  20/20 PASS
Stage 2         27/27 PASS
Stage 3         29/29 PASS
Stage 4         19/19 PASS
Stage 5         14/14 PASS
合计           191/191 PASS
```

## 11. Security / data boundary
- Vision 证据仅白名单字段；无 filepath / public_url / base64 / raw provider response。
- `ImageCandidate.vision` 不进 SSE（`_public_images`/`selection` 只取 id/score/title/tags/public_url）。
- 未修改 `config.yaml`（`VISION_MODEL` 仍 `"null"`）；单测全部 Fake（Fake search / Fake analyzer / Fake LLM）；未调用真实 API；无凭据输出；无数据库/metadata/embedding 修改。

## 12. Non-goals
```
❌ 并行执行（Executor 保持串行，已验证无 ThreadPoolExecutor）
❌ Executor scheduling / 并发机制修改
❌ Context Compiler / Memory / Critic / Reflection / Replanning / MCP / A2A
❌ Synthesizer prompt 重写 / reasoning framework
❌ Legacy agent/tools/search / Router/SSE / 前端 / 数据库 schema / metadata 持久化 / embedding
❌ ImageAnalysis Stage 2 contract / VisionWorker 核心失败语义改动
✅ 全部未做
```

## 13. Known limitations
1. 当 synthesize 只依赖 vision（无 group/retrieve 直接候选源）时，`collect_candidates` 无法取得 base candidate → 退化为确定性空结果。推荐 DAG 始终让 synthesize 依赖 group（或 retrieve），可规避；规划 prompt 未在本 Stage 强制该结构。
2. `collect_candidates` 合并基于「直接依赖」；跨层传递候选不在本 Stage 范围。
3. Vision 证据合并不改变候选排序依据（仍按 score）；证据仅作 Synthesizer 附加输入。

## 14. Acceptance checklist

```
[x] collect_candidates 能消费 VisionOutput（image_id 合并）
[x] VisionOutput 为 Synthesizer 提供 image_id + bounded evidence，无 filepath/URL/base64/raw
[x] image_id 稳定；不创造不存在候选；失败 item 不成为候选；不引入重复候选
[x] item_failed 不成为候选，且 Vision WorkerResult 保持 SUCCEEDED → Synthesizer 可执行
[x] 推荐 DAG retrieve → {group, vision} → synthesize 验证（Validator + Executor）
[x] Case A/B/C/D 全部实测：synth 均 SUCCEEDED
[x] Executor dependency semantics 无冲突（未修改 Executor）
[x] candidate merge：image_id 唯一身份；vision 不覆盖 identity；failed 不注入伪字段；vision-only 不创造候选
[x] Synthesizer 收到 bounded vision；未重写 prompt；validate_selection/fallback 保留
[x] DAG safety tests（12 项要求全覆盖，实际 14 项）
[x] 普通 4-worker DAG 不回归（82 项 + 其余全绿）
[x] Legacy Fast Path 不回归（test_registration gate 回归）
[x] 单测零真实 API；VISION_MODEL/config.yaml 未修改
[x] PHASE2_STAGE5_IMPLEMENTATION_REPORT.md 已生成
```

---

**STAGE 5 = PASS**
Stage 6 = **NOT STARTED**（HARD STOP，等待指令）

## 5. Vision failure semantics（保持 Stage 3 contract）
| Case | Vision WorkerResult | VisionTaskOutput | synth |
|---|---|---|---|
| A 全成功 | SUCCEEDED | success=3 failed=0 systemic=False | SUCCEEDED（含证据） |
| B 部分失败 | SUCCEEDED | success=2 failed=1 systemic=False | SUCCEEDED（失败 item 无证据） |
| C 全 item 失败 | SUCCEEDED | success=0 failed=3 systemic=False | SUCCEEDED（无证据） |
| D systemic | SUCCEEDED | systemic_failed=True, code=CLIENT_UNAVAILABLE | SUCCEEDED（无证据） |

Executor 依赖语义（`any(dep SUCCEEDED)`）与 Stage 3「Worker 恒 SUCCEEDED」设计一致，无冲突，无需修改 Executor。

## 6. Candidate merge semantics
- `image_id` 是唯一 merge 键；Vision 不修改候选身份/score/title，只新增独立 `vision` 字段。
- 失败 item 不注入 title/description/tags/evidence；systemic 输出整体跳过。
- Vision-only id（不在既有候选）不创造 recommendation candidate（candidate-consuming 而非 candidate-discovery）。
- 无重复候选（按 id 去重后数量不变）。

## 7. Synthesizer boundary
- `SynthesizerAgent._user_prompt()`：候选 dict 增加 bounded `vision`（image_id/title/description/tags/evidence/confidence），字段来自 `VisionEvidence`（extra=forbid 白名单）。
- 不重写 prompt；不引入 Context Compiler/Critic/Reflection/Replanning。
- 保留 `validate_selection()` 与 deterministic fallback（fallback 走 `DeterministicSynthesisWorker`，同样经 `collect_candidates` 获得证据）。

## 8. Files changed
| 文件 | 动作 | 原因 |
|---|---|---|
| `orchestration/vision.py` | 修改 | 新增 `VisionEvidence`（bounded 白名单证据模型） |
| `orchestration/workers.py` | 修改 | `ImageCandidate.vision` 字段；`collect_candidates` 支持 VisionTaskOutput；`_attach_vision_evidence` |
| `orchestration/synthesizer.py` | 修改 | 候选 payload 注入 bounded vision evidence |
| `tests/agent_team/test_dag_propagation.py` | 新增 | 14 项 Stage 5 测试 |
| `PHASE2_STAGE5_IMPLEMENTATION_REPORT.md` | 新增 | 本报告 |

未修改：executor.py（串行保持）、models.py、planner.py、Legacy、Router/SSE、前端、数据库、embedding、ImageAnalysis、VisionWorker 失败语义。
