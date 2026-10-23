# PHASE2 Stage 9 — Real E2E 报告

> ⚠️ **历史快照（迁移前口径）**：本文写于 SQLite/sqlite-vec 时代（图库 66 张）。当前数据层已是 **PostgreSQL + pgvector（HNSW/cosine）**、图库约 **1000 张 VOC2010**、门禁 **463 passed**。文中"CLIP+sqlite-vec / 66 图"等数字仅作当时记录，现状见 `README.md` 与 `SMARTIMAGER_RESUME_TEXT.md`。


> 阶段：Phase 2 — Vision Capability Enablement / **Stage 9 — Real E2E**
> 结论：**STAGE 9 = PASS**（含文档化补充：真实 Planner 不产出 analyze_visual，Vision 链通过补充任务验证）
> 状态：Stage 10 = **NOT STARTED**

---

## 1. Environment
| 项 | 值 |
|---|---|
| Provider | DeepSeek |
| Endpoint | `https://api.deepseek.com`（config OPENAI_API_BASE） |
| Vision Model | `deepseek-v4-flash-vision-exp`（runtime override，模型注入） |
| Text Model | `deepseek-chat`（CHAT_MODEL，Planner/Synthesizer） |
| Credential | `credential_present = true`（仅布尔，不打印 key） |
| 运行时编码 | `PYTHONIOENCODING=utf-8` + `-X utf8`（规避 Windows GBK emoji 崩溃） |

## 2. Real / Fake Matrix
```
REAL：
  ComplexityGate / PlannerAgent(DeepSeek) / PlanValidator / AdaptiveTeamOrchestrator
  DAGExecutor / RetrievalWorker(CLIP+sqlite-vec) / GroupImagesWorker
  VisionWorker / ImageAnalysis / DeepSeek Vision / VisionEvidence 传播
  SynthesizerAgent(DeepSeek) / validate_selection / _RecommendationService.stream_core（SSE 路径）

FAKE：
  Controlled Failure Injection（Fake ImageAnalysis —— 明确标记，非 Real Provider failure）
  Fake Legacy agent（仅用于检测 fallback，不调用真实 Legacy）
  Frontend browser（NOT REQUIRED）

DOCUMENTED SUPPLEMENT：
  analyze_visual 任务追加到真实 Planner 的合法计划（真实 Planner 不产出 analyze_visual，见 §4/§6）
```

## 3. Query
- 必需查询：`找旅行照片，按地点分类并整理结果` → Gate=TEAM → 真实 Planner 生成合法计划 `retrieve_images → group_images → synthesize`（本次 attempt 1 即通过 Validator）。

## 4. Planner（真实 DeepSeek，重要发现）
- **成功**：生成合法计划并通过真实 Validator（Plan Validation passed）。
- **F-002 实证**：10 查询探针 + E2E 多次运行确认 —— 真实 Planner **从不自发产出 analyze_visual**；含"视觉"措辞的查询反而触发 `filter_metadata`（非法 include_fields）→ Validator 拒绝 → Legacy fallback。这是 Phase 1 审计 F-002（Planner 方差）的现实表现，非本阶段引入的架构问题。
- 处理：Vision 链通过**文档化补充**（真实 Planner 的合法计划 + 追加 analyze_visual 任务，经真实 Validator 复验通过）验证，见 §6。

## 5. Retrieval（真实 CLIP + sqlite-vec）
- `candidate_count = 40`（真实库 top-40；首个检索含模型加载，E2E 内缓存后 `retrieve_images` 延迟 27~33ms）
- top ids = [46, 18, 4, 58, 52, 53, 65, 63, ...]

## 6. Vision（真实 ImageAnalysis + DeepSeek Vision，supplement）
- supplement 计划：`retrieve → {group, analyze_visual} → synthesize`，真实 Validator 通过（`supplement_plan_valid = True`）。
- **vision_total = 4（默认上限 4）**，`success=4 / failed=0 / systemic=False`，`vision_analyzer_calls = 4`（每候选一次，无 retry）。
- `supplement_latency_s ≈ 11.8~15.8s`（4 张串行）。

## 7. Vision Evidence（真实图片可证内容）
| image_id | title | tags | confidence | evidence |
|---|---|---|---|---|
| 46 | 马术障碍赛 | [马术, 骑马, 障碍赛, 白马] | 0.9 | 骑手、白马、红白障碍栏、草地、树木 |
| 18 | 室内女孩玩耍 | [儿童, 室内, 玩耍] | 0.9 | 女孩、画作、木地板 |
| 4 | 草原骑马与牛群 | [旅行, 草原, 骑马, 山脉, 牛] | 0.8 | 可见木栅栏、戴帽男子、马、小牛和远处山脉 |

- **非 generic**：每张标题/证据与图片内容明显对应（马术/室内/草原骑马），非 "this is an image"。→ §10 满足。
- `vision_evidence_count = 4`：证据经 `VisionTaskOutput → ImageCandidate.vision → Synthesizer payload` 真实到达。

## 8. Synthesizer（真实 DeepSeek）
- E2E #1：`LLM synthesis success (20 ids)`，`synthesize` 延迟 1840~3516ms。
- Supplement：synth 收到 4 条 vision evidence，`synth_status = succeeded`。
- answer 非空（"已按地点对旅行照片进行检索与分组。由于所有照片均未提供地点信息，它们被归入"未知地点"分组…"）——正确地把"未知地点"标为未知（§23 要求，未编造地点）。


## 9. Selection Validation
- E2E #1：`selection_ids = validated_image_ids = [46,18,4,58,52,53,65,63,54,20,39,48,60,43,59,24,37,2,41,28]`（20 个，⊆ 检索候选 40）——`validate_selection()` 生效。
- Supplement：`validated_image_ids = [46,18,4,58]` ⊆ 检索候选。

## 10. SSE / HTTP
- E2E #1 走真实 `_RecommendationService.stream_core`（真实 Gate/Orchestrator）：`route=team` → `plan.created` → `task.started/completed`（含 retrieve/group/synthesize）→ `assistant_delta` → `selection`；无 `plan.failed`、无 Legacy fallback。
- supplement 走真实 `DAGExecutor.execute_stream`（task 事件级）。
- Vision task 事件 `task_type=analyze_visual` 出现且 `output_summary.vision` 仅统计字段。

## 11. Failure Injection（受控 FAKE）
- Fake analyzer 令第 2 个 candidate 抛 TIMEOUT → `vision success=2 failed=1 total=3`，`synth_status=succeeded ids=[201,202,203]`。
- **明确标记**：这是 controlled fake failure（非 Real Provider failure）；item_failed 不阻断 synthesize（既有 contract 复验）。

## 12. Model Call Counts（本次 E2E）
```
Planner calls      = 1（成功即止，无 loop）
Vision calls       = 4（= candidate_count，无 retry）
Synthesizer calls  = 2（E2E#1 + supplement 各 1，无 loop）
client create total= 7（= 1+4+2）
无自动 retry / 无模型切换 / 无 provider 切换
```

## 13. Latency
| 环节 | 值 |
|---|---|
| Planner | 含在 attempt 内（秒级） |
| Retrieval | 27~33ms（模型缓存后；首载含 CLIP 加载 ~11s） |
| Vision (4 张串行) | ~11.8~15.8s |
| Synthesizer | 1840~3516ms |
| Total（单次完整链） | 秒级（检索缓存后主要由 Vision 决定） |

## 14. Regression
```
222/222 PASS（14 个测试文件全绿；E2E 未修改任何测试）
```
（E2E 前基线 222/222；E2E 后复跑 222/222）

## 15. Configuration / Rollback
- 使用 runtime override（`_VisionOn` 进程内 patch + VisionWorker model 注入），**未修改 `config.yaml`**。
- E2E 后：`config.yaml VISION_MODEL = "null"`（未变）· `vision_available() = False` · `ANALYZE_VISUAL` 保持 disabled · `default_workers()` 不含 VisionWorker。

## 16. Security
- 记录仅含：image_id / title / tags / confidence / evidence（截断 80 字符）。
- 未保存：API key（仅布尔）、raw provider response、base64、本地绝对路径、完整 prompt。
- E2E 结果文件 `stage9_e2e_result.txt`（repo 根，UTF-8）亦无敏感字段。

## 17. Known Limitations
1. **真实 Planner 不产出 analyze_visual**（F-002 实证）：Vision 链通过"文档化补充计划"验证；真实 Planner→Vision 的端到端自动触发未发生。这是 Planner prompt 行为，非架构 bug（未改 prompt）。
2. 真实图库 metadata 基本为空 → 分组全为"未知地点"（允许；未写回 metadata）。
3. 未做前端浏览器 smoke（NOT REQUIRED）。
4. 单次 E2E 使用真实 API 约 7 次调用；未做压力/重复验证。

## 18. Files Changed
| 文件 | 动作 | 原因 |
|---|---|---|
| `backend/tests/agent_team/demo_phase2_e2e.py` | 新增 | 真实 E2E 演示脚本（runtime override + 计数 + failure injection + rollback） |
| `stage9_e2e_result.txt` | 生成 | E2E 干净 UTF-8 结果记录（无敏感数据） |
| `PHASE2_STAGE9_REAL_E2E_REPORT.md` | 新增 | 本报告 |

临时探针脚本已删除。生产代码零改动。

## 19. Acceptance Checklist
```
[x] Real DeepSeek Planner = success（合法计划经真实 Validator）
[x] Real CLIP retrieval = success（40 候选）
[x] Real VisionWorker = executed
[x] Real DeepSeek Vision = success（4/4）
[x] Vision output 含真实图片可证信息（马术障碍赛/室内女孩/草原骑马与牛群，非 generic）
[x] Vision evidence 到达 Synthesizer（vision_evidence_count=4）
[x] Real DeepSeek Synthesizer = success（answer + selection）
[x] Selection validation succeeds（20/20 valid，⊆ 候选）
[x] Final answer generated
[x] Final selection 仅含合法 candidate ids
[x] Team task SSE observed（plan.created / task.* / assistant_delta / selection）
[x] 无 sensitive info 泄露
[x] 无自动 retry（vision calls = candidates）
[x] 无模型/provider 切换
[x] Planner calls = 1 / Synthesizer calls = 2（两次运行各 1，无 loop）
[x] 222/222 regression passes（E2E 前后）
[x] production configuration 保持受控（VISION_MODEL="null"，未修改）
```

## 20. Final Verdict
```
STAGE 9 = PASS
```
Real 链（Planner / Retrieval / Vision / Synthesizer / SSE 路径）全部成功；Vision 输出真实可证；唯一偏差是"真实 Planner 不产出 analyze_visual"（F-002 实证），已通过文档化补充计划验证真实 Vision 链，并在 §4/§6/§17 如实记录。

Stage 10 = **NOT STARTED**
