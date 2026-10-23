# PHASE3_SUPERVISOR_PRODUCTION_INTEGRATION_REPORT

> ⚠️ **历史快照 + 口径变更**：本文记录的"默认 False / 需 `ENABLE_SUPERVISOR_RUNTIME=true` 才启用"是 Phase 3.1 落地时的行为。当前代码已翻转默认值：**未设置该变量即为开启**（`recommendation/__init__.py:38-45`），TEAM 路径用 `SupervisorAgent(enhanced=True, backend="langgraph")`；要回退基线需显式设 `ENABLE_SUPERVISOR_RUNTIME=false`。接线点与测试结论不变。


> 项目：SmartImager — 多模态智能图库与对话推荐系统
> 阶段：Phase 3.1 — Supervisor Production Integration（生产 TEAM 链路接线）
> 日期：2026-09-07
> 原则：只做接线，不重新设计架构；默认关闭新链路，保持现有行为完全不变。

---

## 1. 变更内容（最小接线）

**修改 1 个文件**
- `backend/ai_func/recommendation/__init__.py`（`_RecommendationService`）

**新增 1 个接线测试**
- `backend/tests/agent_team/test_supervisor_production_integration.py`（5 用例）

**未改动**：Planner / Validator / DAGExecutor / SpecialistAgent / ToolRegistry / supervisor.py /
orchestrator.py / workers.py / SSE 路由 / DB / 前端。

### 1.1 feature flag

| 项 | 值 |
|----|----|
| 环境变量 | `ENABLE_SUPERVISOR_RUNTIME` |
| 取值（True） | `1` / `true` / `yes` / `on`（不区分大小写） |
| 默认 | False（任何其他/未设置） |

### 1.2 控制器选择（`stream_core` TEAM 分支）

```text
flag=False（默认）→ AdaptiveTeamOrchestrator（Phase 2 基线，调用参数完全不变）
flag=True        → SupervisorAgent(enhanced=True)
                      → Planner → Validator → DAGExecutor → Specialist → Review
```

- `SupervisorAgent` 惰性创建（`service._supervisor_runtime`），支持测试注入；
- Legacy 快速路径与 plan 失败回退逻辑位置/语义未动（`controller.last_plan_ok` 判定）。

---

## 2. 行为保持（SSE / 结果 / fallback）

| 契约 | 保持方式 |
|------|----------|
| SSE event contract | 事件流仍为 `plan.created/plan.failed`、`task.*`、`assistant_delta`、`selection`；Supervisor 额外 `agent.review` 事件被 router 按“未知事件忽略”安全丢弃，不污染前端 |
| assistant_delta / selection | SupervisorAgent 输出结构与 Orchestrator 一致（text / ids / images 白名单校验不变） |
| Legacy fallback | 计划失败仍走 `RecommendAgent.run_stream`（`plan.failed → legacy`），测试 `test_legacy_fallback_preserved_on_plan_failure` 覆盖 |
| Memory | flag=True 时 Supervisor 控制流暂不含 episodic recall/write（Phase 3.1 接线范围外，控制台有提示）；flag=False 原记忆行为不变 |

---

## 3. 接线测试与结果（A/B/C）

运行：`venv\Scripts\python.exe backend/tests/agent_team/test_supervisor_production_integration.py`

| 用例 | 验证 | 结果 |
|------|------|------|
| test_flag_parses_default_off | 默认 False；`true` 时 True | ✅ PASS |
| test_a_flag_off_matches_phase2_baseline | flag=False → selection=[11,12]（Phase2 基线）；无 `agent.review`；Legacy 未触发 | ✅ PASS |
| test_b_flag_on_routes_through_supervisor | flag=True → 出现 `agent.review continue`；run_state.final_state=done；selection=[401,402]；Legacy 未触发 | ✅ PASS |
| test_c_failure_recovery_via_supervisor_review_retry | 首次空检索 → decisions=[revise, continue]；重试后 selection=[401,402]；Legacy 未触发 | ✅ PASS |
| test_legacy_fallback_preserved_on_plan_failure | flag=True + Planner 失败 → `plan.failed` + Legacy 兜底 selection=[999] | ✅ PASS |

运行日志（真实分支证据）：
```text
[ComplexityGate] route=team
[SupervisorRuntime] TEAM path via SupervisorAgent (enhanced=True)
[Specialist:retrieval] RetrievalAgent executed task t1 (retrieve_images) via RetrievalWorker
[ToolRegistry:OK] agent=system tool=search_images latency_ms=0.0
agent.review decision=continue | decision=revise,continue（恢复场景）
```

---

## 4. 回归结果

| 套件 | 结果 |
|------|------|
| test_supervisor_production_integration.py（新增） | 5 passed |
| test_supervisor_state_machine.py | 5 passed |
| test_orchestrator.py（service 集成回归） | 6 passed |
| test_sse_contract.py（SSE 契约回归） | 9 passed |
| test_registration.py | 19 passed |
| test_executor.py / test_workers.py | 8 / 12 passed |
| test_agent_cards.py / test_agent_registry.py / test_specialist_agent.py | 6 / 7 / 10 passed |
| tools（registry / schema / binding） | 11 / 7 / 7 passed |

**合计：新增 5 + 回归 107 = 112 passed / 0 failed**（另含此前 Phase1–3 全量基线 164 passed 无回归）。

---

## 5. 启用方式与风险

- 启用：进程环境变量 `ENABLE_SUPERVISOR_RUNTIME=true`（如
  `$env:ENABLE_SUPERVISOR_RUNTIME='true'; python main.py`）；
- 关闭/回滚：不设置或置 `0` 即回 Phase 2 基线，事件与行为完全一致；
- 已知说明（非缺陷）：flag=True 时 Supervisor 控制流不接入 episodic memory recall/write；
  如需，应在后续接线中为 `SupervisorAgent.run_stream` 增加 memory 钩子（保留本阶段“不改设计”边界）。
