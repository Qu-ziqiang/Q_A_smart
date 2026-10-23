# PHASE2 Stage 4 — Capability and Registration 实现报告

> 阶段：Phase 2 — Vision Capability Enablement / **Stage 4 — Capability and Registration**
> 结论：**PASS**（177/177 测试全绿；capability 单一来源，无 drift；未进入 Stage 5）

---

## 1. Stage 4 Result
**PASS** — VisionWorker 已通过 capability 链路接入 Planner / Validator / Worker registry，且 `VISION_MODEL` 保持 `"null"`（未启用生产 Vision）。

## 2. Pre-change audit

| 审计项 | 修改前真实状态 | 问题 |
|---|---|---|
| `vision_available()` | `models.py`，检查 VISION_MODEL 非空/非 `"null"` + `get_openai_client()` 非 None | 合理，保留为唯一能力来源（不弱化） |
| Planner `_default_capabilities()` | `enabled = [t for t in TaskType if t is not ANALYZE_VISUAL]` —— **硬编码排除**，无视 `vision_available()` | **drift/bug** |
| Validator `registered_types` | `__init__` 默认 `set(REGISTERED_TASK_TYPES)`（静态，不含 Vision）；`validate()` step4 用静态集合 | **drift**：即便 `vision_available_flag=True`，ANALYZE_VISUAL 仍被 `WORKER_UNAVAILABLE` 拒绝 |
| `REGISTERED_TASK_TYPES` | 静态 base set（不含 Vision） | 保留为 base（计划 §630-632 建议） |
| `default_workers()` | 静态 4 个 Worker | 需要 capability gating |
| `ANALYZE_VISUAL` 提前注册 | 无任何位置提前注册 | 符合预期 |
| `TaskType.ANALYZE_VISUAL` | 枚举已定义 | defined-but-disabled ✓ |

## 3. Capability source（单一来源）
- **`vision_available()`**（未修改）：`VISION_MODEL 非空且非 "null"/"none"` **且** `get_openai_client()` 可用。未弱化为 `bool(VISION_MODEL)`（遵守 §13：configured ≠ available）。
- **新增 `registered_task_types(vision_available_flag=None)`**（`models.py`）：base = `REGISTERED_TASK_TYPES`；仅当能力可用时加入 `ANALYZE_VISUAL`。Planner enabled / Validator registered / `default_workers` 三处统一调用/派生，消除 capability drift。

## 4. Planner changes（planner.py）
- `_default_capabilities()`：`enabled = [t for t in TaskType if t in registered_task_types()]`。
- 结果：vision 关闭 → enabled 不含 `analyze_visual`；vision 开启 → 包含。
- 不重新设计 prompt；`PLANNER_SYSTEM_PROMPT_TEMPLATE`、`TASK_PLAN_DRAFT_SCHEMA`、Legacy Fast Path 均未改动。

## 5. Validator changes（models.py PlanValidator）
- `__init__`：`registered_types=None` → validate() 时按 capability 动态解析（默认行为）。
- `validate()`：先解析 `vision_available_flag`，再计算有效 registered 集合：
  - 默认（None）：`registered_task_types(vision_available_flag)`；
  - 显式注入：`set(registered_types)` + 能力开启时补入 `ANALYZE_VISUAL`。
- 原有安全边界全部保留：TaskType/input 类型化/依赖/DAG/循环/深度/孤儿/生产者/未知任务拒绝等未降低。

## 6. Worker registration（workers.py）
- `default_workers()`：4 个 base Worker + **`if vision_available(): workers[ANALYZE_VISUAL] = VisionWorker()`**（capability 在注册层成立，不依赖运行时失败兜底）。
- `orchestration/__init__.py`：导出 `VisionWorker`，可通过正式 registry 被发现（DAGExecutor 分派验证）。

## 7. DAG rules
- `CANDIDATE_CONSUMING_TYPES` 已含 `ANALYZE_VISUAL` → Validator 要求 Vision（传递）依赖候选生产者（`NO_PRODUCER` 拒绝）。
- 孤儿检查 → Vision 必须（传递）到达 synthesize（`ORPHAN_TASK` 拒绝）。
- 推荐结构 `retrieve → {group, vision} → synthesize` 验证合法。
- 未修改 Executor scheduling（保持串行）。

## 8. Disabled capability behavior（真实配置验证）
```
vision_available()                        = False
Planner enabled_task_types                = [filter_metadata, group_images, retrieve_images, synthesize]（无 analyze_visual）
PlanValidator(vision_available_flag=False)= 拒绝 ANALYZE_VISUAL（VISION_UNAVAILABLE / WORKER_UNAVAILABLE）
registered_task_types(False)              = REGISTERED_TASK_TYPES（无 Vision）
default_workers()                          = 4 个 base Worker（无 VisionWorker）
```

## 9. Enabled capability behavior（mock 验证，零真实 API）
```
vision_available()                        = True（mock）
Planner enabled_task_types                = 含 analyze_visual
PlanValidator(vision_available_flag=True) = 接受合法 retrieve→analyze_visual→synthesize 计划
registered_task_types(True)               = 含 ANALYZE_VISUAL
default_workers()                          = 含 VisionWorker 实例
```

## 10. Tests（Stage 4 新增 19 项，全部 PASS，零真实 API）

| 组 | 测试 | 数量 |
|---|---|---|
| A. Capability False | vision_available=False / Planner 排除 / Validator 拒绝 / workers 缺位 / registered 排除 | 5 |
| B. Capability True | Planner 包含 / Validator 接受 / workers 在册 / registered 包含 / 默认使用能力 | 5 |
| C. Registry consistency | disabled 三方一致 / enabled 三方一致 / registry 可发现 | 3 |
| D. 既有任务回归 | disabled 4 worker+普通计划通过 / enabled 4 worker+普通计划通过 | 2 |
| E. Legacy 回归 | Gate simple→legacy / complex→team | 1 |
| Vision DAG | 无 producer→NO_PRODUCER / 不达 synthesize→ORPHAN / 并行结构合法 | 3 |

## 11. Regression

```
Phase 1-A/B/C   82/82 PASS
Stage 1 Vision  20/20 PASS
Stage 2         27/27 PASS
Stage 3         29/29 PASS
Stage 4         19/19 PASS
合计           177/177 PASS
```

## 12. Files changed

| 文件 | 动作 | 原因 |
|---|---|---|
| `orchestration/models.py` | 修改 | 新增 `registered_task_types()`；`PlanValidator` 支持 validate 时 capability-aware 注册 |
| `orchestration/planner.py` | 修改 | `_default_capabilities()` 改用 `registered_task_types()`（修复硬编码 drift） |
| `orchestration/workers.py` | 修改 | `default_workers()` capability-gated 注册 VisionWorker |
| `orchestration/__init__.py` | 修改 | 导出 `VisionWorker` |
| `tests/agent_team/test_registration.py` | 新增 | 19 项 Stage 4 测试 |
| `PHASE2_STAGE4_IMPLEMENTATION_REPORT.md` | 新增 | 本报告 |

## 13. Security / configuration boundary
- **未修改 `config.yaml`**（`VISION_MODEL` 保持 `"null"`）；未打印/写入任何 key；未调用真实 API（测试仅 mock `vision_available`）；未写数据库 / metadata / embedding。
- 单测不执行 Vision 任务（仅断言注册与校验），杜绝意外真实请求。

## 14. Non-goals
```
❌ Vision E2E / Synthesizer 修改 / Vision evidence 注入 / collect_candidates 新语义 / metadata 回填
❌ SSE 新字段 / 前端 / 并行 DAG / Executor 并发 / Context Compiler / Memory / Critic / Replanning / MCP / A2A
❌ Executor scheduling 修改（保持串行）
✅ 全部未做（Stage 5 起再议）
```

## 15. Known limitations
1. `default_workers()` 在每次调用时读取 `vision_available()`（进程内配置变更即时生效；无缓存）。
2. 测试通过 mock 三个模块的 `vision_available` 引用模拟能力开启；真实开启需 Stage 10 修改配置后验证。
3. Validator 显式注入 `registered_types` 时，能力开启仍会补入 ANALYZE_VISUAL（保持一致性）。

## 16. Acceptance checklist

```
[x] vision_available() 能力来源明确且无 drift（单一 registered_task_types()）
[x] Vision disabled：Planner 不提供 / Validator 拒绝 / default_workers 不注册
[x] Vision enabled：Planner 提供 / Validator 接受合法计划 / default_workers 注册
[x] ANALYZE_VISUAL DAG dependency rules 正确（producer / orphan / 并行结构）
[x] VisionWorker 可通过正式 registry 被发现（DAGExecutor 分派验证）
[x] 不修改 Executor scheduling
[x] 不修改 Synthesizer
[x] 不修改 SSE / Router / Frontend
[x] 不修改 Legacy 路径（Gate 回归验证）
[x] Fake / Mock Stage 4 tests 全部通过（19/19）
[x] 原有 158 tests 全部通过（实际 177/177 全绿）
[x] 无真实 API 调用进入普通单测
[x] 无 credential 泄露
[x] 无数据库 / metadata / embedding 修改
```

---

**STAGE 4 = PASS**
Tests: Stage 4 = 19/19 | Regression = 158/158 | Total = 177/177
Files changed: models.py / planner.py / workers.py / __init__.py / test_registration.py / 本报告
ANALYZE_VISUAL: disabled = unavailable/rejected/unregistered | enabled = available/accepted/registered
VisionWorker registration: disabled = absent | enabled = present（VisionWorker 实例）
Stage 5: **NOT STARTED**

