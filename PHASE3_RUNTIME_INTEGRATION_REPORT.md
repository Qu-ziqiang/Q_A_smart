# PHASE3_RUNTIME_INTEGRATION_REPORT

> 项目：SmartImager — 多模态智能图库与对话推荐系统
> 报告性质：Phase 3（Supervisor Agent）**运行链路集成验证 / 审计报告**
> 日期：2026-09-07
> 约束：仅运行审计与测试；**未修改任何生产代码 / 测试 / 数据库 / 前端，未添加功能**

---

## 1. 测试环境与方法

| 项 | 值 |
|----|----|
| 代码基准 | `Smartlmager-sqlite-vec/`（Phase 1 + 2 + 3 现状） |
| 验证对象 | `orchestration/supervisor.py`（SupervisorAgent FSM） |
| 运行模式 | 离线确定性（真实 Planner/Validator/DAGExecutor/AgentRegistry/Specialist/Worker/DeterministicSynthesis 组件；检索叶子与视觉 analyzer 用夹具，注入位同产品既有） |
| 对照基线 | `AdaptiveTeamOrchestrator`（Phase 2 生产 TEAM 路径，未改动） |
| 验证工具 | 只读 harness（系统临时目录，未入库） |

**接线事实（审计前置结论）**：代码审计确认服务层 TEAM 路径仍为
`_RecommendationService._get_orchestrator() → AdaptiveTeamOrchestrator`，
**`SupervisorAgent` 尚未挂入生产调用链**（Phase 3 报告 §11 已列为“接入服务层”后续项）。
因此本次对“Enhanced Supervisor Runtime”的验证方式为：**以 `SupervisorAgent.run_stream`
作为受控入口**驱动完整请求（与 Phase 2 运行时验证同等真实度），并对 `enhanced=False`
与生产基线做同数据对照。若验收口径要求“线上默认请求必须经 SupervisorAgent”，当前为**未接入**
（见 §6 判定）。

---

## 2. 验证链路

### 2.1 目标链路（Supervisor 增强 Runtime）
```text
User
 → SupervisorAgent.run_stream
 → PlannerAgent → PlanValidator
 → DAGExecutor → AgentRegistry → SpecialistAgent（retrieval/organizer/vision）
 → ToolRegistry（adapter）→ Worker
 → Review（确定性规则）→ Final Output（assistant_delta / selection）
```

### 2.2 实测（增强模式，成功场景）——逐段命中 ✅
| 段 | 证据 |
|----|------|
| Supervisor 状态机 | `[states] plan→execute→review→done` |
| Planner/Validator | `plan.created`、validator 通过（服务层同款代码） |
| Executor 拓扑 | t1 retrieve → t2 group → t3 synthesize 全 succeeded |
| Specialist | `RetrievalAgent executed t1`、`OrganizerAgent executed t2` |
| ToolRegistry | `[ToolRegistry:OK] agent=system tool=search_images`；`tool_events=[(search_images,true)]` |
| Review | `agent.review decision=continue reason=执行结果满足要求` |
| Final | `assistant_delta` + `selection=[301,302,303]` |

---

## 3. Success Scenario 记录（“找一些自然风景图片并按照地点分类”）

```text
[states] ['plan','execute','review','done']
[decisions] ['continue']
[final_state] done
[plan] [{"task_id":"t1","task_type":"retrieve_images"},
        {"task_id":"t2","task_type":"group_images"},
        {"task_id":"t3","task_type":"synthesize"}]
[agent_messages]
  ('supervisor','retrieval','task')  ('retrieval','supervisor','report')
  ('supervisor','organizer','task')  ('organizer','supervisor','report')
  ('supervisor','synthesizer','task')('synthesizer','supervisor','report')
[tool_events] [("search_images", true)]
[final_selection] [301, 302, 303]
```

- Supervisor→Specialist 下发 task；Specialist→Supervisor 回报 report（Organizer 正常参与）；
- 检索工具经 ToolRegistry 调用并留痕；
- 确定性 REVIEW `continue` → DONE，最终推荐与检索候选白名单一致。

---

## 4. Recovery Scenario 记录

### 4a. retrieval empty → REVISE → retry → DONE
```text
[decisions] ['revise','continue']
[states]    ['plan','execute','review','revise','execute','review','done']
[search_calls] 2
[final_selection] [301, 302, 303]
```
第一次检索为空 → REVIEW `revise` → 修订（limit 放大）后整体重跑 → 第二次成功 → `continue` → DONE。

### 4b. tool failure → REVISE → retry → ABORT（有界）
```text
[decisions] ['revise','abort']
[final_state] abort
[tool_events] [("search_images", false), ("search_images", false)]
```
工具持续失败：ToolCallEvent 失败留痕 → REVIEW `revise`（第 1 次）→ 重试仍失败
→ 达到 max_iterations → `abort`。**无无限循环**。

---

## 5. Baseline Compatibility（enhanced=False vs Phase 2）

同数据集、同一装配下对照：

| 运行 | decisions | search_calls | 终态 | selection |
|------|-----------|--------------|------|-----------|
| SupervisorAgent enhanced=False（空检索数据） | ['continue'] | 1 | done | —（不重试） |
| AdaptiveTeamOrchestrator（Phase 2 生产基线） | 无 agent.review | 1 | — | [301,302,303] |
| SupervisorAgent enhanced=False（同基线数据） | ['continue'] | 1 | done | [301,302,303] |

- Phase 2 基线事件流无 `agent.review`（Supervisor 专属事件未污染基线）；
- `enhanced=False` 时 Supervisor 单次执行，**selection 与 Phase 2 Orchestrator 完全一致**；
- 结论：默认关闭增强 = 行为与 Phase 2 一致 ✅。

---

## 6. 判定与建议

| 验证项 | 结果 |
|--------|------|
| Supervisor FSM 运行时链路（以 SupervisorAgent 为受控入口） | ✅ PASS |
| Success 场景（含 group→OrganizerAgent） | ✅ PASS |
| Recovery：retrieval empty → REVISE → retry → DONE | ✅ PASS |
| Recovery：tool failure → REVISE → ABORT（有界） | ✅ PASS |
| enhanced=False 与 Phase 2 行为一致（selection 等同） | ✅ PASS |
| 生产服务层请求默认经 SupervisorAgent | ⚠️ **未接入**（当前默认仍 AdaptiveTeamOrchestrator） |
| 回归（Phase 3 状态机单测 + Phase1/2 全量） | ✅ 164 passed / 0 failed（此前已验） |

**判定**：SupervisorAgent 作为受控监督层**运行就绪**（增强链路、恢复路径、工具轨迹、
消息协议均已实证）；但**生产接线仍停留在 Phase 2 基线**。建议作为紧接项把
`SupervisorAgent` 通过 feature flag 接入 `_RecommendationService.stream_core` 的 TEAM 路径
（默认关 = 维持现网行为，开启 = 增强），接线时复跑本报告 Success/Recovery/Baseline 三组
场景作为回归门禁。
