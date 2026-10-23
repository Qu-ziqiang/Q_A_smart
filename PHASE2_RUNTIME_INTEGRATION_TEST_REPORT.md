# PHASE2_RUNTIME_INTEGRATION_TEST_REPORT

> 项目：SmartImager — 多模态智能图库与对话推荐系统
> 报告性质：Phase 2 完成后的 **Multi-Agent Tool Registry Integration 运行链路审计 / 验证报告**
> 日期：2026-09-07
> 约束：仅审计 / 测试 / 报告；**未修改任何生产代码，未新增功能，未调整架构**

---

## 1. 测试环境

| 项 | 值 |
|----|----|
| 代码基准 | `Smartlmager-sqlite-vec/`（backend FastAPI + React） |
| Agent Runtime | `backend/ai_func/recommendation/orchestration/`（Phase 1 + Phase 2 现状） |
| Python | 项目 venv（`-X utf8`） |
| 运行模式 | **离线确定性验证**：真实 Planner/Validator/DAGExecutor/AgentRegistry/SpecialistAgent/Worker/SynthesizerAgent 代码路径；仅最外层叶子替换为内存夹具（检索固定候选行；视觉 analyzer 固定 JSON），零真实 DB / LLM / 视觉 API |
| 验证工具 | 只读 harness（位于系统临时目录，未写入仓库） + 既有测试套件（Phase1+Phase2 全绿） |

**离线替换边界（诚实声明）**：检索数据面（`SearchRepository.unified_search`）与视觉
provider（`ImageAnalysis` 网络调用）是不可确定性复现的外层叶子；验证通过产品内置注入位
`RetrievalWorker(search_tool=…)`、`VisionWorker(analyzer=…)` 注入夹具，**Worker 核心逻辑与
Agent/Registry/Executor/Validator/Planner 均为产品代码原样执行**。

---

## 2. 验证流程

1. 代码级审计：确认 Planner→Validator→Executor→AgentRegistry→SpecialistAgent→ToolRegistry 接线；
2. 链路复现：`AdaptiveTeamOrchestrator.run_stream` 走完整
   `User Query → Planner → Validator → DAGExecutor → SpecialistAgent → Worker → Synthesizer`；
3. 三个真实用户场景（A/B/C）端到端执行，采集计划 / 事件顺序 / Agent 日志 / 工具日志 / 结果；
4. 工具调用层验证：Agent → `ToolRegistry.call`（schema 校验 / ToolCallEvent / success / latency）；
5. MCP schema 导出核对；
6. 架构判断与 Phase 3 准备度评估。

---

## 3. 目标链路 vs 实测链路

### 3.1 需求声称链路（Requirement 1）

```text
User Query → PlannerAgent → PlanValidator → DAGExecutor
          → SpecialistAgent → ToolRegistry → Builtin Tool → Worker → SynthesizerAgent
```

### 3.2 实测链路（Phase 2 现状）

```text
User Query
 → ComplexityGate(team) → AdaptiveTeamOrchestrator.run_stream
 → PlannerAgent.create_plan               ✅
 → PlanValidator.validate                 ✅
 → DAGExecutor.execute_stream             ✅  （拓扑序 / 依赖 / 事件不变）
 → AgentRegistry.get_agent_for_task       ✅
 → SpecialistAgent.execute                ✅  日志: Retrieval/Organizer/Vision executed task
 → Worker.run(task, context)              ✅  真实 Worker（注入叶子 search_tool/analyzer）
 → SynthesizerAgent（LLM 注入 / 确定性回退）✅  终端回答 + selection
```

### 3.3 工具层位置（架构判定前提）

- 实测确认：**plan 执行热路径上 Worker 直接调用其注入实现，ToolRegistry 不位于该热路径**。
  这是 Phase 2 设计的有意取舍（“不修改 Worker 核心逻辑 / 执行稳定性红线”），并已在
  `PHASE2_TOOL_REGISTRY_IMPLEMENTATION_REPORT.md` §10 列为已知差距 #2。
- ToolRegistry 当前承担 **Agent 统一工具入口与治理层**：`SpecialistAgent.call_tool(name, arguments)`
  经注册中心调用 Builtin Tool（薄适配 → 与 Worker 注入位同源的实现），并生成 `ToolCallEvent`。
  该层已通过独立调用验证（见 §6）。

> **判定**：若口径为“plan 执行路径必须逐任务穿过 ToolRegistry” → **部分通过**；
> 若口径为“Agent 具备经 ToolRegistry 的统一、可治理工具调用能力” → **通过**。

---

## 4. Runtime Trace（三个端到端场景）

运行输出摘录（UTF-8，真实组件日志；selection 仅含候选白名单内 id）。

### Scenario A — “帮我找10张关于城市建筑的图片”
```text
[Routing] retrieve_images -> RetrievalAgent
[Runtime Trace]
    plan.created tasks=[('t1','retrieve_images'), ('t2','synthesize')]
    task.started   t1 retrieve_images running
    task.completed t1 retrieve_images succeeded
    task.started   t2 synthesize running
    task.completed t2 synthesize succeeded
    assistant_delta len=20
    selection ids=[101, 102, 104]
[Specialist] [Specialist:retrieval] RetrievalAgent executed task t1 (retrieve_images) via RetrievalWorker
[Result] plan_ok=True selection=[101, 102, 104]
```

### Scenario B — “找一些风景图片，然后按照地点分类”
```text
[Routing] retrieve_images -> RetrievalAgent
          filter_metadata -> OrganizerAgent
          group_images    -> OrganizerAgent
[Runtime Trace]
    plan.created tasks=[('t1','retrieve_images'), ('t2','group_images'), ('t3','synthesize')]
    task.started t1 → completed t1   (检索)
    task.started t2 → completed t2   (group_images)
    task.started t3 → completed t3   (synthesize)
    selection ids=[103]
[Specialist]
    [Specialist:retrieval] RetrievalAgent executed task t1 (retrieve_images) via RetrievalWorker
    [Specialist:organizer] OrganizerAgent executed task t2 (group_images) via GroupImagesWorker
[Result] plan_ok=True selection=[103]
```

### Scenario C — “找一张图片并分析里面有什么”
```text
[Routing] retrieve_images -> RetrievalAgent
          analyze_visual  -> VisionAgent
[Runtime Trace]
    plan.created tasks=[('t1','retrieve_images'), ('t2','analyze_visual'), ('t3','synthesize')]
    task.started t1 → completed t1
    task.started t2 → completed t2   (analyze_visual，VisionWorker + analyzer 注入位)
    task.started t3 → completed t3   (synthesize)
    selection ids=[101, 102, 104]
[Specialist]
    [Specialist:retrieval] RetrievalAgent executed task t1 (retrieve_images) via RetrievalWorker
    [Specialist:vision]    VisionAgent executed task t2 (analyze_visual) via VisionWorker
[Result] plan_ok=True selection=[101, 102, 104]
```

> 事件顺序即 Executor 拓扑序：A: t1→t2；B: t1→t2→t3；C: t1→t2→t3。三场景 `plan_ok=True`。

---

## 5. Agent Routing 验证结果

| 期望 | 实测 | 结果 |
|------|------|------|
| retrieve_images → RetrievalAgent | ✅（三场景） | PASS |
| filter / group task → OrganizerAgent | ✅（filter_metadata、group_images 均路由到 OrganizerAgent） | PASS |
| vision task → VisionAgent | ✅（analyze_visual 路由到 VisionAgent） | PASS |

路由依据：`build_default_agent_registry(workers).get_agent_for_task(task_type)`；
分发表构造：`build_dispatched_workers(base, registry, synthesize_worker=SynthesizerAgent(...))`。

---

## 6. Tool Calling 验证结果

```text
[ToolRegistry:OK] agent=retrieval tool=search_images latency_ms=0.0
    retrieval -> search_images rows=3 (schema validated)
[ToolRegistry:OK] agent=retrieval tool=rerank        latency_ms=0.0
[ToolRegistry:OK] agent=vision tool=analyze_image    latency_ms=0.0
[ToolRegistry:OK] agent=organizer tool=group_images  latency_ms=0.0
    organizer -> group_images groups=['北京', '上海', '成都']
ToolCallEvent trace:
    [ToolRegistry] retrieval -> search_images success=True latency_ms=0.01
    [ToolRegistry] retrieval -> rerank        success=True latency_ms=0.02
    [ToolRegistry] vision -> analyze_image    success=True latency_ms=0.00
    [ToolRegistry] organizer -> group_images  success=True latency_ms=0.02
[Validation] missing required -> ValueError: 缺少必填参数: 'query'
[Failure trace] tool=search_images success=False error=缺少必填参数: 'query'
```

检查项：
- **tool definition 匹配**：`ToolRegistry.get(name)` 返回对应 ToolDefinition，且 AgentCard 绑定
  经 `can_use()` 授权校验；✅
- **input schema 验证**：缺必填 `query` 抛 `ValueError`；✅
- **ToolCallEvent 生成**：每次调用一条事件（tool_name/arguments/agent_id/success/latency_ms/error）；✅
- **latency / success trace 记录**：OK 与 FAIL 两类均落 Trace 与统计；✅

**审计发现（非阻塞）**：内置 `rerank` 工具的 `input_schema.candidates` 声明为 JSON object
数组，而默认实现 `_impl_rerank → rerank_candidates` 期望 `ImageCandidate` 领域对象（含 `.score`
等属性）——存在“schema 层 JSON 化 vs 实现层领域对象”的类型契约不一致（验证仅以 harness 注入适配层
规避，未改产品代码）。建议 Phase 3 前将 rerank 输入收敛为 JSON 候选行并在内部转换
（与 filter/group 工具口径一致）。

---

## 7. MCP Style Schema 验证结果

```text
export_tool_schema(search_images):
    keys=['description', 'inputSchema', 'name']
    {"name": "search_images", "inputSchema.required": ["query"]}
```

结论：输出满足 `{name, description, inputSchema}` 三元组；`inputSchema` 为 JSON Schema 对象，
可直接映射标准 MCP Tool 格式（本阶段未装 MCP SDK / 未启动 Server）。✅

---

## 8. 成功 / 失败汇总

| 验证项 | 结果 |
|--------|------|
| 场景 A：Planner 计划 + RetrievalAgent + Synthesizer 输出 | ✅ PASS |
| 场景 B：RetrievalAgent + OrganizerAgent（group）+ Synthesizer 输出 | ✅ PASS |
| 场景 C：RetrievalAgent + VisionAgent（analyze_visual）+ Synthesizer 输出 | ✅ PASS |
| Agent 路由（retrieve/filter/group/vision → 对应 Specialist） | ✅ PASS |
| 工具调用：definition / schema 校验 / ToolCallEvent / success / latency | ✅ PASS |
| schema 校验失败 → success=False 轨迹 | ✅ PASS |
| MCP schema 导出 `{name, description, inputSchema}` | ✅ PASS |
| 需求 1 的“plan 路径逐任务穿过 ToolRegistry” | ⚠️ 部分通过（设计内差距 #2，见 §3.3） |
| 既有自动化测试基线（Phase1+Phase2） | ✅ 25（tools）+23（agent 层）+核心回归 68 = 116 passed，0 failed（本会话 Phase2 后实测） |

---

## 9. 架构判断

**总体结论：Phase 2 之后的真实运行链路整体成立，达到“Tool-enabled Multi-Agent Architecture”**
**目标；对“规划→执行→合成”主链与“Agent 工具统一调用/治理层”分别判定如下。**

1. **主链（稳定）**：`User Query → PlannerAgent → PlanValidator → DAGExecutor →
   AgentRegistry → SpecialistAgent.execute → Worker.run → SynthesizerAgent` 在三个真实用户
   场景中全部 `plan_ok=True`、任务按拓扑序完成、SSE 事件契约未变。
2. **路由（正确）**：retrieve→RetrievalAgent、filter/group→OrganizerAgent、
   vision→VisionAgent，无错配。
3. **工具层（就绪但热路径外）**：ToolRegistry 提供统一工具描述 / schema 校验 / 调用计量 /
   Trace；`SpecialistAgent.call_tool` 已可用。plan 执行热路径仍由 Worker 直连注入实现，
   属 Phase 2 的**有意设计边界**（稳定性红线），非缺陷；是否在 Phase 3 让 Worker 强制经
   registry 需要先冻结 Worker 注入位口径（见 §10-2）。
4. **可观测/可评估**：ToolCallEvent（agent_id/tool/arguments/success/latency_ms/error）
   已为 Phase 4 Judge 与工具优化提供输入。
5. **诚实性红线未被破坏**：Synthesizer selection 仍限定于候选白名单（validate_selection），
   验证中 selection 与实际检索候选一致。

---

## 10. Phase 3 准备情况

**已就绪（可直接使用）**
- `AgentRegistry`（注册/查询/任务路由）+ `SpecialistAgent.execute`（Worker 分派）；
- `AgentCard`（角色/能力/工具绑定/can_use）+ `AgentMessage` 信封；
- `ToolRegistry`（注册/发现/校验/计量/ToolCallEvent）+ MCP 可映射 schema；
- `ComplexityGate / IntentRewrite / PlanValidator / DAGExecutor / SynthesizerAgent` 全套决策与执行组件。

**建议进入 Phase 3 前处理的 3 项收尾（本报告审计发现）**
1. **rerank 工具类型契约**：`candidates` schema（JSON object）与默认实现期望的
   `ImageCandidate` 对象不一致 → 收敛为 JSON 候选行 + 内部转换；
2. **“执行链强制经 registry”决策**：明确是否让 DAG 内 Worker 的注入位统一由 registry 提供
   （需要冻结 search_tool/analyzer 等注入位与 ToolDefinition 的对应关系，防止行为漂移）；
3. **ToolCallEvent 落库**：供 Supervisor（Phase 3）复盘决策与离线评估直接消费。

**Phase 3 可落点**
- Supervisor 五态小状态机（plan → execute → review → revise/replan/abort，默认 flag 关闭）；
- review 决策确定性触发器可直接消费 Executor 事件与 ToolCallEvent 统计（失败工具 / 空结果 / 延迟）；
- AgentMessage（task/report/decision）作为 Supervisor↔Specialist 通信信封。

---

## 附录 A 本次审计对仓库的改动

- 仅新增本报告文件：`PHASE2_RUNTIME_INTEGRATION_TEST_REPORT.md`
- 只读验证 harness 位于系统临时目录（`%TEMP%\smartimager_verify_phase2.py`），未入库；
- 未修改任何生产代码 / 测试 / 数据库 / 前端 / 架构。

## 附录 B 引用来源

- 源码：`backend/ai_func/recommendation/orchestration/`（planner / models / executor / workers /
  orchestrator / synthesizer / agent_cards / specialist / agent_registry / tools/）
- 报告：`PHASE1_SPECIALIST_AGENT_IMPLEMENTATION_REPORT.md`、`PHASE2_TOOL_REGISTRY_IMPLEMENTATION_REPORT.md`
- 测试：`backend/tests/agent_team/`（含 `tools/`）




