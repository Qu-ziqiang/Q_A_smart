# SmartImager Agent Benchmark Report

> Phase 4.3-B：200-Case Large Scale Agent Benchmark（真实 LLM-as-a-Judge）。
> 运行日期：2026-09 ｜ Judge：OpenAI-compatible（DeepSeek）`deepseek-chat`，
> base_url=`https://api.deepseek.com`，temperature=0。
> 约束：只运行 Evaluation Pipeline + dataset/script/report；**未修改任何 Runtime 代码**
> （Planner/Validator/Executor/Workers/Supervisor/LangGraph/MCP/ToolRegistry/SSE/前端/数据库）。

---

## 1. Benchmark Overview

为什么需要 LLM-as-a-Judge：

1. 200 个场景逐个“人工读 trace 打分”不可扩展、不可复现；
2. 确定性断言只能判断“结构对不对”（任务是否成功、工具是否调用、是否 abort），
   无法判断“做得好不好”（计划是否该更长/更专、路由分工是否完整、结果是否真的匹配意图）；
3. Judge 把一次执行的结构化摘要（EvaluationTrace → JudgeInput）映射为
   pass/flag/fail + 5 维 0–5 分，产出可跨模型复跑、可回归的质量信号。

本次运行还特意保留**双轨**：deterministic（结构）= 200/200 全过；
LLM Judge（质量）= overall 4.228/5，严格 pass 76，flag 124，fail 0——
**这证明了“结构正确 ≠ 质量满分”的评估必要性。**

---

## 2. Evaluation Architecture

```
EvaluationCase（200 条：query/goal/expected_agents/expected_tools/tags）
   │
   ▼ Runtime Execute（确定性场景 harness：SupervisorAgent + LangGraph + Fake 工具）
   │   planner → validator → DAGExecutor → Specialist Agents → ToolRegistry → selection
   ▼ EvaluationTrace（goal/plan_tasks/execution_status/decisions/tool_summary/final_state）
   │
   ▼ JudgeInput（bounded Context：query/goal/plan/agents/tools/execution/recovery/final）
   │
   ▼ LLMJudge（deepseek-chat，temperature=0）
   │   parse_judge_response（严格 JSON；异常收敛 flag）
   ▼ JudgeResult（verdict + 5 维 score + reasons）
   │
   ▼ 聚合 → benchmark_results.json / failure_analysis.json / benchmark_summary.md
```

Judge Prompt：System 角色 = Agent System Evaluation Expert；Context 为裁剪摘要
（不 dump 全量 trace / tool_events / agent_messages）。

---

## 3. Dataset（200 Cases 分布）

`evaluation/datasets/smartimager_agent_benchmark_200.json`

| Category | Count |
|---|---|
| Retrieval（Simple Retrieval，A） | 40 |
| Organization（Retrieval + Organization，B） | 40 |
| Vision（Vision Analysis，C） | 30 |
| Multi-Agent（Multi-Agent Planning / DAG，D） | 40 |
| Recovery（Failure Recovery，E） | 30 |
| Complex（Complex User Intent，F） | 20 |
| **Total** | **200** |

- 每个 case：`case_id/query/goal/expected_agents/expected_tools/tags`；
- Recovery 类覆盖 empty retrieval、tool failure、MCP failure/retry（15 条空检索恢复 +
  15 条工具/MCP 失败中止）；
- Runtime 场景为确定性 Fake（不触碰真实 DB/视觉/MCP 付费链路），保证可复现；
  **不填充虚假“执行结果”**——所有 decisions/final_state/selection 均来自真实执行 trace。


## 4. Rubric（5 维 0–5）

| 维度 | 关注 |
|---|---|
| Planning Quality | 计划是否符合用户目标、任务顺序/粒度是否合理、是否该补充分工任务 |
| Agent Routing | 检索/整理/视觉 Specialist 是否按需被调用，分工是否完整 |
| Tool Usage | 工具调用成功/失败、异常调用、覆盖是否充分 |
| Recovery Behavior | empty retrieval / tool failure / MCP failure 后的 revise/replan/abort 是否合理 |
| Final Result Quality | final_state 与最终 selection 是否满足用户意图 |

---

## 5. Benchmark Result（真实运行，DeepSeek judge）

文件：`backend/ai_func/recommendation/evaluation/results/benchmark_results.json`

| 指标 | 值 |
|---|---|
| total_cases | 200 |
| overall score（5 维均值） | **4.228 / 5.0** |
| verdict | pass=76（38%）｜ flag=124｜ fail=0 |
| success_cases / failed_cases | 76 / 0（strict pass 口径；124 flag 为“可改进”而非失败） |
| deterministic（结构校验） | passed=200 / 200 |
| elapsed | ≈434 s（200 次真实 LLM 调用 + runtime） |

五维平均分：

| dimension | score |
|---|---|
| planning_quality | 4.38 |
| agent_routing | 3.83 |
| tool_usage | 4.66 |
| recovery_behavior | 4.68 |
| final_result_quality | 3.60 |

分类表现：

| category | count | avg score |
|---|---|---|
| retrieval | 40 | 4.02 |
| organization | 40 | **3.62** |
| vision | 30 | 4.99 |
| multi_agent | 40 | 4.36 |
| recovery | 30 | 4.11 |
| complex | 20 | 4.63 |

### 观察

- **vision 类 ≈ 满分**：计划含“检索+视觉核验”，Judge 认为路由分工完整；
- **organization 最低（3.62）**：reason 显示 Judge 认为“只调用 retrieval/organizer、无视觉
  Specialist”，或“工具只用了 search_images 一种”；
- **flag 不表示崩溃**：200 条中 0 个 fail、0 条 abort 之外的异常；124 flag 主要是
  “规划可以更充分/结果还可更匹配”的质量扣分。

---

## 6. Failure Analysis（最低 20 条）

文件：`backend/ai_func/recommendation/evaluation/results/failure_analysis.json`

共同问题（低分 case 的 judge_reason 聚类）：

1. **风格/主题匹配缺少“验证型任务”**（检索类低分主因）：
   e.g. “找高级灰调的高山湖泊”只有 retrieve+synthesize，Judge 认为缺视觉/整理步骤来
   验证“高级灰调”，planning 与 routing 被扣分。
2. **routing 单一**：多条例只出现 retrieval agent；Judge 期望 visual specialist 或 organizer
   参与以“证明内容符合 query”。
3. **工具覆盖单一**：检索类只有 1 次 `search_images`（成功无失败），但 Judge 认为
   “没有体现多轮/细化或过滤工具”。
4. **organization 无视觉**：B 类部分 query 含“照片/归档”，Judge 认为元数据分组虽对，
   但缺少图像内容理解依据（agent_routing 3.83 的主因之一）。

> 归因注意：这些低分案例的 runtime 是**确定性 Fake 场景**（无真实图片可“看”），
> Judge 依据的是执行 trace 里的分工与工具证据——这是 Judge 对“覆盖度”的合理质疑，
> 但不等同于真实图片检索质量差。


## 7. Engineering Insights

- **Planner/多任务**：complex(4.63) 与 multi_agent(4.36) 明显高于单一 retrieval(4.02)，
  说明“结构化 DAG 计划 + 多 Specialist”在该 Judge 口径下被认可；对“带风格/用途约束”的
  请求，计划里补一个视觉核验/整理任务，Judge 平均分显著上升。
- **Supervisor 恢复能力**：recovery_behavior 4.68 是最高维之一；empty→revise→continue、
  tool/MCP failure→revise→abort 的决策序列被 Judge 认可（受控、有预算、有留痕）。
- **ToolRegistry 稳定性**：tool_usage 4.66；200 条 trace 中 0 次未捕获异常、结构全过。
- **LangGraph 状态管理**：200 次 request 级 checkpoint/回环均正常结束（无超迭代、无死循环）；
  recovery 分类 4.11 且 15 条工具失败全部在预算内 abort。
- **确定性 Fake 场景的局限直接影响分数**：vision=4.99 是“分工完整 + 无风险”的组合；
  organization/retrieval 低分主要是“分工证据不足/工具单一”，而非真实检索失败。

---

## 8. Limitations（诚实声明）

1. **Judge 本身存在偏差**：deepseek-chat 对“是否该上 vision”的判断偏乐观/偏覆盖导向，
   导致同一目标在带视觉任务时得分更高；flag(124) 主要反映“覆盖度不够”而非真实失败。
2. **Runtime 场景为确定性 Fake**：本次 Judge 是真实 LLM，但 runtime 未接真实 DB / 真实视觉 /
   真实 MCP 服务；不能外推为“线上真实图片质量”结论，只能评估“Agent 行为架构”。
3. **Dataset 覆盖有限**：200 条由模板组合生成（场景多样但模式集中），未覆盖真实用户长尾、
   图片内容差异与多语言。
4. **Vision 依赖**：C 类/多 Agent 类的高分建立在“任务里写了 vision”之上；若真实环境
   未配置视觉模型（当前 VISION_MODEL=null），这些计划不可执行——报告分数 ≠ 可部署配置。
5. **评分口径**：success_cases 采用 strict pass；flag 语义是“基本可用但可改进”，
   不应被读成失败。整体 4.228 属于“结构非常稳 + 质量有提升空间”。
6. **成本/时间**：200 次真实调用约 7 分钟、token 成本取决于模型；建议后续复用 Mock/缓存
   做回归对比，只对 diff 跑真实 Judge。

---

## 产物

- `backend/ai_func/recommendation/evaluation/results/benchmark_results.json`
- `backend/ai_func/recommendation/evaluation/results/failure_analysis.json`
- `backend/ai_func/recommendation/evaluation/results/benchmark_summary.md`
- `backend/ai_func/recommendation/evaluation/datasets/smartimager_agent_benchmark_200.json`
