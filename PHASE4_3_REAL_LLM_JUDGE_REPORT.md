# PHASE 4.3 REPORT — Real LLM-as-a-Judge Benchmark

> 范围：只在 `backend/ai_func/recommendation/evaluation/` 增加真实外部 LLM Judge 能力；
> **未修改** Planner / Validator / Executor / Workers / Supervisor / LangGraph supervisor_graph /
> MCP / ToolRegistry / SSE / frontend / database。
> 验证：`pytest backend/tests/agent_team` → **462 passed**（含新增 `test_real_llm_judge.py` 8 项）。

---

## 1. 架构变化

新增/修改（全部位于 evaluation 或独立脚本/测试）：

| 文件 | 作用 |
|---|---|
| `evaluation/providers/openai_client.py` | **OpenAICompatibleClient**：真实外部 LLM（任意 OpenAI-compatible /v1/chat/completions）|
| `evaluation/llm_client.py` | 增加 `LLMClientError`（+） |
| `evaluation/prompts/judge_prompt.py` | Prompt 完善：System 角色“Agent System Evaluation Expert”；Context 增加 goal/plan；保留 4.2-B 兼容 key |
| `evaluation/models.py` / `judge.py` / `runner.py` | EvaluationTrace/JudgeInput 增加 goal、plan_tasks（bounded 上下文来源）|
| `evaluation/datasets/agent_benchmark_cases.json` | 30 条 Agent Benchmark Dataset |
| `evaluation/benchmark.py` | BenchmarkResult / run_agent_benchmark / load_benchmark_cases / save_benchmark_result |
| `evaluation/benchmark_harness.py` | 确定性场景 runtime（Fake 工具注入，供演示/测试/脚本）|
| `evaluation/__init__.py` | 导出新 API（+） |
| `scripts/run_agent_benchmark.py` | Benchmark 运行器（真实 LLM / Fake）|
| `backend/tests/agent_team/test_real_llm_judge.py` | 8 项测试（不依赖真实 API）|
| `evaluation/outputs/benchmark_result.json` | 结果输出（默认位置）|

### Framework validation vs Real benchmark（口径区分）

- **Framework validation（无 Key、CI）**：`--provider fake`（MockLLMClient）验证链路正确性
  （prompt → parse → aggregate → pipeline），**不产生质量结论**；
- **Real benchmark（有 Key）**：`--provider openai`，Judge 由真实外部 LLM 打分；
  本仓库的确定性场景保证 Runtime 部分可复现，质量信号来自真实 LLM。

---

## 2. 真实 LLM Judge 流程

```
EvaluationCase
   → EvaluationRunner（runtime_factory 注入确定性场景）
   → EvaluationTrace（goal/plan_tasks/decisions/tool_summary/selection 等）
   → build_judge_input → JudgeInput
   → LLMJudge
        JudgePromptBuilder（Agent System Evaluation Expert + bounded Context）
        → OpenAICompatibleClient.generate(prompt)
             → chat /chat/completions（timeout + 指数退避重试 + 错误聚合 + response logging）
        → parse_judge_response（严格 JSON）
   → JudgeResult（verdict + 5 维 0–5 + reasons）
   → aggregate_judge_results → BenchmarkResult → benchmark_result.json
```

配置只来自环境变量（Task 8，安全）：
```
set LLM_JUDGE_API_KEY=xxx
set LLM_JUDGE_BASE_URL=https://api.openai.com/v1     # 或任意 OpenAI 兼容网关
set LLM_JUDGE_MODEL=gpt-4o-mini
```
客户端能力：timeout、重试（408/429/5xx/网络错误）、LLMClientError 聚合、response logging
只记录 model/status/latency/content 长度；repr 与错误信息均不包含 API Key（有测试断言）。

---

## 3. Dataset 设计（30 条）

文件：`evaluation/datasets/agent_benchmark_cases.json`

每个 case 必含：`case_id / query / goal / expected_agents / expected_tools / tags`，
另附确定性校验字段（expected_final_state / expect_selection / expected_decisions）。

分布：

| 类别（tag） | 数量 | 覆盖内容 |
|---|---|---|
| retrieval | 5 | 建筑/夜景/水乡/园区/展馆等检索 |
| failure_recovery（empty_retry） | 2 | 空检索 → revise → retry |
| failure_recovery（tool_failure） | 3 | 工具持续失败 → revise → abort（预算控制）|
| organization | 5 | 检索 + metadata 过滤 + 按 location 分组 |
| vision | 5 | 检索 + GLM-Vision 式视觉核验 |
| multi_step | 6 | 检索 + 整理 + 视觉核验 + 推荐（多 Agent 协作）|
| complex_planning | 4 | 多任务 DAG（retrieve/filter/group/analyze/synthesize）|

Harness（benchmark_harness.py）按 tags 决定场景 profile，注入确定性 Fake search/analyze，
确保 30 条全部可离线复现（Framework validation 全绿）。

---

## 4. Rubric（0–5，prompt 内联）

planning_quality / agent_routing / tool_usage / recovery_behavior / final_result_quality；
每个维度带 description + evaluation_focus，Judge 只对 bounded 摘要打分，
缺失信息视为未知不据此扣分，非法输出收敛为 verdict=flag。


## 5. Benchmark 结果

### Framework validation（Fake LLM，CI 可复现，本机实测）
```
provider=fake  total_cases=30  elapsed≈1100 ms
scores:    planning_quality=5.0 agent_routing=5.0 tool_usage=5.0
           recovery_behavior=5.0 final_result_quality=5.0
overall:   5.0
verdict:   pass=30 flag=0 fail=0
deterministic: total=30 passed=30 failed=0   ← Framework validation（结构断言）
```
结果文件：`backend/ai_func/recommendation/evaluation/outputs/benchmark_result.json`
（含 total_cases / scores / criterion_average / overall / verdict / pass_rate / fail_rate /
deterministic / per_case）。

### Real benchmark（待提供 Key 后生成真实分数）
命令（真实 LLM Judge，Runtime 场景仍为确定性 Fake）：
```
set LLM_JUDGE_API_KEY=xxx
set LLM_JUDGE_BASE_URL=https://api.openai.com/v1
set LLM_JUDGE_MODEL=gpt-4o-mini
python scripts/run_agent_benchmark.py --provider openai
```
输出覆盖的 5 个维度分数、overall、verdict 分布、deterministic 摘要与 per_case 轨迹；
真实模型温度建议 0；本报告 §5 上述数字仅为 Fake（链路）数字，**不代表真实质量结论**。

---

## 6. 失败案例分析（benchmark 场景视角）

per_case 记录中典型场景的行为与 Judge 关注点：

| 场景 | trace 特征（本仓库真实产生） | Judge 预期关注 |
|---|---|---|
| empty_retry | decisions=[revise, continue]，selection 2 条 | recovery_behavior：revise 是否匹配 empty 信号 |
| tool_failure | ToolCallEvent success=False ×2；decisions=[revise, abort]；selection=[] | tool_usage：失败留痕；recovery：达到预算受控 abort 是否可接受 |
| 常规检索 | continue + selection | planning/agent_routing/tool_usage 基线 |
| vision/multi | analyze_image 逐候选调用，vision 证据合并进候选 | agent_routing：vision 是否正确参与 |
| complex DAG | retrieve→filter→group→analyze→synth | planning_quality：任务顺序与依赖合理性 |

（真实分数待 LLM 输出；上表用于解释“Judge 该看什么”，不是质量结论。）

## 7. Limitations

1. **Framework vs Real 区分**：当前产出的 benchmark_result.json 由 Fake/Mock LLM 生成，
   是链路验证而非质量结论；真实分数需设置 Key 后运行 `--provider openai`。
2. **Runtime 场景仍为确定性 Fake**：Real benchmark 的“真实”在 Judge 层；检索/视觉/网络工具
   未接入真实 DB/MCP/模型（避免测试/演示依赖外部服务）。真实端到端质量评估需接入真工具链。
3. **LLM Judge 是近似信号**：verdict/分数依赖模型与 prompt；项目用确定性断言兜底结构、
   MockJudge 提供确定性基线、非法输出收敛 flag；仍建议真实分数做人工抽样核对。
4. **成本/延迟**：30 case × 1 prompt/case；长 Context 会拉高 token。builder 已做 bounded 摘要。
5. **API Key 安全**：仅环境变量（脚本默认不落盘）；CI/生产请用 secret 托管，禁止提交。

---

## 8. 交付对照与确认

| 要求 | 状态 |
|---|---|
| ✓ Real LLM Client（providers/openai_client.py） | ✅ |
| ✓ OpenAI Compatible（任意 /v1/chat/completions） | ✅ |
| ✓ API Key 环境变量（LLM_JUDGE_*） | ✅（+缺 Key 引导 code=2 测试） |
| ✓ Agent Benchmark Dataset（30 条） | ✅ |
| ✓ LLM Judge（LLMJudge + 真实 client 注入，MockLLMClient 保留） | ✅ |
| ✓ Rubric Score（5 维 0–5 聚合） | ✅ |
| ✓ Benchmark Report（本文件） | ✅ |
| ✓ 不修改 Runtime | ✅（diff 仅在 evaluation/ + scripts/ + tests/） |
| 测试 | `test_real_llm_judge.py` 8 passed；agent_team 全量 **462 passed** |

运行方法：`python scripts/run_agent_benchmark.py --provider fake`（无 Key）或
`--provider openai`（设 LLM_JUDGE_* 后）。结果位置：
`backend/ai_func/recommendation/evaluation/outputs/benchmark_result.json`。
