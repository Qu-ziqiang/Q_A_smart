# AGENT_MEMORY_IMPLEMENTATION_REPORT

> 项目：SmartImager（多模态智能图库与对话推荐系统）
> 阶段：Lightweight Agent Memory（依据 `AGENT_MEMORY_AUDIT_AND_DESIGN.md`）
> 日期：2026-09-01
> 基线：Phase 1-A/B/C + Phase 2 Stage 0–9 + Planner Capability Routing Fix 均已完成
> **2026-09-10 更新**：`agent_episodes` 已随数据层迁移至 **PostgreSQL**（不再是 SQLite 表）；
> 当前门禁为 `pytest backend/tests/agent_team -q` → **463 passed**（本文所记 275/275 为当时口径）。

---

## 1. Implementation Summary

在现有 Agent Team 之上新增 **轻量、bounded、deterministic 的 Agent Memory**：

```text
Conversation History  = conversation_messages（raw chat log，未改造）
ExecutionContext      = Working Memory（直接复用，未新建表）
agent_episodes        = cross-request Episodic Memory（新增持久化表；2026-09-10 已由 SQLite 迁入 PostgreSQL）
```

- 新增 1 张表、1 个 repository、1 个 memory service（确定性摘要，无新 LLM）。
- `conversation_id` 已从 Router 一路线程到 `stream_core → AdaptiveTeamOrchestrator → Memory Service → Planner`。
- 写入门控：**仅成功完成的 Team run** 写入；失败/partial/planner-failed 不写成功 episode。
- Recall：Top-N ≤ 3，渲染总字符 ≤ 800；`current user query > current execution context > memory`。

## 2. Before / After Architecture

### Before
```text
Router(stream)
  → stream_core(messages, state=None)          ← conversation_id 只在 router
      → ComplexityGate.decide
      → AdaptiveTeamOrchestrator.run_stream(messages)
           → Planner.create_plan(goal, recent_messages)
           → Validator → DAGExecutor → ExecutionContext
           → Synthesizer
```

### After
```text
Router(stream, conversation_id=conv_id)
  → stream_core(messages, conversation_id=conv_id)
      → ComplexityGate.decide
      → AdaptiveTeamOrchestrator.run_stream(messages, conversation_id, memory_service)
           ├─ 0) Memory recall（bounded hints）
           ├─ 1) Planner.create_plan(goal, recent_messages, memory_hints)
           ├─ 2) Validator → 3) DAGExecutor → ExecutionContext
           └─ 4) 成功完成 → Memory write（build → should_write → insert）
Router DELETE /conversations/{id} → cascade delete agent_episodes
```

## 3. Existing State Reuse

| 组件 | 角色 | 处理 |
|---|---|---|
| `conversation_messages` | Raw chat log | 保留，未改造（Boundary #1） |
| `request_sessions.state` | 审计占位 | 未使用作 episodic store（Boundary #2） |
| `ExecutionContext` | **Working Memory** | 直接复用；仅 orchestrator 在成功路径读取其 plan/results 提取 episode |
| `TaskPlan` / `WorkerResult` | context 内容 | 只读引用，不落库 |
| `AdaptiveTeamOrchestrator` | 编排 | 最小扩展（conversation_id + memory_service 注入 + recall/write 钩子） |
| `PlannerAgent` | 规划 | 最小输入边界扩展（memory_hints=None 时行为逐字符不变） |
| `Legacy RecommendAgent` | Legacy 路径 | **不修改**（Legacy 不 recall，默认不写） |

## 4. Database Schema

新增 `agent_episodes`（additive DDL，`backend/db_func/core/database.py` 的 init_db 创建；
DDL 单一来源常量位于 `backend/db_func/repositories/agent_episodes.py`）：

```sql
CREATE TABLE IF NOT EXISTS agent_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,     -- 主 scope
    request_id TEXT,
    user_id TEXT,                      -- 仅当客户端传入；不参与 recall
    route TEXT NOT NULL,               -- 'team'
    task_types TEXT NOT NULL,          -- JSON array
    query_summary TEXT NOT NULL,       -- ≤120 chars
    strategy_summary TEXT NOT NULL,    -- ≤200 chars
    result_summary TEXT,               -- ≤200 chars
    preference_hints TEXT,             -- JSON array of short tags
    selected_count INTEGER DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1,
    plan_valid INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL           -- created_at + 30 days
);
-- idx_agent_episodes_conv_created(conversation_id, created_at DESC)
-- idx_agent_episodes_expires(expires_at)
```

**不存**：raw_prompt / raw_response / chain_of_thought / filepath / base64 / API key / 完整 VisionEvidence / 完整 candidate JSON。

## 5. Repository

`backend/db_func/repositories/agent_episodes.py` — `AgentEpisodeRepository(BaseRepository)`：

| 方法 | 行为 |
|---|---|
| `insert(episode, ttl_days=30)` | 参数化 INSERT；自动截断摘要字段；自动计算 expires_at |
| `list_by_conversation(conv, limit=20, now)` | 只返回未过期，created_at DESC，解析 JSON 列 |
| `recall(conv, limit=3, now)` | repo 层 Top-N（未过期） |
| `purge_expired(now)` | 删除全部过期行 |
| `delete_by_conversation(conv)` | DELETE conversation 级联 |
| `prune_conversation(conv, cap=20)` | per-conversation 上限 |
| `count(conv)` | 测试/审计辅助 |

全部参数化 SQL，无字符串拼接；conversation_id 为主 scope。

## 6. Memory Service

`backend/ai_func/recommendation/orchestration/memory.py` — `EpisodicMemoryService`：

- `build_episode_from_context(context, *, route, conversation_id, ...)`：从 ExecutionContext 确定性提取 episode；非成功完成返回 None。
- `should_write_episode(episode, recent_episodes=None, now=None)`：写门控（TEAM only + success + 24h 去重）。
- `recall_episodes(conversation_id, query, top_n=3, budget=800, now=None)`：conversation-scoped recall。
- `render_memory_hints(episodes)`：渲染 bounded hints（单条 ≤ 280 字符）。
- `remember(...)`：完整写链 build → should_write → insert + cap prune。

边界：`Planner → Memory Service → Episode Repository → PostgreSQL`（Planner 不直接碰数据库）。

**Episode 摘要生成（全部 deterministic，无 LLM）**：

```text
query_summary    ← context.goal（截断 ≤120，sanitize）
strategy_summary ← task_type 序列 "a→b→c" + Vision "3/4 成功" 计数
result_summary   ← selected_count / synthesis_degraded / failures codes
preference_hints ← 结构化 tags：uses_metadata_filter / uses_vision_verification /
                   uses_group_by=<field> / prefers_grouped（仅从 plan input 推断）
```

## 7. Write Policy

```text
Orchestrator successful completion
        ↓
build_episode_from_context     ← 仅成功完成才返回 episode
        ↓
should_write_episode           ← TEAM only + success + 24h 无重复
        ↓
AgentEpisodeRepository.insert  ← + per-conversation cap prune
```

成功定义（§8，严格）：`Gate→TEAM + Planner success + Validator success + DAG 到达 synthesis + synthesis/selection 路径成功完成（synthesis SUCCEEDED 且非空 selection）`，且**无任何 FAILED worker**。

**不写入**：plan.failed / validation 失败 / worker fatal / synthesizer fatal / 空 selection / 无 conversation_id / Legacy 路径（TEAM only）。**写时机在 orchestrator 成功路径**，不在 task.started / task.completed / Planner 阶段。

## 8. Recall Policy

```text
Current Query (+ conversation_id)
    ↓
EpisodicMemoryService.recall_episodes(conv, query, top_n=3, budget=800)
    ↓
Bounded hints（单条 ≤ 280 chars，总 ≤ 800 chars）
    ↓
Planner.create_plan(goal, recent_messages, memory_hints)
```

排序：`keyword_overlap(query, query_summary+strategy_summary)` + recency（≤7d +3 / ≤30d +1）+ success(+1) + route team(+0.5)。确定性，无 embedding/vector。

硬限制：**Top-N ≤ 3**；**总渲染 ≤ 800 chars**；每 hint ≤ 280 chars；过期自动排除。

## 9. TTL / Dedup

- **TTL = 30 天**（`expires_at = created_at + 30d`）；recall/查询自动排除 `expires_at <= now`；`purge_expired()` 惰性清理。
- **per-conversation cap = 20**：写入后 `prune_conversation` 保留最新 20 条。
- **重复抑制**：相同 conversation + 相同 task_types 序列 + query_summary Dice 2-gram 相似度 ≥ 0.85 + 24h 时间窗 → 跳过写入。

## 10. Planner Integration

`planner.py` 最小输入边界扩展：

```python
def create_plan(self, goal, recent_messages=None, memory_hints=None, capabilities=None):
    ...
    user_prompt = self._build_user_prompt(goal, recent_messages, memory_hints)
```

- `memory_hints=None` → user prompt **逐字符等同旧版**（回归测试 `test_memory_hints_none_prompt_identical_to_legacy` 证明），capability routing 完全不受影响。
- Memory block 仅在存在有效 hints 时出现，独立段落，`[:3]` 条且每条 `[:280]`。
- Prompt 内嵌优先级规则："当前用户请求优先级最高，若与历史记忆冲突一律以当前目标为准，历史记忆不得覆盖当前请求的 TaskType 选择"。
- 不增加第二次 LLM 调用；不改变 system prompt / tool schema。

## 11. Conversation ID Threading

```text
Router（stream 端点持有 conv_id）
    → recommendation_service.stream_core(messages, conversation_id=conv_id)   ← 新增参数
        → AdaptiveTeamOrchestrator.run_stream(messages, conversation_id, memory_service)
            → Memory recall（Planner 前）
            → Memory write（成功完成后）
```

- 无 conversation_id 时 `memory_service=None`，行为与现网完全一致（回归已验证）。
- Legacy 路径不 recall、默认不写（TEAM only）；Legacy `RecommendAgent` 未修改。

## 12. Delete Cascade

`DELETE /conversations/{conversation_id}` 现在除删除 `conversation_messages` 外，级联调用 `AgentEpisodeRepository.delete_by_conversation(conversation_id)`，防止遗留孤立 Memory。无 ORM migration framework。

## 13. Security Boundary

- 禁止持久化/渲染模式（`_sanitize` 整体替换为 `[REDACTED]`）：`sk-` / `api_key` / `authorization` / `bearer ` / `base64` / `data:image` / `filepath` / `chain_of_thought` / `raw_response` / `raw prompt`。
- Vision：只存 `"Vision 3/4 成功"` 计数；不存 title/description/tags/evidence/confidence/public_url。
- 不存 raw provider response / 完整 candidate JSON / 完整 WorkerResult。
- `user_id` 仅当客户端传入时冗余保存，不伪造 scope；recall 仅按 conversation_id。

## 14. Tests

### 单元 `test_memory.py`（23 项，全部 PASS）

Schema（insert/list、缺 conversation_id 拒绝）· 写策略（成功写 / planner 失败不写 / synth 失败不写 / worker fatal 不写 / 空 selection 不写 / 无 conv 不写 / legacy 不写 / 重复抑制 / 不同 query 允许）· Recall（Top-N≤3 / budget≤800 / 过期排除 / 相关排前 / 会话隔离 / 新→旧排序）· TTL（30 天 / 过期不 recall / purge）· 安全（无禁止字段 / vision 仅计数）· routing 回归（memory_hints=None prompt 等同旧版）。

### 集成 `test_memory_integration.py`（6 项，全部 PASS）

Case1 第一轮 Team 成功 → 1 episode；Case2 第二轮 recall → hints 进入 Planner prompt；Case3 冲突（历史分组 vs 当前不分组）→ prompt 含 precedence 且 plan 无 group_images；Case4 conversation 隔离；Case5 delete 级联；附加 stream_core 线程 conversation_id（含无 conversation_id 时不启用）。

## 15. Regression

| 项目 | 结果 |
|---|---|
| 既有 246/246（Phase1+Phase2+routing） | ✅ 通过（全量 275 = 246 + 29 新增） |
| memory 单元测试 | ✅ 23/23 |
| memory 集成测试 | ✅ 6/6 |
| Planner routing（memory_hints=None） | ✅ 不变——`test_memory_hints_none_prompt_identical_to_legacy` 证明 prompt 逐字符等同旧版；`test_routing_benchmark`（8）+ `test_planner_routing_policy`（16）全部 PASS |
| 完整 56-run 真实 benchmark 重跑 | **未重跑**（§26 允许：使用已冻结 `planner_routing_benchmark_results.json` + targeted regression tests；如实记录，不伪造"Memory benchmark"） |

## 16. Multi-turn Demo

`backend/tests/agent_team/demo_memory_e2e.py`（**Fake Planner + 真实 Memory 路径**；未调用真实 LLM，已在输出中明确标注）：

```text
Conversation A
Turn 1: 找旅行照片并按地点整理
    → Team 成功 → episode 写入（query=找旅行照片并按地点整理；strategy=retrieve→group→synthesize；preference=uses_group_by=location）
Turn 2: 再帮我找类似的旅行照片，并优先做视觉核验
    → Memory recall → Top-1/2 hint（渲染 246 chars ≤ 800）
    → Planner 收到 bounded hint（prompt 含历史摘要 + "以当前目标为准"）
    → Planner 仍按当前 query 选择 analyze_visual（retrieve→analyze_visual→synthesize；Vision 3/3 成功）
    → 第二轮 episode 写入（strategy 含 vision 计数）
[Demo PASS]
```

## 17. Known Limitations

- **写入门控严格**：任何 FAILED worker（含非关键路径）即不写成功 episode——保守，可能漏记"部分成功但可复用策略"的 run。
- **相关性排序是 keyword/recency**：中文查询中近义表达（如"分类"vs"分组"）可能排序不理想；无 semantic recall。
- **Legacy 路径不写入**：仅 TEAM 成功 run 产生 memory；simple retrieval 历史不会沉淀策略。
- **无 user 级 scope**：仅 conversation 级（项目无 auth/user 表，设计如此，不伪造）。
- **30 天 TTL 固定**：无按重要度衰减/永久记忆。
- **新生产代码规模**：`agent_episodes.py` 145 行 + `memory.py` 347 行 = 492 行（含 docstring/空行），超出"尽量 <400"软目标约 23%——已检查无过度设计（每个函数对应设计文档硬性需求；未引入 Memory Agent/vector/新 LLM），如实记录。

## 18. Non-goals

```text
Working Memory = ExecutionContext          （无新 WorkingMemory 表）
Episodic Memory = agent_episodes           （conversation-scoped deterministic summary）
Memory Agent            = NOT IMPLEMENTED
Vector Memory           = NOT IMPLEMENTED
Embedding Memory        = NOT IMPLEMENTED
New LLM / API key       = NOT USED
MCP / A2A / Reflection  = NOT IMPLEMENTED
Checkpoint / Resume     = NOT IMPLEMENTED
Worker/Synthesizer 直接读 Memory = NOT IMPLEMENTED
request_sessions.state / conversation_messages 改造 = NOT DONE
```

## 19. Acceptance Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | ExecutionContext 继续作为 Working Memory | ✅ |
| 2 | 没有 WorkingMemory 新表 | ✅ |
| 3 | agent_episodes 持久化表（现为 PostgreSQL） | ✅ |
| 4 | repository | ✅ |
| 5 | deterministic memory service（无 LLM） | ✅ |
| 6 | Team successful run 才允许写 episode | ✅ |
| 7 | failed / partial / planner-failed 不写成功 episode | ✅ |
| 8 | duplicate suppression（24h + task_types + 相似度≥0.85） | ✅ |
| 9 | TTL = 30 days | ✅ |
| 10 | expired 不 recall | ✅ |
| 11 | per-conversation cap ≤ 20 | ✅ |
| 12 | recall Top-N ≤ 3 | ✅ |
| 13 | rendered memory ≤ 800 chars | ✅ |
| 14 | current query > memory（prompt precedence + 测试） | ✅ |
| 15 | Memory 不强制 task type（Case3 验证） | ✅ |
| 16 | Memory 不覆盖 Planner capability routing（memory_hints=None 回归） | ✅ |
| 17 | conversation_id isolation（Case4） | ✅ |
| 18 | delete conversation 删除 episodes（级联 + Case5） | ✅ |
| 19 | Worker 不直接读取 Memory | ✅ |
| 20 | Synthesizer 不直接读取 Memory | ✅ |
| 21 | Planner 通过 bounded hints 接收 Memory | ✅ |
| 22 | no raw prompt / raw response / CoT / filepath / base64 / API key / 完整 VisionEvidence | ✅（sanitize + 测试） |
| 23 | no new LLM / API key / vector DB / embedding memory | ✅ |
| 24 | memory unit tests PASS | ✅ 23/23 |
| 25 | memory integration tests PASS | ✅ 6/6 |
| 26 | 246/246 回归 PASS | ✅ 275/275（246+29） |

---

*本实施未修改 gate.py / executor.py / workers.py / vision.py / synthesizer.py / Validator / SSE contract / 前端 / 数据库（除 additive agent_episodes DDL）/ 配置（VISION_MODEL 仍为 "null"）。*


