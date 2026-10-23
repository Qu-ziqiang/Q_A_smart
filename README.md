<div align="center">

<img src="assets/logo/logo.png" alt="SmartImager" width="160"/>

# SmartImager

**面向复杂意图的多模态图像检索 Agent 系统**

自然语言一次说清「找图 → 筛选 → 整理 → 看图确认」——由受控多 Agent 运行时完成规划、校验、执行与有界恢复。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)

</div>

---

## 为什么是 SmartImager

传统图搜是「一句话 → Top-K」。真实使用往往更复杂：先按语义或类别召回，再按元数据收紧，再分组整理，必要时对候选做画面核验。把这些写死成流水线会僵硬；让大模型自由 ReAct 调工具又容易越界、难测、难恢复。

SmartImager 选择第三条路：**规划与执行分离的受控 Agent Runtime**——LLM 负责一次产出可校验的任务计划，规则负责校验、执行与恢复。简单请求走轻量检索；复杂请求进入多角色协作。

## 核心能力

| 能力 | 说明 |
|------|------|
| **复杂度门禁** | 规则分流：简单查询不规划，复杂组合进入 Team 路径 |
| **一次规划** | 强制 Tool Calling 提交 Typed Task Plan，温度 0，封闭任务类型 |
| **确定性校验** | PlanValidator 校验依赖、环、生产者、视觉门禁；失败整计划拒绝 |
| **三角色执行** | Retrieval / Organizer / Vision 按 `task_type` 路由，权限隔离 |
| **有界恢复** | 规则 Supervisor：空召回硬插查询扩写、失败修订、有限重规划，最多两轮 |
| **多模态检索** | CLIP 三路向量（标题 / 描述 / 图像）+ PostgreSQL · pgvector · HNSW |
| **精确类别** | 数据集域 + `image_labels`（如 VOC 词表），多类默认 AND |
| **按需视觉** | 看图重排与窄视觉任务（读字 / 主体 / 空间等）按门禁启用，控制成本 |
| **双轨评测** | 结构断言 + LLM-as-a-Judge（规划 / 路由 / 工具 / 恢复 / 终态） |

## 架构一览

```text
用户自然语言
        │
        ▼
 ComplexityGate ──► LEGACY：轻量 CLIP 检索（不规划）
        │
        └─────────► TEAM
                      │
              ┌───────┴───────┐
              │  Planner LLM  │  submit_task_plan（一次）
              └───────┬───────┘
                      ▼
              PlanValidator（规则）
                      │
                      ▼
              DAG Executor（拓扑串行）
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   Retrieval     Organizer      Vision
   改写/扩写       过滤/分组      描述/核验
   域/类别/文搜     去重/排序      读字/主体/空间
   相似图/重排      精选
         └────────────┼────────────┘
                      ▼
                 Synthesize
                      │
                      ▼
            Supervisor REVIEW（规则）
         注入 expand │ revise │ replan │ abort
```

**设计原则**

- LLM 只出现在规划、合成与离线评测；控制面可测、可穷举。
- 任务类型封闭；新增能力必须同步枚举、输入契约、Worker 与 AgentCard。
- 空/差召回由 Supervisor **硬插入** `expand_query`，不把「要不要扩写」交给 Planner 赌运气。

## 技术栈

| 层 | 选型 |
|----|------|
| API / 流式 | FastAPI · SSE |
| 前端 | React 18 |
| 规划 / 合成 | DeepSeek（Tool Calling） |
| 向量 | CLIP · PostgreSQL · pgvector（HNSW / cosine） |
| 图库视觉 | GLM-4V（跟随 `vision_available()`） |
| 看图重排 | DeepSeek 多模态（失败回退公式） |
| 契约 | Pydantic Typed Plan / Input |
| 可选 | LangGraph（仅 Supervisor 状态机）· MCP（协议验证，非默认链路） |

## 仓库结构

```text
Smartlmager-sqlite-vec/
├── backend/                 # FastAPI · 检索 · Agent Runtime · 评测
│   ├── ai_func/recommendation/orchestration/   # Gate / Planner / Validator / Supervisor
│   ├── db_func/             # PostgreSQL · pgvector · datasets · labels
│   └── tests/agent_team/    # 控制面与集成测试
├── frontend/                # React 管理与对话推荐界面
├── scripts/                 # 环境初始化 · 数据迁移 · 评测入口
├── docs/                    # API 文档
├── docker-compose.yml       # pgvector 一键依赖
├── main.py / start.py       # 启动入口
└── requirements.txt
```

> 目录名保留历史 `sqlite-vec` 字样；**运行时向量库为 PostgreSQL + pgvector**。

## 快速开始

### 1. 数据库

```bash
docker compose up -d
```

默认拉起带 pgvector 的 PostgreSQL。连接信息写入本地配置（**勿提交** `.env` / `config.yaml`）。

### 2. 后端

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate
pip install -r requirements.txt
# 按需安装 PyTorch（CUDA 版本见 requirements 注释）

python start.py
# 或: python main.py
```

### 3. 前端

```bash
cd frontend
npm install
npm run dev
```

### 4. 配置要点

- CLIP 模型首次启动会拉取权重（体积较大，已 `.gitignore` `models/`）。
- 规划 / 视觉 / 重排需配置对应 API Key（仅本地环境文件）。
- 图库数据与向量需自行入库；公开 VOC 等数据集可参考 `scripts/` 下 ingest 脚本。

## 评测口径（摘要）

离线评测与演示库指标，**不是线上 DAU**：

| 指标 | 结果 | 说明 |
|------|------|------|
| 万级三路检索 | p95 ≈ **122ms** | CLIP + HNSW |
| 看图重排 | Hit@1 **76.8% → 89.9%** | 400 张子集 · 约 1.2s/张 · 按需 |
| 规划延迟 | p50 ≈ **1.9s** / p95 ≈ **2.8s** | 仅规划段，非端到端 |
| 结构断言 | **200 / 200** | Fake Runtime · 控制面契约 |
| Judge 路由 | 短板 **3.83**（200）· A/B **4.05 → 4.20**（240） | 两套题，勿合并口述 |
| 计划膨胀 | 均 Task **4.85 → 3.26** · plan_ok **100%** | 分轨政策 + plan_floor |

## 安全与隐私

本仓库**不包含**：

- `.env` / API Key
- 本地数据库文件与 `data/` 图库
- 大模型权重目录 `models/`
- 含密钥的运行时 `config.yaml`

请使用 `config.example.yaml` 与环境变量自行配置。

## License

[Apache License 2.0](LICENSE)

---

<div align="center">

**SmartImager** — Controlled multimodal agents for image discovery.

</div>
