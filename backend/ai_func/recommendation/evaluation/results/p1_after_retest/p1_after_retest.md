# T1 P1 After-only 复测（无 Judge）

对照：`task_catalog_ab_deepseek/p1_plan_bloat.json`  
新结果：本目录（不覆盖旧 A/B）。

## 改动

- 只重跑 After 臂：DeepSeek Planner + Fake 工具，不打 Judge。
- Prompt 收紧已在上轮落地；本轮未开 `SM_PLAN_LINT`（均 Task 已 <4.0）。
- 软 lint 代码已就位（flag 默认关，可回退）。

## 数字

| | before | after |
|---|---:|---:|
| 简单子集均 Task | 4.85 | **2.967** |
| 纯 retrieve→synth | 8.5% | **38.6%** |
| 全量 plan_ok | 98.33% | **93.75%** |
| 全量 dedup / select / rerank / verify | 187 / 132 / 68 / 20 | **16 / 13 / 4 / 18** |

简单均 Task **显著下降**（4.85→2.967）。plan_ok **下降 4.6pp**，未达「不降」。  
routing 分未打（按纪律不为分加套餐）。

Planner 真延迟：p50 **1877ms** / p95 **3029ms**（n=259）。单独 Planner p95 已 >2.7s，**不能写 P95 2.7s**。

## 验收

- 通过：均 Task 显著下降。
- 未通过：plan_ok 不降。
- 不上软 lint（仍膨胀阈值 >4.0 才上）。
