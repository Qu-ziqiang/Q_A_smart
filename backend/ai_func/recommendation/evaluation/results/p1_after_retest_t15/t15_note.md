# T1.5 空计划硬底线复测

对照：`p1_after_retest/`（4.85→2.967，plan_ok 93.75%）  
本目录：同口径 After-only 240，DeepSeek，无 Judge。

## 改动（确定性，未盲调 prompt）

1. Validator 硬底线：空计划 / 无可执行 Task / 校验失败 → 带错误重试一次；仍失败 → retrieve→synthesize。不交空计划。
2. 政策分级：Gate 简单注入最小必要（禁止默认套餐）；Gate TEAM 注入完整目录政策（禁止空计划，不省略点名步骤）。

## 数字

| | T1 复测 | T1.5 |
|---|---:|---:|
| 全量 plan_ok | 93.75% | **100%** |
| 简单均 Task | 2.967 | **3.261** |
| 纯 retrieve | 38.6% | 37.3% |
| floor retry / fallback | — | 16 / 2 |

验收：plan_ok ≥98% **通过**；简单均 Task ≤3.2 **未过**（3.261）。  
简历可写 plan_ok 100% 持平/回升；不要写「压到 3.2 以下」。
