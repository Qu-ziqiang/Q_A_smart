# T3 反例加厚

数据集：`evaluation/datasets/gate_contrast_legacy.json`（60→120）  
结果：本目录（不覆盖 `gate_track/`）

## 改动

- 加 60 条字面像视觉/确认、应走 Legacy 的反例。
- 规则未改（dev FP 仍为 0，无需动正则）。
- test 冻结复测：旧 15 条 + 新切分，合计 32。

## 数字

| | before | after |
|---|---:|---:|
| 反例总量 | 60 | **120** |
| test | 15 | **32** |
| contrast test FP | 0 | **0** |
| contrast dev FP | 0 | **0** |
| gold 28 FN/FP | 0/0 | **0/0** |
| Gate p95 | 0.008ms | **0.011ms** |

未到 300，不做 CLIP logistic 路由头。
