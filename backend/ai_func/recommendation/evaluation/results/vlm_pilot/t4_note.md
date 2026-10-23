# T4 VLM 试点

数据集：`evaluation/datasets/vlm_pilot_queries.json`（12 条 gold 视觉核验 + 24 条补全 = 36）  
结果：本目录。`config.yaml` 的 `VISION_MODEL` 仍为 `"null"`，仅进程内 pin `deepseek-v4-pro`。

## 改动

- 不改生产默认。Verify 路径用 analyzer wrapper 注入模型（避免 yaml `"null"` 被 API 拒绝）。
- 无 GLM key，走 DeepSeek 视觉。
- 验收用确定性断言 + 延迟，不打 Judge。

## 数字

| 指标 | 值 |
|---|---:|
| n | 36 |
| Gate TEAM | **1.0**（36/36） |
| 结构化断言成功 | **36/36 = 1.0** |
| VLM p50 / p95 | **24.8s / 67.1s**（每 query 最多 2 张） |
| 单 query 墙钟均值 | 31.3s |

断言 = 解析出 `passed: bool` 且 status≠failed。`passed=false` 仍算成功（条件不成立 ≠ 解析失败）。  
延迟太高，不包装成「已上线视觉核验很快」。
