# Task Catalog A/B — LLM Judge

After = 10 TaskTypes（含 rerank/dedup/sort/select/verify）；Before = 5 TaskTypes（retrieve/filter/group/analyze_visual/synthesize）。同一 Judge、同一 Fake 工具、真实 Planner。

- overall: 4.17 → 4.263 (Δ +0.093)
- pass_rate: 0.6375 → 0.6292 (Δ -0.0083)
- plan_ok_rate: 0.9792 → 0.9833 (Δ +0.0041)

## Dimensions
- planning_quality: 4.487 → 4.721 (Δ +0.234)
- agent_routing: 4.046 → 4.196 (Δ +0.15)
- tool_usage: 4.558 → 4.575 (Δ +0.017)
- recovery_behavior: 4.567 → 4.579 (Δ +0.012)
- final_result_quality: 3.192 → 3.246 (Δ +0.054)
