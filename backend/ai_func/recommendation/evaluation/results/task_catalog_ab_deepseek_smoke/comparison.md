# Task Catalog A/B — LLM Judge

After = 10 TaskTypes（含 rerank/dedup/sort/select/verify）；Before = 5 TaskTypes（retrieve/filter/group/analyze_visual/synthesize）。同一 Judge、同一 Fake 工具、真实 Planner。

- overall: 3.867 → 4.6 (Δ +0.733)
- pass_rate: 0.6667 → 1.0 (Δ +0.3333)
- plan_ok_rate: 1.0 → 1.0 (Δ +0.0)

## Dimensions
- planning_quality: 3.667 → 5.0 (Δ +1.333)
- agent_routing: 2.667 → 4.0 (Δ +1.333)
- tool_usage: 4.667 → 5.0 (Δ +0.333)
- recovery_behavior: 4.667 → 5.0 (Δ +0.333)
- final_result_quality: 3.667 → 4.0 (Δ +0.333)
