# Task Catalog A/B — LLM Judge

After = 10 TaskTypes（含 rerank/dedup/sort/select/verify）；Before = 5 TaskTypes（retrieve/filter/group/analyze_visual/synthesize）。同一 Judge、同一 Fake 工具、真实 Planner。

- overall: 3.457 → 0.0 (Δ -3.457)
- pass_rate: 0.5792 → 0.0 (Δ -0.5792)
- plan_ok_rate: 0.7917 → 0.0 (Δ -0.7917)

## Dimensions
- planning_quality: 3.521 → 0.0 (Δ -3.521)
- agent_routing: 3.479 → 0.0 (Δ -3.479)
- tool_usage: 3.775 → 0.0 (Δ -3.775)
- recovery_behavior: 3.65 → 0.0 (Δ -3.65)
- final_result_quality: 2.858 → 0.0 (Δ -2.858)
