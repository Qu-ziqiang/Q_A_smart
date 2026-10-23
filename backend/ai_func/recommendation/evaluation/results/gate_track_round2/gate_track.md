# Gate Track (P0)

产品指标：FN/FP、TEAM 率、Legacy 跳过 Planner 条数、Gate 延迟。不进 Judge。调正则只许看 contrast_dev。

- gold_28 FN=0 FP=0 TEAM=0.7143
- contrast_dev FP=0 n=88
- contrast_test FP=0 n=32 ids=[]
- all TEAM=0.1351 Legacy skip Planner=128
- Gate latency ms p50=0.008 p95=0.011 p99=0.013

## Latency
stage | n | mean | p50 | p95 | p99 | max
---|---:|---:|---:|---:|---:|---:
gate | 148 | 0.008 | 0.008 | 0.011 | 0.013 | 0.021
