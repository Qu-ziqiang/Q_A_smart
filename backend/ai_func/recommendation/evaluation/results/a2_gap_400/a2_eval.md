# A2 分差触发评测（400 张子集）

- n=70  tau=0.006  model=glm-4v-flash
- 触发率 0.3143（目标参考 20–40%）
- 简单 query 误触发 0.25
- Hit@1 baseline 0.7681 → always R2 0.8986 → A2 0.8116
- MRR baseline 0.8572 → always R2 0.9377 → A2 0.8825
- 触发子集 n=22 Hit@1 0.8636 → 1.0
- always R2 p50/p95 12139.9/15364.0 ms
- A2 均摊 p50/p95 0.0/14252.0 ms
