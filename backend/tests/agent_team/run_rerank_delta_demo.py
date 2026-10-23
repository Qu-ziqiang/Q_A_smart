"""Rerank Delta 演示 runner（Evaluation-only，零 API、零生产代码依赖）。

演示内容（全部为本地固定 toy ranking，无真实检索、无 GLM、不伪造 Ground Truth）：
    Baseline → R1
    Baseline → R2
    R1 → R2（必要时）
并展示 candidate_pool_mismatch 的隔离报告。

用法（repo 根）：
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/run_rerank_delta_demo.py
"""

import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))
_AGENT_TEAM = os.path.dirname(os.path.abspath(__file__))
if _AGENT_TEAM not in sys.path:
    sys.path.insert(0, _AGENT_TEAM)

import rerank_delta as rd

# 固定候选池（本地演示用；正式评估请接入真实 Gold Set + Baseline/R1/R2 排名）
BASE_POOL = ["A", "B", "C", "D", "E", "F", "G", "H"]

# 每个 query：relevant_ids + 各策略产生的排序（同 pool，只演示不同排序）
TOY = [
    {"query": "q1-sunset", "relevant_ids": ["F", "C"],
     "r1": ["C", "F", "A", "B", "D", "E", "G", "H"],
     "r2": ["F", "C", "B", "A", "D", "E", "G", "H"]},
    {"query": "q2-car", "relevant_ids": ["E", "A", "G"],
     "r1": ["A", "E", "G", "B", "C", "D", "F", "H"],
     "r2": ["G", "A", "E", "B", "C", "D", "F", "H"]},
    {"query": "q3-cat", "relevant_ids": ["D"],
     "r1": ["D", "A", "B", "C", "E", "F", "G", "H"],
     "r2": ["X", "D", "A", "B", "C", "E", "F", "G"]},  # 故意 pool mismatch（r2 缺 H 多 X）
]


def main():
    base_pairs = {
        "Baseline vs R1": [{"query": t["query"], "baseline_ids": BASE_POOL,
                            "rerank_ids": t["r1"], "relevant_ids": t["relevant_ids"]} for t in TOY],
        "Baseline vs R2": [{"query": t["query"], "baseline_ids": BASE_POOL,
                            "rerank_ids": t["r2"], "relevant_ids": t["relevant_ids"]} for t in TOY],
        "R1 vs R2": [{"query": t["query"], "baseline_ids": t["r1"],
                      "rerank_ids": t["r2"], "relevant_ids": t["relevant_ids"]} for t in TOY],
    }
    all_summaries = []
    for label, qrs in base_pairs.items():
        summaries = rd.evaluate_pair(label, qrs)
        all_summaries += summaries
        g = rd.global_summary(summaries, pair=label)
        print(f"\n===== {label} =====")
        if g["queries_pool_mismatch"]:
            print("candidate_pool_mismatch: " + ", ".join(
                f"{m['query']}({m['pool_diff']})" for m in g["queries_pool_mismatch"]))
        print(rd.format_query_table(summaries))
        print()
        print(rd.format_first_rr_table(summaries))
        print()
        print(rd.format_movement_table(rd.movement_rows(summaries, rd.IMPROVED, 10), rd.IMPROVED))
        print()
        print(rd.format_movement_table(rd.movement_rows(summaries, rd.DEGRADED, 10), rd.DEGRADED))
        print()
        print("global: total_relevant_images=%d improved=%d unchanged=%d degraded=%d "
              "improvement_rate=%.2f degradation_rate=%.2f" % (
                  g["total_relevant_images"], g["improved_count"], g["unchanged_count"],
                  g["degraded_count"], g["improvement_rate"], g["degradation_rate"]))
        print("query_level:", g["query_level"])
    print("\n== demo done (local toy rankings only; no GLM / no Ground Truth claim) ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
