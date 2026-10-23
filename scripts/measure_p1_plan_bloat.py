"""P1：从已有 After JSON 量计划膨胀（不调 LLM）。

验收：简单子集平均任务数、纯 retrieve 计划占比。
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_DEFAULT = os.path.join(
    _REPO, "backend", "ai_func", "recommendation",
    "evaluation", "results", "task_catalog_ab_deepseek", "after.json",
)
_OPTIONAL = ("rerank_candidates", "deduplicate_images", "select_images",
             "sort_images", "verify_visual_condition")


def _is_simple(rec: dict) -> bool:
    if rec.get("category") == "retrieval":
        return True
    q = rec.get("query") or ""
    return ("分组" not in q and "确认" not in q and "判断" not in q
            and "比较" not in q and "按" not in q)


def analyze(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records") or []
    simple = [r for r in records if _is_simple(r)]
    pool = simple or records

    def stats(rows):
        lengths = [len(r.get("plan_task_types") or []) for r in rows]
        pure = 0
        optional_hits = Counter()
        for r in rows:
            types = [t for t in (r.get("plan_task_types") or []) if t]
            core = [t for t in types if t != "synthesize"]
            if core == ["retrieve_images"]:
                pure += 1
            for t in types:
                if t in _OPTIONAL:
                    optional_hits[t] += 1
        n = len(rows) or 1
        return {
            "n": len(rows),
            "avg_tasks": round(sum(lengths) / n, 3),
            "pure_retrieve_rate": round(pure / n, 4),
            "optional_counts": dict(optional_hits),
        }

    return {
        "source": path,
        "all": stats(records),
        "simple_subset": stats(pool),
        "note": "基线来自 DeepSeek After 臂。Prompt 收紧后应再跑 After-only 对比本文件。",
    }


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT
    if not os.path.isfile(path):
        print(f"missing {path}", file=sys.stderr)
        return 2
    report = analyze(path)
    out = os.path.join(os.path.dirname(path), "p1_plan_bloat.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[p1] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
