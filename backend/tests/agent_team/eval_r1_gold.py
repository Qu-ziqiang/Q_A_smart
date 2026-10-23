"""Rerank R1 正式 Evaluation（Baseline vs R1，18 条 Strong VOC2007 Gold Query）。

规则：
- GT：PASCAL VOC2007 官方 annotation（已字节级核验，frozen 于 gold_voc2007_healthy.json）
- 每 Query：只做一次原始检索（limit 固定 40，禁改 retrieval 参数）
  → 同池去重为 unique content → Baseline 序；同池对象 → R1 rerank_candidates（默认权重，不改）
- R1 只改变排序；candidate 集合不变（content 级校验 set==）
- 指标在 candidate pool 内计算：Hit@1/3/5、MRR、P@5；Candidate Recall 区分“召回”与“重排”
- Rank Delta 复用 rerank_delta.py（pool 内 relevant 才参与）
- R2 / GLM：NOT RUN（无任何付费 API）

用法（repo 根）：
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/eval_r1_gold.py
输出：终端表格 + JSON 结果（报告据此生成）。
"""

import json
import os
import sys
import time

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))
_AGENT_TEAM = os.path.dirname(os.path.abspath(__file__))
if _AGENT_TEAM not in sys.path:
    sys.path.insert(0, _AGENT_TEAM)

import rerank_delta as rd
from backend.ai_func.recommendation.tools import tool_search_images
from backend.ai_func.recommendation.orchestration.rerank import rerank_candidates
from backend.ai_func.recommendation.orchestration.workers import RetrievalWorker

RETRIEVAL_LIMIT = 40
GOLD = os.environ.get("GOLD_FILE") or os.path.join(_AGENT_TEAM, "gold_voc2007_healthy.json")
OUT = os.environ.get("EVAL_OUT") or os.path.join(
    os.environ.get("TEMP", r"C:\Users\31669\AppData\Local\Temp"), "r1_eval_results.json"
)


def search_targets():
    raw = os.environ.get("SEARCH_TARGETS", "title,description,image")
    return [t.strip() for t in raw.split(",") if t.strip()] or ["image"]


def content_id(filename):
    f = (filename or "").strip()
    lower = f.lower()
    if not lower.endswith(".jpg") and not lower.endswith(".jpeg"):
        return None
    return os.path.splitext(f)[0]


def dedupe_content(rows_cands):
    seen, out = set(), []
    for row, cand in rows_cands:
        key = content_id(row.get("filename") or "")
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append((key, row, cand))
    return out


def rank_metrics(ids, gt):
    hits = [i + 1 for i, x in enumerate(ids) if x in gt]
    first = hits[0] if hits else None
    return {
        "hit1": 1 if first == 1 else 0,
        "hit3": 1 if first is not None and first <= 3 else 0,
        "hit5": 1 if first is not None and first <= 5 else 0,
        "mrr": (1.0 / first) if first else 0.0,
        "p5": sum(1 for i, x in enumerate(ids[:5]) if x in gt) / 5.0,
        "first_rank": first,
    }


def main():
    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)
    queries = gold["queries"]
    rows_list = []

    for qi, q in enumerate(queries, 1):
        query = q["query"]
        gt = set(q["gt_unique"])
        t0 = time.perf_counter()
        rows = tool_search_images(query=query, targets=search_targets(),
                                  filters={}, limit=RETRIEVAL_LIMIT) or []
        t_retr = (time.perf_counter() - t0) * 1000.0
        cands = [RetrievalWorker._to_candidate(r) for r in rows]
        content = dedupe_content(list(zip(rows, cands)))
        pool_ids = [k for k, _, _ in content]
        pool_cands = [c for _, _, c in content]

        t1 = time.perf_counter()
        r1_cands = list(rerank_candidates(query, pool_cands))
        t_r1 = (time.perf_counter() - t1) * 1000.0
        id_of_cand = {c.id: k for k, _, c in content}
        r1_ids = [id_of_cand[c.id] for c in r1_cands]

        assert set(pool_ids) == set(r1_ids), "R1 改变候选集合!"
        assert len(pool_ids) == len(r1_ids)

        gt_in_pool = [x for x in pool_ids if x in gt]
        b = rank_metrics(pool_ids, gt)
        r = rank_metrics(r1_ids, gt)
        summ = rd.query_summary(query, pool_ids, r1_ids, gt_in_pool, pair="Baseline vs R1")

        rows_list.append({
            "query": query, "class": q["class"], "gt_total": len(gt),
            "gt_in_pool": len(gt_in_pool),
            "pool_size": len(pool_ids),
            "baseline": b, "r1": r,
            "first_relevant_rank_delta": summ["first_relevant_rank_delta"],
            "improved_count": summ["improved_count"], "unchanged_count": summ["unchanged_count"],
            "degraded_count": summ["degraded_count"],
            "average_rank_delta": summ["average_rank_delta"],
            "median_rank_delta": summ["median_rank_delta"],
            "baseline_ids": pool_ids, "r1_ids": r1_ids,
            "retrieval_ms": round(t_retr, 1), "rerank_ms": round(t_r1, 1),
        })
        print(f"[{qi:02d}] {query:16s} gt={len(gt):2d} inPool={len(gt_in_pool):2d} "
              f"B(h1/h3/h5/mrr)= {b['hit1']}/{b['hit3']}/{b['hit5']}/{b['mrr']:.3f} "
              f"R1= {r['hit1']}/{r['hit3']}/{r['hit5']}/{r['mrr']:.3f} "
              f"firstDelta={summ['first_relevant_rank_delta']} "
              f"retr={t_retr:.0f}ms r1={t_r1:.0f}ms", flush=True)

    n = len(rows_list)

    def mean(key, side):
        return sum(r[side][key] for r in rows_list) / n

    tot_gt = sum(r["gt_total"] for r in rows_list)
    tot_in_pool = sum(r["gt_in_pool"] for r in rows_list)
    agg = {
        "queries": n,
        "gold_file": GOLD,
        "search_targets": search_targets(),
        "retrieval_limit": RETRIEVAL_LIMIT,
        "candidate_recall_pooled": tot_in_pool / tot_gt,
        "candidate_recall_mean": sum(r["gt_in_pool"] / r["gt_total"] for r in rows_list) / n,
        "macro": {
            "baseline": {k: round(mean(k, "baseline"), 4) for k in ("hit1", "hit3", "hit5", "mrr", "p5")},
            "r1": {k: round(mean(k, "r1"), 4) for k in ("hit1", "hit3", "hit5", "mrr", "p5")},
        },
        "latency_ms": {
            "retrieval_mean": sum(r["retrieval_ms"] for r in rows_list) / n,
            "rerank_mean": sum(r["rerank_ms"] for r in rows_list) / n,
            "total_mean": sum(r["retrieval_ms"] + r["rerank_ms"] for r in rows_list) / n,
        },
        "query_rows": rows_list,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=1)
    print("\n== pooled candidate recall =", round(tot_in_pool / tot_gt, 4))
    print("== macro baseline =", agg["macro"]["baseline"])
    print("== macro r1       =", agg["macro"]["r1"])
    print("== latency(ms) =", agg["latency_ms"])
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

