"""Rerank R2 正式 Evaluation（Baseline vs R1 vs R2，Top-10 固定池，18 Gold Query）。

规则：
- Gold：backend/tests/agent_team/gold_voc2007_healthy.json（冻结，不改）
- 每 Query 只执行一次 CLIP Retrieval（fetch 上限 40 rows）；取前 10 个 unique content 作为固定池
- Baseline = 池原始序；R1 = rerank_candidates(默认权重)；R2 = GLM-4V-Flash (MultimodalReranker 生产路径)
- R2 不重新 Retrieval；池集合三者相等且长度=10（断言）
- R2 环境变量：MM_RERANK_PROVIDER/API_KEY/BASE_URL/MODEL（模型 glm-4v-flash）
- R2 最多 18×10=180 次真实 API 调用（顺序逐张，无并发）
- 指标在池内计算；Candidate Recall@10 与 Rerank 质量分开报告
- Rank Delta 复用 rerank_delta.py（仅池内 relevant 参与）

用法：venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/eval_r2_gold.py
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
from backend.ai_func.recommendation.orchestration.model_reranker import (
    ENV_API_KEY, ENV_BASE_URL, ENV_MODEL, ENV_PROVIDER,
    build_multimodal_reranker,
)
from backend.ai_func.recommendation.orchestration.rerank import rerank_candidates
from backend.ai_func.recommendation.orchestration.workers import RetrievalWorker

FETCH_LIMIT = 40
POOL_SIZE = 10
GOLD = os.environ.get("GOLD_FILE") or os.path.join(_AGENT_TEAM, "gold_voc2007_healthy.json")
OUT = os.environ.get("EVAL_OUT") or os.path.join(
    os.environ.get("TEMP", r"C:\Users\31669\AppData\Local\Temp"), "r2_eval_results.json"
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


def fetch_pool(query):
    """单次 Retrieval → 前 POOL_SIZE 个 unique content（固定池）。"""
    rows = tool_search_images(query=query, targets=search_targets(),
                              filters={}, limit=FETCH_LIMIT) or []
    cands = [RetrievalWorker._to_candidate(r) for r in rows]
    seen, out = set(), []
    for row, cand in zip(rows, cands):
        key = content_id(row.get("filename") or "")
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append((key, row, cand))
    pool = out[:POOL_SIZE]
    if len(pool) != POOL_SIZE:
        raise SystemExit("pool < %d for query %r (rows=%d unique=%d)" % (POOL_SIZE, query, len(rows), len(out)))
    return pool


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


def movement(a_ids, b_ids):
    """全池 item 级移动统计（含非 relevant）：平均绝对位移/最大升/最大降/排序是否改变。"""
    pa = {x: i for i, x in enumerate(a_ids)}
    pb = {x: i for i, x in enumerate(b_ids)}
    diffs = [pa[x] - pb[x] for x in a_ids]
    return {
        "order_changed": a_ids != b_ids,
        "top5_changed": a_ids[:5] != b_ids[:5],
        "mean_abs_move": round(sum(abs(d) for d in diffs) / len(diffs), 3),
        "max_up": max([d for d in diffs if d > 0] or [0]),
        "max_down": min([d for d in diffs if d < 0] or [0]),
    }


def main():
    env_ok = all(os.environ.get(k) for k in (ENV_PROVIDER, ENV_API_KEY, ENV_BASE_URL, ENV_MODEL))
    print("R2 env complete:", env_ok, "model:", os.environ.get(ENV_MODEL, ""))
    print("预计 API 调用上限 = 18 Query × 10 = 180 次（顺序逐张）")
    if not env_ok:
        print("FAIL: MM_RERANK_* 未完整配置，未发起任何 API 调用")
        return 2
    mm = build_multimodal_reranker()
    if mm is None or mm.name != "openai_chat":
        print("FAIL: 期望 openai_chat provider")
        return 2
    prov = mm.provider
    if getattr(prov, "_reason", ""):
        print("FAIL provider unavailable:", prov._reason)
        return 2

    # instrument：统计 provider 逐图延迟/调用数（不改生产代码，只包一层计时）
    orig_relevance = prov.relevance
    r2_stats = {"calls": 0, "errors": 0, "lats_ms": []}

    def wrapped(q, image):
        t0 = time.perf_counter()
        try:
            return orig_relevance(q, image)
        except Exception:
            r2_stats["errors"] += 1
            raise
        finally:
            r2_stats["lats_ms"].append((time.perf_counter() - t0) * 1000.0)
            r2_stats["calls"] += 1

    prov.relevance = wrapped

    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)
    queries = gold["queries"]
    rows_list = []
    for qi, q in enumerate(queries, 1):
        query = q["query"]
        gt = set(q["gt_unique"])
        t_retr0 = time.perf_counter()
        pool = fetch_pool(query)
        retrieval_ms = (time.perf_counter() - t_retr0) * 1000.0
        base_ids = [k for k, _, _ in pool]
        pool_cands = [c for _, _, c in pool]
        gt_in_pool = [x for x in base_ids if x in gt]
        errors_before = r2_stats["errors"]

        b_m = rank_metrics(base_ids, gt)
        t_r1_0 = time.perf_counter()
        r1_cands = list(rerank_candidates(query, list(pool_cands)))
        r1_ms = (time.perf_counter() - t_r1_0) * 1000.0
        id_of_cand = {c.id: k for k, _, c in pool}
        r1_ids = [id_of_cand[c.id] for c in r1_cands]
        r1_m = rank_metrics(r1_ids, gt)

        t0 = time.perf_counter()
        r2_cands = mm.rerank(query, list(pool_cands))
        t_r2 = (time.perf_counter() - t0) * 1000.0
        r2_ids = [id_of_cand[c.id] for c in r2_cands]
        r2_m = rank_metrics(r2_ids, gt)
        r2_fallback = r2_stats["errors"] > errors_before

        assert set(base_ids) == set(r1_ids) == set(r2_ids), "pool 不一致!"
        assert len(r1_ids) == len(r2_ids) == POOL_SIZE

        s_r1 = rd.query_summary(query, base_ids, r1_ids, gt_in_pool, pair="B vs R1")
        s_r2 = rd.query_summary(query, base_ids, r2_ids, gt_in_pool, pair="B vs R2")
        rows_list.append({
            "query": query, "class": q["class"], "gt_total": len(gt), "gt_in_pool": len(gt_in_pool),
            "baseline_ids": base_ids, "r1_ids": r1_ids, "r2_ids": r2_ids,
            "baseline": b_m, "r1": r1_m, "r2": r2_m,
            "delta_first_r1": s_r1["first_relevant_rank_delta"],
            "delta_first_r2": s_r2["first_relevant_rank_delta"],
            "delta_item_r1": {"improved": s_r1["improved_count"], "unchanged": s_r1["unchanged_count"],
                              "degraded": s_r1["degraded_count"]},
            "delta_item_r2": {"improved": s_r2["improved_count"], "unchanged": s_r2["unchanged_count"],
                              "degraded": s_r2["degraded_count"]},
            "movement_r1": movement(base_ids, r1_ids),
            "movement_r2": movement(base_ids, r2_ids),
            "retrieval_ms": round(retrieval_ms, 1), "r1_ms": round(r1_ms, 1),
            "r2_ms": round(t_r2, 1), "r2_fallback": r2_fallback,
        })
        print(f"[{qi:02d}] {query:16s} gtIn={len(gt_in_pool)} "
              f"B({b_m['hit1']}/{b_m['hit3']}/{b_m['mrr']:.3f}) "
              f"R1({r1_m['hit1']}/{r1_m['hit3']}/{r1_m['mrr']:.3f}) "
              f"R2({r2_m['hit1']}/{r2_m['hit3']}/{r2_m['mrr']:.3f}) "
              f"firstB={b_m['first_rank']} firstR1={r1_m['first_rank']} firstR2={r2_m['first_rank']} "
              f"mvR2={movement(base_ids, r2_ids)['mean_abs_move']}", flush=True)

    n = len(rows_list)
    agg = {"queries": n, "pool_size": POOL_SIZE, "gold_file": GOLD,
           "search_targets": search_targets(), "query_rows": rows_list}
    agg["macro"] = {side: {k: round(sum(r[side][k] for r in rows_list) / n, 4)
                           for k in ("hit1", "hit3", "hit5", "mrr", "p5")}
                    for side in ("baseline", "r1", "r2")}
    agg["candidate_recall10_pooled"] = round(sum(r["gt_in_pool"] for r in rows_list) /
                                             sum(r["gt_total"] for r in rows_list), 4)
    agg["order_changed_queries"] = {
        "r1": sum(1 for r in rows_list if r["baseline_ids"] != r["r1_ids"]),
        "r2": sum(1 for r in rows_list if r["baseline_ids"] != r["r2_ids"]),
    }
    agg["top5_changed_queries"] = {
        "r1": sum(1 for r in rows_list if r["baseline_ids"][:5] != r["r1_ids"][:5]),
        "r2": sum(1 for r in rows_list if r["baseline_ids"][:5] != r["r2_ids"][:5]),
    }
    agg["r2_stats"] = {"calls": r2_stats["calls"], "errors": r2_stats["errors"],
                       "avg_latency_per_image_ms": round(sum(r2_stats["lats_ms"]) / len(r2_stats["lats_ms"]), 1)
                       if r2_stats["lats_ms"] else None,
                       "total_ms": round(sum(r2_stats["lats_ms"]), 1)}
    agg["latency_mean_ms"] = {
        "retrieval": round(sum(r["retrieval_ms"] for r in rows_list) / n, 1),
        "r1_rerank": round(sum(r["r1_ms"] for r in rows_list) / n, 1),
        "r2_rerank": round(sum(r["r2_ms"] for r in rows_list) / n, 1),
    }
    agg["fallback_queries"] = [r["query"] for r in rows_list if r["r2_fallback"]]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=1)
    print("\n== macro ==", agg["macro"])
    print("== candidate_recall10_pooled =", agg["candidate_recall10_pooled"])
    print("== order_changed_queries =", agg["order_changed_queries"])
    print("== top5_changed_queries =", agg["top5_changed_queries"])
    print("== r2_stats =", agg["r2_stats"])
    print("== latency_mean_ms =", agg["latency_mean_ms"])
    print("== fallback_queries =", agg["fallback_queries"])
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
