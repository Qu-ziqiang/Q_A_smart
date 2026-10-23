"""Rerank Delta Evaluation —— 只读纯函数分析层（Evaluation/Benchmark 专用，非生产代码）。

目标：对「Baseline → R1」与「Baseline → R2」等 Rerank 变化做 Rank Delta 分析，
不修改 rerank.py / model_reranker.py / workers.py 等任何生产逻辑；
不调用真实 GLM；不自行生成/伪造 Ground Truth。

核心定义（1-based rank）：
    rank_delta = baseline_rank - rerank_rank
        > 0  improved（排名提前）
        = 0  unchanged
        < 0  degraded（排名下降）
    某侧找不到相关图时 rank 取 None（绝不用 0 占位），status 标记 missing_*。

First Relevant Rank：
    baseline_first_relevant_rank = min(相关图在 baseline 中的 rank)（无则 None）
    rerank_first_relevant_rank   = min(相关图在 rerank 中的 rank)（无则 None）
    first_relevant_rank_delta    = baseline_first_relevant_rank - rerank_first_relevant_rank
    两者都有名次时才有 delta。MRR 侧展示 RR = 1 / first_relevant_rank（缺失记 0.0）。

一致性：R1/R2 契约要求候选集合不变。分析前检查 set(baseline)==set(rerank)；
不一致的 Query 记为 candidate_pool_mismatch，其 item 不进入全局 image-level 统计。
"""

import statistics
from typing import Dict, List, Optional

IMPROVED = "improved"
UNCHANGED = "unchanged"
DEGRADED = "degraded"
MISSING_BASELINE = "missing_baseline"
MISSING_RERANK = "missing_rerank"
MISSING_BOTH = "missing_both"

STANDARD_STATUSES = (IMPROVED, UNCHANGED, DEGRADED)
MISSING_STATUSES = (MISSING_BASELINE, MISSING_RERANK, MISSING_BOTH)


def first_position(ids: List, target) -> Optional[int]:
    """target 在 ids 中的第一个出现位置（1-based）；不存在返回 None。
    重复 ID 处理：取第一次出现（文档化规则），重复事实由 duplicates 单独暴露。"""
    try:
        return ids.index(target) + 1
    except ValueError:
        return None


def duplicate_ids(ids: List) -> List:
    """返回出现次数 >1 的 ID（按 str 排序保证确定性）。"""
    return sorted({x for x in ids if ids.count(x) > 1}, key=str)


def pool_diff(baseline_ids: List, rerank_ids: List) -> Dict[str, List]:
    return {
        "only_baseline": sorted(set(baseline_ids) - set(rerank_ids), key=str),
        "only_rerank": sorted(set(rerank_ids) - set(baseline_ids), key=str),
    }


def candidate_pool_consistent(baseline_ids: List, rerank_ids: List) -> bool:
    """R1/R2 契约：只改排序，不改候选集合。set 相等即一致。"""
    return set(baseline_ids) == set(rerank_ids)


def _record(rid, b, r) -> Dict:
    if b is not None and r is not None:
        delta = b - r
        status = IMPROVED if delta > 0 else (DEGRADED if delta < 0 else UNCHANGED)
    elif b is None and r is None:
        delta, status = None, MISSING_BOTH
    elif b is None:
        delta, status = None, MISSING_BASELINE
    else:
        delta, status = None, MISSING_RERANK
    return {
        "relevant_id": rid,
        "baseline_rank": b,
        "rerank_rank": r,
        "rank_delta": delta,
        "status": status,
    }


def compute_rank_delta(baseline_ids: List, rerank_ids: List,
                       relevant_ids: List) -> Dict:
    """对每个 relevant image 计算 rank delta。

    返回结构化结果：
      - pool_consistent: bool
      - pool_diff: {"only_baseline": [...], "only_rerank": [...]}
      - duplicates: {"baseline": [...], "rerank": [...]}
      - items: [ {relevant_id, baseline_rank, rerank_rank, rank_delta, status}, ... ]
    rank 缺失一律 None；delta 仅在两侧都有 rank 时存在。
    """
    dup = {
        "baseline": duplicate_ids(list(baseline_ids)),
        "rerank": duplicate_ids(list(rerank_ids)),
    }
    items = []
    for rid in relevant_ids:
        b = first_position(baseline_ids, rid)
        r = first_position(rerank_ids, rid)
        items.append(_record(rid, b, r))
    return {
        "pool_consistent": candidate_pool_consistent(list(baseline_ids), list(rerank_ids)),
        "pool_diff": pool_diff(list(baseline_ids), list(rerank_ids)),
        "duplicates": dup,
        "items": items,
    }


def standard_items(items: List[Dict]) -> List[Dict]:
    """两侧都有排名的 item（可参与标准 delta 统计）。"""
    return [it for it in items if it["status"] in STANDARD_STATUSES]


def _first_rank(items: List[Dict], key: str) -> Optional[int]:
    present = [it[key] for it in items if it[key] is not None]
    return min(present) if present else None


def reciprocal_rank(rank: Optional[int]) -> float:
    """MRR 用 reciprocal rank：rank 缺失（未出现）按 0.0。"""
    return 1.0 / rank if rank else 0.0


def _median(values: List) -> Optional[float]:
    return statistics.median(values) if values else None


def _mean(values: List) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def query_summary(query: str, baseline_ids: List, rerank_ids: List,
                  relevant_ids: List, pair: str = "") -> Dict:
    """单个 Query 的 Rank Delta 汇总。

    覆盖：relevant 总数 / improved·unchanged·degraded 数 / missing_* 数 /
    avg·median delta / max_rank_gain / max_rank_drop /
    First Relevant Rank + RR（供人工理解 MRR 变化来源）。
    """
    res = compute_rank_delta(list(baseline_ids), list(rerank_ids), list(relevant_ids))
    items = res["items"]
    std = standard_items(items)
    deltas = [it["rank_delta"] for it in std]
    gains = [d for d in deltas if d is not None and d > 0]
    drops = [d for d in deltas if d is not None and d < 0]
    bf = _first_rank(items, "baseline_rank")
    rf = _first_rank(items, "rerank_rank")
    return {
        "query": query,
        "pair": pair,
        "pool_consistent": res["pool_consistent"],
        "pool_diff": res["pool_diff"],
        "duplicates": res["duplicates"],
        "relevant_total": len(items),
        "improved_count": sum(1 for it in items if it["status"] == IMPROVED),
        "unchanged_count": sum(1 for it in items if it["status"] == UNCHANGED),
        "degraded_count": sum(1 for it in items if it["status"] == DEGRADED),
        "missing_baseline_count": sum(1 for it in items if it["status"] == MISSING_BASELINE),
        "missing_rerank_count": sum(1 for it in items if it["status"] == MISSING_RERANK),
        "missing_both_count": sum(1 for it in items if it["status"] == MISSING_BOTH),
        "average_rank_delta": _mean(deltas),
        "median_rank_delta": _median(deltas),
        "max_rank_gain": max(gains) if gains else None,
        "max_rank_drop": min(drops) if drops else None,
        "baseline_first_relevant_rank": bf,
        "rerank_first_relevant_rank": rf,
        "first_relevant_rank_delta": (bf - rf) if (bf is not None and rf is not None) else None,
        "baseline_reciprocal_rank": reciprocal_rank(bf),
        "rerank_reciprocal_rank": reciprocal_rank(rf),
        "items": items,
    }


def evaluate_pair(pair: str, query_results: List) -> List[Dict]:
    """把一个对比对（如 'Baseline vs R1'）的一组 query 全部加上 pair 标签。

    query_results: [{query, baseline_ids, rerank_ids, relevant_ids}, ...]
    返回可送入 global_summary 的 summary 列表。
    """
    return [
        query_summary(qr["query"], qr["baseline_ids"], qr["rerank_ids"],
                      qr["relevant_ids"], pair=pair)
        for qr in query_results
    ]


def _status_counts(items: List[Dict]) -> Dict[str, int]:
    counts = {s: 0 for s in STANDARD_STATUSES + MISSING_STATUSES}
    for it in items:
        counts[it["status"]] += 1
    return counts


def global_summary(summaries: List[Dict], pair: Optional[str] = None) -> Dict:
    """跨 Query 的全局统计。

    候选池不一致的 Query 先被隔离（candidate_pool_mismatch），
    其 item 不进入 image-level 的标准统计，也不进入 rates。
    pair 指定时只统计该对比对的 summaries。

    Query-level 指标基于 First Relevant Rank Delta：
      improved/unchanged/degraded 按 first delta 符号；无法判定（任一侧缺失）记 unknown。
    Image-level 指标基于每张 relevant image 的 rank_delta。
    """
    ss = [s for s in summaries if not pair or s.get("pair") == pair]
    mismatches = [{"pair": s.get("pair", ""), "query": s.get("query"),
                   "pool_diff": s.get("pool_diff")} for s in ss if not s.get("pool_consistent", True)]
    consistent = [s for s in ss if s.get("pool_consistent", True)]
    items_all = [it for s in consistent for it in s.get("items", [])]
    total_relevant = len(items_all)
    counts = _status_counts(items_all)
    std_deltas = [it["rank_delta"] for it in items_all if it["status"] in STANDARD_STATUSES]

    q_known = [s for s in consistent if s.get("first_relevant_rank_delta") is not None]
    q_unknown = len(consistent) - len(q_known)
    q_imp = sum(1 for s in q_known if s["first_relevant_rank_delta"] > 0)
    q_unch = sum(1 for s in q_known if s["first_relevant_rank_delta"] == 0)
    q_deg = sum(1 for s in q_known if s["first_relevant_rank_delta"] < 0)

    n_consistent = len(consistent)
    return {
        "pair": pair or "(all)",
        "queries_total": len(ss),
        "queries_consistent": n_consistent,
        "queries_pool_mismatch": mismatches,
        "total_relevant_images": total_relevant,
        "improved_count": counts[IMPROVED],
        "unchanged_count": counts[UNCHANGED],
        "degraded_count": counts[DEGRADED],
        "missing_baseline_count": counts[MISSING_BASELINE],
        "missing_rerank_count": counts[MISSING_RERANK],
        "missing_both_count": counts[MISSING_BOTH],
        "average_rank_delta": _mean(std_deltas),
        "median_rank_delta": _median(std_deltas),
        # 公式分母 = 该 pool 内全部 relevant images（含 missing 项）；不把 Query 数当分母
        "improvement_rate": (counts[IMPROVED] / total_relevant) if total_relevant else 0.0,
        "degradation_rate": (counts[DEGRADED] / total_relevant) if total_relevant else 0.0,
        "image_level_improvement_rate": (counts[IMPROVED] / total_relevant) if total_relevant else 0.0,
        "query_level": {
            "queries_with_first_known": len(q_known),
            "queries_with_first_unknown": q_unknown,
            "improved_queries": q_imp,
            "unchanged_queries": q_unch,
            "degraded_queries": q_deg,
            "query_improvement_rate": (q_imp / n_consistent) if n_consistent else 0.0,
            "query_degradation_rate": (q_deg / n_consistent) if n_consistent else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# 输出表（Markdown 风格文本）
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def movement_rows(summaries: List[Dict], kind: str = IMPROVED, top_n: int = 10) -> List[Dict]:
    """Top movements：improved 按 delta 降序；degraded 按 delta 升序（最负在前）。仅统计 pool 一致 query。"""
    rows = []
    for s in summaries:
        if not s.get("pool_consistent", True):
            continue
        for it in s.get("items", []):
            if it["status"] != kind or it["rank_delta"] is None:
                continue
            rows.append({
                "query": s.get("query"),
                "pair": s.get("pair", ""),
                "relevant_id": it["relevant_id"],
                "baseline_rank": it["baseline_rank"],
                "rerank_rank": it["rerank_rank"],
                "rank_delta": it["rank_delta"],
            })
    rows.sort(key=lambda r: r["rank_delta"], reverse=(kind == IMPROVED))
    return rows[:top_n]


def format_query_table(summaries: List[Dict]) -> str:
    lines = [
        "| Query | Pair | Baseline First Rank | Rerank First Rank | First Rank Delta "
        "| Improved | Unchanged | Degraded |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {_fmt(s.get('query'))} | {_fmt(s.get('pair')) or '—'} "
            f"| {_fmt(s.get('baseline_first_relevant_rank'))} "
            f"| {_fmt(s.get('rerank_first_relevant_rank'))} "
            f"| {_fmt(s.get('first_relevant_rank_delta'))} "
            f"| {s.get('improved_count', 0)} | {s.get('unchanged_count', 0)} "
            f"| {s.get('degraded_count', 0)} |"
        )
    return "\n".join(lines)


def format_first_rr_table(summaries: List[Dict]) -> str:
    """MRR 变化来源人工阅读表（不重复实现 MRR 聚合）。"""
    lines = [
        "| Query | Baseline First Relevant Rank | Rerank First Relevant Rank "
        "| First Rank Delta | Baseline RR | Rerank RR |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {_fmt(s.get('query'))} "
            f"| {_fmt(s.get('baseline_first_relevant_rank'))} "
            f"| {_fmt(s.get('rerank_first_relevant_rank'))} "
            f"| {_fmt(s.get('first_relevant_rank_delta'))} "
            f"| {_fmt(s.get('baseline_reciprocal_rank'))} "
            f"| {_fmt(s.get('rerank_reciprocal_rank'))} |"
        )
    return "\n".join(lines)


def format_movement_table(rows: List[Dict], kind: str = IMPROVED) -> str:
    lines = [
        "| Query | Pair | Image ID | Baseline Rank | Rerank Rank | Delta |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {_fmt(r['query'])} | {_fmt(r['pair']) or '—'} | {_fmt(r['relevant_id'])} "
            f"| {_fmt(r['baseline_rank'])} | {_fmt(r['rerank_rank'])} | {_fmt(r['rank_delta'])} |"
        )
    title = "Top Improvements" if kind == IMPROVED else "Top Degradations"
    return f"{title}:\n" + "\n".join(lines)


