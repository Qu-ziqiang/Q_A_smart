"""Rerank Delta 纯函数单元测试（test_rerank_delta.py）。

运行（repo 根）：
    venv\\Scripts\\python.exe -X utf8 -m pytest backend/tests/agent_team/test_rerank_delta.py -q
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


def _by_id(items):
    return {it["relevant_id"]: it for it in items}


def _approx(a, b):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-9)


# ---------------------------------------------------------------------------
# 1) 定义正确性：题目给定 toy ranking
#    Baseline [A,B,C,D,E] / Rerank [C,A,E,B,D] / Relevant [A,C]
# ---------------------------------------------------------------------------
def test_toy_spec_deltas_exact():
    base = ["A", "B", "C", "D", "E"]
    rer = ["C", "A", "E", "B", "D"]
    rel = ["A", "C"]
    res = rd.compute_rank_delta(base, rer, rel)
    assert res["pool_consistent"] is True
    m = _by_id(res["items"])
    assert m["A"]["baseline_rank"] == 1 and m["A"]["rerank_rank"] == 2
    assert m["A"]["rank_delta"] == -1 and m["A"]["status"] == rd.DEGRADED
    assert m["C"]["baseline_rank"] == 3 and m["C"]["rerank_rank"] == 1
    assert m["C"]["rank_delta"] == 2 and m["C"]["status"] == rd.IMPROVED
    s = rd.query_summary("q", base, rer, rel)
    assert s["baseline_first_relevant_rank"] == 1
    assert s["rerank_first_relevant_rank"] == 1
    assert s["first_relevant_rank_delta"] == 0


def test_plus_five_improvement():
    base = list(range(1, 11))
    rer = list(range(1, 11))
    rer.remove(8)
    rer.insert(2, 8)  # 8: baseline rank 8 → rerank rank 3
    it = rd.compute_rank_delta(base, rer, [8])["items"][0]
    assert it == {"relevant_id": 8, "baseline_rank": 8, "rerank_rank": 3,
                  "rank_delta": 5, "status": rd.IMPROVED}


def test_minus_three_degradation():
    base = list(range(1, 11))
    rer = list(range(1, 11))
    rer.remove(2)
    rer.insert(4, 2)  # 2: baseline rank 2 → rerank rank 5
    it = rd.compute_rank_delta(base, rer, [2])["items"][0]
    assert it == {"relevant_id": 2, "baseline_rank": 2, "rerank_rank": 5,
                  "rank_delta": -3, "status": rd.DEGRADED}


def test_unchanged():
    base = ["a", "b", "c"]
    it = rd.compute_rank_delta(base, list(base), ["b"])["items"][0]
    assert it["rank_delta"] == 0 and it["status"] == rd.UNCHANGED


# ---------------------------------------------------------------------------
# 2) 多个 relevant / query 汇总（不只取第一张）
# ---------------------------------------------------------------------------
def test_multiple_relevant_stats_cover_all():
    base = [1, 2, 3, 4, 5, 6]
    rer = [6, 1, 4, 2, 3, 5]  # 4: rank4→3(+1)、6: rank6→1(+5)
    s = rd.query_summary("q", base, rer, [4, 6])
    assert s["relevant_total"] == 2
    assert s["improved_count"] == 2 and s["unchanged_count"] == 0
    assert s["degraded_count"] == 0
    assert s["average_rank_delta"] == 3.0 and s["median_rank_delta"] == 3.0
    assert s["max_rank_gain"] == 5 and s["max_rank_drop"] is None
    assert s["baseline_first_relevant_rank"] == 4
    assert s["rerank_first_relevant_rank"] == 1
    assert s["first_relevant_rank_delta"] == 3


def test_first_relevant_rank_delta_positive():
    base = list(range(1, 11))
    rer = list(range(1, 11))
    rer.remove(8)
    rer.insert(1, 8)  # relevant {5,8}: baseline first=5 → rerank first=2
    s = rd.query_summary("q", base, rer, [5, 8])
    assert s["baseline_first_relevant_rank"] == 5
    assert s["rerank_first_relevant_rank"] == 2
    assert s["first_relevant_rank_delta"] == 3
    assert _approx(s["baseline_reciprocal_rank"], 0.2)
    assert _approx(s["rerank_reciprocal_rank"], 0.5)


def test_all_unchanged_query():
    base = list("ABCDE")
    s = rd.query_summary("q", base, list(base), ["A", "C"])
    assert s["improved_count"] == 0 and s["unchanged_count"] == 2 and s["degraded_count"] == 0
    assert s["average_rank_delta"] == 0.0 and s["median_rank_delta"] == 0.0
    assert s["first_relevant_rank_delta"] == 0


def test_top_k_outside_still_full_rank_not_missing():
    base = list(range(1, 31))
    rer = list(range(1, 31))
    rer.remove(25)
    rer.insert(21, 25)  # rank 25 → 22（仍在 pool 内，不得当作 missing）
    s = rd.query_summary("q", base, rer, [25])
    it = s["items"][0]
    assert it["baseline_rank"] == 25 and it["rerank_rank"] == 22
    assert it["status"] == rd.IMPROVED and it["rank_delta"] == 3


# ---------------------------------------------------------------------------
# 3) missing / 空输入 / 重复 ID
# ---------------------------------------------------------------------------
def test_missing_relevant_statuses():
    base = ["A", "B"]
    rer = ["A", "C"]  # 同时制造 pool mismatch
    res = rd.compute_rank_delta(base, rer, ["X", "B", "C", "A"])
    m = _by_id(res["items"])
    assert m["X"]["status"] == rd.MISSING_BOTH and m["X"]["baseline_rank"] is None
    assert m["X"]["rank_delta"] is None
    assert m["B"]["status"] == rd.MISSING_RERANK and m["B"]["rerank_rank"] is None
    assert m["C"]["status"] == rd.MISSING_BASELINE and m["C"]["baseline_rank"] is None
    assert m["A"]["status"] == rd.UNCHANGED and m["A"]["rank_delta"] == 0


def test_empty_candidates_both_sides():
    res = rd.compute_rank_delta([], [], [1, 2])
    assert res["pool_consistent"] is True
    assert all(it["status"] == rd.MISSING_BOTH for it in res["items"])
    s = rd.query_summary("q", [], [], [1, 2])
    assert s["relevant_total"] == 2 and s["missing_both_count"] == 2
    assert s["baseline_first_relevant_rank"] is None
    assert s["rerank_first_relevant_rank"] is None
    assert s["first_relevant_rank_delta"] is None
    assert s["baseline_reciprocal_rank"] == 0.0 and s["rerank_reciprocal_rank"] == 0.0


def test_empty_relevant_ids():
    s = rd.query_summary("q", ["a"], ["a"], [])
    assert s["relevant_total"] == 0 and s["improved_count"] == 0
    assert s["average_rank_delta"] is None and s["first_relevant_rank_delta"] is None


def test_duplicate_ids_in_candidates():
    base = [1, 1, 2, 3]
    rer = [2, 1, 1, 3]
    res = rd.compute_rank_delta(base, rer, [1])
    assert res["duplicates"]["baseline"] == [1]
    assert res["duplicates"]["rerank"] == [1]
    it = res["items"][0]
    assert it["baseline_rank"] == 1 and it["rerank_rank"] == 2  # 取第一次出现位置
    assert it["rank_delta"] == -1


def test_pool_mismatch_reported_and_excluded_globally():
    s1 = rd.query_summary("bad", [1, 2, 3], [1, 2, 4], [1])
    assert s1["pool_consistent"] is False
    assert s1["pool_diff"]["only_baseline"] == [3]
    assert s1["pool_diff"]["only_rerank"] == [4]
    s2 = rd.query_summary("ok", [1, 2], [1, 2], [2])
    g = rd.global_summary([s1, s2])
    assert g["queries_total"] == 2 and g["queries_consistent"] == 1
    assert len(g["queries_pool_mismatch"]) == 1 and g["queries_pool_mismatch"][0]["query"] == "bad"
    assert g["total_relevant_images"] == 1  # 不一致 query 的 item 不进全局统计


# ---------------------------------------------------------------------------
# 4) 全局统计 / pair 分离 / 确定性 / 表渲染
# ---------------------------------------------------------------------------
def test_global_summary_counts_and_rates():
    # q1：relevant {4,6}；4: rank4→3(+1)、6: rank6→1(+5) → 2 improved
    s1 = rd.query_summary("q1", [1, 2, 3, 4, 5, 6], [6, 1, 4, 2, 3, 5], [4, 6])
    # q2：{b1,b3} 排名不变 → 2 unchanged
    s2 = rd.query_summary("q2", ["b1", "b2", "b3", "b4"],
                          ["b1", "b2", "b3", "b4"], ["b1", "b3"])
    g = rd.global_summary([s1, s2])
    assert g["total_relevant_images"] == 4
    assert g["improved_count"] == 2 and g["unchanged_count"] == 2 and g["degraded_count"] == 0
    assert _approx(g["improvement_rate"], 0.5)
    assert g["degradation_rate"] == 0.0
    assert _approx(g["average_rank_delta"], 1.5)
    assert _approx(g["median_rank_delta"], 0.5)
    assert g["query_level"]["improved_queries"] == 1
    assert g["query_level"]["unchanged_queries"] == 1


def test_pair_separation():
    qr1 = {"query": "q1", "baseline_ids": [1, 2, 3], "rerank_ids": [1, 2, 3],
           "relevant_ids": [1]}
    qr2 = {"query": "q1", "baseline_ids": [1, 2, 3], "rerank_ids": [3, 2, 1],
           "relevant_ids": [1]}
    pair_r1 = rd.evaluate_pair("Baseline vs R1", [qr1])
    pair_r2 = rd.evaluate_pair("Baseline vs R2", [qr2])
    g1 = rd.global_summary(pair_r1 + pair_r2, pair="Baseline vs R1")
    g2 = rd.global_summary(pair_r1 + pair_r2, pair="Baseline vs R2")
    assert g1["queries_total"] == 1 and g1["unchanged_count"] == 1
    assert g2["queries_total"] == 1 and g2["degraded_count"] == 1
    assert g2["query_level"]["degraded_queries"] == 1


def test_deterministic_repeated_runs():
    base = ["A", "B", "C", "D", "E"]
    rer = ["C", "A", "E", "B", "D"]
    a = rd.query_summary("q", base, rer, ["A", "C"])
    b = rd.query_summary("q", base, rer, ["A", "C"])
    assert a == b


def test_single_candidate_pool():
    s = rd.query_summary("q", ["only"], ["only"], ["only"])
    assert s["relevant_total"] == 1 and s["unchanged_count"] == 1
    assert s["first_relevant_rank_delta"] == 0


def test_format_tables_basic():
    s = rd.query_summary("q1", [1, 2, 3], [3, 2, 1], [1, 2, 3])
    qt = rd.format_query_table([s])
    assert "| Query |" in qt and "Baseline First Rank" in qt
    rr = rd.format_first_rr_table([s])
    assert "Baseline RR" in rr
    imp = rd.format_movement_table(rd.movement_rows([s], rd.IMPROVED, 10), rd.IMPROVED)
    deg = rd.format_movement_table(rd.movement_rows([s], rd.DEGRADED, 10), rd.DEGRADED)
    assert "Top Improvements" in imp and "Top Degradations" in deg


# ---------------------------------------------------------------------------
def _main():
    failed = passed = 0
    g = globals()
    for name in sorted(g):
        if name.startswith("test_"):
            try:
                g[name]()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n== rerank_delta: {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

