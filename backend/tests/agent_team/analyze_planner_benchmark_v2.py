"""Analyze Planner Benchmark v2（+ 与 v1 对比）——离线、无 API。

读取 gold（planner_benchmark_v2.py / routing_benchmark_28.py）与 results JSON。
输出：TEMP/planner_benchmark_v2_analysis.json + 控制台。
"""

import json
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))
_AGENT_TEAM = os.path.dirname(os.path.abspath(__file__))
if _AGENT_TEAM not in sys.path:
    sys.path.insert(0, _AGENT_TEAM)

from planner_benchmark_v2 import GOLD_V2_QUERIES
from routing_benchmark_28 import GOLD_QUERIES as GOLD_V1

TMP = os.environ.get("TEMP", r"C:\Users\31669\AppData\Local\Temp")


def norm_deps(tasks):
    by = {}
    for t in tasks:
        by.setdefault(t["task_type"], set()).update(t["depends_on"])
    return {k: sorted(v) for k, v in by.items()}


def graph_signature(tasks):
    return frozenset((t["task_type"], frozenset(t["depends_on"])) for t in tasks)


def parse_gold(gold):
    out = {}
    for q in gold:
        if "expected_dependencies" in q and q["expected_dependencies"]:
            gd = q["expected_dependencies"]
        else:  # v1：由有序 expected_task_types 推导线性链 gold
            gd = {}
            prev = None
            for t in q["expected_task_types"]:
                gd[t] = [prev] if prev else []
                prev = t
        tasks = [{"task_type": t, "depends_on": gd.get(t, [])} for t in q["expected_task_types"]]
        out[q["id"]] = {
            "types": q["expected_task_types"],
            "vision_required": bool(q["vision_required"]),
            "deps": norm_deps(tasks),
            "difficulty": q.get("difficulty", "v1"),
        }
    return out


def row_tasks(r):
    tids = r["task_ids"]
    types_by_id = dict(zip(tids, r["task_types"]))
    return [{"task_type": tt,
             "depends_on": sorted({types_by_id[d] for d in r["dependencies"].get(tid, []) if d in types_by_id})}
            for tid, tt in zip(tids, r["task_types"])]


def run_features(r, gq):
    f = {"parsed": False, "valid": r["validator_result"] == "valid",
         "task_set_ok": bool(r["score"]["exact_task_types_match"]),
         "vision_selected": bool(r["score"]["vision_selected"]),
         "vision_ok": bool(r["score"]["vision_required"]) == bool(r["score"]["vision_selected"]),
         "task_count": len(r["task_types"]),
         "signature": None, "dep_exact": False,
         "dep_all_present": False, "dep_missing": 0, "plan_exact": False,
         "synthesis_ok": False, "primary_err": "none"}
    if r.get("exception_type") or not r["task_ids"]:
        f["primary_err"] = "parse_exception"
        return f
    tasks = row_tasks(r)
    f["parsed"] = True
    f["signature"] = (graph_signature(tasks), r["validator_result"])
    dep_map = norm_deps(tasks)
    f["dep_exact"] = dep_map == gq["deps"]
    f["dep_all_present"] = all(set(dep_map.get(t, [])) >= set(ds) for t, ds in gq["deps"].items())
    f["dep_missing"] = sum(len(set(gq["deps"][t]) - set(dep_map.get(t, []))) for t in gq["deps"])
    synth = [tt for tt in r["task_types"] if tt == "synthesize"]
    f["synthesis_ok"] = len(synth) == 1
    f["plan_exact"] = (f["task_set_ok"] and f["vision_ok"] and f["dep_exact"] and f["synthesis_ok"])
    if not f["task_set_ok"]:
        f["primary_err"] = "task_selection"
    elif not f["vision_ok"]:
        f["primary_err"] = "vision_decision"
    elif not f["dep_exact"]:
        f["primary_err"] = "dependency"
    elif not f["plan_exact"]:
        f["primary_err"] = "other"
    return f


def analyze(gold, runs, label):
    g = parse_gold(gold)
    n = len(runs)
    agg = {"label": label, "n_queries": len(g), "n_runs": n,
           "counts": {}, "per_diff": {}, "consistency": {"consistent": 0, "total": 0, "pairs": []}}
    c = agg["counts"]
    per_diff = agg["per_diff"]
    count_errs = []
    for r in runs:
        gq = g[r["query_id"]]
        diff = gq["difficulty"]
        d = per_diff.setdefault(diff, {"runs": 0})
        d["runs"] += 1
        f = run_features(r, gq)
        for k in ("parsed", "valid", "task_set_ok", "vision_ok", "dep_exact", "dep_all_present",
                  "plan_exact", "synthesis_ok"):
            c[k] = c.get(k, 0) + (1 if f[k] else 0)
            d[k] = d.get(k, 0) + (1 if f[k] else 0)
        c["vision_required"] = c.get("vision_required", 0) + (1 if gq["vision_required"] else 0)
        c["vision_selected"] = c.get("vision_selected", 0) + (1 if f["vision_selected"] else 0)
        c["dep_missing"] = c.get("dep_missing", 0) + f["dep_missing"]
        c["primary_err_" + f["primary_err"]] = c.get("primary_err_" + f["primary_err"], 0) + 1
        count_errs.append(abs(f["task_count"] - len(gq["types"])))
        # consistency pair
        sg = f["signature"]
        if sg is not None:
            sg0 = tuple(sorted((tt, tuple(sorted(deps))) for tt, deps in sg[0]))
            sg = (sg0, sg[1])
        agg["consistency"]["pairs"].append({"qid": r["query_id"], "run": r["run_index"], "sig": sg})
    pairs = agg["consistency"]["pairs"]
    by_q = {}
    for p in pairs:
        by_q.setdefault(p["qid"], []).append(p["sig"])
    for qid, sigs in sorted(by_q.items()):
        agg["consistency"]["total"] += 1
        if len(sigs) == 2 and sigs[0] is not None and sigs[1] is not None and sigs[0] == sigs[1]:
            agg["consistency"]["consistent"] += 1
    v_req = c.get("vision_required", 0)
    v_sel = c.get("vision_selected", 0)
    v_tp = c.get("vision_ok", 0)  # placeholder replaced below
    # vision TP: vision required & selected 且 vision_ok 覆盖；单独算
    v_tp = sum(1 for r in runs
               if g[r["query_id"]]["vision_required"] and run_features(r, g[r["query_id"]])["vision_selected"])
    v_fp = sum(1 for r in runs
               if (not g[r["query_id"]]["vision_required"]) and run_features(r, g[r["query_id"]])["vision_selected"])
    m = {"exact_task_types_match": round(c["task_set_ok"] / n, 4),
         "valid_plan_rate": round(c["valid"] / n, 4),
         "dependency_exact_match": round(c["dep_exact"] / n, 4),
         "dependency_all_expected_present": round(c["dep_all_present"] / n, 4),
         "dep_missing_edges_total": c.get("dep_missing", 0),
         "plan_exact_match": round(c["plan_exact"] / n, 4),
         "vision_recall": round(v_tp / v_req, 4) if v_req else None,
         "vision_precision": round(v_tp / v_sel, 4) if v_sel else None,
         "vision_positive_runs": v_req, "vision_selected_runs": v_sel,
         "vision_false_positive_runs": v_fp,
         "avg_tasks_per_plan": round(sum(len(r["task_types"]) for r in runs if r["task_ids"]) /
                                    max(1, sum(1 for r in runs if r["task_ids"])), 2),
         "avg_task_count_error": round(sum(count_errs) / max(1, len(count_errs)), 2),
         "plan_consistency_rate": round(agg["consistency"]["consistent"] / agg["consistency"]["total"], 4),
         "error_attribution": {k.split("_", 1)[1]: v for k, v in c.items() if k.startswith("primary_err_")},
         }
    agg["metrics"] = m
    for d in per_diff.values():
        dn = d["runs"]
        d["exact_task_types_match"] = round(d.get("task_set_ok", 0) / dn, 4)
        d["valid_plan_rate"] = round(d.get("valid", 0) / dn, 4)
        d["dependency_exact_match"] = round(d.get("dep_exact", 0) / dn, 4)
        d["plan_exact_match"] = round(d.get("plan_exact", 0) / dn, 4)
        for k in list(d):
            if k not in ("runs", "exact_task_types_match", "valid_plan_rate",
                         "dependency_exact_match", "plan_exact_match"):
                d.pop(k, None)
    return agg


def main():
    v2 = json.load(open(os.path.join(_REPO_ROOT, "planner_benchmark_v2_results.json"), encoding="utf-8"))
    v1 = json.load(open(os.path.join(_REPO_ROOT, "planner_routing_benchmark_results.json"), encoding="utf-8"))
    a_v2 = analyze(GOLD_V2_QUERIES, v2["runs"], "v2")
    a_v1 = analyze(GOLD_V1, v1["runs"], "v1")
    out = {"v1": a_v1, "v2": a_v2}
    with open(os.path.join(TMP, "planner_benchmark_v2_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("=== V1 ===")
    print(json.dumps(a_v1["metrics"], ensure_ascii=False, indent=1))
    print("=== V2 ===")
    print(json.dumps(a_v2["metrics"], ensure_ascii=False, indent=1))
    print("=== V2 per difficulty ===")
    print(json.dumps(a_v2["per_diff"], ensure_ascii=False, indent=1))
    fn = sum(1 for g in v2["gate_rows"] if g["expected_route"] == "team" and g["actual_route"] == "legacy")
    fp = sum(1 for g in v2["gate_rows"] if g["expected_route"] == "legacy" and g["actual_route"] == "team")
    print("V2 GATE: FN=", fn, "FP=", fp)
    out["v2_gate"] = {"fn": fn, "fp": fp}
    with open(os.path.join(TMP, "planner_benchmark_v2_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

