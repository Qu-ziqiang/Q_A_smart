import json
import os
import sys

_REPO = "D:/document/py_prj/ImageTextSystem/Smartlmager-sqlite-vec"
sys.path.insert(0, _REPO)
from backend.tests.agent_team.routing_benchmark_28 import GOLD_QUERIES

with open(os.path.join(_REPO, "planner_routing_benchmark_results.json"), encoding="utf-8") as f:
    data = json.load(f)

gold = {q["id"]: q for q in GOLD_QUERIES}
lines = []
lines.append("| id | cat | gold(types) | r1 types | r2 types | r1 valid | r2 valid | r1 err | r2 err | vision(r1/r2) |")
lines.append("|---|-----|-------------|----------|----------|----------|----------|--------|--------|---------------|")

inconsistent = []
for qid in sorted(gold):
    g = gold[qid]
    rs = {r["run_index"]: r for r in data["runs"] if r["query_id"] == qid}
    r1, r2 = rs[1], rs[2]
    t1 = ",".join(sorted(r1["task_types"])) or "(none)"
    t2 = ",".join(sorted(r2["task_types"])) or "(none)"
    e1 = "|".join(r1["validation_errors"][:2])[:60] if r1["validation_errors"] else (r1["safe_error"] or "-")
    e2 = "|".join(r2["validation_errors"][:2])[:60] if r2["validation_errors"] else (r2["safe_error"] or "-")
    v1 = r1["validator_result"] or "exception"
    v2 = r2["validator_result"] or "exception"
    vis = f"{'Y' if r1['score']['vision_selected'] else 'N'}/{'Y' if r2['score']['vision_selected'] else 'N'}"
    lines.append(f"| q{qid} | {g['category']} | {g['expected_route']} {','.join(sorted(g['expected_task_types']))} | {t1} | {t2} | {v1} | {v2} | {e1} | {e2} | {vis} |")
    if set(r1["task_types"]) != set(r2["task_types"]) or (v1 != v2):
        inconsistent.append(qid)

lines.append("")
lines.append(f"INCONSISTENT query_ids (r1 != r2 task set or validity): {inconsistent}")

# vision 分类样本
lines.append("\n## Vision FN / FP / invalid 样本")
vp = data["metrics"]["vision_false_negative"]
fp_ids = []
fn_ids = []
inv_ids = []
for qid in sorted(gold):
    rs = {r["run_index"]: r for r in data["runs"] if r["query_id"] == qid}
    for ri in (1, 2):
        r = rs[ri]
        if r["score"]["vision_false_positive"]:
            fp_ids.append(f"q{qid}.{ri}")
        if r["score"]["vision_false_negative"]:
            fn_ids.append(f"q{qid}.{ri}")
        if r["validator_result"] != "valid":
            inv_ids.append(f"q{qid}.{ri}")
lines.append(f"Vision FN: {fn_ids}")
lines.append(f"Vision FP: {fp_ids}")
lines.append(f"Invalid runs: {inv_ids}")

# gate rows
lines.append("\n## Gate rows (Layer A)")
for g in data["gate_rows"]:
    flag = "  <-- FN" if g["expected_route"] == "team" and g["actual_route"] == "legacy" else ""
    flag += "  <-- FP" if g["expected_route"] == "legacy" and g["actual_route"] == "team" else ""
    lines.append(f"q{g['query_id']} {g['category']} expected={g['expected_route']} actual={g['actual_route']}{flag}")

with open(os.path.join(_REPO, "benchmark_analysis_table.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written")
