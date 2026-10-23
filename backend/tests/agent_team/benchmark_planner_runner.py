"""Planner Capability Routing — Real Benchmark Runner（Evaluation only）。

28 queries × 2 runs = 56 次真实 Planner 调用（仅文本 LLM deepseek-chat，temperature=0）。
不调用 Vision / Retrieval / Synthesizer / Executor；不改配置 / DB / 生产代码。

每次 run：query → PlannerAgent.create_plan → PlanValidator.validate(vision_available_flag=True) → score_plan。
Layer A：ComplexityGate.decide（无 API）。
Layer C：gate → planner → validator 生产路径分析。

固定 capability snapshot：vision_available=True（runtime patch，仅本进程；不改 config.yaml）。

运行（repo 根，需显式授权真实 API）：
    $env:PYTHONIOENCODING='utf-8'
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/benchmark_planner_runner.py
"""

import json
import os
import sys
import time

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

import backend.ai_func.recommendation.orchestration.models as models_mod
import backend.ai_func.recommendation.orchestration.planner as planner_mod
from backend.ai_func.recommendation.orchestration import (
    ComplexityGate,
    PlanValidator,
    PlannerAgent,
)
from backend.ai_func.recommendation.orchestration.models import TaskType
from backend.tests.agent_team.routing_benchmark_28 import (
    GOLD_QUERIES,
    VISION_POSITIVE_CATEGORIES,
    score_plan,
    vision_negative_set,
    vision_positive_set,
)


class _VisionOn:
    def __enter__(self):
        self._orig = (models_mod.vision_available, planner_mod.vision_available)
        models_mod.vision_available = lambda: True
        planner_mod.vision_available = lambda: True
        return self

    def __exit__(self, *a):
        models_mod.vision_available, planner_mod.vision_available = self._orig


# ---------------------------------------------------------------------------
# 完整性检查（不修改 gold）
# ---------------------------------------------------------------------------
def integrity_check():
    errors = []
    if len(GOLD_QUERIES) != 28:
        errors.append(f"dataset 数量 {len(GOLD_QUERIES)} != 28")
    from collections import Counter
    cats = Counter(q["category"] for q in GOLD_QUERIES)
    if set(cats) != {"A", "B", "C", "D", "E", "F", "G"} or any(v != 4 for v in cats.values()):
        errors.append(f"类别结构异常 {dict(cats)}")
    ids = [q["id"] for q in GOLD_QUERIES]
    if ids != list(range(1, 29)):
        errors.append("id 不连续唯一")
    for q in GOLD_QUERIES:
        if not q.get("query") or not q.get("expected_task_types") or q.get("expected_route") not in ("legacy", "team"):
            errors.append(f"query #{q.get('id')} 缺字段")
        if "retrieve_images" not in q["expected_task_types"] or "synthesize" not in q["expected_task_types"]:
            errors.append(f"query #{q['id']} expected 缺 retrieve/synthesize")
    if len(vision_positive_set()) != 12 or len(vision_negative_set()) != 16:
        errors.append("vision positive/negative 集合大小异常")
    return errors


def _has_ancestor(tasks, node_ids, target_types, visited=None):
    by_id = {t["task_id"]: t for t in tasks}
    visited = visited or set()
    stack = list(node_ids)
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        t = by_id.get(cur)
        if t is None:
            continue
        if t["task_type"] in target_types:
            return True
        stack.extend(t.get("depends_on", []))
    return False


def _dependency_ok(tasks):
    """合法依赖检查：analyze_visual 与 synthesize 都必须（传递）可达非 vision 候选生产者。"""
    has_vision = any(t["task_type"] == "analyze_visual" for t in tasks)
    if not has_vision:
        return True
    producers = {TaskType.RETRIEVE_IMAGES.value, TaskType.GROUP_IMAGES.value,
                 TaskType.FILTER_METADATA.value}
    for t in tasks:
        if t["task_type"] == "analyze_visual" or t["task_type"] == TaskType.SYNTHESIZE.value:
            if not _has_ancestor(tasks, t.get("depends_on", []), producers):
                return False
    return True


def run_planner_once(planner, validator, query, run_index):
    row = {
        "query_id": query["id"], "category": query["category"], "query": query["query"],
        "run_index": run_index,
        "raw_plan_valid": None, "task_types": [], "task_ids": [], "dependencies": {},
        "route": None, "validator_result": None, "validation_errors": [],
        "planner_latency_ms": None, "llm_call_count": None,
        "exception_type": None, "safe_error": None,
    }
    t0 = time.time()
    try:
        draft = planner.create_plan(goal=query["query"])
        row["planner_latency_ms"] = int((time.time() - t0) * 1000)
        row["llm_call_count"] = 1
        row["task_types"] = [t.task_type.value for t in draft.tasks]
        row["task_ids"] = [t.task_id for t in draft.tasks]
        row["dependencies"] = {t.task_id: list(t.depends_on) for t in draft.tasks}
        row["raw_plan_valid"] = True
        r = validator.validate(draft, vision_available_flag=True)
        row["validator_result"] = "valid" if r.valid else "invalid"
        row["validation_errors"] = list(r.errors)
    except Exception as e:
        row["planner_latency_ms"] = int((time.time() - t0) * 1000)
        row["raw_plan_valid"] = False
        row["exception_type"] = type(e).__name__
        row["safe_error"] = str(e)[:160]
    row["score"] = score_plan(query, row["task_types"], row["validator_result"] == "valid")
    row["dependency_ok"] = None
    if row["raw_plan_valid"] and row["validator_result"] == "valid":
        row["dependency_ok"] = _dependency_ok([
            {"task_id": tid, "task_type": tt, "depends_on": row["dependencies"].get(tid, [])}
            for tid, tt in zip(row["task_ids"], row["task_types"])
        ])
    return row

# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def compute_metrics(runs):
    n = len(runs)
    exact = sum(1 for r in runs if r["score"]["exact_task_types_match"])
    valid = sum(1 for r in runs if r["validator_result"] == "valid")
    vision_req = [r for r in runs if r["score"]["vision_required"]]
    vision_sel = [r for r in runs if r["score"]["vision_selected"]]
    tp = sum(1 for r in vision_req if r["score"]["vision_selected"])
    fn = len(vision_req) - tp
    fp = sum(1 for r in runs if r["score"]["vision_false_positive"])
    recall = tp / len(vision_req) if vision_req else None
    precision = tp / len(vision_sel) if vision_sel else None
    task_counts = [len(r["task_types"]) for r in runs if r["validator_result"] == "valid"]
    avg_tasks = (sum(task_counts) / len(task_counts)) if task_counts else None
    gold_avg = sum(len(q["expected_task_types"]) for q in GOLD_QUERIES) / len(GOLD_QUERIES)
    dep_ok = sum(1 for r in runs if r["dependency_ok"] is True)
    dep_checked = sum(1 for r in runs if r["dependency_ok"] is not None)
    return {
        "n_runs": n,
        "exact_task_types_accuracy": round(exact / n, 4),
        "valid_plan_rate": round(valid / n, 4),
        "vision_positive_runs": len(vision_req),
        "vision_required_runs_recall_tp": tp,
        "vision_false_negative": fn,
        "vision_recall": round(recall, 4) if recall is not None else None,
        "vision_selected_runs": len(vision_sel),
        "vision_false_positive": fp,
        "vision_precision": round(precision, 4) if precision is not None else None,
        "avg_tasks_per_plan": round(avg_tasks, 2) if avg_tasks is not None else None,
        "gold_avg_tasks": round(gold_avg, 2),
        "dependency_ok_rate": round(dep_ok / dep_checked, 4) if dep_checked else None,
        "llm_calls": sum(1 for r in runs if r["llm_call_count"] == 1),
    }


def compute_gate_metrics(gate_rows):
    fn = sum(1 for r in gate_rows if r["expected_route"] == "team" and r["actual_route"] == "legacy")
    fp = sum(1 for r in gate_rows if r["expected_route"] == "legacy" and r["actual_route"] == "team")
    team_expected = [r for r in gate_rows if r["expected_route"] == "team"]
    team_reach = sum(1 for r in team_expected if r["actual_route"] == "team")
    return {
        "gate_false_negative": fn,
        "gate_false_positive": fp,
        "team_expected": len(team_expected),
        "team_reachability": f"{team_reach}/{len(team_expected)}",
        "vision_positive_gate_blocked": sum(
            1 for r in gate_rows if r["expected_route"] == "team" and r["actual_route"] == "legacy"
            and r["category"] in VISION_POSITIVE_CATEGORIES),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=== Planner Capability Routing Real Benchmark ===")
    errs = integrity_check()
    if errs:
        print("INTEGRITY FAIL:", errs)
        return 2
    print("integrity check: OK (28 queries / 7 categories / scorer deterministic / planner-only)")

    # pre-fix baseline availability
    base_files = [
        f for f in os.listdir(_REPO_ROOT)
        if ("baseline" in f.lower() or "benchmark_result" in f.lower()) and f.endswith((".json", ".log", ".txt"))
    ]
    print("REAL PRE-FIX BASELINE = NOT AVAILABLE" if not base_files
          else f"pre-fix baseline files: {base_files}")

    gate = ComplexityGate()
    gate_rows = []
    for q in GOLD_QUERIES:
        try:
            route = gate.decide(query=q["query"]).route.value
        except Exception as e:
            route = "legacy"
        gate_rows.append({"query_id": q["id"], "category": q["category"], "query": q["query"],
                          "expected_route": q["expected_route"], "actual_route": route})

    runs = []
    with _VisionOn():
        planner = PlannerAgent()
        validator = PlanValidator()
        for q in GOLD_QUERIES:
            for run_index in (1, 2):
                row = run_planner_once(planner, validator, q, run_index)
                runs.append(row)
                print(f"  run q{q['id']}.{run_index}: valid={row['validator_result']} "
                      f"tasks={row['task_types']} err={row['safe_error'] or row['validation_errors'][:1]}")

    # 立即落盘 runs（防后续计算崩溃丢失 56 次真实调用结果）
    partial_path = os.path.join(_REPO_ROOT, "planner_routing_benchmark_runs.json")
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "gate_rows": gate_rows}, f, ensure_ascii=False, indent=2)
    print(f"runs saved to {partial_path}")

    metrics = compute_metrics(runs)
    gate_metrics = compute_gate_metrics(gate_rows)

    result = {
        "pre_fix_baseline": "NOT AVAILABLE",
        "metrics": metrics,
        "gate_metrics": gate_metrics,
        "runs": runs,
        "gate_rows": gate_rows,
    }
    out_path = os.path.join(_REPO_ROOT, "planner_routing_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n=== METRICS ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n=== GATE (Layer A) ===")
    print(json.dumps(gate_metrics, ensure_ascii=False, indent=2))
    print(f"\nresults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

