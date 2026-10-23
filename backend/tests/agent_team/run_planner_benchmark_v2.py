"""Planner Benchmark v2 — Real Runner（Evaluation only）。

60 queries × 2 runs = 120 次真实 Planner 调用（deepseek-chat, temperature=0, 参数不变）。
每 run：gate.decide（无 API）→ PlannerAgent.create_plan → PlanValidator.validate(vision=True) → score_plan。
固定 capability snapshot：vision_available=True（runtime patch，不改 config.yaml）。

运行（repo 根，真实 API 成本≈120 次 LLM call）：
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/run_planner_benchmark_v2.py
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
from backend.ai_func.recommendation.orchestration import ComplexityGate, PlanValidator, PlannerAgent
from backend.tests.agent_team.planner_benchmark_v2 import GOLD_V2_QUERIES, integrity_check
from backend.tests.agent_team.routing_benchmark_28 import score_plan

OUT = os.path.join(_REPO_ROOT, "planner_benchmark_v2_results.json")


class _VisionOn:
    def __enter__(self):
        self._orig = (models_mod.vision_available, planner_mod.vision_available)
        models_mod.vision_available = lambda: True
        planner_mod.vision_available = lambda: True
        return self

    def __exit__(self, *a):
        models_mod.vision_available, planner_mod.vision_available = self._orig


def _has_ancestor(tasks, node_ids, target_types, visited=None):
    by_id = {t["task_id"]: t for t in tasks}
    visited = set(visited or set())
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


def run_planner_once(planner, validator, q, run_index):
    row = {
        "query_id": q["id"], "query": q["query"], "difficulty": q["difficulty"],
        "run_index": run_index,
        "gate_route": None, "route": None,
        "raw_plan_valid": None, "task_types": [], "task_ids": [], "dependencies": {},
        "validator_result": None, "validation_errors": [], "exception_type": None,
        "safe_error": None, "planner_latency_ms": None, "llm_call_count": None,
        "score": None, "dependency_ok": None,
        "expected_task_types": q["expected_task_types"],
        "expected_dependencies": q["expected_dependencies"],
        "expected_synthesis_task": q["expected_synthesis_task"],
    }
    t0 = time.time()
    try:
        draft = planner.create_plan(goal=q["query"])
        row["planner_latency_ms"] = int((time.time() - t0) * 1000)
        row["llm_call_count"] = 1
        row["task_types"] = [t.task_type.value for t in draft.tasks]
        row["task_ids"] = [t.task_id for t in draft.tasks]
        row["dependencies"] = {t.task_id: list(t.depends_on) for t in draft.tasks}
        row["raw_plan_valid"] = True
        vres = validator.validate(draft, vision_available_flag=True)
        row["validator_result"] = "valid" if vres.valid else "invalid"
        row["validation_errors"] = list(vres.errors)
    except Exception as e:
        row["planner_latency_ms"] = int((time.time() - t0) * 1000)
        row["raw_plan_valid"] = False
        row["exception_type"] = type(e).__name__
        row["safe_error"] = str(e)[:160]
    row["score"] = score_plan(q, row["task_types"], row["validator_result"] == "valid")
    if row["raw_plan_valid"] and row["validator_result"] == "valid":
        tasks = [{"task_id": tid, "task_type": tt,
                  "depends_on": row["dependencies"].get(tid, [])}
                 for tid, tt in zip(row["task_ids"], row["task_types"])]
        row["dependency_ok"] = True
        for t in tasks:
            if t["task_type"] in ("analyze_visual", "synthesize"):
                if not _has_ancestor(tasks, t["depends_on"], {"retrieve_images", "filter_metadata", "group_images"}):
                    row["dependency_ok"] = False
    return row


def main():
    errs = integrity_check()
    if errs:
        print("INTEGRITY FAIL:", errs)
        return 2
    print("integrity OK: 60 queries")
    print("预计真实 Planner 调用 = 60 × 2 = 120 次（deepseek-chat，temperature=0，参数不变）")

    gate = ComplexityGate()
    gate_rows = []
    for q in GOLD_V2_QUERIES:
        try:
            route = gate.decide(query=q["query"]).route.value
        except Exception:
            route = "legacy"
        gate_rows.append({"query_id": q["id"], "query": q["query"], "difficulty": q["difficulty"],
                          "expected_route": q["expected_route"], "actual_route": route})

    runs = []
    with _VisionOn():
        planner = PlannerAgent()
        validator = PlanValidator()
        for q in GOLD_V2_QUERIES:
            for run_index in (1, 2):
                row = run_planner_once(planner, validator, q, run_index)
                runs.append(row)
                print(f"  q{q['id']}.{run_index} [{q['difficulty']}] valid={row['validator_result']} "
                      f"tasks={row['task_types']} err={(row['safe_error'] or row['validation_errors'][:1])}",
                      flush=True)
    # 立即落盘
    result = {"gate_rows": gate_rows, "runs": runs}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
