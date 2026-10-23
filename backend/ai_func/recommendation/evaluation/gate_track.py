"""P0 Gate 评测轨：先分流再记 FN/FP，不进 Judge。

正例：routing_benchmark_28 的 expected_route。
反例：gate_contrast_legacy.json（字面像视觉、应 Legacy）。
split：contrast id 哈希 %10 < 7 = dev，其余 test。调规则只碰 dev。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

from ..orchestration.gate import ComplexityGate
from ..orchestration.models import RouteType
from .latency import StageTimer

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRAST = os.path.join(_HERE, "datasets", "gate_contrast_legacy.json")


def contrast_split(qid: str) -> str:
    h = int(hashlib.sha256(qid.encode("utf-8")).hexdigest(), 16)
    return "dev" if (h % 10) < 7 else "test"


def load_contrast(path: str = _CONTRAST) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for q in data.get("queries") or []:
        qid = str(q["id"])
        rows.append({
            "id": qid,
            "query": q["query"],
            "expected_route": "legacy",
            "split": contrast_split(qid),
            "source": "contrast",
        })
    return rows


def load_gold_28() -> List[Dict[str, Any]]:
    from backend.tests.agent_team.routing_benchmark_28 import GOLD_QUERIES

    rows = []
    for q in GOLD_QUERIES:
        rows.append({
            "id": f"gold_{q['id']}",
            "query": q["query"],
            "expected_route": q["expected_route"],
            "split": "gold",
            "source": "routing_28",
            "vision_required": q.get("vision_required"),
        })
    return rows


def evaluate_rows(rows: List[Dict[str, Any]], gate: ComplexityGate | None = None
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gate = gate or ComplexityGate()
    timer = StageTimer()
    records = []
    fn = fp = tp_team = tn_legacy = 0
    for row in rows:
        with timer.span("gate"):
            decision = gate.decide(query=row["query"])
        got = decision.route.value if isinstance(decision.route, RouteType) else str(decision.route)
        exp = row["expected_route"]
        err = None
        if exp == "team" and got != "team":
            fn += 1
            err = "FN"
        elif exp == "legacy" and got != "legacy":
            fp += 1
            err = "FP"
        elif got == "team":
            tp_team += 1
        else:
            tn_legacy += 1
        records.append({
            **row,
            "got_route": got,
            "error": err,
            "matched_features": list(decision.matched_features or []),
        })
    n = len(rows) or 1
    team_n = sum(1 for r in records if r["got_route"] == "team")
    summary = {
        "n": len(rows),
        "fn": fn,
        "fp": fp,
        "fn_rate": round(fn / n, 4),
        "fp_rate": round(fp / n, 4),
        "team_rate": round(team_n / n, 4),
        "legacy_rate": round(1.0 - team_n / n, 4),
        "planner_calls_saved_vs_all_team": team_n,  # 实际会调 Planner 的条数
        "legacy_skip_planner": len(rows) - team_n,
        "latency": timer.summary().get("gate", {}),
    }
    return records, summary


def run_gate_track() -> Dict[str, Any]:
    gold = load_gold_28()
    contrast = load_contrast()
    gold_rec, gold_sum = evaluate_rows(gold)
    dev = [r for r in contrast if r["split"] == "dev"]
    test = [r for r in contrast if r["split"] == "test"]
    _, dev_sum = evaluate_rows(dev)
    test_rec, test_sum = evaluate_rows(test)
    all_rec, all_sum = evaluate_rows(gold + contrast)
    return {
        "gold_28": gold_sum,
        "contrast_dev": dev_sum,
        "contrast_test": test_sum,
        "all": all_sum,
        "gold_fn_ids": [r["id"] for r in gold_rec if r["error"] == "FN"],
        "contrast_test_fp_ids": [r["id"] for r in test_rec if r["error"] == "FP"],
        "note": (
            "产品指标：FN/FP、TEAM 率、Legacy 跳过 Planner 条数、Gate 延迟。"
            "不进 Judge。调正则只许看 contrast_dev。"
        ),
    }
