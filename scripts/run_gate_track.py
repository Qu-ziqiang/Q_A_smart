"""P0 Gate 评测轨。零 LLM。

  venv\\Scripts\\python.exe scripts/run_gate_track.py
"""

from __future__ import annotations

import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from backend.ai_func.recommendation.evaluation.gate_track import run_gate_track
from backend.ai_func.recommendation.evaluation.latency import format_summary


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="P0 Gate track")
    ap.add_argument(
        "--out-dir",
        default=os.path.join(
            _REPO, "backend", "ai_func", "recommendation",
            "evaluation", "results", "gate_track",
        ),
    )
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    report = run_gate_track()
    path = os.path.join(out_dir, "gate_track.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md = os.path.join(out_dir, "gate_track.md")
    lat = report["all"].get("latency") or {}
    lines = [
        "# Gate Track (P0)",
        "",
        report["note"],
        "",
        f"- gold_28 FN={report['gold_28']['fn']} FP={report['gold_28']['fp']} TEAM={report['gold_28']['team_rate']}",
        f"- contrast_dev FP={report['contrast_dev']['fp']} n={report['contrast_dev']['n']}",
        f"- contrast_test FP={report['contrast_test']['fp']} n={report['contrast_test']['n']} ids={report['contrast_test_fp_ids']}",
        f"- all TEAM={report['all']['team_rate']} Legacy skip Planner={report['all']['legacy_skip_planner']}",
        f"- Gate latency ms p50={lat.get('p50_ms')} p95={lat.get('p95_ms')} p99={lat.get('p99_ms')}",
        "",
        "## Latency",
        format_summary({"gate": lat} if lat else {}),
        "",
    ]
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[gate_track] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
