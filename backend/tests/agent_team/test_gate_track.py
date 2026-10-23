"""P0 Gate 评测轨 + 反例切分。零 LLM。"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.evaluation.gate_track import (
    contrast_split,
    load_contrast,
    run_gate_track,
)
from backend.ai_func.recommendation.evaluation.latency import StageTimer, percentile


def test_contrast_split_is_deterministic():
    assert contrast_split("neg_01") == contrast_split("neg_01")
    splits = {contrast_split(f"neg_{i:02d}") for i in range(1, 61)}
    assert splits <= {"dev", "test"}


def test_contrast_has_dev_and_test():
    rows = load_contrast()
    assert len(rows) >= 100
    n_test = sum(1 for r in rows if r["split"] == "test")
    assert n_test >= 30, n_test
    assert any(r["split"] == "dev" for r in rows)


def test_gate_track_gold_fn_zero():
    report = run_gate_track()
    assert report["gold_28"]["fn"] == 0
    assert report["gold_28"]["fp"] == 0


def test_gate_track_contrast_test_frozen_no_fp():
    """test 切分冻结：当前规则不得在 test 上误伤。"""
    report = run_gate_track()
    assert report["contrast_test"]["fp"] == 0, report["contrast_test_fp_ids"]


def test_gate_latency_under_5ms_p95():
    report = run_gate_track()
    p95 = float(report["all"]["latency"].get("p95_ms") or 0)
    assert p95 < 5.0, p95


def test_percentile_helper():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    t = StageTimer()
    with t.span("x"):
        pass
    assert t.summary()["x"]["n"] == 1
