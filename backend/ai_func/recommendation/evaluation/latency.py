"""阶段延迟仪表：P50/P95，与 Judge 脱钩。

用法：
    rec = StageTimer()
    with rec.span("gate"):
        gate.decide(...)
    rec.summary() -> {stage: {n, p50_ms, p95_ms, mean_ms}}
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


class StageTimer:
    def __init__(self) -> None:
        self._ms: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def span(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._ms[stage].append((time.perf_counter() - t0) * 1000.0)

    def record(self, stage: str, ms: float) -> None:
        self._ms[stage].append(float(ms))

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for stage, vals in sorted(self._ms.items()):
            out[stage] = {
                "n": len(vals),
                "mean_ms": round(sum(vals) / len(vals), 3) if vals else 0.0,
                "p50_ms": round(percentile(vals, 50), 3),
                "p95_ms": round(percentile(vals, 95), 3),
                "p99_ms": round(percentile(vals, 99), 3),
                "max_ms": round(max(vals), 3) if vals else 0.0,
            }
        return out


def format_summary(summary: Dict[str, Dict[str, float]]) -> str:
    lines = ["stage | n | mean | p50 | p95 | p99 | max", "---|---:|---:|---:|---:|---:|---:"]
    for stage, s in summary.items():
        lines.append(
            f"{stage} | {s['n']} | {s['mean_ms']} | {s['p50_ms']} | "
            f"{s['p95_ms']} | {s['p99_ms']} | {s['max_ms']}"
        )
    return "\n".join(lines)
