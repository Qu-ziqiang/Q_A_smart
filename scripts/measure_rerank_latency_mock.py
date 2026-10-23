"""A1/A2 仪表（mock VLM）：批处理 vs 逐张墙钟；分差触发跳过率。

不打真实 GLM。真测需 MM_RERANK_* + 标注集。
"""

from __future__ import annotations

import os
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from backend.ai_func.recommendation.evaluation.latency import StageTimer
from backend.ai_func.recommendation.orchestration.model_reranker import MultimodalReranker
from backend.ai_func.recommendation.orchestration.opt_flags import (
    reset_opt_stats,
    should_trigger_mm_rerank,
)
from backend.ai_func.recommendation.orchestration.workers import ImageCandidate


class _SleepProvider:
    name = "sleep"

    def __init__(self, per_image=0.05, batch=0.08):
        self.per_image = per_image
        self.batch = batch

    def relevance(self, query, image):
        time.sleep(self.per_image)
        return 0.5

    def rank_batch(self, query, candidates):
        time.sleep(self.batch)
        return list(range(len(candidates)))


def _cands(n=10):
    return [ImageCandidate(id=i, score=0.9 - i * 0.01, title=str(i),
                           description="", tags=[], metadata={}) for i in range(n)]


def main() -> int:
    timer = StageTimer()
    cands = _cands(10)
    os.environ.pop("SM_RERANK_BATCH", None)
    rk = MultimodalReranker(_SleepProvider())
    with timer.span("r2_sequential_10"):
        rk.rerank("q", cands)
    os.environ["SM_RERANK_BATCH"] = "1"
    rk2 = MultimodalReranker(_SleepProvider())
    with timer.span("r2_batch_10"):
        rk2.rerank("q", cands)
    os.environ.pop("SM_RERANK_BATCH", None)

    reset_opt_stats()
    os.environ["SM_RERANK_GAP_TRIGGER"] = "1"
    os.environ["SM_RERANK_GAP_TAU"] = "0.08"
    close = _cands(5)
    for c, s in zip(close, (0.51, 0.50, 0.49, 0.48, 0.47)):
        c.score = s
    wide = _cands(5)
    for c, s in zip(wide, (0.95, 0.70, 0.60, 0.50, 0.40)):
        c.score = s
    trig = should_trigger_mm_rerank(close)
    skip = not should_trigger_mm_rerank(wide)
    os.environ.pop("SM_RERANK_GAP_TRIGGER", None)

    report = {
        "latency": timer.summary(),
        "a2_close_triggers": trig,
        "a2_wide_skips": skip,
        "note": "mock sleep，不是 GLM 实测。目标对照：逐张 ~10*per_image vs 一次 batch。",
    }
    out = os.path.join(
        _REPO, "backend", "ai_func", "recommendation",
        "evaluation", "results", "rerank_latency_mock.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import json
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
