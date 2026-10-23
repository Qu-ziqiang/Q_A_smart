"""优化开关（全部默认关闭，旧路径保持 A/B 对照）。

环境变量（1/true/on 为开）：
  SM_RERANK_GAP_TRIGGER   A2：CLIP top1-top5 分差 < τ 才走 R2
  SM_RERANK_GAP_TAU       分差阈值，默认 0.08
  SM_RERANK_BATCH         A1：多图单次重排
  SM_RERANK_BATCH_MAX     单次最多图数，默认 10；更大则两轮淘汰
  SM_DAG_PARALLEL         B1：同层非视觉任务并行（最多 2）
  SM_PLAN_CACHE           C2：Planner 计划缓存
  SM_PLAN_LINT            P1：Gate 判简单时剥离可选 Task（可回退）
"""

from __future__ import annotations

import os
from typing import Any, List, Optional


def _on(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def rerank_gap_trigger_enabled() -> bool:
    return _on("SM_RERANK_GAP_TRIGGER")


def rerank_gap_tau() -> float:
    return _float("SM_RERANK_GAP_TAU", 0.08)


def rerank_batch_enabled() -> bool:
    return _on("SM_RERANK_BATCH")


def rerank_batch_max() -> int:
    return max(2, _int("SM_RERANK_BATCH_MAX", 10))


def dag_parallel_enabled() -> bool:
    return _on("SM_DAG_PARALLEL")


def plan_cache_enabled() -> bool:
    return _on("SM_PLAN_CACHE")


def plan_lint_enabled() -> bool:
    return _on("SM_PLAN_LINT")


def clip_top_gap(candidates: List[Any], top_k: int = 5) -> Optional[float]:
    """CLIP/检索分 top1 与 topK 的差；不足 2 个有效分返回 None。"""
    scores = []
    for c in candidates or []:
        s = getattr(c, "score", None)
        if s is None:
            continue
        try:
            scores.append(float(s))
        except (TypeError, ValueError):
            continue
    if len(scores) < 2:
        return None
    ranked = sorted(scores, reverse=True)
    top1 = ranked[0]
    tail = ranked[min(top_k, len(ranked)) - 1]
    return top1 - tail


def should_trigger_mm_rerank(candidates: List[Any], tau: Optional[float] = None) -> bool:
    """A2：flag 关 → 总是触发（旧行为）；flag 开 → 仅当 gap < τ。"""
    if not rerank_gap_trigger_enabled():
        return True
    gap = clip_top_gap(candidates)
    if gap is None:
        return True
    threshold = rerank_gap_tau() if tau is None else tau
    return gap < threshold


class _Counters:
    rerank_considered = 0
    rerank_triggered = 0
    rerank_skipped = 0
    batch_ok = 0
    batch_fallback = 0
    cache_hit = 0
    cache_miss = 0
    plan_lint_stripped = 0
    plan_floor_retry = 0
    plan_floor_fallback = 0


STATS = _Counters()


def reset_opt_stats() -> None:
    STATS.rerank_considered = 0
    STATS.rerank_triggered = 0
    STATS.rerank_skipped = 0
    STATS.batch_ok = 0
    STATS.batch_fallback = 0
    STATS.cache_hit = 0
    STATS.cache_miss = 0
    STATS.plan_lint_stripped = 0
    STATS.plan_floor_retry = 0
    STATS.plan_floor_fallback = 0


def rerank_trigger_rate() -> float:
    if STATS.rerank_considered <= 0:
        return 0.0
    return STATS.rerank_triggered / STATS.rerank_considered
