"""C2 计划缓存：同一句少调一次 Planner。与 Memory episode 不是一层。

- 精确匹配（规范化 query）优先
- 否则相似度 ≥ 0.95（可注入 embed_fn；默认 token Jaccard）
- TTL 默认 1h
- 工具注册指纹变更即失效
- Replan（带 observation）不读写缓存
"""

from __future__ import annotations

import re
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import registered_task_types
from .opt_flags import STATS, plan_cache_enabled

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[，。！？、；：,.!?;:\"'()\[\]{}]")
DEFAULT_TTL_SECONDS = 3600.0
DEFAULT_SIMILARITY = 0.95


def normalize_query(query: str) -> str:
    text = (query or "").strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def registry_fingerprint(vision_available_flag: Optional[bool] = None) -> str:
    types = registered_task_types(vision_available_flag)
    return ",".join(sorted(t.value for t in types))


def _jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cosine(u: List[float], v: List[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = sum(x * y for x, y in zip(u, v))
    nu = sum(x * x for x in u) ** 0.5
    nv = sum(y * y for y in v) ** 0.5
    if nu <= 0 or nv <= 0:
        return 0.0
    return dot / (nu * nv)


class PlanCache:
    """内存计划缓存。测试可注入 embed_fn / clock。"""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        similarity: float = DEFAULT_SIMILARITY,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        now_fn: Optional[Callable[[], float]] = None,
    ):
        self.ttl_seconds = ttl_seconds
        self.similarity = similarity
        self.embed_fn = embed_fn
        self._now = now_fn or time.time
        self._exact: Dict[Tuple[str, str], Tuple[float, Any]] = {}
        self._semantic: List[Tuple[str, str, Optional[List[float]], float, Any]] = []

    def clear(self) -> None:
        self._exact.clear()
        self._semantic.clear()

    def get(self, query: str, registry_fp: str) -> Optional[Any]:
        key = normalize_query(query)
        if not key:
            return None
        now = self._now()
        hit = self._exact.get((key, registry_fp))
        if hit is not None:
            exp, draft = hit
            if exp >= now:
                return deepcopy(draft)
            self._exact.pop((key, registry_fp), None)
        best: Optional[Any] = None
        best_sim = 0.0
        alive: List[Tuple[str, str, Optional[List[float]], float, Any]] = []
        qvec = self.embed_fn(key) if self.embed_fn is not None else None
        for norm, fp, vec, exp, draft in self._semantic:
            if exp < now or fp != registry_fp:
                continue
            alive.append((norm, fp, vec, exp, draft))
            if norm == key:
                continue
            if qvec is not None and vec is not None:
                sim = _cosine(qvec, vec)
            else:
                sim = _jaccard(key, norm)
            if sim >= self.similarity and sim > best_sim:
                best_sim = sim
                best = draft
        self._semantic = alive
        return deepcopy(best) if best is not None else None

    def put(self, query: str, draft: Any, registry_fp: str) -> None:
        key = normalize_query(query)
        if not key or draft is None:
            return
        exp = self._now() + self.ttl_seconds
        stored = deepcopy(draft)
        self._exact[(key, registry_fp)] = (exp, stored)
        vec = self.embed_fn(key) if self.embed_fn is not None else None
        self._semantic.append((key, registry_fp, vec, exp, stored))


_GLOBAL = PlanCache()


def get_plan_cache() -> PlanCache:
    return _GLOBAL


def lookup_plan(query: str, registry_fp: Optional[str] = None) -> Optional[Any]:
    if not plan_cache_enabled():
        return None
    fp = registry_fp if registry_fp is not None else registry_fingerprint()
    hit = _GLOBAL.get(query, fp)
    if hit is not None:
        STATS.cache_hit += 1
        return hit
    STATS.cache_miss += 1
    return None


def store_plan(query: str, draft: Any, registry_fp: Optional[str] = None) -> None:
    if not plan_cache_enabled() or draft is None:
        return
    fp = registry_fp if registry_fp is not None else registry_fingerprint()
    _GLOBAL.put(query, draft, fp)
