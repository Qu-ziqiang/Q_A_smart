"""Deterministic Retrieval Rerank（R1，第二阶段轻量排序）。

约束（R1 核心）：
- 只对 RetrievalWorker 已召回的候选做**重排**：候选集合不变、数量不变、无新 ID、无删除。
- 确定性、可解释、无新模型 / 无 LLM / 无 Cross-Encoder。
- 复用既有信号：CLIP vector score（candidate.score = 1 - cosine distance）、
  title/description/tags 文本、metadata 字段；keyword 复用 memory._keyword_overlap。

scoring：
    rerank_score = w_vec * vector_score + w_kw * keyword_overlap + w_meta * metadata_match
- vector_score   候选池内 min-max 归一化到 [0,1]（不重算 embedding）
- keyword_overlap 复用 memory._tokens / _keyword_overlap（CJK bigram + latin 词）∈ [0,1]
- metadata_match  仅当显式传入 metadata constraints 时 >0；普通 query 为 0（中性）

回退保证（任一命中即保持原顺序，绝不改变集合）：
1. 候选为空 / 只有 1 个
2. 全部权重为 0
3. 任一候选缺少数值 score（真实链路恒有 score；缺分数则不做重排）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from .memory import _keyword_overlap

if TYPE_CHECKING:  # 仅类型标注；运行时避免 workers <-> rerank 循环导入
    from .workers import ImageCandidate

# 初始默认权重（见模块 docstring / 实施报告说明选择理由）
DEFAULT_WEIGHTS: Dict[str, float] = {"vector": 1.0, "keyword": 0.35, "metadata": 0.2}

_EPS = 1e-9


def _as_float(value) -> Optional[float]:
    """None / 非法 → None；合法数值转 float。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vector_component(candidates: List[ImageCandidate]) -> List[float]:
    """candidate.score 池内 min-max 归一化到 [0,1]；全同则返回 0。"""
    scores = [_as_float(c.score) for c in candidates]
    if any(s is None for s in scores):
        return [0.0] * len(candidates)
    nums = [s for s in scores if s is not None]
    lo, hi = min(nums), max(nums)
    if hi - lo < _EPS:
        return [0.0] * len(candidates)
    return [(s - lo) / (hi - lo) if s is not None else 0.0 for s in scores]


def _doc_text(candidate: ImageCandidate) -> str:
    tags = " ".join(str(t) for t in (candidate.tags or []) if t)
    return " ".join(x for x in (candidate.title or "", candidate.description or "", tags) if x)


def _metadata_score(candidate: ImageCandidate, constraints: List) -> float:
    """命中的 metadata constraint 比例 ∈ [0,1]；无 constraints → 0（中性）。"""
    if not constraints:
        return 0.0
    from .workers import MetadataWorker  # 延迟导入避免 workers<->rerank 循环

    hit = 0
    for p in constraints:
        field = getattr(p, "field", None)
        if field is None:
            continue
        val = MetadataWorker._field_value(candidate, field)
        if val is not None and MetadataWorker._apply_op(getattr(p, "operator", ""), val, getattr(p, "value", None)):
            hit += 1
    return hit / len(constraints)


def _all_zero(weights: Dict[str, float]) -> bool:
    return abs(float(weights.get("vector", 0.0))) < _EPS and \
        abs(float(weights.get("keyword", 0.0))) < _EPS and \
        abs(float(weights.get("metadata", 0.0))) < _EPS


def rerank_candidates(
    query: str,
    candidates: List[ImageCandidate],
    intent=None,
    weights: Optional[Dict[str, float]] = None,
) -> List[ImageCandidate]:
    """返回与输入相同对象集合的重排列表。输入对象不被修改。

    不改变 candidate ID 集合 / 数量；同分按原顺序稳定排序。
    """
    if not candidates or len(candidates) < 2:
        return candidates
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    if _all_zero(w):
        return candidates

    # 缺数值 score（真实链路恒有）→ 不做重排，保持原序
    if any(_as_float(c.score) is None for c in candidates):
        return candidates

    vec = _vector_component(candidates)
    constraints = list(getattr(intent, "metadata_constraints", None) or []) if intent is not None else []
    scored = []
    for i, c in enumerate(candidates):
        kw = _keyword_overlap(query or "", _doc_text(c))
        meta = _metadata_score(c, constraints)
        total = float(w.get("vector", 0.0)) * vec[i] \
            + float(w.get("keyword", 0.0)) * kw \
            + float(w.get("metadata", 0.0)) * meta
        scored.append((total, i, c))
    # 稳定排序：同 total 保持原 candidates 顺序（i 仅作显式 tie-break）
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored]
