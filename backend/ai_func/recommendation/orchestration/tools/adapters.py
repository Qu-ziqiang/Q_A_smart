"""Phase 3.2 — Tool payload 归一化适配器（JSON ↔ 领域对象）。

目的：ToolRegistry 面向 LLM / MCP Client 只暴露 JSON 可序列化参数
（例如 rerank.candidates = [{id, score, metadata, ...}]），由 adapter 在
“ToolRegistry → builtin adapter → existing worker function”之间完成转换，
避免 Registry 认识 Python 领域对象。

normalize_candidates(payload) 接受：
  - dict:  {"candidates": [JSON object, ...]}（外部 LLM/Tool Calling 形式）
  - list:  [JSON object | ImageCandidate, ...]（内部 Python 调用方便捷形式）
返回：
  - List[ImageCandidate]
"""

from __future__ import annotations

from typing import Any, Dict, List, Union


def normalize_candidates(payload: Union[Dict[str, Any], List[Any]]) -> List[Any]:
    """把 candidates payload 归一化为 ImageCandidate 列表。"""
    from ..workers import ImageCandidate

    if isinstance(payload, dict):
        items = payload.get("candidates") or []
    elif isinstance(payload, list):
        items = payload
    else:
        raise TypeError(
            f"payload 需要 {{'candidates': [...]}} 或 list，实际 {type(payload).__name__}"
        )
    if not isinstance(items, list):
        raise TypeError(f"candidates 必须是数组，实际 {type(items).__name__}")

    out: List[Any] = []
    for item in items:
        if isinstance(item, ImageCandidate):
            out.append(item)  # 内部调用方已传领域对象（兼容）
            continue
        if not isinstance(item, dict):
            raise TypeError(
                f"candidate 必须是 JSON object，实际 {type(item).__name__}"
            )
        try:
            out.append(ImageCandidate(**item))
        except Exception as e:
            raise TypeError(f"candidate 字段无法构建 ImageCandidate: {type(e).__name__}: {e}")
    return out
