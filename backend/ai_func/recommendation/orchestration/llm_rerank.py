"""看图重排：DeepSeek 系列的视觉模型 deepseek-flash 看原图后重排。

不看标题凑顺序。调用失败时退回公式，不改候选 id 集合。
图库识别（元素、场景、颜色、图中文字）不走这里，走 GLM。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, List, Optional

from .model_reranker import _image_source
from .rerank import rerank_candidates

_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_MAX_IMAGES = 8
_DEFAULT_MODEL = "deepseek-flash"


def _order_from_text(raw: str, candidates: List[Any]) -> List[Any]:
    text = (raw or "").strip()
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("no json")
    data = json.loads(match.group(0))
    order = data.get("order") if isinstance(data, dict) else None
    if not isinstance(order, list):
        raise ValueError("missing order")
    by_id = {int(c.id): c for c in candidates}
    seen = set()
    ranked = []
    for item in order:
        try:
            i = int(item)
        except (TypeError, ValueError):
            continue
        if i in by_id and i not in seen:
            ranked.append(by_id[i])
            seen.add(i)
    for c in candidates:
        if int(c.id) not in seen:
            ranked.append(c)
    if len(ranked) != len(candidates):
        raise ValueError("id set changed")
    return ranked


def _content(query: str, candidates: List[Any]) -> List[dict]:
    blocks: List[dict] = [{
        "type": "text",
        "text": (
            f"检索查询：{query}\n"
            "下面每段文字后是对应图片。请按画面与查询的相关程度从高到低排序。"
            "只能使用给出的 id，不能增删。"
            '只输出 JSON：{"order": [id, ...]}'
        ),
    }]
    for cand in candidates:
        blocks.append({"type": "text", "text": f"id={int(cand.id)}"})
        source = _image_source(cand)
        if source:
            url = source.get("image_url") or {}
            if isinstance(url, dict):
                url = {**url, "detail": "low"}
                source = {**source, "image_url": url}
            blocks.append(source)
    return blocks


def _call_deepseek_vision(query: str, candidates: List[Any]) -> str:
    from ....config import settings
    from ....config.init_service import get_openai_client

    client = get_openai_client()
    if client is None:
        raise RuntimeError("deepseek client missing")
    config = settings.get_config()
    model = (getattr(config, "RERANK_VISION_MODEL", None) or _DEFAULT_MODEL).strip()
    head = list(candidates)[:_MAX_IMAGES]
    if not any(_image_source(c) for c in head):
        raise RuntimeError("no image for vision rerank")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "你是看图重排器。只根据画面判断相关度，不根据文件名猜测。",
            },
            {"role": "user", "content": _content(query, head)},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content or ""


def rerank_with_deepseek_vision(
    query: str,
    candidates: List[Any],
    llm: Optional[Callable[[str, List[Any]], str]] = None,
) -> tuple:
    """返回 (新顺序, 标记)。deepseek_vision 表示看图模型，formula 表示退回公式。"""
    if not candidates or len(candidates) < 2:
        return list(candidates), "deepseek_vision"
    head = list(candidates)[:_MAX_IMAGES]
    tail = list(candidates)[_MAX_IMAGES:]
    try:
        raw = llm(query, head) if llm is not None else _call_deepseek_vision(query, head)
        return _order_from_text(raw, head) + tail, "deepseek_vision"
    except Exception as exc:
        print(f"[rerank] deepseek vision failed → formula: {type(exc).__name__}: {exc}")
        return list(rerank_candidates(query, candidates)), "formula"
