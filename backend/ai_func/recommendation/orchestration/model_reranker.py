"""Injected Multimodal Reranker — R2 接口（模型型 / 外部 API）。

R2 是 R1（确定性 rerank，rerank.py）的第二种方案：
  Vector Retrieval → Top-N → MultimodalReranker(provider) → relevance score → 同池新序

设计要点：
- R1 保留不动；R2 通过 RetrievalWorker 已预留的 `reranker=` 注入位接入。
- 不在本地下载 / 运行任何大型多模态模型；实际 relevance 由外部 API provider 提供。
- provider 用 adapter 抽象：切换模型只改 provider/配置，不改 RetrievalWorker。
- API Key 一律读环境变量（MM_RERANK_*），禁止写入代码。
- relevance 归一化到 [0,1]（非法值抛 RelevanceError → 走 fallback）。
- 任何 provider 失败 → fallback 到 R1（rerank_candidates）；绝不让 Agent 请求失败。
- contract 不变：候选数量/ID 集合不变，只改排序。

真实 provider 只需后续配置（见模块末尾 ENV 常量说明），本模块不发起任何付费调用。
"""

from __future__ import annotations

import base64
import math
import mimetypes
import os
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Tuple

from .rerank import rerank_candidates
from .opt_flags import (
    STATS,
    rerank_batch_enabled,
    rerank_batch_max,
    should_trigger_mm_rerank,
)

if TYPE_CHECKING:  # 仅类型标注；运行时避免 workers <-> model_reranker 循环
    from .workers import ImageCandidate

# ---------------------------------------------------------------------------
# 环境变量（接入真实 provider 时只需配置这些；当前一律不配置 → 自动走 R1）
# ---------------------------------------------------------------------------
ENV_PROVIDER = "MM_RERANK_PROVIDER"          # 例如 "openai_chat"（空 = 关闭 R2）
ENV_API_KEY = "MM_RERANK_API_KEY"
ENV_BASE_URL = "MM_RERANK_BASE_URL"
ENV_MODEL = "MM_RERANK_MODEL"
ENV_TIMEOUT = "MM_RERANK_TIMEOUT_SECONDS"
DEFAULT_TIMEOUT_SECONDS = 30.0

_JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


class RelevanceError(Exception):
    """provider 不可用 / 响应无法转稳定 relevance 的统一失败。"""


def parse_rank_order(data: Any, n: int) -> List[int]:
    """解析 {"order": [...], "reason": "..."}。支持 0-based 与 1-based；非法则抛 RelevanceError。"""
    if not isinstance(data, dict):
        raise RelevanceError("batch rank 响应不是对象")
    raw = data.get("order")
    if raw is None:
        raw = data.get("ranking")
    if not isinstance(raw, list) or not raw:
        raise RelevanceError("batch rank 缺少 order 数组")
    idxs: List[int] = []
    for item in raw:
        try:
            idxs.append(int(item))
        except (TypeError, ValueError) as e:
            raise RelevanceError(f"batch rank order 含非整数: {item!r}") from e
    if any(i == 0 for i in idxs):
        zero_based = idxs
    elif all(1 <= i <= n for i in idxs):
        zero_based = [i - 1 for i in idxs]
    else:
        zero_based = idxs
    if not any(0 <= i < n for i in zero_based):
        raise RelevanceError("batch rank order 无有效下标")
    return zero_based


def parse_rank_json(raw: str, n: int) -> List[int]:
    import json

    text = (raw or "").strip()
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise RelevanceError("batch rank 未返回 JSON")
    try:
        data = json.loads(m.group(0))
    except Exception as e:
        raise RelevanceError(f"batch rank JSON 解析失败: {str(e)[:120]}") from e
    return parse_rank_order(data, n)


def _apply_order(candidates: List["ImageCandidate"], order: List[int]) -> List["ImageCandidate"]:
    """order 为 0-based 下标排列；非法/缺失下标忽略，漏网的按原序补齐。"""
    n = len(candidates)
    seen = set()
    out: List[Any] = []
    for idx in order or []:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(candidates[i])
    for i, c in enumerate(candidates):
        if i not in seen:
            out.append(c)
    return out


def _env(key: str) -> str:
    return (os.environ.get(key) or "").strip()


def _to_unit_score(raw: float) -> float:
    """把任意数值响应收敛到稳定 [0,1]；非有限值视为 provider 错误。"""
    try:
        value = float(raw)
    except (TypeError, ValueError) as e:
        raise RelevanceError(f"relevance 非数值: {raw!r}") from e
    if not math.isfinite(value):
        raise RelevanceError(f"relevance 非有限值: {value!r}")
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Provider 协议
# ---------------------------------------------------------------------------
class MultimodalProvider(Protocol):
    """多模态 provider adapter 协议：给定 text query + image candidate → 原始相关性分。

    约定：
    - relevance() 必须能抛出 RelevanceError（含不可用/超时/解析失败/网络错误）。
    - 返回分值会被 _to_unit_score 收敛到 [0,1]（超出即 clamp，非有限即报错）。
    """

    name: str

    def relevance(self, query: str, image: "ImageCandidate") -> float:
        ...


def _image_source(image: "ImageCandidate") -> Optional[Dict[str, str]]:
    """候选图 → 单张图片的 API 输入（优先本地 filepath 转 data-url，否则 public_url）。"""
    fp = (getattr(image, "filepath", None) or "").strip()
    if fp and os.path.exists(fp):
        mime = mimetypes.guess_type(fp)[0] or "image/jpeg"
        try:
            with open(fp, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            raise RelevanceError(f"读取本地图片失败: {e}") from e
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
    url = (getattr(image, "public_url", None) or "").strip()
    if url:
        return {"type": "image_url", "image_url": {"url": url}}
    raise RelevanceError("候选无可用图片来源（filepath/public_url 均缺失）")


# ---------------------------------------------------------------------------
# 统一接口（注入到 RetrievalWorker 的可调用重排器）
# ---------------------------------------------------------------------------
class MultimodalReranker:
    """可注入模型型重排器：score(query, image)->float 单位化；rerank 同池稳定排序。

    构造：MultimodalReranker(provider[, fallback])
    - provider：MultimodalProvider adapter（relevance() 抛错 → fallback）
    - fallback：失败时使用的重排函数，默认 R1 rerank_candidates（保留 R1）
    """

    def __init__(self, provider: MultimodalProvider, fallback: Optional[Callable] = None):
        if provider is None:
            raise TypeError("provider 不能为 None（用 build_multimodal_reranker 工厂）")
        self.provider = provider
        self._fallback = fallback if fallback is not None else rerank_candidates

    @property
    def name(self) -> str:
        return getattr(self.provider, "name", "unknown")

    def score(self, query: str, image: "ImageCandidate") -> float:
        """query-image relevance ∈ [0,1]；provider 错误向上抛（由 rerank 统一 fallback）。"""
        raw = self.provider.relevance(query, image)
        return _to_unit_score(raw)

    def rerank(self, query: str, candidates: List["ImageCandidate"],
               fallback: Optional[Callable] = None) -> List["ImageCandidate"]:
        """重排同池。默认逐张打分；SM_RERANK_BATCH=1 时走多图单次/两轮淘汰。"""
        if not candidates or len(candidates) < 2:
            return candidates
        fn = fallback if fallback is not None else self._fallback
        if rerank_batch_enabled() and hasattr(self.provider, "rank_batch"):
            try:
                ordered = self._rerank_batch(query, candidates)
                STATS.batch_ok += 1
                return ordered
            except Exception as e:
                STATS.batch_fallback += 1
                print(f"[MultimodalReranker:{self.name}] batch error → R1/CLIP: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                if callable(fn):
                    return list(fn(query, candidates))
                return list(candidates)
        return self._rerank_sequential(query, candidates, fn)

    def _rerank_sequential(self, query: str, candidates: List["ImageCandidate"],
                           fn: Optional[Callable]) -> List["ImageCandidate"]:
        scored: List[Tuple[float, int, object]] = []
        try:
            for i, c in enumerate(candidates):
                scored.append((self.score(query, c), i, c))
        except Exception as e:
            print(f"[MultimodalReranker:{self.name}] provider error → fallback: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            if callable(fn):
                return list(fn(query, candidates))
            return list(candidates)
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [c for _, _, c in scored]

    def _rerank_batch(self, query: str, candidates: List["ImageCandidate"]) -> List["ImageCandidate"]:
        cap = rerank_batch_max()
        if len(candidates) <= cap:
            order = self.provider.rank_batch(query, candidates)
            return _apply_order(candidates, order)
        # 两轮：先取 top-cap，再精排
        first = self.provider.rank_batch(query, candidates)
        shortlist = _apply_order(candidates, first)[:cap]
        if len(shortlist) < 2:
            return shortlist
        second = self.provider.rank_batch(query, shortlist)
        return _apply_order(shortlist, second) + [
            c for c in _apply_order(candidates, first)[cap:]
        ]


class _UnavailableProvider:
    """占位 provider：未配置/未知 provider 时所有调用抛 RelevanceError（→ R1 fallback）。"""

    name = "unavailable"

    def __init__(self, reason: str):
        self._reason = reason

    def relevance(self, query: str, image) -> float:
        raise RelevanceError(f"multimodal reranker 不可用: {self._reason}")


class OpenAICompatChatRerankerProvider:
    """OpenAI-compatible 视觉 chat adapter（示例 provider，名称 "openai_chat"）。

    输入：query + 单张图（data-url 或 public_url）
    输出：模型返回 JSON {"relevance": 0..1}；解析失败/不可用抛 RelevanceError → R1 fallback。
    配置（全部环境变量，当前不配置即不可用）：
      MM_RERANK_MODEL / MM_RERANK_API_KEY / MM_RERANK_BASE_URL（可选超时）
    """

    name = "openai_chat"
    _SYSTEM = ("你是多模态图片相关性评分器。只判断图片内容与检索意图是否相关。"
               "只输出 JSON：{\"relevance\": 0到1的小数}，0=完全不相关，1=完全相关，不要输出其它文字。")
    _BATCH_SYSTEM = (
        "你是多模态图片排序器。根据检索意图给图排序。"
        "只输出 JSON：{\"order\": [按相关性从高到低的图片编号], \"reason\": \"一句理由\"}。"
        "编号从 1 开始且必须是输入图的排列，不要输出其它文字。"
    )

    def __init__(self) -> None:
        self._model = _env(ENV_MODEL)
        self._api_key = _env(ENV_API_KEY)
        self._base_url = _env(ENV_BASE_URL)
        self.last_raw: Optional[str] = None  # 最近一次原始响应文本（观察/smoke 用，无敏感信息）
        try:
            self._timeout = float(_env(ENV_TIMEOUT) or DEFAULT_TIMEOUT_SECONDS)
        except ValueError:
            self._timeout = DEFAULT_TIMEOUT_SECONDS
        missing = [k for k, v in ((ENV_MODEL, self._model), (ENV_API_KEY, self._api_key),
                                  (ENV_BASE_URL, self._base_url)) if not v]
        self._reason = f"缺少环境变量: {', '.join(missing)}" if missing else ""

    def relevance(self, query: str, image) -> float:
        if self._reason:
            raise RelevanceError(self._reason)
        from openai import OpenAI  # 延迟导入，避免模块级网络/重依赖

        source = _image_source(image)
        try:
            client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)
            resp = client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"检索意图：{query}"},
                        source,
                    ]},
                ],
            )
            raw = (resp.choices[0].message.content or "")
            self.last_raw = raw
        except Exception as e:  # 网络/超时/provider 4xx
            raise RelevanceError(f"openai_chat 调用失败: {type(e).__name__}: {str(e)[:120]}") from e
        m = _JSON_OBJ_RE.search(raw)
        if not m:
            raise RelevanceError("openai_chat 未返回 JSON")
        import json

        try:
            data = json.loads(m.group(0))
            return float(data.get("relevance"))
        except Exception as e:
            raise RelevanceError(f"openai_chat relevance 解析失败: {str(e)[:120]}") from e

    def rank_batch(self, query: str, candidates: List[Any]) -> List[int]:
        """一次多图排序，返回 0-based 下标。解析失败重试一次，仍失败抛 RelevanceError。"""
        last: Optional[Exception] = None
        for _attempt in range(2):
            try:
                raw = self._chat_batch(query, candidates)
                return parse_rank_json(raw, len(candidates))
            except RelevanceError as e:
                last = e
        raise last if last else RelevanceError("batch rank 失败")

    def _chat_batch(self, query: str, candidates: List[Any]) -> str:
        if self._reason:
            raise RelevanceError(self._reason)
        from openai import OpenAI

        content: List[Any] = [{"type": "text", "text": f"检索意图：{query}\n共 {len(candidates)} 张图，编号 1..{len(candidates)}。"}]
        for i, image in enumerate(candidates, start=1):
            content.append({"type": "text", "text": f"图 {i}："})
            content.append(_image_source(image))
        try:
            client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)
            resp = client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self._BATCH_SYSTEM},
                    {"role": "user", "content": content},
                ],
            )
            raw = (resp.choices[0].message.content or "")
            self.last_raw = raw
            return raw
        except Exception as e:
            raise RelevanceError(f"openai_chat batch 调用失败: {type(e).__name__}: {str(e)[:120]}") from e


# ---------------------------------------------------------------------------
# provider 注册表 + env 工厂
# ---------------------------------------------------------------------------
PROVIDER_BUILDERS: Dict[str, Callable[[], MultimodalProvider]] = {
    "openai_chat": OpenAICompatChatRerankerProvider,
    # 未来新增 provider：在此注册即可，不改 RetrievalWorker
    # "embedding_similarity": <builder>,
}


def build_multimodal_reranker(provider_name: Optional[str] = None,
                              fallback: Optional[Callable] = None) -> Optional[MultimodalReranker]:
    """按 provider 名构造 MultimodalReranker；未配置/未知 provider 返回带 unavailable provider 的实例。

    provider_name=None 时读取 MM_RERANK_PROVIDER；为空 → 返回 None（调用方用 R1）。
    """
    name = provider_name if provider_name is not None else _env(ENV_PROVIDER)
    if not name:
        return None
    builder = PROVIDER_BUILDERS.get(str(name).strip().lower())
    if builder is None:
        return MultimodalReranker(_UnavailableProvider(f"未知 provider: {name}"), fallback=fallback)
    return MultimodalReranker(builder(), fallback=fallback)


def default_model_reranker() -> Callable[[str, list], list]:
    """RetrievalWorker 默认重排器：
    - 未配置 MM_RERANK_PROVIDER → 直接返回 R1 rerank_candidates（现状/无回归）。
    - 已配置 → 返回模型型重排 dispatcher；其内部任何 provider 失败自动回退 R1。
    """
    mm = build_multimodal_reranker()
    if mm is None:
        return rerank_candidates

    def _dispatch(query: str, candidates: list) -> list:
        STATS.rerank_considered += 1
        if not should_trigger_mm_rerank(candidates):
            STATS.rerank_skipped += 1
            return list(candidates)
        STATS.rerank_triggered += 1
        return mm.rerank(query, candidates)

    return _dispatch


