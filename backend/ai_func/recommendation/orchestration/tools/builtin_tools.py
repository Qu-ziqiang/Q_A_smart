"""Phase 2 MCP-style Tool — 内置工具注册（复用既有能力，不改 Worker 实现）。

把既有 5 个能力注册为统一 ToolDefinition（MCP-style 元数据）+ 实现：
  - search_images  (retrieval)   ← tool_search_images / SearchRepository.unified_search
  - rerank         (retrieval)   ← rerank_candidates（R1 确定性重排；R2 见 model_reranker）
  - analyze_image  (vision)      ← ImageAnalysis（VisionWorker 的 provider 客户端）
  - filter_images  (organization)← MetadataWorker 的确定性过滤语义（复用静态实现）
  - group_images   (organization)← GroupImagesWorker 的确定性分组语义（复用静态实现）

原则：
  - Worker 核心逻辑不动；工具实现以“薄适配”复用既有模块；
  - 依赖在函数内惰性导入，避免 import 时重依赖 / 循环；
  - 测试 / Demo 可用 overrides 注入 Fake 实现，不触碰真实 DB / 视觉 API。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .models import (
    CATEGORY_ORGANIZATION,
    CATEGORY_RETRIEVAL,
    CATEGORY_VISION,
    ToolDefinition,
)
from .registry import ToolRegistry


# ---------------------------------------------------------------------------
# input_schema（JSON Schema，受限子集；与 legacy inline tool 描述同义）
# ---------------------------------------------------------------------------
def _schema_search_images() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "用户的检索语句（主题/对象/场景/用途描述）"},
            "vector_targets": {"type": "array", "items": {"type": "string"},
                               "description": "向量字段白名单: title | description | image"},
            "filters": {"type": "object", "description": "结构化过滤条件"},
            "limit": {"type": "integer", "description": "候选上限 1..40"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def _schema_rerank() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索主句"},
            "candidates": {"type": "array", "items": {"type": "object"},
                           "description": "候选池（同池重排，不增删候选）"},
            "intent": {"type": "object", "description": "可选 NormalizedIntent（metadata_constraints 用于重排）"},
            "weights": {"type": "object", "description": "可选权重覆盖 {vector, keyword, metadata}"},
        },
        "required": ["query", "candidates"],
        "additionalProperties": False,
    }


def _schema_analyze_image() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "image_url": {"type": "string", "description": "图片 URL（与 local_image_path 二选一）"},
            "local_image_path": {"type": "string", "description": "本地图片路径（jpg/jpeg/png/webp）"},
            "model": {"type": "string", "description": "覆盖视觉模型名（可选）"},
            "detail": {"type": "string", "enum": ["low", "high", "auto"], "description": "细节级别"},
            "prompt": {"type": "string", "description": "覆盖分析提示词（可选）"},
            "timeout": {"type": "number", "description": "请求超时秒数（可选）"},
        },
        "required": [],
        "additionalProperties": False,
    }


def _schema_filter_images() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidates": {"type": "array", "items": {"type": "object"},
                           "description": "候选行列表（id/tags/metadata/width/height/created_at/file_type）"},
            "predicates": {"type": "array", "items": {"type": "object"},
                           "description": "[{field: created_at|width|height|orientation|file_type|tags|location, "
                                          "operator: eq|in|contains|gte|lte, value}]"},
        },
        "required": ["candidates", "predicates"],
        "additionalProperties": False,
    }


def _schema_group_images() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidates": {"type": "array", "items": {"type": "object"},
                           "description": "候选行列表"},
            "group_by": {"type": "string",
                         "enum": ["location", "year", "month", "orientation", "file_type", "tag"]},
            "unknown_group_name": {"type": "string", "description": "缺失分组键时的兜底组名"},
        },
        "required": ["candidates", "group_by"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# 默认实现（薄适配：复用既有模块；依赖惰性导入）
# ---------------------------------------------------------------------------
def _impl_search_images(query: str, vector_targets: Optional[List[str]] = None,
                        filters: Optional[Dict[str, Any]] = None, limit: int = 20) -> List[Dict[str, Any]]:
    from ...tools import tool_search_images  # backend/ai_func/recommendation/tools.py
    return tool_search_images(query=query, targets=list(vector_targets or []),
                              filters=filters or {}, limit=limit) or []


def _impl_rerank(query: str, candidates: List[Any], intent: Any = None,
                 weights: Optional[Dict[str, float]] = None) -> List[Any]:
    from .adapters import normalize_candidates
    from ..rerank import rerank_candidates

    # Phase 3.2 契约：candidates 为 JSON object 数组；adapter 归一化为 ImageCandidate[]，
    # 再由既有 rerank_candidates 重排（ToolRegistry 不接触领域对象）。
    normalized = normalize_candidates(candidates)
    return rerank_candidates(query, normalized, intent=intent, weights=weights)


def _impl_analyze_image(image_url: Optional[str] = None, local_image_path: Optional[str] = None,
                        model: Optional[str] = None, detail: str = "low",
                        prompt: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
    from ....image_analysis import ImageAnalysis  # ai_func/image_analysis.py
    return ImageAnalysis().analyze_image(
        image_url=image_url, local_image_path=local_image_path, model=model,
        detail=detail, prompt=prompt, timeout=timeout, raise_on_error=True,
    )


def _impl_filter_images(candidates: List[Any], predicates: List[Any]) -> Dict[str, Any]:
    from ..models import MetadataPredicate
    from ..workers import MetadataWorker, RetrievalWorker

    def _to_candidate(row):
        if hasattr(row, "metadata"):  # 已是 ImageCandidate / 类似对象
            return row
        return RetrievalWorker._to_candidate(row)  # dict row → ImageCandidate

    normalized = []
    for p in predicates:
        normalized.append(p if isinstance(p, MetadataPredicate) else MetadataPredicate.model_validate(p))
    kept_rows, rejected_rows, rejected_ids = [], [], []
    for row in candidates:
        cand = _to_candidate(row)
        if all(
            MetadataWorker._apply_op(p.operator, MetadataWorker._field_value(cand, p.field), p.value)
            for p in normalized
        ):
            kept_rows.append(row)
        else:
            rejected_rows.append(row)
            if isinstance(row, dict):
                rejected_ids.append(row.get("id"))
    return {
        "kept": kept_rows,
        "rejected": rejected_rows,
        "kept_ids": [r.get("id") if isinstance(r, dict) else getattr(r, "id", None) for r in kept_rows],
        "rejected_ids": rejected_ids,
    }


def _impl_group_images(candidates: List[Any], group_by: str,
                       unknown_group_name: str = "未分类") -> Dict[str, Any]:
    from ..workers import GroupImagesWorker, RetrievalWorker

    def _to_candidate(row):
        if hasattr(row, "metadata"):
            return row
        return RetrievalWorker._to_candidate(row)

    fallback = (unknown_group_name or "未分类").strip() or "未分类"
    buckets: Dict[str, list] = {}
    order: List[str] = []
    for row in candidates:
        cand = _to_candidate(row)
        key = GroupImagesWorker._group_key(cand, group_by)
        key = key or fallback
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)
    groups = [
        {"key": key, "image_ids": [
            r.get("id") if isinstance(r, dict) else getattr(r, "id", None) for r in buckets[key]
        ], "count": len(buckets[key])}
        for key in order
    ]
    return {"group_by": group_by, "groups": groups, "total": len(candidates)}


# ---------------------------------------------------------------------------
# 定义 + 默认实现表 + 注册入口
# ---------------------------------------------------------------------------
def builtin_tool_definitions() -> Dict[str, ToolDefinition]:
    """五个内置工具的 ToolDefinition（按 name 索引）。"""
    return {
        "search_images": ToolDefinition(
            name="search_images",
            description="按文本语义检索图片（向量召回 + 元数据过滤，返回候选行列表）。"
                        "复用 tool_search_images / SearchRepository.unified_search。",
            input_schema=_schema_search_images(),
            output_schema="List[dict]（id/score/title/description/tags/metadata/public_url/filepath）",
            category=CATEGORY_RETRIEVAL,
            cost_level="cheap",
        ),
        "rerank": ToolDefinition(
            name="rerank",
            description="对已召回候选做同池重排（确定性 R1；候选集合不增删）。",
            input_schema=_schema_rerank(),
            output_schema="List[ImageCandidate]（同池新序）",
            category=CATEGORY_RETRIEVAL,
            cost_level="medium",
        ),
        "analyze_image": ToolDefinition(
            name="analyze_image",
            description="对单张图片做视觉分析，返回 title/description/tags/evidence/confidence。"
                        "复用 ImageAnalysis（VisionWorker 使用的 provider 客户端）。",
            input_schema=_schema_analyze_image(),
            output_schema="dict（title/description/tags/error）或 raise ImageAnalysisError",
            category=CATEGORY_VISION,
            cost_level="expensive",
        ),
        "filter_images": ToolDefinition(
            name="filter_images",
            description="基于结构化元数据对候选做确定性过滤（语义与 MetadataWorker 一致）。",
            input_schema=_schema_filter_images(),
            output_schema="dict（kept/rejected/kept_ids/rejected_ids）",
            category=CATEGORY_ORGANIZATION,
            cost_level="cheap",
        ),
        "group_images": ToolDefinition(
            name="group_images",
            description="按结构化字段对候选做确定性分组（语义与 GroupImagesWorker 一致）。",
            input_schema=_schema_group_images(),
            output_schema="dict（group_by/groups[{key,image_ids,count}]/total）",
            category=CATEGORY_ORGANIZATION,
            cost_level="cheap",
        ),
    }


def default_implementations() -> Dict[str, Callable]:
    return {
        "search_images": _impl_search_images,
        "rerank": _impl_rerank,
        "analyze_image": _impl_analyze_image,
        "filter_images": _impl_filter_images,
        "group_images": _impl_group_images,
    }


def register_builtin_tools(registry: ToolRegistry,
                           overrides: Optional[Dict[str, Callable]] = None) -> ToolRegistry:
    """把五个内置工具注册进 registry；overrides 可替换实现（测试 / Demo 注入 Fake）。"""
    definitions = builtin_tool_definitions()
    impls = default_implementations()
    for name, definition in definitions.items():
        implementation = (overrides or {}).get(name, impls.get(name))
        registry.register(definition, implementation=implementation)
    return registry


_DEFAULT_REGISTRY: Optional[ToolRegistry] = None


def default_tool_registry() -> ToolRegistry:
    """进程内共享默认 ToolRegistry（首次注册内置工具）。"""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        reg = ToolRegistry()
        register_builtin_tools(reg)
        _DEFAULT_REGISTRY = reg
    return _DEFAULT_REGISTRY


