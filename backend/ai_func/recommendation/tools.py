from typing import List, Dict, Any, Optional
from ...db_func.repositories.search import SearchRepository

# 注意：此文件位于 backend/ai_func/recommendation/tools.py 下
# 作为“工具层”，被 Agent 调用


def tool_search_images(query: str, targets: List[str], filters: Optional[Dict[str, Any]] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """图片检索工具

    增加向量目标白名单过滤，防止将标签/风格等中文字段误当成向量表名。
    """
    repo = SearchRepository()
    allowed = {"title", "description", "image"}
    norm_targets = [t for t in (targets or []) if isinstance(t, str) and t in allowed]
    if not norm_targets:
        norm_targets = ["title", "description", "image"]
    return repo.unified_search(
        query_type="text",
        query_content=query,
        search_targets=norm_targets,
        filters=filters or {},
        limit=limit,
        offset=0,
    ) or []


def tool_search_by_labels(
    labels: List[str],
    *,
    dataset_id: Optional[int] = None,
    match: str = "all",
    limit: int = 40,
) -> List[Dict[str, Any]]:
    """精确类别筛选（默认 AND），不走 CLIP。"""
    return SearchRepository().search_by_labels(
        labels, dataset_id=dataset_id, match=match, limit=limit
    ) or []


def tool_retrieve_similar(
    image_id: int,
    *,
    dataset_id: Optional[int] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """以图搜图：同数据集内图像向量近邻。"""
    filters: Dict[str, Any] = {}
    if dataset_id is not None:
        filters["dataset_id"] = int(dataset_id)
    return SearchRepository().search_by_image_id(
        int(image_id),
        vector_type="image",
        k=int(limit),
        filters=filters,
        exclude_self=True,
    ) or []

