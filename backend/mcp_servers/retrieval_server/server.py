"""SmartImager — Retrieval MCP Server（FastMCP v1 / stdio 传输）。

工具：
  - search_images(query: str, top_k: int=20)
       复用既有 backend/ai_func/recommendation/tools.py::tool_search_images
       （内部走 SearchRepository.unified_search + sqlite-vec，不重复实现向量检索）
  - rerank_candidates(candidates: list, strategy: str="default")
       复用 orchestration/rerank.py::rerank_candidates（JSON → ImageCandidate adapter）

运行（repo 根目录）：
    venv\\Scripts\\python.exe -m backend.mcp_servers.retrieval_server.server

测试夹具（离线确定性，供协议/集成测试）：
    SMARTIMAGER_MCP_RETRIEVAL_SOURCE=fake   → 返回内存固定候选
    query 含 "fail"            → 抛错（验证失败处理）
    query 含 "recover" 且 top_k<30 → 抛错（验证 Supervisor REVISE 后成功）
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

mcp = FastMCP("smartimager-retrieval",
              instructions="SmartImager 图片检索工具：向量召回 + 确定性重排。")


# ---------------------------------------------------------------------------
# 离线夹具（仅测试用，避免真实 DB / 模型依赖）
# ---------------------------------------------------------------------------
_FAKE_ROWS: List[Dict[str, Any]] = [
    {"id": 901, "score": 0.95, "title": "北京城市天际线", "description": "CBD 高层建筑",
     "tags": ["建筑", "城市"], "metadata": {"location": "北京"}, "width": 1920, "height": 1280,
     "file_type": "jpg", "created_at": "2025-06-01T10:00:00Z",
     "public_url": "/static/901.jpg", "filepath": "data/images/901.jpg"},
    {"id": 902, "score": 0.90, "title": "上海陆家嘴", "description": "摩天楼群",
     "tags": ["建筑", "城市"], "metadata": {"location": "上海"}, "width": 1920, "height": 1080,
     "file_type": "jpg", "created_at": "2025-06-02T10:00:00Z",
     "public_url": "/static/902.jpg", "filepath": "data/images/902.jpg"},
    {"id": 903, "score": 0.82, "title": "杭州西湖", "description": "湖面与远山",
     "tags": ["风景"], "metadata": {"location": "杭州"}, "width": 1600, "height": 1200,
     "file_type": "jpg", "created_at": "2025-06-03T10:00:00Z",
     "public_url": "/static/903.jpg", "filepath": "data/images/903.jpg"},
]


def _fake_mode() -> bool:
    return os.environ.get("SMARTIMAGER_MCP_RETRIEVAL_SOURCE", "").strip().lower() == "fake"


def _row_by_image_id(image_id: str) -> Dict[str, Any]:
    """图片元数据行（fake 模式查内存；真实模式查 SearchRepository）。"""
    if _fake_mode():
        for row in _FAKE_ROWS:
            if str(row["id"]) == str(image_id).strip():
                return row
        raise ValueError(f"图片不存在: {image_id!r}")
    from backend.db_func.repositories.search import SearchRepository

    try:
        rows = SearchRepository().search_by_image_id(int(image_id), vector_type="image", k=1)
    except (TypeError, ValueError):
        rows = []
    if not rows:
        raise ValueError(f"图片不存在: {image_id!r}")
    return dict(rows[0])


def _search_impl(query: str, top_k: int) -> List[Dict[str, Any]]:
    if _fake_mode():
        if "fail" in query:
            raise RuntimeError("retrieval tool down (fake)")
        if "recover" in query and top_k < 30:
            raise RuntimeError("retrieval transient failure (fake, top_k<30)")
        if "风景" in query or "scenic" in query:
            rows = [r for r in _FAKE_ROWS if "风景" in (r.get("tags") or [])]
        else:
            rows = [r for r in _FAKE_ROWS if "建筑" in (r.get("tags") or [])]
        return rows[:top_k]

    # 真实路径：复用既有检索工具（tool_search_images → SearchRepository.unified_search）
    from backend.ai_func.recommendation.tools import tool_search_images

    return tool_search_images(query=query, targets=["title", "description", "image"],
                              filters={}, limit=top_k) or []


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def search_images(query: str, top_k: int = 20) -> str:
    """按文本语义检索图片（返回候选 JSON 行列表）。"""
    rows = _search_impl(query=query, top_k=int(top_k))
    return json.dumps({"ok": True, "tool": "search_images", "count": len(rows),
                       "rows": rows}, ensure_ascii=False)


@mcp.tool()
def rerank_candidates(candidates: list, strategy: str = "default") -> str:
    """对候选做同池确定性重排（candidates 为 JSON object 数组）。"""
    from backend.ai_func.recommendation.orchestration.rerank import rerank_candidates
    from backend.ai_func.recommendation.orchestration.tools.adapters import normalize_candidates

    normalized = normalize_candidates({"candidates": candidates})
    ranked = rerank_candidates("", normalized)  # strategy 扩展位（默认确定性 R1）
    payload = [c.model_dump(mode="json") for c in ranked]
    return json.dumps({"ok": True, "tool": "rerank_candidates", "count": len(payload),
                       "candidates": payload}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# MCP Resources（Phase 5 Task 6，最小实现）
# ---------------------------------------------------------------------------
@mcp.resource("image://{image_id}", title="图片媒体元数据",
              description="按 image://{image_id} 返回图片行元数据（可读媒体资源位）。")
def image_resource(image_id: str) -> str:
    """媒体资源：image://{id} → 图片行元数据（JSON）。"""
    return json.dumps(_row_by_image_id(image_id), ensure_ascii=False)


@mcp.resource("metadata://{image_id}", title="图片库元数据记录",
              description="按 metadata://{image_id} 返回检索库中该图片的元数据记录。")
def metadata_resource(image_id: str) -> str:
    """元数据资源：metadata://{id} → 图片元数据记录（JSON）。"""
    return json.dumps(_row_by_image_id(image_id), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
