"""SmartImager — Vision MCP Server（FastMCP v1 / stdio 传输）。

工具：
  - analyze_image(image_id: str, task: str)
       复用既有 ImageAnalysis 视觉能力（不修改 VisionWorker）；
       真实路径：image_id → 本地图片路径 → ImageAnalysis().analyze_image(...)

运行（repo 根目录）：
    venv\\Scripts\\python.exe -m backend.mcp_servers.vision_server.server

测试夹具（离线确定性）：
    SMARTIMAGER_MCP_VISION_SOURCE=fake
    image_id == "fail" → 抛错（验证失败处理）
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

mcp = FastMCP("smartimager-vision",
              instructions="SmartImager 图片视觉分析工具。")


def _fake_mode() -> bool:
    return os.environ.get("SMARTIMAGER_MCP_VISION_SOURCE", "").strip().lower() == "fake"


def _analyze_impl(image_id: str, task: str) -> Dict[str, Any]:
    if _fake_mode():
        if str(image_id).strip() == "fail":
            raise RuntimeError("vision tool down (fake)")
        return {
            "image_id": image_id,
            "task": task,
            "title": "画面主体",
            "description": f"对图片 {image_id} 的视觉核验：{task}",
            "tags": ["分析"],
            "evidence": "根据给定任务完成描述性核验",
            "confidence": 0.9,
        }

    # 真实路径：image_id → 图片行（含本地路径）→ ImageAnalysis
    from backend.db_func.repositories.search import SearchRepository
    from backend.ai_func.image_analysis import ImageAnalysis

    try:
        img_id = int(image_id)
    except (TypeError, ValueError):
        raise ValueError(f"非法 image_id: {image_id!r}")
    rows = SearchRepository().search_by_image_id(img_id, vector_type="image", k=1)
    if not rows:
        raise ValueError(f"图片不存在: {image_id!r}")
    filepath = (rows[0].get("filepath") or "").strip()
    if not filepath:
        raise ValueError(f"图片缺少本地路径: {image_id!r}")
    prompt = f"请针对任务『{task}』分析该图片，输出 title/description/tags/evidence/confidence。"
    result = ImageAnalysis().analyze_image(local_image_path=filepath, prompt=prompt,
                                           raise_on_error=True)
    return {"image_id": image_id, "task": task, **result}


@mcp.tool()
def analyze_image(image_id: str, task: str) -> str:
    """对指定图片做视觉核验并返回结构化 JSON。"""
    payload = _analyze_impl(image_id=image_id, task=task)
    return json.dumps({"ok": True, "tool": "analyze_image", **payload}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
