"""Routing Benchmark — 28-query gold dataset + 离线评分器（审计 §10）。

分层：
  Layer A — Gate：expected_route（legacy|team）
  Layer B — Planner：expected_task_types（固定 vision_available=True snapshot 下直接调 Planner）

Gold dataset 每类 4 条（A 检索 / B 元数据 / C 分组 / D 视觉核验 / E 检索+视觉比较 /
F 检索+元数据 / G 检索+元数据+视觉）。Vision-positive = D+E+G = 12；Vision-negative = A+B+C+F = 16。

重要：
  - 本文件不含真实 API 调用；真实 probe 需显式授权（28×2 次 Planner 调用）。
  - 普通 regression 只校验数据集完整性与评分器。
"""

from typing import Any, Dict, List

GOLD_QUERIES: List[Dict[str, Any]] = [
    # ---- A. Retrieval-only（4）----
    {"id": 1, "category": "A", "query": "找几张海边日落照片",
     "expected_route": "legacy", "expected_task_types": ["retrieve_images", "synthesize"],
     "vision_required": False, "note": "纯检索，不需要 Vision"},
    {"id": 2, "category": "A", "query": "找城市夜景照片",
     "expected_route": "legacy", "expected_task_types": ["retrieve_images", "synthesize"],
     "vision_required": False, "note": "纯检索"},
    {"id": 3, "category": "A", "query": "推荐几张可爱的猫照片",
     "expected_route": "legacy", "expected_task_types": ["retrieve_images", "synthesize"],
     "vision_required": False, "note": "纯检索"},
    {"id": 4, "category": "A", "query": "找水墨画风格的山景图",
     "expected_route": "legacy", "expected_task_types": ["retrieve_images", "synthesize"],
     "vision_required": False, "note": "风格语义召回"},
    # ---- B. Metadata-only（4）----
    {"id": 5, "category": "B", "query": "找 2025 年上传的照片",
     "expected_route": "legacy", "expected_task_types": ["retrieve_images", "filter_metadata", "synthesize"],
     "vision_required": False, "note": "年份是已有元数据"},
    {"id": 6, "category": "B", "query": "找横向 JPG 照片",
     "expected_route": "legacy", "expected_task_types": ["retrieve_images", "filter_metadata", "synthesize"],
     "vision_required": False, "note": "方向/格式是已有元数据"},
    {"id": 7, "category": "B", "query": "找宽度至少 1920 的照片",
     "expected_route": "legacy", "expected_task_types": ["retrieve_images", "filter_metadata", "synthesize"],
     "vision_required": False, "note": "宽度是已有元数据"},
    {"id": 8, "category": "B", "query": "找已有标签包含“旅行”的照片",
     "expected_route": "legacy", "expected_task_types": ["retrieve_images", "filter_metadata", "synthesize"],
     "vision_required": False, "note": "“已有标签”是元数据"},
    # ---- C. Group（4）----
    {"id": 9, "category": "C", "query": "找旅行照片，按地点分组",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "group_images", "synthesize"],
     "vision_required": False, "note": "分组=元数据，非 Vision"},
    {"id": 10, "category": "C", "query": "找图库照片，按年份整理",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "group_images", "synthesize"],
     "vision_required": False, "note": "按年份分组"},
    {"id": 11, "category": "C", "query": "找动物照片，按文件类型分类",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "group_images", "synthesize"],
     "vision_required": False, "note": "按文件类型分组"},
    {"id": 12, "category": "C", "query": "找风景照片，按横向和竖向分组",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "group_images", "synthesize"],
     "vision_required": False, "note": "orientation 分组，元数据"},
    # ---- D. Vision verification（4）----
    {"id": 13, "category": "D", "query": "找派对照片，再确认画面中是否有生日蛋糕",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "像素核验：画面中是否有生日蛋糕"},
    {"id": 14, "category": "D", "query": "找候选照片，再判断是否能看见红色自行车",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "颜色/物体核验"},
    {"id": 15, "category": "D", "query": "找室内照片，再识别画面中是否有儿童",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "画面人物核验"},
    {"id": 16, "category": "D", "query": "找风景照片，再判断天空是否阴沉",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "场景/天气核验"},
    # ---- E. Retrieval + visual comparison（4）----
    {"id": 17, "category": "E", "query": "先找建筑照片，再比较哪几张构图最对称",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "构图比较=像素核验"},
    {"id": 18, "category": "E", "query": "找人像候选，然后判断哪些人物眼睛清晰可见",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "细节可见性=像素核验"},
    {"id": 19, "category": "E", "query": "找商品照片，再比较包装主色是否为蓝色",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "颜色比较=像素核验"},
    {"id": 20, "category": "E", "query": "找旅行照片，再确认哪些画面里确实同时有山和湖",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "多物体共存=像素核验"},

    # ---- F. Retrieval + Metadata（4）----
    {"id": 21, "category": "F", "query": "找 2025 年旅行照片，再按地点分组",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "filter_metadata", "group_images", "synthesize"],
     "vision_required": False, "note": "年份过滤 + 地点分组，均元数据"},
    {"id": 22, "category": "F", "query": "找横向照片，按文件类型分类",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "filter_metadata", "group_images", "synthesize"],
     "vision_required": False, "note": "orientation + file_type 均元数据"},
    {"id": 23, "category": "F", "query": "找已有标签为动物的照片，再按年份整理",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "filter_metadata", "group_images", "synthesize"],
     "vision_required": False, "note": "已有标签 + 年份，均元数据"},
    {"id": 24, "category": "F", "query": "找宽度至少 1920 的照片，然后按月份归组",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "filter_metadata", "group_images", "synthesize"],
     "vision_required": False, "note": "宽度 + 月份，均元数据"},
    # ---- G. Retrieval + Metadata + Vision（4）----
    {"id": 25, "category": "G", "query": "找 2025 年照片，再确认画面中有人骑马",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "filter_metadata", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "年份元数据 + 骑马像素核验"},
    {"id": 26, "category": "G", "query": "从横向 JPG 照片中找出画面有海滩的，再推荐",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "filter_metadata", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "格式元数据 + 海滩像素核验"},
    {"id": 27, "category": "G", "query": "先按已有旅行标签筛选，再确认哪些照片有雪山",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "filter_metadata", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "已有标签 + 雪山像素核验"},
    {"id": 28, "category": "G", "query": "找宽度至少 1920 的图片，再判断构图是否适合横幅",
     "expected_route": "team", "expected_task_types": ["retrieve_images", "filter_metadata", "analyze_visual", "synthesize"],
     "vision_required": True, "note": "宽度元数据 + 构图像素核验"},
]

VISION_POSITIVE_CATEGORIES = {"D", "E", "G"}
VISION_NEGATIVE_CATEGORIES = {"A", "B", "C", "F"}


def vision_positive_set() -> List[Dict[str, Any]]:
    return [q for q in GOLD_QUERIES if q["category"] in VISION_POSITIVE_CATEGORIES]


def vision_negative_set() -> List[Dict[str, Any]]:
    return [q for q in GOLD_QUERIES if q["category"] in VISION_NEGATIVE_CATEGORIES]


def score_plan(query: Dict[str, Any], actual_task_types: List[str], validation_ok: bool = True) -> Dict[str, Any]:
    """离线评分：单条 query 的 planner 输出打分。"""
    expected = set(query["expected_task_types"])
    actual = set(actual_task_types or [])
    vision_required = bool(query["vision_required"])
    vision_selected = "analyze_visual" in actual
    return {
        "query_id": query["id"],
        "exact_task_types_match": actual == expected,
        "missing": sorted(expected - actual),
        "extra": sorted(actual - expected),
        "vision_required": vision_required,
        "vision_selected": vision_selected,
        "vision_false_negative": vision_required and not vision_selected,
        "vision_false_positive": (not vision_required) and vision_selected,
        "valid": bool(validation_ok),
        "actual_task_types": sorted(actual),
    }


def aggregate_scores(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    return {
        "n": n,
        "exact_task_types_match": sum(1 for r in results if r["exact_task_types_match"]),
        "vision_false_negative": sum(1 for r in results if r["vision_false_negative"]),
        "vision_false_positive": sum(1 for r in results if r["vision_false_positive"]),
        "invalid": sum(1 for r in results if not r["valid"]),
    }


if __name__ == "__main__":
    print("28-query routing benchmark dataset（审计 §10）")
    print(f"total={len(GOLD_QUERIES)}  vision_positive={len(vision_positive_set())}  "
          f"vision_negative={len(vision_negative_set())}")
    for q in GOLD_QUERIES:
        print(f"  [{q['category']:>1}] #{q['id']:>2} {q['query']}  → {q['expected_route']} "
              f"{q['expected_task_types']}  vision={'Y' if q['vision_required'] else 'N'}")
    print("\n真实 probe 需要显式授权（28×2 次真实 Planner 调用）；普通 regression 不调用真实 API。")
