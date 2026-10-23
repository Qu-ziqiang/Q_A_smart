"""Planner Benchmark v2 —— 60-Query Gold Dataset（Evaluation only，不修改生产代码）。

与 v1（routing_benchmark_28.py，28 条 frozen）的关系：
- v1 保留不变（作为 frozen baseline / 旧 benchmark）；
- v2 扩大规模（Easy 15 / Medium 20 / Hard 15 / Very Hard 10）+ 显式 dependency gold
  + synthesis target + difficulty，用于评估 Planner 的任务拆解与依赖生成能力边界。

Gold 原则（防 circular evaluation）：
- expected_task_types 与 expected_dependencies 由**人工设计的“理想业务计划”**定义，
  不是从 Planner 输出回填；
- 只允许现有 5 种能力：retrieve_images / filter_metadata / group_images /
  analyze_visual / synthesize；
- dependency gold 采用“按 query 业务顺序的线性链”表示（R 开头、S 结尾），
  以唯一 canonical 形式保证可比性；测评时另报告边级 dependency accuracy 以容忍
  语义等价但拓扑不同的合法计划（见 evaluator）。

每条 query 字段：
  id / query / difficulty(easy|medium|hard|very_hard) / expected_route /
  expected_task_types(有序=理想执行顺序) / vision_required /
  expected_dependencies({task_type: [直接依赖的 task_type]}) /
  expected_synthesis_task / note

本文件自身只含数据集与完整性检查，不发任何真实 API 调用。
"""

from typing import Dict, List, Tuple

R = "retrieve_images"
F = "filter_metadata"
G = "group_images"
V = "analyze_visual"
S = "synthesize"

ALLOWED = {R, F, G, V, S}


def _chain_gold(types: List[str]) -> Dict[str, List[str]]:
    """由有序 expected_task_types 生成 canonical dependency gold（线性链）。"""
    deps = {}
    prev = None
    for t in types:
        deps[t] = [prev] if prev is not None else []
        prev = t
    return deps


def _build(rows: List[Tuple[int, str, List[str], str, str]]) -> List[Dict]:
    out = []
    for i, (qid, difficulty, types, query, note) in enumerate(rows, 1):
        if qid != i:
            raise ValueError("dataset 顺序/编号不连续")
        assert types[0] == R and types[-1] == S and len(set(types)) == len(types)
        assert set(types) <= ALLOWED
        has_group = G in types
        has_vision = V in types
        route = "team" if (has_group or has_vision) else "legacy"
        deps = _chain_gold(types)
        # 校验链一致性：除 R 外每个类型有且仅有一个直接前置，且无环
        seen = set()
        prev = None
        for t in types:
            if t in seen:
                raise ValueError("duplicate task type in gold")
            seen.add(t)
            if prev is not None:
                assert deps[t] == [prev]
            else:
                assert deps[t] == []
            prev = t
        out.append({
            "id": qid,
            "query": query,
            "difficulty": difficulty,
            "expected_route": route,
            "expected_task_types": types,
            "vision_required": has_vision,
            "expected_dependencies": deps,
            "expected_synthesis_task": S,
            "note": note,
        })
    return out


_EASY: List[Tuple[int, str, List[str], str, str]] = [
    (1, "easy", [R, S], "找一些猫的照片。", "纯检索单路径"),
    (2, "easy", [R, S], "找几张海边日落的照片。", "纯检索单路径"),
    (3, "easy", [R, S], "找城市夜景的照片。", "纯检索单路径"),
    (4, "easy", [R, S], "推荐几张可爱的狗狗照片。", "纯检索单路径"),
    (5, "easy", [R, S], "找雪山风景照。", "纯检索单路径"),
    (6, "easy", [R, S], "找星空摄影作品。", "纯检索单路径"),
    (7, "easy", [R, S], "找复古咖啡馆氛围的照片。", "纯检索单路径"),
    (8, "easy", [R, S], "找自行车骑行主题的照片。", "纯检索单路径"),
    (9, "easy", [R, S], "找几张花卉特写照片。", "纯检索单路径"),
    (10, "easy", [R, S], "找一组美食摆拍照片。", "纯检索单路径"),
    (11, "easy", [R, F, S], "找2025年上传的照片。", "单条元数据过滤"),
    (12, "easy", [R, F, S], "找横向的JPG照片。", "orientation+file_type 元数据"),
    (13, "easy", [R, F, S], "找宽度至少1920像素的照片。", "width 元数据"),
    (14, "easy", [R, G, S], "找旅行照片，按地点分组整理结果。", "简单分组"),
    (15, "easy", [R, G, S], "找动物照片，按文件类型分类整理。", "简单分组"),
]

_MEDIUM: List[Tuple[int, str, List[str], str, str]] = [
    (16, "medium", [R, G, S], "找风景照片，按拍摄年份分组并整理结果。", "按年份分组"),
    (17, "medium", [R, G, S], "找建筑照片，按拍摄月份分组并整理结果。", "按月份分组"),
    (18, "medium", [R, G, S], "找照片，按已有标签分组整理结果。", "按标签分组"),
    (19, "medium", [R, F, S], "找标签包含“旅行”且2025年拍摄的照片。", "双元数据条件"),
    (20, "medium", [R, F, S], "找横向且宽度至少1920像素的照片。", "双元数据条件"),
    (21, "medium", [R, F, G, S], "找2025年拍摄的照片，按地点分组后整理结果。", "过滤+分组"),
    (22, "medium", [R, F, G, S], "找横向照片，按拍摄年份分组后整理结果。", "过滤+分组"),
    (23, "medium", [R, F, G, S], "找标签含“旅行”的照片，按月份分组后整理结果。", "过滤+分组"),
    (24, "medium", [R, F, G, S], "找宽度至少1920的照片，按文件类型分组整理结果。", "过滤+分组"),
    (25, "medium", [R, F, G, S], "找2025年且横向的照片，按地点分组后整理推荐。", "过滤+分组+合成"),
    (26, "medium", [R, V, S], "找几张生日派对照片，确认画面中是否有生日蛋糕。", "像素核验"),
    (27, "medium", [R, V, S], "找城市街头照片，确认画面中是否有红色出租车。", "像素核验"),
    (28, "medium", [R, V, S], "找登山照片，确认画面里是否有雪山。", "像素核验"),
    (29, "medium", [R, V, S], "找室内照片，确认画面中是否有壁炉。", "像素核验"),
    (30, "medium", [R, F, V, S], "找横向照片，确认画面里有没有彩虹。", "过滤+像素核验"),
    (31, "medium", [R, F, V, S], "找2025年照片，确认画面里是否有海。", "过滤+像素核验"),
    (32, "medium", [R, F, V, S], "找宽度至少1920的照片，确认画面中是否有广告牌文字。", "过滤+像素核验"),
    (33, "medium", [R, F, V, S], "找标签含“宠物”的照片，确认画面里是否有猫。", "过滤+像素核验"),
    (34, "medium", [R, F, G, S], "找横向且2025年的照片，按地点分组后整理推荐。", "双过滤+分组"),
    (35, "medium", [R, G, V, S], "找动物照片，先按地点分组，再确认每组画面中是否有狗，最后整理结果。", "分组+像素核验"),
]

_HARD: List[Tuple[int, str, List[str], str, str]] = [
    (36, "hard", [R, V, S], "找一组体育赛事照片，逐一确认画面里是否有奖杯，再推荐有奖杯的照片。", "核验后过滤候选"),
    (37, "hard", [R, F, S], "找2023年至2025年间拍摄、横向且标签包含“建筑”的照片。", "多字段复杂过滤"),
    (38, "hard", [R, F, G, S], "找标签含“旅行”且横向的照片，先按拍摄年份分组，再整理推荐每组最佳。", "过滤+分组+挑选"),
    (39, "hard", [R, F, V, S], "找2025年、横向的户外照片，确认画面里是否有雪山，再推荐适合做封面的照片。", "过滤+核验+精选"),
    (40, "hard", [R, G, V, S], "找风景照片，先按地点分组，再确认每组画面里是否有海景，最后整理结果。", "分组+组内核验"),
    (41, "hard", [R, F, G, S], "找2025年、横向且宽度至少1920的照片，按地点分组后整理。", "三重过滤+分组"),
    (42, "hard", [R, F, V, S], "找标签含“旅行”且横向的照片，确认画面里有没有山，再整理推荐。", "过滤+核验"),
    (43, "hard", [R, G, V, S], "找宠物照片，先按地点分组，再确认每组画面中是否有猫，最后整理成报告。", "分组+组内核验"),
    (44, "hard", [R, F, V, S], "找2025年、标签含“旅行”的照片，确认画面中是否有历史建筑。", "过滤+核验"),
    (45, "hard", [R, F, G, S], "找宽度至少1920且横向的照片，按月份分组并整理结果。", "过滤+分组"),
    (46, "hard", [R, V, S], "找办公室照片，确认画面里是否有白板，只保留有白板的候选再推荐。", "核验后过滤候选"),
    (47, "hard", [R, F, V, S], "找2025年拍摄的风景照片，确认画面里是否有瀑布，只保留有瀑布的候选再推荐。", "过滤+核验+过滤"),
    (48, "hard", [R, G, V, S], "找旅行照片，按拍摄年份分组，再确认每组画面中是否有当地特色小吃，最后整理推荐。", "分组+组内核验"),
    (49, "hard", [R, F, G, S], "找标签含“美食”且横向的照片，按拍摄年份分组后整理推荐。", "过滤+分组"),
    (50, "hard", [R, V, S], "找滑雪照片，确认画面中是否有跳台，并推荐包含跳台的照片。", "核验后过滤候选"),
]

_VERY_HARD: List[Tuple[int, str, List[str], str, str]] = [
    (51, "very_hard", [R, F, G, V, S],
     "找2024年至2025年拍摄的旅行风景照片，只要横向的，确认画面中是否有雪山，再按拍摄年份分组，每组推荐几张并说明原因。",
     "全五能力链"),
    (52, "very_hard", [R, F, G, V, S],
     "从图库中找出2025年、标签含“宠物”且宽度至少1920的照片，确认画面里有没有猫，最后按地点分组并整理推荐。",
     "全五能力链"),
    (53, "very_hard", [R, G, V, S],
     "找美食照片，先按地点分组，再确认每组画面里是否有火锅，最后每组给出推荐并说明理由。",
     "分组+组内核验+分组推荐"),
    (54, "very_hard", [R, F, G, V, S],
     "找标签含“建筑”且横向的照片，先按年份分组，再确认每组画面中是否有哥特式教堂，最后整理成带推荐理由的清单。",
     "全五能力链"),
    (55, "very_hard", [R, F, G, V, S],
     "找2025年拍摄的户外运动照片，确认画面中是否有滑雪者，按地点分组后每组推荐并说明原因。",
     "全五能力链"),
    (56, "very_hard", [R, G, V, S],
     "找旅行照片，按年份分组后，确认各组画面中是否有海边落日，并整理推荐。",
     "分组+组内核验"),
    (57, "very_hard", [R, F, G, V, S],
     "找宽度至少1920的室内照片，确认画面里是否有书架，再按地点分组并整理推荐理由。",
     "全五能力链"),
    (58, "very_hard", [R, F, G, V, S],
     "找标签含“旅行”且横向的照片，先确认画面里是否有古桥，再把有古桥的照片按拍摄年份分组，并整理推荐。",
     "全五能力链"),
    (59, "very_hard", [R, F, G, V, S],
     "找2025年、横向、宽度至少1920的照片，先按地点分组，再确认每组中是否有雪山并推荐。",
     "全五能力链"),
    (60, "very_hard", [R, G, V, S],
     "找图库里适合做壁纸的照片，先按拍摄年份分组，再确认各组画面中是否有星空，最后每组推荐一张并说明理由。",
     "分组+组内核验"),
]

GOLD_V2_QUERIES: List[Dict] = _build(_EASY + _MEDIUM + _HARD + _VERY_HARD)

DIFFICULTY_EXPECT = {"easy": 15, "medium": 20, "hard": 15, "very_hard": 10}
ALLOWED_TYPES = ALLOWED


def integrity_check() -> List[str]:
    errors = []
    from collections import Counter

    n = len(GOLD_V2_QUERIES)
    if n != 60:
        errors.append(f"total {n} != 60")
    diff = Counter(q["difficulty"] for q in GOLD_V2_QUERIES)
    for k, v in DIFFICULTY_EXPECT.items():
        if diff.get(k) != v:
            errors.append(f"difficulty {k}: {diff.get(k)} != {v}")
    ids = [q["id"] for q in GOLD_V2_QUERIES]
    if ids != list(range(1, 61)):
        errors.append("id 不连续唯一")
    for q in GOLD_V2_QUERIES:
        ts = q["expected_task_types"]
        if ts[0] != R or ts[-1] != S or len(set(ts)) != len(ts):
            errors.append(f"#{q['id']} expected_task_types 非法")
        if "synthesize" not in q["expected_synthesis_task"]:
            errors.append(f"#{q['id']} synthesis target 非法")
        exp_vis = V in ts
        if q["vision_required"] != exp_vis:
            errors.append(f"#{q['id']} vision_required 与 task_types 不一致")
        exp_route = "team" if (G in ts or V in ts) else "legacy"
        if q["expected_route"] != exp_route:
            errors.append(f"#{q['id']} route 与 task_types 不一致")
        # dependency gold 必须是其自身 task_types 子集的合法链（无环/无未知类型）
        for t, deps in q["expected_dependencies"].items():
            if t not in ts or any(d not in ts for d in deps):
                errors.append(f"#{q['id']} dep 引用未知类型")
    return errors


if __name__ == "__main__":
    errs = integrity_check()
    print("INTEGRITY:", "PASS" if not errs else errs)
    from collections import Counter

    print("total =", len(GOLD_V2_QUERIES), Counter(q["difficulty"] for q in GOLD_V2_QUERIES))
    print("vision_required =", sum(1 for q in GOLD_V2_QUERIES if q["vision_required"]))
    print("route team =", sum(1 for q in GOLD_V2_QUERIES if q["expected_route"] == "team"))

