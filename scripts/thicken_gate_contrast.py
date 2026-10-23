"""把 Gate 反例加厚到 100+，test 切分 ≥30。只改数据集，不改规则。"""

from __future__ import annotations

import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from backend.ai_func.recommendation.evaluation.gate_track import contrast_split

_PATH = os.path.join(
    _REPO, "backend", "ai_func", "recommendation",
    "evaluation", "datasets", "gate_contrast_legacy.json",
)

_EXTRA = [
    ("neg_61", "找莫兰迪色调静物"),
    ("neg_62", "推荐几张日系小清新室内"),
    ("neg_63", "找暗调电影海报"),
    ("neg_64", "找高对比度黑白街拍，只要风格检索"),
    ("neg_65", "找低饱和度海边"),
    ("neg_66", "看起来很有氛围感的咖啡馆"),
    ("neg_67", "找视觉上舒服的莫兰迪配色图"),
    ("neg_68", "构图感强的街拍素材，只要检索"),
    ("neg_69", "确认一下搜索词写成猫就可以"),
    ("neg_70", "判断文件是不是 JPG 格式"),
    ("neg_71", "识别扩展名是 png 的图片"),
    ("neg_72", "确认上传日期是 2024"),
    ("neg_73", "是否有竖向的壁纸可以看看"),
    ("neg_74", "找已有标签带风景的图"),
    ("neg_75", "找宽度大于 1280 的图"),
    ("neg_76", "找高度至少 720 的照片"),
    ("neg_77", "给我几张森系人像"),
    ("neg_78", "找港风夜景霓虹"),
    ("neg_79", "找奶油色家居图"),
    ("neg_80", "找雾霾蓝天空"),
    ("neg_81", "找油画质感花束"),
    ("neg_82", "找像素风游戏截图风格海报"),
    ("neg_83", "找ins风穿搭平铺"),
    ("neg_84", "找日落下的剪影，不要核验天空颜色"),
    ("neg_85", "找红色裙子海报，不要判断是不是红的"),
    ("neg_86", "找对称教堂照片，不要比较构图"),
    ("neg_87", "找山湖同框风景，只要语义召回"),
    ("neg_88", "找海滩椰树明信片"),
    ("neg_89", "找骑马杂志封面"),
    ("neg_90", "找横幅广告底图"),
    ("neg_91", "再给我两张狗"),
    ("neg_92", "然后再找几张花就行"),
    ("neg_93", "确认收藏夹名字叫旅行"),
    ("neg_94", "判断一下文件名有没有 sunset"),
    ("neg_95", "识别目录里带 2025 的文件"),
    ("neg_96", "比较一下哪个文件夹照片更多，看数量就行"),
    ("neg_97", "找适合当桌面的星空"),
    ("neg_98", "找看起来柔和的窗边光"),
    ("neg_99", "找干净白底化妆品图"),
    ("neg_100", "找正方形头像框素材，只要尺寸"),
    ("neg_101", "找胶片颗粒感人像"),
    ("neg_102", "找赛博夜雨街道"),
    ("neg_103", "找暖黄灯光小酒馆"),
    ("neg_104", "找冷青调地铁站"),
    ("neg_105", "找水彩风植物插画"),
    ("neg_106", "找卡通猫咪贴纸"),
    ("neg_107", "找极简建筑外立面"),
    ("neg_108", "找复古火车车厢"),
    ("neg_109", "找金色麦田航拍风"),
    ("neg_110", "找蓝色海洋波纹"),
    ("neg_111", "有没有适合做头图的城市"),
    ("neg_112", "我想看一些安静的书店照片"),
    ("neg_113", "找高饱和度热带水果"),
    ("neg_114", "找低对比度清晨薄雾"),
    ("neg_115", "找电影感窗边人像"),
    ("neg_116", "确认我要搜的是樱花不是桃花"),
    ("neg_117", "判断搜索范围只要 2023 年"),
    ("neg_118", "识别标签字段里写着动物的"),
    ("neg_119", "找自行车零件静物"),
    ("neg_120", "找蛋糕甜品俯拍"),
]


def main() -> int:
    with open(_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    have = {q["id"] for q in data["queries"]}
    added = 0
    for qid, query in _EXTRA:
        if qid in have:
            continue
        data["queries"].append({"id": qid, "query": query})
        have.add(qid)
        added += 1
    with open(_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    n_dev = n_test = 0
    test_ids = []
    for q in data["queries"]:
        if contrast_split(q["id"]) == "test":
            n_test += 1
            test_ids.append(q["id"])
        else:
            n_dev += 1
    print(json.dumps({
        "path": _PATH,
        "added": added,
        "total": len(data["queries"]),
        "dev": n_dev,
        "test": n_test,
        "test_ids": test_ids,
    }, ensure_ascii=False, indent=2))
    if len(data["queries"]) < 100 or n_test < 30:
        print("FAIL: need total>=100 and test>=30", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
