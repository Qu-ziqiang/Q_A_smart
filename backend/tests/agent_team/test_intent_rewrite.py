"""Intent Rewrite — 单元测试（test_intent_rewrite.py）。

覆盖：metadata 条件 / group_by / visual verification / 普通检索 / greeting·无关 Query /
确定性（同 query 多次结果一致）/ 28 条 routing gold 对齐（能力标记 == expected_task_types）。

运行（repo 根）：
    venv\\Scripts\\python.exe -X utf8 -m pytest backend/tests/agent_team/test_intent_rewrite.py -q
"""

import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.intent import IntentRewrite, NormalizedIntent
from backend.ai_func.recommendation.orchestration.models import (
    MetadataPredicate,
    _ALLOWED_GROUP_BY,
)
from backend.tests.agent_team.routing_benchmark_28 import GOLD_QUERIES

_REWRITER = IntentRewrite()
_GROUP_BY_VALUES = set(getattr(_ALLOWED_GROUP_BY, "__args__", ()))


def _field_names(intent: NormalizedIntent) -> set:
    return {c.field for c in intent.metadata_constraints}


def _tag_value(intent: NormalizedIntent) -> str:
    for c in intent.metadata_constraints:
        if c.field == "tags":
            return str(c.value)
    return ""


# ---------------------------------------------------------------------------
# metadata 条件
# ---------------------------------------------------------------------------
def test_metadata_year_created_at():
    i = _REWRITER.rewrite("找 2025 年上传的照片")
    assert any(c.field == "created_at" and c.operator == "contains" and c.value == "2025"
               for c in i.metadata_constraints)


def test_metadata_orientation_and_file_type():
    i = _REWRITER.rewrite("找横向 JPG 照片")
    fields = _field_names(i)
    assert "orientation" in fields and "file_type" in fields
    assert any(c.field == "orientation" and c.value == "landscape" for c in i.metadata_constraints)
    assert any(c.field == "file_type" and c.value == "jpg" for c in i.metadata_constraints)


def test_metadata_width_gte():
    i = _REWRITER.rewrite("找宽度至少 1920 的照片")
    assert any(c.field == "width" and c.operator == "gte" and c.value == 1920
               for c in i.metadata_constraints)


def test_metadata_tags_contains():
    i = _REWRITER.rewrite("找已有标签包含“旅行”的照片")
    assert _tag_value(i) == "旅行"


def test_metadata_constraint_uses_existing_vocab():
    for c in _REWRITER.rewrite("找横向 JPG 照片").metadata_constraints:
        assert isinstance(c, MetadataPredicate)


# ---------------------------------------------------------------------------
# group_by
# ---------------------------------------------------------------------------
def test_group_location():
    i = _REWRITER.rewrite("找旅行照片，按地点分组")
    assert i.group_by == "location"
    assert not i.needs_visual_verification


def test_group_year_and_month_and_file_type_and_orientation():
    assert _REWRITER.rewrite("找图库照片，按年份整理").group_by == "year"
    assert _REWRITER.rewrite("找动物照片，按文件类型分类").group_by == "file_type"
    assert _REWRITER.rewrite("找风景照片，按横向和竖向分组").group_by == "orientation"
    assert _REWRITER.rewrite("找宽度至少 1920 的照片，然后按月份归组").group_by == "month"


def test_group_by_in_vocab():
    g = _REWRITER.rewrite("找旅行照片，按地点分组").group_by
    assert g in _GROUP_BY_VALUES


def test_group_phrase_not_counted_as_metadata_constraint():
    i = _REWRITER.rewrite("找风景照片，按横向和竖向分组")
    assert "orientation" not in _field_names(i)
    assert i.group_by == "orientation"


def test_metadata_and_group_coexist():
    i = _REWRITER.rewrite("找 2025 年旅行照片，再按地点分组")
    assert any(c.field == "created_at" for c in i.metadata_constraints)
    assert i.group_by == "location"

# ---------------------------------------------------------------------------
# visual verification
# ---------------------------------------------------------------------------
_VISION_QUERIES = [
    "找派对照片，再确认画面中是否有生日蛋糕",
    "找候选照片，再判断是否能看见红色自行车",
    "找室内照片，再识别画面中是否有儿童",
    "找风景照片，再判断天空是否阴沉",
    "先找建筑照片，再比较哪几张构图最对称",
    "找人像候选，然后判断哪些人物眼睛清晰可见",
    "找商品照片，再比较包装主色是否为蓝色",
    "找旅行照片，再确认哪些画面里确实同时有山和湖",
    "找 2025 年照片，再确认画面中有人骑马",
    "从横向 JPG 照片中找出画面有海滩的，再推荐",
    "先按已有旅行标签筛选，再确认哪些照片有雪山",
    "找宽度至少 1920 的图片，再判断构图是否适合横幅",
]


def test_visual_verification_detected_with_criteria():
    for q in _VISION_QUERIES:
        i = _REWRITER.rewrite(q)
        assert i.needs_visual_verification, q
        assert i.visual_criteria, q


def test_metadata_and_group_do_not_trigger_vision():
    for q in [
        "找旅行照片，按地点分组",
        "找图库照片，按年份整理",
        "找 2025 年旅行照片，再按地点分组",
        "找横向照片，按文件类型分类",
        "找宽度至少 1920 的照片，然后按月份归组",
    ]:
        assert not _REWRITER.rewrite(q).needs_visual_verification, q


# ---------------------------------------------------------------------------
# 普通检索
# ---------------------------------------------------------------------------
def test_plain_retrieval_no_extra_caps():
    for q in ["找几张海边日落照片", "找城市夜景照片", "推荐几张可爱的猫照片", "找水墨画风格的山景图"]:
        i = _REWRITER.rewrite(q)
        assert i.needs_images is True
        assert i.metadata_constraints == []
        assert i.group_by is None
        assert i.needs_visual_verification is False
        assert i.retrieval_query


def test_retrieval_query_takes_first_clause():
    i = _REWRITER.rewrite("找旅行照片，按地点分组")
    assert "旅行照片" in i.retrieval_query
    assert "分组" not in i.retrieval_query


# ---------------------------------------------------------------------------
# greeting / 无关 query
# ---------------------------------------------------------------------------
def test_greeting_needs_images_false():
    for q in ["你好", "您好", "在吗", "早上好", "谢谢", "hi", "hello"]:
        assert _REWRITER.rewrite(q).needs_images is False, q


def test_irrelevant_needs_images_false():
    for q in ["今天天气怎么样", "讲个笑话", "你是谁", "帮我介绍一下你自己"]:
        assert _REWRITER.rewrite(q).needs_images is False, q


def test_greeting_with_search_signal_still_images():
    i = _REWRITER.rewrite("你好，帮我找几张可爱的猫照片")
    assert i.needs_images is True


def test_raw_and_normalized_query_kept():
    i = _REWRITER.rewrite("  帮我找几张海边日落照片  ")
    assert i.raw_query == "帮我找几张海边日落照片"
    assert i.normalized_query == "找几张海边日落照片"
    assert i.normalized_query != i.raw_query


# ---------------------------------------------------------------------------
# 确定性
# ---------------------------------------------------------------------------
def test_rewrite_deterministic():
    q = "找宽度至少 1920 的照片，再确认画面中有没有人骑马"
    a = _REWRITER.rewrite(q).model_dump()
    b = _REWRITER.rewrite(q).model_dump()
    assert a == b


def test_rewrite_normalized_idempotent():
    i = _REWRITER.rewrite("帮我找横向 JPG 照片谢谢")
    again = _REWRITER.rewrite(i.normalized_query)
    assert again.normalized_query == i.normalized_query


def test_rewrite_messages_uses_last_user():
    msgs = [{"role": "user", "content": "帮我找旅行照片，按地点分组"}]
    i = _REWRITER.rewrite_messages(msgs)
    assert i is not None and i.group_by == "location"
    assert _REWRITER.rewrite_messages([]) is None
    assert _REWRITER.rewrite_messages([{"role": "assistant", "content": "x"}]) is None


# ---------------------------------------------------------------------------
# 28 条 routing gold 对齐：Intent 能力标记 == gold expected_task_types
# ---------------------------------------------------------------------------
def test_gold_28_capability_alignment():
    mismatches = []
    for q in GOLD_QUERIES:
        i = _REWRITER.rewrite(q["query"])
        expected = set(q["expected_task_types"])
        meta = "filter_metadata" in expected
        grp = "group_images" in expected
        vis = "analyze_visual" in expected
        if bool(i.metadata_constraints) != meta:
            mismatches.append((q["id"], "metadata"))
        if (i.group_by is not None) != grp:
            mismatches.append((q["id"], "group"))
        if i.needs_visual_verification != vis:
            mismatches.append((q["id"], "vision"))
        if vis and not i.visual_criteria:
            mismatches.append((q["id"], "vision_criteria_empty"))
    assert mismatches == [], f"gold 对齐失败: {mismatches}"


def test_gold_28_always_needs_images_and_retrieval():
    for q in GOLD_QUERIES:
        i = _REWRITER.rewrite(q["query"])
        assert i.needs_images is True, q["id"]
        assert i.retrieval_query, q["id"]


# ---------------------------------------------------------------------------
def _main():
    failed = passed = 0
    g = globals()
    for name in sorted(g):
        if name.startswith("test_"):
            try:
                g[name]()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n== intent rewrite: {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

