"""Phase 1 Specialist — AgentCard 单元测试。

覆盖：
  - AgentCard 创建与字段默认值
  - 未知字段拒绝（extra=forbid，延续既有模型纪律）
  - 默认三张 Agent Card（retrieval / organizer / vision）字段语义
  - VisionAgent.enabled 跟随视觉能力门禁（vision_ok 参数，不读真实配置）

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_agent_cards.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from pydantic import ValidationError

from backend.ai_func.recommendation.orchestration import (
    AgentCard,
    build_default_agent_cards,
    default_agent_card_map,
)
from backend.ai_func.recommendation.orchestration.models import TaskType

_ROLE_ORDER = ("retrieval", "organizer", "vision")


def test_create_minimal_card():
    card = AgentCard(
        agent_id="retrieval",
        name="RetrievalAgent",
        description="semantic retrieval / vector search / reranking",
        supported_tasks=[TaskType.RETRIEVE_IMAGES],
        available_tools=["search_images", "rerank"],
        input_schema="RetrieveImagesInput",
        output_schema="RetrievalOutput",
        cost_level="cheap",
    )
    assert card.agent_id == "retrieval"
    assert card.name == "RetrievalAgent"
    assert card.description == "semantic retrieval / vector search / reranking"
    assert card.supported_tasks == [TaskType.RETRIEVE_IMAGES]
    assert card.available_tools == ["search_images", "rerank"]
    assert card.input_schema == "RetrieveImagesInput"
    assert card.output_schema == "RetrievalOutput"
    assert card.cost_level == "cheap"
    assert card.enabled is True  # 默认启用


def test_card_field_defaults_for_optional_fields():
    card = AgentCard(agent_id="x", name="X", description="d")
    assert card.supported_tasks == []
    assert card.available_tools == []
    assert card.input_schema == ""
    assert card.output_schema == ""
    assert card.cost_level == "cheap"
    assert card.enabled is True


def test_card_rejects_unknown_fields():
    try:
        AgentCard(agent_id="x", name="X", description="d", evil_field=1)
    except ValidationError as e:
        assert "extra_forbidden" in str(e).lower() or "extra" in str(e).lower()
    else:
        raise AssertionError("extra=forbid 未生效")


def test_default_cards_three_roles_and_fields():
    cards = build_default_agent_cards(vision_ok=False)
    assert [c.agent_id for c in cards] == list(_ROLE_ORDER)
    by_id = {c.agent_id: c for c in cards}

    retrieval = by_id["retrieval"]
    assert retrieval.name == "RetrievalAgent"
    assert TaskType.REWRITE_QUERY in retrieval.supported_tasks
    assert TaskType.EXPAND_QUERY in retrieval.supported_tasks
    assert TaskType.SCOPE_DATASET in retrieval.supported_tasks
    assert TaskType.FILTER_BY_LABELS in retrieval.supported_tasks
    assert TaskType.RETRIEVE_IMAGES in retrieval.supported_tasks
    assert TaskType.RETRIEVE_SIMILAR in retrieval.supported_tasks
    assert TaskType.RERANK_CANDIDATES in retrieval.supported_tasks
    assert retrieval.available_tools == ["search_images", "rerank"]
    assert retrieval.cost_level == "cheap"
    assert retrieval.enabled is True

    organizer = by_id["organizer"]
    assert organizer.name == "OrganizerAgent"
    assert organizer.supported_tasks == [
        TaskType.FILTER_METADATA,
        TaskType.DEDUPLICATE_IMAGES,
        TaskType.GROUP_IMAGES,
        TaskType.SORT_IMAGES,
        TaskType.SELECT_IMAGES,
    ]
    assert organizer.available_tools == ["filter_images", "group_images"]
    assert organizer.cost_level == "cheap"
    assert organizer.enabled is True

    vision = by_id["vision"]
    assert vision.name == "VisionAgent"
    assert vision.supported_tasks == [
        TaskType.ANALYZE_VISUAL,
        TaskType.VERIFY_VISUAL_CONDITION,
        TaskType.READ_VISIBLE_TEXT,
        TaskType.LIST_OBJECTS,
        TaskType.COUNT_NAMED_OBJECT,
        TaskType.JUDGE_MAIN_SUBJECT,
        TaskType.VERIFY_SPATIAL,
    ]
    assert vision.available_tools == ["analyze_image"]
    assert vision.input_schema == "AnalyzeVisualInput / VerifyVisualInput"
    assert vision.output_schema == "VisionTaskOutput / VerifyTaskOutput"
    assert vision.cost_level == "expensive"


def test_default_cards_vision_enabled_follows_flag():
    # vision_ok=False → VisionAgent 存在但 disabled（能力门禁映射到卡片）
    cards_off = build_default_agent_cards(vision_ok=False)
    vision_off = next(c for c in cards_off if c.agent_id == "vision")
    assert vision_off.enabled is False
    # 其余角色不受影响
    assert all(c.enabled for c in cards_off if c.agent_id != "vision")

    cards_on = build_default_agent_cards(vision_ok=True)
    vision_on = next(c for c in cards_on if c.agent_id == "vision")
    assert vision_on.enabled is True


def test_default_agent_card_map_indexed_by_id():
    m = default_agent_card_map(vision_ok=False)
    assert set(m.keys()) == set(_ROLE_ORDER)
    assert m["retrieval"].name == "RetrievalAgent"
    assert m["vision"].enabled is False


# ---------------------------------------------------------------------------
def _main():
    failed = 0
    passed = 0
    mod_globals = globals()
    for name in sorted(mod_globals):
        if name.startswith("test_"):
            fn = mod_globals[name]
            try:
                fn()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print(f"\n== {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
