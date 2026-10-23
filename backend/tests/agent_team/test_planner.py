"""Phase 1-A — PlannerAgent 单元测试（全部使用 Fake/Mock LLM，不消耗真实 API）。

运行方式（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_planner.py
"""

import copy
import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    PlanValidator,
    PlannerAgent,
    PlannerFailure,
    TaskPlanDraft,
    TaskType,
)

_VALID_PLAN_JSON = {
    "goal": "找旅行照片，按地点分类，然后推荐最合适的几张",
    "synthesis_task_id": "t3",
    "tasks": [
        {
            "task_id": "t1",
            "task_type": "retrieve_images",
            "objective": "检索旅行照片",
            "input": {"query": "旅行照片", "limit": 30},
            "depends_on": [],
        },
        {
            "task_id": "t2",
            "task_type": "group_images",
            "objective": "按地点分组",
            "input": {"group_by": "location"},
            "depends_on": ["t1"],
        },
        {
            "task_id": "t3",
            "task_type": "synthesize",
            "objective": "汇总生成最终推荐",
            "input": {"recommendation_limit": 10, "presentation": "grouped"},
            "depends_on": ["t2"],
        },
    ],
}


class FakeLLM:
    """Fake LLM：直接返回预设文本；可配置抛异常。"""

    def __init__(self, response="", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


# ---------------------------------------------------------------------------
# 合法输出
# ---------------------------------------------------------------------------
def test_planner_valid_plan_passes_validator():
    fake = FakeLLM(response=json.dumps(_VALID_PLAN_JSON, ensure_ascii=False))
    planner = PlannerAgent(llm=fake)
    draft = planner.create_plan(goal="找旅行照片，按地点分类，然后推荐最合适的几张")
    assert isinstance(draft, TaskPlanDraft)
    assert len(draft.tasks) == 3

    r = PlanValidator().validate(draft)
    assert r.valid, r.errors
    assert r.plan.synthesis_task_id == "t3"


def test_planner_accepts_fenced_json():
    fake = FakeLLM(
        response="好的，这是计划：\n```json\n"
        + json.dumps(_VALID_PLAN_JSON, ensure_ascii=False)
        + "\n```"
    )
    planner = PlannerAgent(llm=fake)
    draft = planner.create_plan(goal="x")
    assert len(draft.tasks) == 3


def test_planner_injects_recent_messages():
    fake = FakeLLM(response=json.dumps(_VALID_PLAN_JSON, ensure_ascii=False))
    planner = PlannerAgent(llm=fake)
    planner.create_plan(
        goal="找旅行照片",
        recent_messages=[
            {"role": "user", "content": "帮我找旅行照片"},
            {"role": "assistant", "content": "好的"},
        ],
    )
    assert len(fake.calls) == 1
    messages = fake.calls[0]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert "旅行照片" in messages[1]["content"]


# ---------------------------------------------------------------------------
# 非法输出 → PlannerFailure
# ---------------------------------------------------------------------------
def test_planner_invalid_json():
    fake = FakeLLM(response="这不是 JSON")
    planner = PlannerAgent(llm=fake)
    try:
        planner.create_plan(goal="x")
        assert False, "应当抛出 PlannerFailure"
    except PlannerFailure:
        pass


def test_planner_invalid_task_type():
    bad = copy.deepcopy(_VALID_PLAN_JSON)
    bad["tasks"][0]["task_type"] = "fly_to_moon"
    fake = FakeLLM(response=json.dumps(bad, ensure_ascii=False))
    planner = PlannerAgent(llm=fake)
    try:
        planner.create_plan(goal="x")
        assert False, "应当抛出 PlannerFailure"
    except PlannerFailure as e:
        assert "结构校验失败" in str(e)


def test_planner_missing_required_field():
    bad = copy.deepcopy(_VALID_PLAN_JSON)
    bad["tasks"][0].pop("task_id")
    fake = FakeLLM(response=json.dumps(bad, ensure_ascii=False))
    planner = PlannerAgent(llm=fake)
    try:
        planner.create_plan(goal="x")
        assert False, "应当抛出 PlannerFailure"
    except PlannerFailure:
        pass


def test_planner_unknown_field_rejected():
    bad = copy.deepcopy(_VALID_PLAN_JSON)
    bad["tasks"][0]["evil_field"] = "should be rejected"
    fake = FakeLLM(response=json.dumps(bad, ensure_ascii=False))
    planner = PlannerAgent(llm=fake)
    try:
        planner.create_plan(goal="x")
        assert False, "应当抛出 PlannerFailure"
    except PlannerFailure:
        pass


def test_planner_llm_raises():
    fake = FakeLLM(error=RuntimeError("network down"))
    planner = PlannerAgent(llm=fake)
    try:
        planner.create_plan(goal="x")
        assert False, "应当抛出 PlannerFailure"
    except PlannerFailure as e:
        assert "LLM 调用失败" in str(e)


def test_planner_empty_goal():
    planner = PlannerAgent(llm=FakeLLM(response="{}"))
    try:
        planner.create_plan(goal="   ")
        assert False, "应当抛出 PlannerFailure"
    except PlannerFailure:
        pass


def test_planner_output_not_object():
    fake = FakeLLM(response='["a", "b"]')
    planner = PlannerAgent(llm=fake)
    try:
        planner.create_plan(goal="x")
        assert False, "应当抛出 PlannerFailure"
    except PlannerFailure:
        pass


def test_planner_invalid_task_id_format():
    bad = copy.deepcopy(_VALID_PLAN_JSON)
    bad["tasks"][0]["task_id"] = "T1-Upper"
    fake = FakeLLM(response=json.dumps(bad, ensure_ascii=False))
    planner = PlannerAgent(llm=fake)
    try:
        planner.create_plan(goal="x")
        assert False, "应当抛出 PlannerFailure"
    except PlannerFailure:
        pass


# ---------------------------------------------------------------------------
# 纯 python 运行器
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
