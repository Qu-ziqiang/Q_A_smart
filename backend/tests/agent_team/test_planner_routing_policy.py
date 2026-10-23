"""Planner Capability Routing — 最小修复测试（审计 F-002b/c）。

覆盖：catalog 覆盖 / disabled-enabled 渲染 / Prompt 语义区分（Metadata vs Vision,
Retrieval vs Vision）/ schema enum 收窄与描述同源 / 单次 LLM 调用 / tool_choice 强制 /
输出仍进 Validator / 无“无条件 Vision”错误规则。

全部 Fake/Mock，零真实 API。运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_planner_routing_policy.py
"""

import json
import os
import sys
import traceback
from types import SimpleNamespace

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

import backend.ai_func.recommendation.orchestration.models as models_mod
import backend.ai_func.recommendation.orchestration.planner as planner_mod
from backend.ai_func.recommendation.orchestration import PlanValidator, TaskPlanDraft, TaskType
from backend.ai_func.recommendation.orchestration.planner import (
    TASK_CAPABILITY_CATALOG,
    PlannerAgent,
    _build_task_plan_schema,
    _render_capability_semantics,
    _render_input_contracts,
)

_REQUIRED_KEYS = {"purpose", "evidence_source", "use_when", "avoid_when",
                  "dependency_rule", "cost", "input_contract"}

_VALID_PLAN_JSON = {
    "goal": "找旅行照片，按地点分类，然后推荐最合适的几张",
    "synthesis_task_id": "t3",
    "tasks": [
        {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
         "input": {"query": "旅行照片", "limit": 30}, "depends_on": []},
        {"task_id": "t2", "task_type": "group_images", "objective": "分组",
         "input": {"group_by": "location"}, "depends_on": ["t1"]},
        {"task_id": "t3", "task_type": "synthesize", "objective": "汇总",
         "input": {"recommendation_limit": 10}, "depends_on": ["t2"]},
    ],
}


class _FakeLLM:
    def __init__(self, response=""):
        self.response = response
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.response


class _VisionOn:
    def __enter__(self):
        self._orig = (models_mod.vision_available, planner_mod.vision_available)
        models_mod.vision_available = lambda: True
        planner_mod.vision_available = lambda: True
        return self

    def __exit__(self, *a):
        models_mod.vision_available, planner_mod.vision_available = self._orig


def _enabled_caps():
    return PlannerAgent._default_capabilities()


# ---------------------------------------------------------------------------
# Catalog 覆盖与完整性
# ---------------------------------------------------------------------------
def test_catalog_covers_all_task_types():
    for t in TaskType:
        spec = TASK_CAPABILITY_CATALOG.get(t.value)
        assert spec is not None, f"catalog 缺少 {t.value}"
        assert _REQUIRED_KEYS <= set(spec.keys()), f"{t.value} 缺少字段"
        assert isinstance(spec["use_when"], list) and isinstance(spec["avoid_when"], list)


def test_catalog_single_source_input_contracts():
    caps = _enabled_caps()
    enabled = caps["enabled_task_types"]
    schema = _build_task_plan_schema(enabled)
    input_desc = schema["properties"]["tasks"]["items"]["properties"]["input"]["description"]
    for t in enabled:
        spec = TASK_CAPABILITY_CATALOG.get(t)
        if spec and spec.get("input_contract"):
            assert spec["input_contract"][:20] in input_desc, f"schema input 描述应含 {t} 契约"


def test_disabled_vision_not_in_rendered_catalog():
    import backend.ai_func.recommendation.orchestration.models as models_mod
    orig_models = models_mod.vision_available
    orig_planner = planner_mod.vision_available
    models_mod.vision_available = lambda: False
    planner_mod.vision_available = lambda: False
    try:
        caps = _enabled_caps()
    finally:
        models_mod.vision_available = orig_models
        planner_mod.vision_available = orig_planner
    enabled = caps["enabled_task_types"]
    assert "analyze_visual" not in enabled
    # 其他能力的 avoid_when 允许"提及"analyze_visual，但 catalog 不应渲染 analyze_visual 条目
    semantics = _render_capability_semantics(enabled)
    assert "- analyze_visual:" not in semantics
    assert "从候选图片的像素" not in semantics
    contracts = _render_input_contracts(enabled)
    assert "- analyze_visual:" not in contracts
    assert "max_candidates(1..8" not in contracts
    assert "analyze_visual" not in caps["task_capabilities"]
    schema = _build_task_plan_schema(enabled)
    assert "analyze_visual" not in schema["properties"]["tasks"]["items"]["properties"]["task_type"]["enum"]


def test_enabled_vision_in_rendered_catalog():
    with _VisionOn():
        caps = _enabled_caps()
    enabled = caps["enabled_task_types"]
    assert "analyze_visual" in enabled
    semantics = _render_capability_semantics(enabled)
    assert "从候选图片的像素" in semantics
    assert "昂贵" in semantics
    assert "候选消费型" in semantics
    contracts = _render_input_contracts(enabled)
    assert "max_candidates(1..8" in contracts
    assert any(c["task_type"] == "analyze_visual" for c in caps["task_capabilities"])


def test_default_capabilities_keeps_contract_keys():
    caps = _enabled_caps()
    for k in ("enabled_task_types", "vision_available", "max_tasks", "max_plan_depth", "task_capabilities"):
        assert k in caps


# ---------------------------------------------------------------------------
# Prompt 语义区分
# ---------------------------------------------------------------------------
def test_prompt_distinguishes_metadata_vs_vision():
    caps = _enabled_caps()
    prompt = PlannerAgent._build_system_prompt(caps)
    assert "读不到图片像素" in prompt          # Metadata 局限
    assert "从像素验证" in prompt             # Vision 职责
    assert "不要映射为 filter_metadata" in prompt


def test_prompt_distinguishes_retrieval_vs_vision():
    caps = _enabled_caps()
    prompt = PlannerAgent._build_system_prompt(caps)
    assert "不读取像素细节" in prompt          # Retrieval 局限
    assert "像素级核验" in prompt


def test_prompt_has_positive_and_negative_examples():
    caps = _enabled_caps()
    prompt = PlannerAgent._build_system_prompt(caps)
    assert "例3 (检索 + 视觉核验)" in prompt
    assert "反例：「找红色花朵照片」不自动需要 Vision" in prompt
    assert "「确认候选中花朵是否确实为红色」才需要 Vision" in prompt


def test_prompt_forbids_default_rerank_dedup_select_combo():
    caps = _enabled_caps()
    prompt = PlannerAgent._build_system_prompt(caps)
    assert "禁止默认套餐" in prompt
    assert "retrieve_images → synthesize" in prompt
    assert "不要加 rerank/dedup/select" in prompt


def test_prompt_no_unconditional_vision_rule():
    caps = _enabled_caps()
    prompt = PlannerAgent._build_system_prompt(caps)
    assert "不要因为出现“视觉/颜色/图片”一词就无条件加入" in prompt
    assert "必须加入 analyze_visual" not in prompt
    assert "所有视觉查询都必须" not in prompt


def test_prompt_includes_max_plan_depth():
    caps = _enabled_caps()
    prompt = PlannerAgent._build_system_prompt(caps)
    assert "{{max_plan_depth}}" not in prompt   # 未格式化占位符不应残留
    assert "依赖链深度不得超过" in prompt
    assert str(caps["max_plan_depth"]) in prompt


# ---------------------------------------------------------------------------
# schema enum 收窄 + 一致性
# ---------------------------------------------------------------------------
def test_schema_enum_consistency_with_prompt():
    orig = (models_mod.vision_available, planner_mod.vision_available)
    models_mod.vision_available = lambda: False
    planner_mod.vision_available = lambda: False
    try:
        caps_off = _enabled_caps()
    finally:
        models_mod.vision_available, planner_mod.vision_available = orig
    schema_off = _build_task_plan_schema(caps_off["enabled_task_types"])
    enum_off = set(schema_off["properties"]["tasks"]["items"]["properties"]["task_type"]["enum"])
    assert enum_off == set(caps_off["enabled_task_types"])
    with _VisionOn():
        caps_on = _enabled_caps()
    schema_on = _build_task_plan_schema(caps_on["enabled_task_types"])
    enum_on = set(schema_on["properties"]["tasks"]["items"]["properties"]["task_type"]["enum"])
    assert enum_on == set(caps_on["enabled_task_types"])
    assert enum_off < enum_on


def test_global_schema_not_mutated():
    from backend.ai_func.recommendation.orchestration.planner import TASK_PLAN_DRAFT_SCHEMA
    _build_task_plan_schema(["retrieve_images"])
    enum = TASK_PLAN_DRAFT_SCHEMA["properties"]["tasks"]["items"]["properties"]["task_type"]["enum"]
    assert set(enum) == {t.value for t in TaskType}   # global 模板仍含全部类型

    with _VisionOn():
        caps = _enabled_caps()
    enabled = caps["enabled_task_types"]
    assert "analyze_visual" in enabled
    semantics = _render_capability_semantics(enabled)
    assert "从候选图片的像素" in semantics
    assert "昂贵" in semantics
    assert "候选消费型" in semantics
    contracts = _render_input_contracts(enabled)
    assert "max_candidates(1..8" in contracts
    assert any(c["task_type"] == "analyze_visual" for c in caps["task_capabilities"])


def test_default_capabilities_keeps_contract_keys():
    caps = _enabled_caps()
    for k in ("enabled_task_types", "vision_available", "max_tasks", "max_plan_depth", "task_capabilities"):
        assert k in caps

# ---------------------------------------------------------------------------
# 单次 LLM 调用 / tool_choice / Validator
# ---------------------------------------------------------------------------
def test_planner_still_single_llm_call():
    fake = _FakeLLM(json.dumps(_VALID_PLAN_JSON, ensure_ascii=False))
    PlannerAgent(llm=fake).create_plan(goal="找旅行照片")
    assert len(fake.calls) == 1


def test_output_still_goes_to_validator():
    fake = _FakeLLM(json.dumps(_VALID_PLAN_JSON, ensure_ascii=False))
    draft = PlannerAgent(llm=fake).create_plan(goal="找旅行照片")
    assert isinstance(draft, TaskPlanDraft)
    r = PlanValidator().validate(draft)
    assert r.valid, r.errors


def _fake_resp(plan_json):
    args = json.dumps(plan_json, ensure_ascii=False)
    fn = SimpleNamespace(name="submit_task_plan", arguments=args)
    tc = SimpleNamespace(function=fn)
    msg = SimpleNamespace(tool_calls=[tc], content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _fake_resp(_VALID_PLAN_JSON)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat(_FakeCompletions())


def test_client_path_uses_filtered_schema_and_forced_tool_choice():
    client = _FakeClient()
    planner = PlannerAgent(client=client, model="m")
    planner.create_plan(goal="找旅行照片")
    kwargs = client.chat.completions.calls[0]
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "submit_task_plan"}}
    tools = kwargs["tools"]
    assert len(tools) == 1 and tools[0]["function"]["name"] == "submit_task_plan"
    enum = tools[0]["function"]["parameters"]["properties"]["tasks"]["items"]["properties"]["task_type"]["enum"]
    assert "analyze_visual" in enum
    assert {"retrieve_images", "synthesize"} <= set(enum)


def test_client_path_schema_includes_vision_when_enabled():
    with _VisionOn():
        client = _FakeClient()
        planner = PlannerAgent(client=client, model="m")
        planner.create_plan(goal="找旅行照片")
        kwargs = client.chat.completions.calls[0]
        enum = kwargs["tools"][0]["function"]["parameters"]["properties"]["tasks"]["items"]["properties"]["task_type"]["enum"]
        assert "analyze_visual" in enum


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

