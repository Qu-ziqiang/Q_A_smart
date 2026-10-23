"""Phase 4.3 — Agent Benchmark 确定性场景运行时（Harness）。

用途：在不触真实 DB / 视觉 API 的前提下，为 benchmark 提供可复现的 Agent Runtime
（SupervisorAgent + 确定性 Fake 工具）。Judge 层可以是 Fake 或真实外部 LLM。

说明：本文件在 evaluation 包内只提供“场景 → runtime”的确定性装配，
不改动任何 Agent Runtime 生产代码。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _rows():
    return [
        {"id": 601, "score": 0.92, "title": "北京城市天际线A", "description": "CBD 高层建筑",
         "tags": ["建筑", "城市"], "metadata": {"location": "北京"}, "width": 1920, "height": 1280,
         "file_type": "jpg", "created_at": "2025-06-01T10:00:00Z",
         "public_url": "/static/601.jpg", "filepath": ""},
        {"id": 602, "score": 0.87, "title": "上海陆家嘴B", "description": "摩天楼群",
         "tags": ["建筑", "城市"], "metadata": {"location": "上海"}, "width": 1920, "height": 1080,
         "file_type": "jpg", "created_at": "2025-06-02T10:00:00Z",
         "public_url": "/static/602.jpg", "filepath": ""},
    ]


class FakeSearch:
    """search_images 的确定性实现（按调用次序返回）。"""

    def __init__(self, results_by_call=None, fail=False):
        self.results = list(results_by_call) if results_by_call is not None else [_rows()]
        self.calls = 0
        self.fail = fail

    def __call__(self, query=None, vector_targets=None, filters=None, limit=20):
        if self.fail:
            self.calls += 1
            raise RuntimeError("search tool down (benchmark fake)")
        idx = min(self.calls, len(self.results) - 1) if self.results else 0
        self.calls += 1
        return [dict(r) for r in self.results[idx]]


class FakeAnalyze:
    """analyze_image 的确定性实现（RegistryAnalyzer → registry 调用路径）。"""

    def __call__(self, image_url=None, local_image_path=None, model=None,
                 detail="low", prompt=None, timeout=None):
        return {
            "title": "视觉核验标题",
            "description": "核验：图片内容为城市/建筑，符合任务描述。",
            "tags": ["建筑", "城市"],
            "evidence": "按给定任务完成的描述性核验",
            "confidence": 0.9,
        }


class _VisionPatchedRuntime:
    """视觉用例运行时的门禁补丁封装：run_stream 期间开启 vision 能力，结束即恢复。"""

    _MODULE_NAMES = (
        "backend.ai_func.recommendation.orchestration.models",
        "backend.ai_func.recommendation.orchestration.planner",
        "backend.ai_func.recommendation.orchestration.workers",
    )

    def __init__(self, inner: Any, vision: bool):
        self._inner = inner
        self._vision = vision
        self._saved = {}

    def _apply(self) -> None:
        if not self._vision:
            return
        import importlib

        for name in self._MODULE_NAMES:
            mod = importlib.import_module(name)
            self._saved[name] = getattr(mod, "vision_available", None)
            mod.vision_available = lambda: True
        # 不原地修改原 set 对象（避免其他模块 import 时捕获的引用被污染），
        # 而是整体替换成“原集合 + ANALYZE_VISUAL”的新集合，结束后还原引用。
        models_mod = importlib.import_module(
            "backend.ai_func.recommendation.orchestration.models")
        from backend.ai_func.recommendation.orchestration.models import TaskType

        self._saved["_registered"] = models_mod.REGISTERED_TASK_TYPES
        models_mod.REGISTERED_TASK_TYPES = set(models_mod.REGISTERED_TASK_TYPES)
        models_mod.REGISTERED_TASK_TYPES.add(TaskType.ANALYZE_VISUAL)

    def _restore(self) -> None:
        import importlib

        for name, value in self._saved.items():
            if name == "_registered":
                mod = importlib.import_module(
                    "backend.ai_func.recommendation.orchestration.models")
                mod.REGISTERED_TASK_TYPES = value
                continue
            mod = importlib.import_module(name)
            mod.vision_available = value
        self._saved = {}

    def run_stream(self, messages=None, **kwargs):
        self._apply()
        try:
            yield from self._inner.run_stream(messages, **kwargs)
        finally:
            self._restore()

    def __getattr__(self, item: str):  # 代理 run_state / last_plan_ok 等
        return getattr(self._inner, item)


def _plan_flags(case) -> Dict[str, bool]:
    tags = set(case.tags)
    return {
        "vision": bool({"vision", "multi_step", "complex_planning"} & tags),
        "organize": bool({"organization", "multi_step", "complex_planning"} & tags),
        "empty_retry": "empty_retry" in tags,
        "tool_failure": "tool_failure" in tags,
    }


def _plan_json(case) -> Dict[str, Any]:
    flags = _plan_flags(case)
    goal = (case.goal or case.query or "检索图片").strip()
    query = (case.query or goal or "图片")[:200]
    tasks: List[Dict[str, Any]] = [
        {"task_id": "t1", "task_type": "retrieve_images",
         "objective": "语义检索候选图片", "depends_on": [],
         "input": {"query": query, "limit": 20}},
    ]
    if flags["organize"]:
        tasks.append({"task_id": "t2", "task_type": "filter_metadata",
                      "objective": "按元数据过滤", "depends_on": ["t1"],
                      "input": {"predicates": [
                          {"field": "file_type", "operator": "eq", "value": "jpg"}],
                          "include_fields": []}})
        tasks.append({"task_id": "t3", "task_type": "group_images",
                      "objective": "按地点分组", "depends_on": ["t2"],
                      "input": {"group_by": "location", "unknown_group_name": "未分类"}})
    if flags["vision"]:
        vid = "t4" if flags["organize"] else "t2"
        tasks.append({"task_id": vid, "task_type": "analyze_visual",
                      "objective": "视觉核验候选图片", "depends_on": ["t1"],
                      "input": {"criteria": ["画面内容"], "max_candidates": 2,
                                "detail": "low"}})
    if flags["organize"] and flags["vision"]:
        synth_id = "t5"
        tasks.append({"task_id": "t5", "task_type": "synthesize",
                      "objective": "汇总推荐", "depends_on": ["t3", "t4"],
                      "input": {"recommendation_limit": 5, "presentation": "flat"}})
    elif flags["organize"]:
        synth_id = "t4"
        tasks.append({"task_id": "t4", "task_type": "synthesize",
                      "objective": "汇总推荐", "depends_on": ["t3"],
                      "input": {"recommendation_limit": 5, "presentation": "flat"}})
    elif flags["vision"]:
        synth_id = "t3"
        tasks.append({"task_id": "t3", "task_type": "synthesize",
                      "objective": "汇总推荐", "depends_on": ["t1", "t2"],
                      "input": {"recommendation_limit": 5, "presentation": "flat"}})
    else:
        synth_id = "t2"
        tasks.append({"task_id": "t2", "task_type": "synthesize",
                      "objective": "汇总推荐", "depends_on": ["t1"],
                      "input": {"recommendation_limit": 5, "presentation": "flat"}})
    return {"goal": goal, "synthesis_task_id": synth_id, "tasks": tasks}


def build_case_runtime(case) -> Any:
    """构造一次确定性 Agent Runtime（SupervisorAgent + Fake 工具注入）。"""
    from backend.ai_func.recommendation.orchestration import (
        PlanValidator, PlannerAgent, SupervisorAgent, ToolRegistry,
        register_builtin_tools,
    )
    from backend.ai_func.recommendation.orchestration.supervisor import (
        build_registry_backed_workers,
    )
    from backend.ai_func.recommendation.orchestration.workers import (
        DeterministicSynthesisWorker,
    )

    flags = _plan_flags(case)
    search = FakeSearch()
    if flags["empty_retry"]:
        search = FakeSearch([[], _rows()])
    elif flags["tool_failure"]:
        search = FakeSearch(fail=True)
    analyze = FakeAnalyze()

    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={
        "search_images": search,
        "analyze_image": analyze,
    })
    workers = build_registry_backed_workers(tool_reg, with_vision=flags["vision"])
    planner = PlannerAgent(llm=lambda messages: json.dumps(
        _plan_json(case), ensure_ascii=False))
    return _VisionPatchedRuntime(
        SupervisorAgent(
            planner=planner,
            validator=PlanValidator(),
            workers=workers,
            tool_registry=tool_reg,
            synthesize_worker=DeterministicSynthesisWorker(),
            enhanced=True,
            max_iterations=2,
            backend="langgraph",
        ),
        vision=flags["vision"],
    )


class _CatalogPinnedRuntime:
    """真实 Planner + Fake 工具；按 before/after 收窄 Task 目录。"""

    def __init__(self, inner: Any, arm: str, vision: bool):
        self._inner = inner
        self._arm = arm
        self._vision = vision

    def run_stream(self, messages=None, **kwargs):
        from .catalog_ab import pin_registered_catalog

        with pin_registered_catalog(self._arm, self._vision):
            yield from self._inner.run_stream(messages, **kwargs)

    def __getattr__(self, item: str):
        return getattr(self._inner, item)


def build_real_planner_runtime(
    case,
    arm: str,
    *,
    planner_client: Any = None,
    planner_model: str | None = None,
    planner_llm=None,
) -> Any:
    """A/B 评测运行时：真实 Planner（可注入 client/llm）+ Fake 检索/视觉。"""
    from backend.ai_func.recommendation.orchestration import (
        PlanValidator, SupervisorAgent, ToolRegistry, register_builtin_tools,
    )
    from backend.ai_func.recommendation.orchestration.supervisor import (
        build_registry_backed_workers,
    )
    from backend.ai_func.recommendation.orchestration.workers import (
        DeterministicSynthesisWorker,
    )
    from .catalog_ab import CatalogPinnedPlanner

    flags = _plan_flags(case)
    vision = bool(flags["vision"])
    search = FakeSearch()
    if flags["empty_retry"]:
        search = FakeSearch([[], _rows()])
    elif flags["tool_failure"]:
        search = FakeSearch(fail=True)
    analyze = FakeAnalyze()

    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={
        "search_images": search,
        "analyze_image": analyze,
    })
    workers = build_registry_backed_workers(tool_reg, with_vision=vision)
    planner_kwargs: Dict[str, Any] = {"timeout": 60}
    if planner_llm is not None:
        planner_kwargs["llm"] = planner_llm
    else:
        planner_kwargs["client"] = planner_client
        planner_kwargs["model"] = planner_model
    planner = CatalogPinnedPlanner(arm, vision, **planner_kwargs)
    return _CatalogPinnedRuntime(
        SupervisorAgent(
            planner=planner,
            validator=PlanValidator(),
            workers=workers,
            tool_registry=tool_reg,
            synthesize_worker=DeterministicSynthesisWorker(),
            enhanced=True,
            max_iterations=2,
            backend="langgraph",
        ),
        arm=arm,
        vision=vision,
    )

