"""Phase 2 Stage 4 — Capability and Registration 单元测试。

覆盖：
  A. capability=False：Planner 不可用 / Validator 拒绝 / default_workers 不注册
  B. capability=True（mock vision_available）：三者全部可用
  C. Registry consistency（Planner = Validator = Workers 无 drift）
  D. 既有任务回归（RETRIEVE / FILTER / GROUP / SYNTHESIZE）
  E. Legacy Fast Path 回归
  + Vision DAG 规则（producer / orphan / 并行结构）

全部 Fake/Mock，不调用真实 API；不修改配置。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_registration.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

import backend.ai_func.recommendation.orchestration.models as models_mod
import backend.ai_func.recommendation.orchestration.planner as planner_mod
import backend.ai_func.recommendation.orchestration.workers as workers_mod
from backend.ai_func.recommendation.orchestration import (
    ComplexityGate,
    PlannerAgent,
    PlanValidator,
    TaskDraft,
    TaskPlanDraft,
    TaskType,
    VisionWorker,
    default_workers,
)
from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.models import (
    REGISTERED_TASK_TYPES,
    registered_task_types,
    vision_available,
)


class _VisionOn:
    """把三个模块的 vision_available 引用统一替换为 True（单一 mock 点）。"""

    def __enter__(self):
        self._orig = {
            "models": models_mod.vision_available,
            "planner": planner_mod.vision_available,
            "workers": workers_mod.vision_available,
        }
        models_mod.vision_available = lambda: True
        planner_mod.vision_available = lambda: True
        workers_mod.vision_available = lambda: True
        return self

    def __exit__(self, *exc):
        models_mod.vision_available = self._orig["models"]
        planner_mod.vision_available = self._orig["planner"]
        workers_mod.vision_available = self._orig["workers"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _draft(synth_id="t_s", tasks=None, goal="g"):
    return TaskPlanDraft(goal=goal, synthesis_task_id=synth_id, tasks=tasks or [])


def _task(tid, ttype, objective="o", input=None, depends=None):
    return TaskDraft(
        task_id=tid, task_type=ttype, objective=objective,
        input=input or {}, depends_on=depends or [],
    )


# ---------------------------------------------------------------------------
# A. Capability False（真实配置：VISION_MODEL="null"）
# ---------------------------------------------------------------------------
class _VisionOff:
    def __enter__(self):
        self._orig = {
            "models": models_mod.vision_available,
            "planner": planner_mod.vision_available,
            "workers": workers_mod.vision_available,
        }
        models_mod.vision_available = lambda: False
        planner_mod.vision_available = lambda: False
        workers_mod.vision_available = lambda: False
        return self

    def __exit__(self, *exc):
        models_mod.vision_available = self._orig["models"]
        planner_mod.vision_available = self._orig["planner"]
        workers_mod.vision_available = self._orig["workers"]


def test_live_glm_vision_is_enabled():
    assert vision_available() is True


def test_disabled_planner_excludes_analyze_visual():
    with _VisionOff():
        caps = PlannerAgent._default_capabilities()
    assert "analyze_visual" not in caps["enabled_task_types"]
    assert caps["vision_available"] is False


def test_disabled_validator_rejects_analyze_visual():
    d = _draft(synth_id="t3", tasks=[
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _task("t2", TaskType.ANALYZE_VISUAL, input={"criteria": ["c"]}, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, depends=["t2"]),
    ])
    r = PlanValidator().validate(d, vision_available_flag=False)
    assert not r.valid
    assert any("VISION_UNAVAILABLE" in e or "WORKER_UNAVAILABLE" in e for e in r.errors)


def test_disabled_default_workers_absent():
    with _VisionOff():
        w = default_workers()
    assert TaskType.ANALYZE_VISUAL not in w


def test_disabled_registered_task_types_excludes_vision():
    reg = registered_task_types(False)
    assert TaskType.ANALYZE_VISUAL not in reg
    assert reg == set(REGISTERED_TASK_TYPES)


# ---------------------------------------------------------------------------
# B. Capability True（mock vision_available，零真实 API）
# ---------------------------------------------------------------------------
def test_enabled_planner_includes_analyze_visual():
    with _VisionOn():
        caps = PlannerAgent._default_capabilities()
        assert "analyze_visual" in caps["enabled_task_types"]
        assert caps["vision_available"] is True


def test_enabled_validator_accepts_analyze_visual_plan():
    d = _draft(synth_id="t3", tasks=[
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _task("t2", TaskType.ANALYZE_VISUAL,
              input={"criteria": ["c"], "max_candidates": 4}, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t2"]),
    ])
    r = PlanValidator().validate(d, vision_available_flag=True)
    assert r.valid, r.errors
    assert r.plan is not None
    assert r.plan.tasks[1].task_type == TaskType.ANALYZE_VISUAL


def test_enabled_default_workers_includes_vision():
    with _VisionOn():
        w = default_workers()
        assert TaskType.ANALYZE_VISUAL in w
        assert isinstance(w[TaskType.ANALYZE_VISUAL], VisionWorker)


def test_enabled_registered_task_types_includes_vision():
    with _VisionOn():
        reg = registered_task_types()
        assert TaskType.ANALYZE_VISUAL in reg


def test_registered_task_types_default_uses_capability():
    assert TaskType.ANALYZE_VISUAL in registered_task_types()
    with _VisionOff():
        assert TaskType.ANALYZE_VISUAL not in registered_task_types()


# ---------------------------------------------------------------------------
# C. Registry consistency（无 drift）
# ---------------------------------------------------------------------------
def test_registry_consistency_disabled():
    with _VisionOff():
        planner_enabled = {TaskType(v) for v in PlannerAgent._default_capabilities()["enabled_task_types"]}
        validator_reg = registered_task_types(False)
        workers_reg = set(default_workers().keys())
    assert planner_enabled == validator_reg == workers_reg
    assert TaskType.ANALYZE_VISUAL not in planner_enabled


def test_registry_consistency_enabled():
    with _VisionOn():
        planner_enabled = {TaskType(v) for v in PlannerAgent._default_capabilities()["enabled_task_types"]}
        validator_reg = registered_task_types(True)
        workers_reg = set(default_workers().keys())
        assert planner_enabled == validator_reg == workers_reg
        assert TaskType.ANALYZE_VISUAL in planner_enabled


def test_vision_worker_discoverable_via_registry():
    with _VisionOn():
        workers = default_workers()
        worker = workers[TaskType.ANALYZE_VISUAL]
        assert worker.task_type == TaskType.ANALYZE_VISUAL
        # 通过正式 registry 构造 Executor 即可分派该 Worker
        ex = DAGExecutor(workers=workers)
        assert ex.workers[TaskType.ANALYZE_VISUAL] is worker

# ---------------------------------------------------------------------------
# D. 既有任务回归
# ---------------------------------------------------------------------------
def test_existing_tasks_regression_disabled():
    w = default_workers()
    for t in (TaskType.RETRIEVE_IMAGES, TaskType.FILTER_METADATA,
              TaskType.GROUP_IMAGES, TaskType.SYNTHESIZE):
        assert t in w
    d = _draft(synth_id="t3", tasks=[
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _task("t2", TaskType.GROUP_IMAGES, input={"group_by": "location"}, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t2"]),
    ])
    r = PlanValidator().validate(d)
    assert r.valid, r.errors


def test_existing_tasks_regression_enabled():
    with _VisionOn():
        w = default_workers()
        for t in (TaskType.RETRIEVE_IMAGES, TaskType.FILTER_METADATA,
                  TaskType.GROUP_IMAGES, TaskType.SYNTHESIZE):
            assert t in w
        d = _draft(synth_id="t3", tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
            _task("t2", TaskType.GROUP_IMAGES, input={"group_by": "location"}, depends=["t1"]),
            _task("t3", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t2"]),
        ])
        r = PlanValidator().validate(d, vision_available_flag=True)
        assert r.valid, r.errors


# ---------------------------------------------------------------------------
# E. Legacy Fast Path 回归
# ---------------------------------------------------------------------------
def test_legacy_fast_path_unaffected():
    gate = ComplexityGate()
    dec = gate.decide(query="你好")
    assert dec.route.value == "legacy"
    dec2 = gate.decide(query="找旅行照片，按地点分类，再推荐每组最合适的")
    assert dec2.route.value == "team"


# ---------------------------------------------------------------------------
# Vision DAG rules
# ---------------------------------------------------------------------------
def test_vision_requires_producer():
    d = _draft(synth_id="t3", tasks=[
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _task("t2", TaskType.ANALYZE_VISUAL, input={"criteria": ["c"]}, depends=[]),
        _task("t3", TaskType.SYNTHESIZE, depends=["t2"]),
    ])
    r = PlanValidator().validate(d, vision_available_flag=True)
    assert not r.valid
    assert any("NO_PRODUCER" in e for e in r.errors)


def test_vision_must_reach_synthesize():
    d = _draft(synth_id="t3", tasks=[
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _task("t2", TaskType.ANALYZE_VISUAL, input={"criteria": ["c"]}, depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, depends=["t1"]),  # t2 不（传递）到达 t3
    ])
    r = PlanValidator().validate(d, vision_available_flag=True)
    assert not r.valid
    assert any("ORPHAN_TASK" in e for e in r.errors)


def test_vision_group_parallel_structure_accepted():
    # retrieve → {group, vision} → synthesize（推荐结构，验证合法）
    d = _draft(synth_id="t4", tasks=[
        _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "q"}),
        _task("t2", TaskType.GROUP_IMAGES, input={"group_by": "location"}, depends=["t1"]),
        _task("t3", TaskType.ANALYZE_VISUAL, input={"criteria": ["旅行"]}, depends=["t1"]),
        _task("t4", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t2", "t3"]),
    ])
    r = PlanValidator().validate(d, vision_available_flag=True)
    assert r.valid, r.errors


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

