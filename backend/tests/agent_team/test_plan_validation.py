"""Phase 1-A — PlanValidator 单元测试。

运行方式（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_plan_validation.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.models import MAX_PLAN_DEPTH, MAX_TASKS
from backend.ai_func.recommendation.orchestration import (
    PlanValidator,
    TaskDraft,
    TaskPlanDraft,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.models import RetrieveImagesInput


def _draft(goal="g", synth_id="t_s", tasks=None):
    return TaskPlanDraft(goal=goal, synthesis_task_id=synth_id, tasks=tasks or [])


def _task(tid, ttype, objective="o", input=None, depends=None):
    return TaskDraft(
        task_id=tid,
        task_type=ttype,
        objective=objective,
        input=input or {},
        depends_on=depends or [],
    )


def _validator():
    return PlanValidator()


# ---------------------------------------------------------------------------
# 合法计划
# ---------------------------------------------------------------------------
def test_valid_linear_plan():
    d = _draft(
        synth_id="t3",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "旅行", "limit": 30}),
            _task("t2", TaskType.FILTER_METADATA, depends=["t1"]),
            _task("t3", TaskType.SYNTHESIZE, input={"recommendation_limit": 10}, depends=["t2"]),
        ],
    )
    r = _validator().validate(d)
    assert r.valid, r.errors
    assert r.plan is not None
    assert r.plan.plan_id.startswith("plan_")
    assert r.plan.plan_version == "1.0"


def test_valid_retrieve_group_synthesize():
    d = _draft(
        synth_id="t3",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "旅行"}),
            _task("t2", TaskType.GROUP_IMAGES, input={"group_by": "location"}, depends=["t1"]),
            _task("t3", TaskType.SYNTHESIZE, input={"presentation": "grouped"}, depends=["t2"]),
        ],
    )
    r = _validator().validate(d)
    assert r.valid, r.errors


def test_typed_input_assigned():
    d = _draft(
        synth_id="t2",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "猫", "limit": 15}),
            _task("t2", TaskType.SYNTHESIZE, depends=["t1"]),
        ],
    )
    r = _validator().validate(d)
    assert r.valid, r.errors
    retr = [t for t in r.plan.tasks if t.task_type == TaskType.RETRIEVE_IMAGES][0]
    assert isinstance(retr.input, RetrieveImagesInput)
    assert retr.input.query == "猫"
    assert retr.input.limit == 15


# ---------------------------------------------------------------------------
# 非法计划
# ---------------------------------------------------------------------------
def test_empty_plan():
    r = _validator().validate(_draft(tasks=[]))
    assert not r.valid


def test_single_task_too_few():
    d = _draft(
        synth_id="t1",
        tasks=[_task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"})],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("MIN_TASKS" in e for e in r.errors)


def test_duplicate_task_id():
    d = _draft(
        synth_id="t2",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"}),
            _task("t1", TaskType.FILTER_METADATA, depends=["t1"]),
            _task("t2", TaskType.SYNTHESIZE, depends=["t1"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("DUPLICATE_TASK_ID" in e for e in r.errors)


def test_missing_dependency():
    d = _draft(
        synth_id="t2",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"}),
            _task("t2", TaskType.SYNTHESIZE, depends=["t99"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("UNKNOWN_DEPENDENCY" in e for e in r.errors)


def test_self_dependency():
    d = _draft(
        synth_id="t2",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"}),
            _task("t2", TaskType.SYNTHESIZE, depends=["t2"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("SELF_DEPENDENCY" in e for e in r.errors)


def test_cycle_simple():
    d = _draft(
        synth_id="t2",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"}, depends=["t2"]),
            _task("t2", TaskType.SYNTHESIZE, depends=["t1"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("CYCLE_DETECTED" in e for e in r.errors)


def test_cycle_longer():
    d = _draft(
        synth_id="t4",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"}, depends=["t4"]),
            _task("t2", TaskType.FILTER_METADATA, depends=["t1"]),
            _task("t3", TaskType.GROUP_IMAGES, input={"group_by": "tag"}, depends=["t2"]),
            _task("t4", TaskType.SYNTHESIZE, depends=["t3"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("CYCLE_DETECTED" in e for e in r.errors)


def test_too_many_tasks():
    n = MAX_TASKS + 1
    tasks = [_task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"})]
    for i in range(2, n):
        tasks.append(_task(f"t{i}", TaskType.FILTER_METADATA, depends=[f"t{i-1}"]))
    tasks.append(_task(f"t{n}", TaskType.SYNTHESIZE, depends=[f"t{n-1}"]))
    d = _draft(synth_id=f"t{n}", tasks=tasks)
    r = _validator().validate(d)
    assert not r.valid
    assert any("MAX_TASKS" in e for e in r.errors)


def test_too_deep():
    depth = MAX_PLAN_DEPTH + 1
    tasks = [_task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"})]
    for i in range(2, depth):
        tasks.append(_task(f"t{i}", TaskType.FILTER_METADATA, depends=[f"t{i-1}"]))
    tasks.append(_task(f"t{depth}", TaskType.SYNTHESIZE, depends=[f"t{depth-1}"]))
    d = _draft(synth_id=f"t{depth}", tasks=tasks)
    r = _validator().validate(d)
    assert not r.valid
    assert any("MAX_PLAN_DEPTH" in e for e in r.errors)


def test_no_retrieve():
    d = _draft(
        synth_id="t2",
        tasks=[
            _task("t1", TaskType.FILTER_METADATA),
            _task("t2", TaskType.SYNTHESIZE, depends=["t1"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("NO_RETRIEVE" in e for e in r.errors)


def test_multiple_synthesize():
    d = _draft(
        synth_id="t3",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"}),
            _task("t2", TaskType.SYNTHESIZE, depends=["t1"]),
            _task("t3", TaskType.SYNTHESIZE, depends=["t1"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("SYNTHESIZE_COUNT" in e for e in r.errors)


def test_synthesis_target_mismatch():
    d = _draft(
        synth_id="tX",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"}),
            _task("t2", TaskType.SYNTHESIZE, depends=["t1"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("SYNTHESIS_TARGET" in e for e in r.errors)


def test_synthesize_not_terminal():
    d = _draft(
        synth_id="t2",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"}),
            _task("t2", TaskType.SYNTHESIZE, depends=["t1"]),
            _task("t3", TaskType.FILTER_METADATA, depends=["t2"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("SYNTHESIZE_NOT_TERMINAL" in e for e in r.errors)


def test_orphan_task():
    d = _draft(
        synth_id="t3",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"}),
            _task("t2", TaskType.FILTER_METADATA, depends=["t1"]),  # 无法到达 t3
            _task("t3", TaskType.SYNTHESIZE, depends=["t1"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("ORPHAN_TASK" in e for e in r.errors)


def test_consumer_without_producer():
    d = _draft(
        synth_id="t3",
        tasks=[
            _task("t1", TaskType.FILTER_METADATA),  # 无 producer
            _task("t2", TaskType.SYNTHESIZE, depends=["t1"]),
            _task("t3", TaskType.GROUP_IMAGES, input={"group_by": "tag"}, depends=["t2"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("NO_PRODUCER" in e for e in r.errors)


def test_vision_unavailable():
    d = _draft(
        synth_id="t3",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={"query": "x"}),
            _task("t2", TaskType.ANALYZE_VISUAL, depends=["t1"]),
            _task("t3", TaskType.SYNTHESIZE, depends=["t2"]),
        ],
    )
    r = _validator().validate(d, vision_available_flag=False)
    assert not r.valid
    assert any("VISION_UNAVAILABLE" in e for e in r.errors)


def test_invalid_task_input():
    # retrieve_images 缺少必填 query → input 类型化失败
    d = _draft(
        synth_id="t2",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, input={}),
            _task("t2", TaskType.SYNTHESIZE, depends=["t1"]),
        ],
    )
    r = _validator().validate(d)
    assert not r.valid
    assert any("INVALID_TASK_INPUT" in e for e in r.errors)


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

