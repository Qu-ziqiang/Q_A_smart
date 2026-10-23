"""Phase 1 Specialist — SpecialistAgent 单元 + DAG 集成测试。

覆盖：
  - SpecialistAgent(card, worker) 构造与字段
  - execute() 委托 Worker.run(task, context)，输出一致
  - run() 为 Worker 协议兼容别名（DAGExecutor 可直接分派）
  - Organizer 多 Worker（filter/group）按 task_type 分发
  - 不支持的任务类型显式报错；构造校验
  - execute 打印角色日志（Requirement 7 可观测）
  - 集成：DAGExecutor（未改）→ AgentRegistry → SpecialistAgent → Worker 完整链路
    （retrieval + organizer + vision 三种 Specialist）

全部 Fake/Mock，零真实 API / DB；DAGExecutor 与 PlanValidator 均为生产代码。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_specialist_agent.py
"""

import contextlib
import io
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    AgentCard,
    AgentRegistry,
    PlanValidator,
    SpecialistAgent,
    TaskDraft,
    TaskPlanDraft,
    TaskType,
    build_dispatched_workers,
)
from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.vision import build_vision_output
from backend.ai_func.recommendation.orchestration.workers import (
    GroupOutput,
    ImageCandidate,
    ImageGroup,
    RetrievalOutput,
    SynthesisOutput,
    TaskStatus,
    WorkerResult,
)


class _FW:
    """Fake Worker：记录调用 (task, context) 并执行 fn。"""

    def __init__(self, ttype, fn):
        self.task_type = ttype
        self._fn = fn
        self.calls = []

    def run(self, task, context):
        self.calls.append((task, context))
        return self._fn(task, context)


def _card(agent_id, tasks, enabled=True):
    return AgentCard(
        agent_id=agent_id,
        name={"retrieval": "RetrievalAgent", "organizer": "OrganizerAgent", "vision": "VisionAgent"}[agent_id],
        description=f"{agent_id} role card",
        supported_tasks=tasks,
        available_tools=[],
        input_schema="In",
        output_schema="Out",
        cost_level="cheap",
        enabled=enabled,
    )


def _ok(ttype, output):
    return WorkerResult(task_id="t", task_type=ttype, status=TaskStatus.SUCCEEDED, output=output)


class _Task:
    """最小 Task 替身（execute 只读取 task_id / task_type）。"""

    def __init__(self, task_id, task_type):
        self.task_id = task_id
        self.task_type = task_type


def test_constructor_holds_card_and_worker():
    w = _FW(TaskType.RETRIEVE_IMAGES, lambda task, ctx: _ok(TaskType.RETRIEVE_IMAGES, RetrievalOutput()))
    agent = SpecialistAgent(_card("retrieval", [TaskType.RETRIEVE_IMAGES]), w)
    assert agent.card.agent_id == "retrieval"
    assert agent.card.name == "RetrievalAgent"
    assert agent.worker is w
    assert agent.task_type == TaskType.RETRIEVE_IMAGES


def test_constructor_rejects_bad_inputs():
    w = _FW(TaskType.RETRIEVE_IMAGES, lambda task, ctx: None)
    try:
        SpecialistAgent("card", w)
    except TypeError:
        pass
    else:
        raise AssertionError("card 非 AgentCard 应抛 TypeError")
    try:
        SpecialistAgent(_card("retrieval", [TaskType.RETRIEVE_IMAGES]), None)
    except ValueError:
        pass
    else:
        raise AssertionError("worker=None 应抛 ValueError")
    try:
        # mapping 缺少 supported_tasks 声明的 filter_metadata
        SpecialistAgent(_card("organizer", [TaskType.FILTER_METADATA, TaskType.GROUP_IMAGES]),
                        {TaskType.GROUP_IMAGES: w})
    except ValueError:
        pass
    else:
        raise AssertionError("缺 Worker 的 supported_tasks 应抛 ValueError")


def test_execute_delegates_to_worker_output_identical():
    sentinel = _ok(TaskType.RETRIEVE_IMAGES, RetrievalOutput(candidates=[], query="q", total=0))
    w = _FW(TaskType.RETRIEVE_IMAGES, lambda task, ctx: sentinel)
    agent = SpecialistAgent(_card("retrieval", [TaskType.RETRIEVE_IMAGES]), w)
    task, ctx = _Task("t1", TaskType.RETRIEVE_IMAGES), object()
    out = agent.execute(task, ctx)
    assert out is sentinel
    assert len(w.calls) == 1
    assert w.calls[0] == (task, ctx)  # 原样透传 task / context


def test_run_protocol_alias_matches_execute():
    sentinel = _ok(TaskType.RETRIEVE_IMAGES, RetrievalOutput())
    w = _FW(TaskType.RETRIEVE_IMAGES, lambda task, ctx: sentinel)
    agent = SpecialistAgent(_card("retrieval", [TaskType.RETRIEVE_IMAGES]), w)
    task = _Task("t1", TaskType.RETRIEVE_IMAGES)
    assert agent.run(task, object()) is sentinel  # DAGExecutor 依赖 run() 契约


def test_organizer_dispatches_two_workers_by_task_type():
    sentinel_filter = _ok(TaskType.FILTER_METADATA, RetrievalOutput())
    sentinel_group = _ok(TaskType.GROUP_IMAGES, GroupOutput(group_by="location", groups=[]))
    w_filter = _FW(TaskType.FILTER_METADATA, lambda task, ctx: sentinel_filter)
    w_group = _FW(TaskType.GROUP_IMAGES, lambda task, ctx: sentinel_group)
    organizer = SpecialistAgent(
        _card("organizer", [TaskType.FILTER_METADATA, TaskType.GROUP_IMAGES]),
        {TaskType.FILTER_METADATA: w_filter, TaskType.GROUP_IMAGES: w_group},
    )
    assert organizer.task_type is None  # 多任务 Agent 不暴露单一 task_type
    assert organizer.execute(_Task("t2", TaskType.FILTER_METADATA), object()) is sentinel_filter
    assert organizer.execute(_Task("t3", TaskType.GROUP_IMAGES), object()) is sentinel_group
    assert len(w_filter.calls) == 1 and len(w_group.calls) == 1


def test_execute_unsupported_task_type_raises():
    w = _FW(TaskType.RETRIEVE_IMAGES, lambda task, ctx: None)
    agent = SpecialistAgent(_card("retrieval", [TaskType.RETRIEVE_IMAGES]), w)
    try:
        agent.execute(_Task("t9", TaskType.ANALYZE_VISUAL), object())
    except TypeError:
        pass
    else:
        raise AssertionError("不支持的任务类型应显式报错")


def test_execute_logs_role_line():
    w = _FW(TaskType.RETRIEVE_IMAGES, lambda task, ctx: _ok(TaskType.RETRIEVE_IMAGES, RetrievalOutput()))
    agent = SpecialistAgent(_card("retrieval", [TaskType.RETRIEVE_IMAGES]), w)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agent.execute(_Task("t1", TaskType.RETRIEVE_IMAGES), object())
    text = buf.getvalue()
    assert "RetrievalAgent executed task t1 (retrieve_images)" in text


# ---------------------------------------------------------------------------
# 集成：DAGExecutor（生产代码）→ AgentRegistry → SpecialistAgent → Worker
# ---------------------------------------------------------------------------
def _plan_for(task_json, vision_ok=False):
    draft = TaskPlanDraft.model_validate({
        "goal": "找旅行照片并按地点整理，同时确认画面",
        "synthesis_task_id": task_json["synthesis_task_id"],
        "tasks": task_json["tasks"],
    })
    vres = PlanValidator().validate(draft, vision_available_flag=vision_ok)
    assert vres.valid, vres.errors
    return vres.plan


def _retrieval_worker():
    cand = ImageCandidate(id=1, score=0.9, public_url="/static/1.jpg", metadata={"location": "北京"})
    return _FW(TaskType.RETRIEVE_IMAGES, lambda task, ctx: _ok(
        TaskType.RETRIEVE_IMAGES,
        RetrievalOutput(candidates=[cand], query="旅行照片", total=1),
    ))


def test_dag_executor_routes_retrieval_and_organizer_via_agents():
    plan_json = {
        "synthesis_task_id": "t4",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
             "input": {"query": "旅行照片", "limit": 10}, "depends_on": []},
            {"task_id": "t2", "task_type": "group_images", "objective": "分组",
             "input": {"group_by": "location"}, "depends_on": ["t1"]},
            {"task_id": "t4", "task_type": "synthesize", "objective": "汇总",
             "input": {"recommendation_limit": 5}, "depends_on": ["t2"]},
        ],
    }
    plan = _plan_for(plan_json, vision_ok=False)

    w_retr = _retrieval_worker()
    w_group = _FW(TaskType.GROUP_IMAGES, lambda task, ctx: _ok(
        TaskType.GROUP_IMAGES,
        GroupOutput(group_by="location", groups=[ImageGroup(key="北京", image_ids=[1])]),
    ))
    w_synth = _FW(TaskType.SYNTHESIZE, lambda task, ctx: _ok(
        TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[1]),
    ))
    base = {TaskType.RETRIEVE_IMAGES: w_retr, TaskType.GROUP_IMAGES: w_group,
            TaskType.SYNTHESIZE: w_synth}
    registry = AgentRegistry()
    registry.register_agent(SpecialistAgent(
        AgentCard(agent_id="retrieval", name="RetrievalAgent", description="r",
                  supported_tasks=[TaskType.RETRIEVE_IMAGES]),
        w_retr,
    ))
    registry.register_agent(SpecialistAgent(
        AgentCard(agent_id="organizer", name="OrganizerAgent", description="o",
                  supported_tasks=[TaskType.GROUP_IMAGES]),
        w_group,
    ))
    dispatched = build_dispatched_workers(base, registry, synthesize_worker=w_synth)
    ex = DAGExecutor(workers=dispatched)

    events = list(ex.execute_stream(plan))
    names = [e["event"] for e in events]
    assert names.count("task.completed") == 3
    assert ex.context.results["t1"].status == TaskStatus.SUCCEEDED
    assert ex.context.results["t2"].status == TaskStatus.SUCCEEDED
    assert ex.context.results["t4"].status == TaskStatus.SUCCEEDED
    # Agent 确实路由到了底层 Worker
    assert len(w_retr.calls) == 1 and len(w_group.calls) == 1


def test_dag_executor_routes_vision_via_agent():
    plan_json = {
        "synthesis_task_id": "t3",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
             "input": {"query": "海边", "limit": 10}, "depends_on": []},
            {"task_id": "t2", "task_type": "analyze_visual", "objective": "核验海边",
             "input": {"criteria": ["海边"], "max_candidates": 2}, "depends_on": ["t1"]},
            {"task_id": "t3", "task_type": "synthesize", "objective": "汇总",
             "input": {"recommendation_limit": 5}, "depends_on": ["t2"]},
        ],
    }
    plan = _plan_for(plan_json, vision_ok=True)

    w_retr = _retrieval_worker()
    w_vision = _FW(TaskType.ANALYZE_VISUAL, lambda task, ctx: _ok(
        TaskType.ANALYZE_VISUAL, build_vision_output([]),
    ))
    w_synth = _FW(TaskType.SYNTHESIZE, lambda task, ctx: _ok(
        TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[1]),
    ))
    base = {TaskType.RETRIEVE_IMAGES: w_retr, TaskType.ANALYZE_VISUAL: w_vision,
            TaskType.SYNTHESIZE: w_synth}
    registry = AgentRegistry()
    registry.register_agent(SpecialistAgent(
        AgentCard(agent_id="retrieval", name="RetrievalAgent", description="r",
                  supported_tasks=[TaskType.RETRIEVE_IMAGES]),
        w_retr,
    ))
    registry.register_agent(SpecialistAgent(
        AgentCard(agent_id="vision", name="VisionAgent", description="v",
                  supported_tasks=[TaskType.ANALYZE_VISUAL]),
        w_vision,
    ))
    dispatched = build_dispatched_workers(base, registry, synthesize_worker=w_synth)
    ex = DAGExecutor(workers=dispatched)

    events = list(ex.execute_stream(plan))
    names = [e["event"] for e in events]
    assert names.count("task.completed") == 3
    assert ex.context.results["t2"].status == TaskStatus.SUCCEEDED
    assert len(w_vision.calls) == 1


def test_worker_crash_still_isolated_by_executor():
    # 包装层不应改变 Executor 的异常隔离语义
    def boom(task, ctx):
        raise RuntimeError("worker boom")

    w_retr = _FW(TaskType.RETRIEVE_IMAGES, boom)
    plan_json = {
        "synthesis_task_id": "t2",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
             "input": {"query": "x", "limit": 5}, "depends_on": []},
            {"task_id": "t2", "task_type": "synthesize", "objective": "汇总",
             "input": {"recommendation_limit": 5}, "depends_on": ["t1"]},
        ],
    }
    plan = _plan_for(plan_json, vision_ok=False)
    w_synth = _FW(TaskType.SYNTHESIZE, lambda task, ctx: _ok(
        TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[]),
    ))
    base = {TaskType.RETRIEVE_IMAGES: w_retr, TaskType.SYNTHESIZE: w_synth}
    registry = AgentRegistry()
    registry.register_agent(SpecialistAgent(
        AgentCard(agent_id="retrieval", name="RetrievalAgent", description="r",
                  supported_tasks=[TaskType.RETRIEVE_IMAGES]),
        w_retr,
    ))
    ex = DAGExecutor(workers=build_dispatched_workers(base, registry, synthesize_worker=w_synth))
    list(ex.execute_stream(plan))
    assert ex.context.results["t1"].status == TaskStatus.FAILED
    assert ex.context.results["t1"].error.code == "WORKER_CRASH"


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



