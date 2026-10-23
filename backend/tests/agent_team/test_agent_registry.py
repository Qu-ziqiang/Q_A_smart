"""Phase 1 Specialist — AgentRegistry 单元测试。

覆盖：
  - register_agent / get_agent / list_agents / get_agent_for_task
  - TaskType → Specialist 映射（含 Organizer 双任务归组）
  - disabled Agent 可查询但不参与任务分发
  - build_default_agent_registry 注入 worker 表（含视觉）生成三个默认 Specialist
  - build_dispatched_workers 分发（未注册任务回退原 Worker）

全部 Fake/Mock，零真实 API / DB。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_agent_registry.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    AgentCard,
    AgentRegistry,
    SpecialistAgent,
    build_default_agent_registry,
    build_dispatched_workers,
)
from backend.ai_func.recommendation.orchestration.models import TaskType
from backend.ai_func.recommendation.orchestration.workers import (
    GroupOutput,
    RetrievalOutput,
    SynthesisOutput,
    TaskStatus,
    WorkerResult,
)


class _FW:
    """Fake Worker：普通函数包装，记录 (task, context) 调用。"""

    def __init__(self, ttype, fn, tag=""):
        self.task_type = ttype
        self._fn = fn
        self.tag = tag
        self.calls = []

    def run(self, task, context):
        self.calls.append((task, context))
        return self._fn(task, context)


def _card(agent_id, tasks, tools):
    return AgentCard(
        agent_id=agent_id,
        name=f"{agent_id.title()}Agent",
        description=f"{agent_id} role",
        supported_tasks=tasks,
        available_tools=tools,
        input_schema="In",
        output_schema="Out",
        cost_level="cheap",
    )


def _ok(ttype, output):
    return WorkerResult(task_id="t", task_type=ttype, status=TaskStatus.SUCCEEDED, output=output)


def _retrieval_worker():
    def fn(task, context):
        return _ok(TaskType.RETRIEVE_IMAGES, RetrievalOutput(candidates=[], query="q", total=0))
    return _FW(TaskType.RETRIEVE_IMAGES, fn, tag="retrieval")


def _group_worker():
    def fn(task, context):
        return _ok(TaskType.GROUP_IMAGES, GroupOutput(group_by="location", groups=[]))
    return _FW(TaskType.GROUP_IMAGES, fn, tag="group")


def _synth_worker():
    def fn(task, context):
        return _ok(TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[]))
    return _FW(TaskType.SYNTHESIZE, fn, tag="synth")


def _three_agents_registry():
    registry = AgentRegistry()
    registry.register_agent(SpecialistAgent(
        _card("retrieval", [TaskType.RETRIEVE_IMAGES], ["search_images", "rerank"]),
        _retrieval_worker(),
    ))
    registry.register_agent(SpecialistAgent(
        _card("organizer", [TaskType.FILTER_METADATA, TaskType.GROUP_IMAGES], ["filter_images", "group_images"]),
        {TaskType.FILTER_METADATA: _retrieval_worker(), TaskType.GROUP_IMAGES: _group_worker()},
    ))
    registry.register_agent(SpecialistAgent(
        _card("vision", [TaskType.ANALYZE_VISUAL], ["analyze_image"]),
        _retrieval_worker(),
    ))
    return registry


def test_register_and_query_by_id():
    registry = AgentRegistry()
    agent = SpecialistAgent(_card("retrieval", [TaskType.RETRIEVE_IMAGES], []), _retrieval_worker())
    registry.register_agent(agent)
    assert registry.get_agent("retrieval") is agent
    assert registry.get_agent("nope") is None
    assert len(registry.list_agents()) == 1

def test_disabled_agent_not_dispatched_but_visible():
    registry = AgentRegistry()
    disabled = SpecialistAgent(
        _card("vision", [TaskType.ANALYZE_VISUAL], ["analyze_image"]).model_copy(update={"enabled": False}),
        _retrieval_worker(),
    )
    registry.register_agent(disabled)
    assert registry.get_agent("vision") is disabled          # 可查询
    assert len(registry.list_agents()) == 1                   # 可列举
    assert registry.get_agent_for_task(TaskType.ANALYZE_VISUAL) is None  # 不分发


def test_duplicate_agent_id_overrides_and_rebuilds_index():
    registry = AgentRegistry()
    a1 = SpecialistAgent(_card("retrieval", [TaskType.RETRIEVE_IMAGES], []), _retrieval_worker())
    a2 = SpecialistAgent(_card("retrieval", [TaskType.RETRIEVE_IMAGES], []), _retrieval_worker())
    registry.register_agent(a1)
    registry.register_agent(a2)
    assert registry.get_agent("retrieval") is a2
    assert registry.get_agent_for_task(TaskType.RETRIEVE_IMAGES) is a2


def test_default_registry_built_from_workers_including_vision():
    workers = {
        TaskType.RETRIEVE_IMAGES: _retrieval_worker(),
        TaskType.FILTER_METADATA: _retrieval_worker(),
        TaskType.GROUP_IMAGES: _group_worker(),
        TaskType.ANALYZE_VISUAL: _retrieval_worker(),  # 模拟视觉可用
        TaskType.SYNTHESIZE: _synth_worker(),
    }
    registry = build_default_agent_registry(workers=workers)
    by_id = {a.card.agent_id: a for a in registry.list_agents()}
    assert set(by_id) == {"retrieval", "organizer", "vision"}
    assert registry.get_agent_for_task(TaskType.RETRIEVE_IMAGES).card.agent_id == "retrieval"
    assert registry.get_agent_for_task(TaskType.GROUP_IMAGES).card.agent_id == "organizer"
    assert registry.get_agent_for_task(TaskType.ANALYZE_VISUAL).card.agent_id == "vision"
    assert by_id["vision"].card.enabled is True


def test_default_registry_omits_vision_when_no_worker():
    workers = {
        TaskType.RETRIEVE_IMAGES: _retrieval_worker(),
        TaskType.FILTER_METADATA: _retrieval_worker(),
        TaskType.GROUP_IMAGES: _group_worker(),
        TaskType.SYNTHESIZE: _synth_worker(),
    }
    registry = build_default_agent_registry(workers=workers)
    by_id = {a.card.agent_id for a in registry.list_agents()}
    assert by_id == {"retrieval", "organizer"}
    assert registry.get_agent_for_task(TaskType.ANALYZE_VISUAL) is None


def test_build_dispatched_workers_routes_via_agents_and_falls_back():
    base = {
        TaskType.RETRIEVE_IMAGES: _retrieval_worker(),
        TaskType.FILTER_METADATA: _retrieval_worker(),
        TaskType.GROUP_IMAGES: _group_worker(),
        TaskType.ANALYZE_VISUAL: _retrieval_worker(),
        TaskType.SYNTHESIZE: _synth_worker(),
    }
    registry = build_default_agent_registry(workers=base)
    dispatched = build_dispatched_workers(base, registry, synthesize_worker=_synth_worker())
    assert isinstance(dispatched[TaskType.RETRIEVE_IMAGES], SpecialistAgent)
    assert isinstance(dispatched[TaskType.FILTER_METADATA], SpecialistAgent)
    assert isinstance(dispatched[TaskType.GROUP_IMAGES], SpecialistAgent)
    assert isinstance(dispatched[TaskType.ANALYZE_VISUAL], SpecialistAgent)
    # SYNTHESIZE 由 synthesize_worker 覆盖（保留既有语义）
    assert not isinstance(dispatched[TaskType.SYNTHESIZE], SpecialistAgent)
    assert dispatched[TaskType.SYNTHESIZE].tag == "synth"


def test_build_dispatched_workers_fallback_when_agent_missing():
    base = {TaskType.RETRIEVE_IMAGES: _retrieval_worker()}
    registry = AgentRegistry()  # 空注册表
    dispatched = build_dispatched_workers(base, registry)
    assert dispatched[TaskType.RETRIEVE_IMAGES] is base[TaskType.RETRIEVE_IMAGES]


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



def test_register_rejects_non_specialist():
    registry = AgentRegistry()
    try:
        registry.register_agent("not an agent")
    except TypeError:
        pass
    else:
        raise AssertionError("register_agent 应拒绝非 SpecialistAgent")


def test_get_agent_for_task_mapping():
    registry = _three_agents_registry()
    assert registry.get_agent_for_task(TaskType.RETRIEVE_IMAGES).agent_id == "retrieval"
    assert registry.get_agent_for_task(TaskType.FILTER_METADATA).agent_id == "organizer"
    assert registry.get_agent_for_task(TaskType.GROUP_IMAGES).agent_id == "organizer"
    assert registry.get_agent_for_task(TaskType.ANALYZE_VISUAL).agent_id == "vision"
    # synthesize 未注册 Specialist → None
    assert registry.get_agent_for_task(TaskType.SYNTHESIZE) is None
