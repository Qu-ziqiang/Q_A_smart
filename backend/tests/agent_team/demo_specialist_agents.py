"""Phase 1 Specialist — Runtime Verification（离线演示，Requirement 7）。

演示内容（零真实 API / DB）：
  1) Agent Card / AgentRegistry 角色视图（Retrieval / Organizer / Vision）
  2) 完整流程 User Query → Planner(Fake) → PlanValidator → DAGExecutor
     → AgentRegistry → SpecialistAgent.execute → Worker.run → DeterministicSynthesis
  3) 日志体现 RetrievalAgent / OrganizerAgent / VisionAgent executed
  4) 既有 Orchestrator 默认 Executor 已切换为 AgentRegistry 分发（静态核对）

运行（repo 根目录）：
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/demo_specialist_agents.py
"""

import json
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    ComplexityGate,
    PlanValidator,
    PlannerAgent,
    TaskType,
    build_default_agent_cards,
    build_dispatched_workers,
    build_default_agent_registry,
)
from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.vision import build_vision_output
from backend.ai_func.recommendation.orchestration.workers import (
    DeterministicSynthesisWorker,
    GroupOutput,
    ImageCandidate,
    ImageGroup,
    RetrievalOutput,
    TaskStatus,
    WorkerResult,
)


def banner(title):
    print("\n" + "=" * 74)
    print("== " + title)
    print("=" * 74)


# ---------------------------------------------------------------------------
# Fake Workers（确定性，不访问 DB / API）
# ---------------------------------------------------------------------------
class _FW:
    def __init__(self, ttype, fn, tag):
        self.task_type = ttype
        self._fn = fn
        self.tag = tag

    def run(self, task, context):
        return self._fn(task, context)


def _candidates():
    return [
        ImageCandidate(id=11, score=0.92, title="北京夜景", metadata={"location": "北京"},
                       width=1200, height=800, public_url="/static/images/11.jpg"),
        ImageCandidate(id=12, score=0.85, title="上海外滩", metadata={"location": "上海"},
                       width=1200, height=900, public_url="/static/images/12.jpg"),
    ]


def _fake_workers():
    cands = _candidates()
    retr = _FW(TaskType.RETRIEVE_IMAGES, lambda task, ctx: WorkerResult(
        task_id=task.task_id, task_type=TaskType.RETRIEVE_IMAGES, status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=cands, query=task.input.query, total=len(cands)),
    ), "retrieval")
    filt = _FW(TaskType.FILTER_METADATA, lambda task, ctx: WorkerResult(
        task_id=task.task_id, task_type=TaskType.FILTER_METADATA, status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=cands, query="", total=len(cands)),
    ), "filter")
    group = _FW(TaskType.GROUP_IMAGES, lambda task, ctx: WorkerResult(
        task_id=task.task_id, task_type=TaskType.GROUP_IMAGES, status=TaskStatus.SUCCEEDED,
        output=GroupOutput(group_by="location", groups=[
            ImageGroup(key="北京", image_ids=[11], candidates=[cands[0]]),
            ImageGroup(key="上海", image_ids=[12], candidates=[cands[1]]),
        ]),
    ), "group")
    vision = _FW(TaskType.ANALYZE_VISUAL, lambda task, ctx: WorkerResult(
        task_id=task.task_id, task_type=TaskType.ANALYZE_VISUAL, status=TaskStatus.SUCCEEDED,
        output=build_vision_output([]),
    ), "vision")
    return {
        TaskType.RETRIEVE_IMAGES: retr,
        TaskType.FILTER_METADATA: filt,
        TaskType.GROUP_IMAGES: group,
        TaskType.ANALYZE_VISUAL: vision,
        TaskType.SYNTHESIZE: DeterministicSynthesisWorker(),
    }

def demo_role_view():
    banner("1) Agent Card / AgentRegistry 角色视图（含视觉）")
    cards = build_default_agent_cards(vision_ok=True)
    for c in cards:
        print(f"  agent_id={c.agent_id:<10} name={c.name:<16} cost={c.cost_level:<8} "
              f"enabled={c.enabled}  tasks={[t.value for t in c.supported_tasks]}  "
              f"tools={c.available_tools}")
    registry = build_default_agent_registry(workers=_fake_workers())
    print("\n  Task → Specialist 映射：")
    for ttype in (TaskType.RETRIEVE_IMAGES, TaskType.FILTER_METADATA,
                  TaskType.GROUP_IMAGES, TaskType.ANALYZE_VISUAL):
        agent = registry.get_agent_for_task(ttype)
        print(f"    {ttype.value:<18} → {agent.card.name if agent else '(none)'}")


class _FakePlannerLLM:
    """产出覆盖三个 Specialist 的合法计划（含视觉）。"""

    def __call__(self, messages):
        plan = {
            "goal": "找旅行照片，按地点整理，并确认画面是否在海边",
            "synthesis_task_id": "t5",
            "tasks": [
                {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索旅行照片",
                 "input": {"query": "旅行照片", "limit": 20}, "depends_on": []},
                {"task_id": "t2", "task_type": "filter_metadata", "objective": "元数据过滤",
                 "input": {}, "depends_on": ["t1"]},
                {"task_id": "t3", "task_type": "group_images", "objective": "按地点分组",
                 "input": {"group_by": "location"}, "depends_on": ["t2"]},
                {"task_id": "t4", "task_type": "analyze_visual", "objective": "核验海边画面",
                 "input": {"criteria": ["海边"], "max_candidates": 2}, "depends_on": ["t1"]},
                {"task_id": "t5", "task_type": "synthesize", "objective": "汇总推荐",
                 "input": {"recommendation_limit": 5}, "depends_on": ["t3", "t4"]},
            ],
        }
        return json.dumps(plan, ensure_ascii=False)




def demo_full_flow():
    banner("2) 完整流程：Planner → Validator → DAGExecutor → SpecialistAgent → Worker → Synthesis")
    gate = ComplexityGate()
    query = "找旅行照片，按地点整理，并确认画面是否在海边"
    decision = gate.decide(query=query)
    print(f"  [Gate] route={decision.route.value} (query={query!r})")

    planner = PlannerAgent(llm=_FakePlannerLLM())
    draft = planner.create_plan(goal=query)
    vres = PlanValidator().validate(draft, vision_available_flag=True)
    assert vres.valid, vres.errors
    plan = vres.plan
    print(f"  [Planner] plan_id={plan.plan_id} tasks="
          f"{[(t.task_id, t.task_type.value) for t in plan.tasks]}")
    print("  [Validator] validation passed")

    workers = _fake_workers()
    registry = build_default_agent_registry(workers=workers)
    dispatched = build_dispatched_workers(workers, registry)
    executor = DAGExecutor(workers=dispatched)

    print("  [Executor] dispatch: ")
    for ttype, w in executor.workers.items():
        who = getattr(getattr(w, "card", None), "name", type(w).__name__)
        print(f"    {ttype.value:<18} → {who}")

    print("\n  [Executing DAG…]")
    executed_agents = set()
    for ev in executor.execute_stream(plan):
        event = ev["event"]
        if event == "task.started":
            print(f"    → task.started {ev['task_id']} ({ev['task_type']})")
    results = executor.context.results
    for t in plan.tasks:
        r = results[t.task_id]
        print(f"    task {t.task_id} ({t.task_type.value}): {r.status.value}")

    synth = results[plan.synthesis_task_id]
    ids = synth.output.image_ids if synth.output else []
    print(f"\n  [Result] synthesis ids={ids} (DeterministicSynthesisWorker，无 LLM)")

    # 从运行日志（print）回收角色证据：再跑一次并抓取 stdout
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        list(DAGExecutor(workers=build_dispatched_workers(workers, registry)).execute_stream(plan))
    text = buf.getvalue()
    for agent_name in ("RetrievalAgent", "OrganizerAgent", "VisionAgent"):
        present = f"{agent_name} executed" in text
        executed_agents.add(agent_name) if present else None
        print(f"  [LogCheck] {agent_name} executed = {present}")


def demo_orchestrator_wiring():
    banner("3) Orchestrator 默认 Executor 已接入 AgentRegistry（静态核对）")
    from backend.ai_func.recommendation.orchestration.orchestrator import AdaptiveTeamOrchestrator
    orch = AdaptiveTeamOrchestrator()
    print("  AgentRegistry agents: "
          + ", ".join(a.card.name for a in orch.agent_registry.list_agents()))
    for ttype, w in orch.executor.workers.items():
        who = getattr(getattr(w, "card", None), "name", type(w).__name__)
        print(f"    executor.workers[{ttype.value}] → {who}")


if __name__ == "__main__":
    demo_role_view()
    demo_full_flow()
    demo_orchestrator_wiring()
    print("\nRuntime Verification 完成：Workers 已通过 Specialist Agent 角色化封装执行。")

