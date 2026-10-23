"""Phase 1-A 真实演示脚本（无需启动服务器）。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/demo_phase1a.py

演示内容：
  1) Complexity Gate：简单/复杂/模糊请求路由
  2) PlanValidator：合法 / 非法 / 循环 / 视觉不可用 计划
  3) PlannerAgent：尝试真实 LLM 生成 Typed TaskPlan（失败则明确打印，不阻塞）
  4) 服务层集成：stream_core 的 Team 事件 + Legacy 兜底（使用 Fake Agent/Fake Planner，确定性）
"""

import json
import os
import sys
import time

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    ComplexityGate,
    DeterministicSynthesisWorker,
    GroupImagesWorker,
    PlanValidator,
    PlannerAgent,
    PlannerFailure,
    RetrievalWorker,
    TaskDraft,
    TaskPlanDraft,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.models import RouteType


def banner(title):
    print("\n" + "=" * 70)
    print("== " + title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1) Complexity Gate
# ---------------------------------------------------------------------------
def demo_gate():
    banner("1) Complexity Gate 路由")
    gate = ComplexityGate()
    queries = [
        ("简单：找猫的照片", "找猫的照片"),
        ("简单：问候", "你好"),
        ("模糊：推荐点好看的图", "推荐点好看的图"),
        ("复杂：按地点分类", "找出旅行照片，按地点分类，然后推荐最合适的几张"),
        ("复杂：多阶段", "先找出旅行照片，再按时间归组，每组推荐两张"),
        ("复杂：多集合", "分别找猫和狗的照片"),
        ("复杂：E2E", "从旅行照片中找出横向图片，按地点分类，每组推荐最多 2 张"),
    ]
    for label, q in queries:
        d = gate.decide(query=q)
        route = "LEGACY" if d.route == RouteType.LEGACY else "TEAM"
        print(f"  [{route}] {label}")
        if d.matched_features:
            print(f"          matched_features={d.matched_features}")


# ---------------------------------------------------------------------------
# 2) PlanValidator
# ---------------------------------------------------------------------------
def demo_validator():
    banner("2) PlanValidator 校验")
    v = PlanValidator()

    def _task(tid, ttype, input=None, depends=None):
        return TaskDraft(
            task_id=tid, task_type=ttype, objective="o",
            input=input or {}, depends_on=depends or [],
        )

    valid = TaskPlanDraft(
        goal="找旅行照片按地点分类推荐", synthesis_task_id="t4",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, {"query": "旅行照片", "limit": 30}),
            _task("t2", TaskType.FILTER_METADATA, depends=["t1"]),
            _task("t3", TaskType.GROUP_IMAGES, {"group_by": "location"}, ["t2"]),
            _task("t4", TaskType.SYNTHESIZE, {"recommendation_limit": 10}, ["t3"]),
        ],
    )
    r = v.validate(valid)
    print(f"  合法计划      -> valid={r.valid}  plan_id={r.plan.plan_id if r.plan else None}")
    if r.plan:
        for t in r.plan.tasks:
            print(f"      {t.task_id}  {t.task_type.value:<18} input={type(t.input).__name__}")

    cyc = TaskPlanDraft(
        goal="x", synthesis_task_id="t2",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, {"query": "x"}, ["t2"]),
            _task("t2", TaskType.SYNTHESIZE, depends=["t1"]),
        ],
    )
    r = v.validate(cyc)
    print(f"  循环计划      -> valid={r.valid}  errors={r.errors[:2]}")

    bad = TaskPlanDraft(
        goal="x", synthesis_task_id="t2",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, {"query": "x"}),
            _task("t2", TaskType.SYNTHESIZE, depends=["t99"]),
        ],
    )
    r = v.validate(bad)
    print(f"  非法依赖      -> valid={r.valid}  errors={r.errors[:2]}")

    vis = TaskPlanDraft(
        goal="x", synthesis_task_id="t3",
        tasks=[
            _task("t1", TaskType.RETRIEVE_IMAGES, {"query": "x"}),
            _task("t2", TaskType.ANALYZE_VISUAL, depends=["t1"]),
            _task("t3", TaskType.SYNTHESIZE, depends=["t2"]),
        ],
    )
    r = v.validate(vis)
    print(f"  视觉任务      -> valid={r.valid}  errors={r.errors[:2]}")


# ---------------------------------------------------------------------------
# 3) PlannerAgent（真实 LLM，尽力而为）
# ---------------------------------------------------------------------------
def demo_real_planner():
    banner("3) PlannerAgent 真实 LLM（尽力而为）")
    query = "找出旅行照片，按地点分类，然后推荐最合适的几张"
    try:
        planner = PlannerAgent()
        t0 = time.time()
        draft = planner.create_plan(goal=query)
        r = PlanValidator().validate(draft)
        print(f"  planner 调用耗时: {time.time() - t0:.2f}s")
        print(f"  draft valid={r.valid}  tasks={len(draft.tasks)}")
        if r.valid and r.plan:
            print(f"  plan_id={r.plan.plan_id}")
            for t in r.plan.tasks:
                print(f"      {t.task_id}  {t.task_type.value:<18} depends={t.depends_on}  objective={t.objective}")
        else:
            print(f"  validation errors={r.errors}")
    except PlannerFailure as e:
        print(f"  [PlannerFailure] {e}")
        print("  （本环境可能无外网或 API Key 无效；可改为注入 Fake LLM 验证，见测试）")
    except Exception as e:
        print(f"  [Unexpected] {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 4) 服务层集成演示（Fake Planner + Fake 检索工具，确定性）
# ---------------------------------------------------------------------------
class _FakeAgent:
    """模拟 Legacy RecommendAgent，只产出 selection 事件。"""

    def run_stream(self, messages, vector_targets=None, filters=None, limit=20):
        yield json.dumps({"event": "assistant_delta", "delta": "（Legacy 兜底回答）"}, ensure_ascii=False) + "\n\n"
        yield json.dumps({"event": "selection", "ids": [1, 2], "images": [{"id": 1, "score": 0.9}]}, ensure_ascii=False) + "\n\n"


class _FakePlannerLLM:
    def __call__(self, messages):
        plan = {
            "goal": "找旅行照片按地点分类推荐",
            "synthesis_task_id": "t3",
            "tasks": [
                {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索旅行照片",
                 "input": {"query": "旅行照片", "limit": 30}, "depends_on": []},
                {"task_id": "t2", "task_type": "group_images", "objective": "按地点分组",
                 "input": {"group_by": "location"}, "depends_on": ["t1"]},
                {"task_id": "t3", "task_type": "synthesize", "objective": "汇总推荐",
                 "input": {"recommendation_limit": 10}, "depends_on": ["t2"]},
            ],
        }
        return json.dumps(plan, ensure_ascii=False)


def _fake_search_tool(query, targets, filters, limit):
    return [
        {"id": 11, "score": 0.91, "title": "北京夜景", "description": "", "tags": ["city"],
         "metadata": {"location": "北京"}, "width": 1200, "height": 800, "file_type": "jpg",
         "public_url": "/static/images/11.jpg", "filepath": "data/images/11.jpg"},
        {"id": 12, "score": 0.85, "title": "上海外滩", "description": "", "tags": ["city"],
         "metadata": {"location": "上海"}, "width": 1200, "height": 900, "file_type": "jpg",
         "public_url": "/static/images/12.jpg", "filepath": "data/images/12.jpg"},
        {"id": 13, "score": 0.80, "title": "北京胡同", "description": "", "tags": [],
         "metadata": {}, "width": 800, "height": 1200, "file_type": "jpg",
         "public_url": "/static/images/13.jpg", "filepath": "data/images/13.jpg"},
    ]


def demo_service_integration():
    banner("4) 服务层集成（Fake Planner + Fake 检索，确定性）")
    from backend.ai_func.recommendation import _RecommendationService
    from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
    from backend.ai_func.recommendation.orchestration.orchestrator import AdaptiveTeamOrchestrator

    service = _RecommendationService()
    service._agent = _FakeAgent()
    # Team 路径：Fake Planner + Fake 检索工具 + 真实 Group/合成 Worker
    executor = DAGExecutor(workers={
        TaskType.RETRIEVE_IMAGES: RetrievalWorker(search_tool=_fake_search_tool),
        TaskType.GROUP_IMAGES: GroupImagesWorker(),
        TaskType.SYNTHESIZE: DeterministicSynthesisWorker(),
    })
    service._orchestrator = AdaptiveTeamOrchestrator(
        planner=PlannerAgent(llm=_FakePlannerLLM()),
        executor=executor,
    )

    simple_msgs = [{"role": "user", "content": "找猫的照片"}]
    print("  简单查询 stream_core 事件:")
    for raw in service.stream_core(messages=simple_msgs):
        ev = json.loads(raw)["event"]
        print(f"    -> {ev}")

    print()
    complex_msgs = [{"role": "user", "content": "找出旅行照片，按地点分类，然后推荐最合适的几张"}]
    print("  复杂查询 stream_core 事件:")
    for raw in service.stream_core(messages=complex_msgs):
        obj = json.loads(raw)
        ev = obj["event"]
        extra = ""
        if ev == "plan.created":
            extra = f" plan_id={obj['plan_id']} tasks={len(obj['tasks'])}"
        elif ev == "task.completed":
            extra = f" {obj.get('output_summary', {})}"
        elif ev == "selection":
            extra = f" ids={obj['ids']}"
        print(f"    -> {ev}{extra}")


if __name__ == "__main__":
    demo_gate()
    demo_validator()
    demo_real_planner()
    demo_service_integration()
    print("\n演示完成。")
