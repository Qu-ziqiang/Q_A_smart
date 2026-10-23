"""Phase 1-B 真实 E2E 演示脚本（§22 / §25）。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/demo_phase1b.py

演示内容：
  1) 简单查询 → Gate → Legacy
  2) 复杂查询 → 真实 DeepSeek Planner → Validator → 真实 RetrievalWorker
     （复用 tool_search_images / SearchRepository / sqlite-vec）→ GroupImagesWorker
     → DeterministicSynthesisWorker → selection
  3) 失败注入：Partial Failure（metadata 失败，group 继续）
  4) 失败注入：Dependency Failure（retrieve 失败 → group SKIPPED）
"""

import json
import os
import sys
import time

# 兼容 GBK 控制台：emoji/中文输出不崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import ComplexityGate, RouteType
from backend.ai_func.recommendation.orchestration.orchestrator import AdaptiveTeamOrchestrator


def banner(title):
    print("\n" + "=" * 72)
    print("== " + title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1) 简单查询 → Legacy
# ---------------------------------------------------------------------------
def demo_simple_legacy():
    banner("1) 简单查询 → ComplexityGate → LEGACY")
    gate = ComplexityGate()
    for q in ["找猫的照片", "你好", "推荐点好看的图"]:
        d = gate.decide(query=q)
        print(f"  [{d.route.value}] {q}")


# ---------------------------------------------------------------------------
# 2) 真实复杂查询 E2E（DeepSeek Planner + 真实 Workers）
# ---------------------------------------------------------------------------
def demo_real_complex_e2e():
    banner("2) 复杂查询：真实 DeepSeek Planner + 真实 Retrieval/Group Worker")
    query = "找旅行照片，按地点分类并整理结果"
    d = ComplexityGate().decide(query=query)
    print(f"  Gate: route={d.route.value}  matched={d.matched_features}")
    if d.route != RouteType.TEAM:
        print("  ⚠️ 未进入 Team，无法演示 Executor。")
        return

    t0 = time.time()
    orchestrator = AdaptiveTeamOrchestrator()  # 真实 Planner + default_workers
    events = []
    try:
        for raw in orchestrator.run_stream([{"role": "user", "content": query}]):
            events.append(json.loads(raw))
    except Exception as e:
        print(f"  ❌ 执行异常: {type(e).__name__}: {e}")
        return
    total = time.time() - t0

    print(f"  总延迟: {total:.2f}s")
    print(f"  事件序列: {[e['event'] for e in events]}")

    plan = orchestrator.last_plan
    if plan:
        print(f"  plan_id={plan.plan_id}  tasks={[t.task_id + ':' + t.task_type.value for t in plan.tasks]}")

    exec_result = orchestrator.executor.result
    if exec_result:
        print(f"  ExecutionResult: "
              f"succeeded={exec_result.succeeded_count} failed={exec_result.failed_count} "
              f"skipped={exec_result.skipped_count}")
        print(f"  task_statuses={exec_result.task_statuses}")

    retrieval_count = None
    group_count = None
    for e in events:
        if e["event"] == "task.completed":
            print(f"    task {e['task_id']}（{e['task_type']}）耗时 {e.get('duration_ms')}ms "
                  f"summary={e.get('output_summary')}")
            if e["task_type"] == "retrieve_images":
                retrieval_count = e["output_summary"].get("candidate_count")
            if e["task_type"] == "group_images":
                group_count = e["output_summary"].get("group_count")
    sel = [e for e in events if e["event"] == "selection"]
    if sel:
        print(f"  selection ids={sel[-1]['ids']}（{len(sel[-1]['ids'])} 张）")
    print(f"  检索候选数={retrieval_count}  分组数={group_count}")


# ---------------------------------------------------------------------------
# 3/4) 失败注入（Fake Workers，确定性）
# ---------------------------------------------------------------------------
def _mk_fake_workers(meta_fail=False, retr_fail=False):
    from backend.ai_func.recommendation.orchestration.models import (
        FilterMetadataInput,
        GroupImagesInput,
        RetrieveImagesInput,
        SynthesizeInput,
        Task,
        TaskPlan,
        TaskType,
    )
    from backend.ai_func.recommendation.orchestration.workers import (
        GroupOutput,
        ImageCandidate,
        MetadataOutput,
        RetrievalOutput,
        SynthesisOutput,
        TaskError,
        TaskStatus,
        WorkerResult,
    )

    def _ok(tid, ttype, output):
        return WorkerResult(task_id=tid, task_type=ttype, status=TaskStatus.SUCCEEDED, output=output)

    def _fail(tid, ttype, code):
        return WorkerResult(task_id=tid, task_type=ttype, status=TaskStatus.FAILED,
                            error=TaskError(code=code, message=code))

    def retr(task, ctx):
        if retr_fail:
            return _fail(task.task_id, TaskType.RETRIEVE_IMAGES, "RETRIEVAL_ERROR")
        return _ok(task.task_id, TaskType.RETRIEVE_IMAGES,
                   RetrievalOutput(candidates=[
                       ImageCandidate(id=1, score=0.9, public_url="/s/1.jpg", metadata={"location": "北京"}),
                       ImageCandidate(id=2, score=0.8, public_url="/s/2.jpg", metadata={"location": "上海"}),
                   ], query="q", total=2))

    def meta(task, ctx):
        if meta_fail:
            return _fail(task.task_id, TaskType.FILTER_METADATA, "METADATA_READ_ERROR")
        return _ok(task.task_id, TaskType.FILTER_METADATA, MetadataOutput(candidates=[], rejected_ids=[], applied_predicates=[]))

    def group(task, ctx):
        return _ok(task.task_id, TaskType.GROUP_IMAGES, GroupOutput(group_by="location", groups=[]))

    def synth(task, ctx):
        return _ok(task.task_id, TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[1, 2], candidates=[]))

    workers = {
        TaskType.RETRIEVE_IMAGES: _FW(TaskType.RETRIEVE_IMAGES, retr),
        TaskType.FILTER_METADATA: _FW(TaskType.FILTER_METADATA, meta),
        TaskType.GROUP_IMAGES: _FW(TaskType.GROUP_IMAGES, group),
        TaskType.SYNTHESIZE: _FW(TaskType.SYNTHESIZE, synth),
    }
    plan = TaskPlan(
        plan_version="1.0", plan_id="plan_demo", goal="演示",
        synthesis_task_id="t4",
        tasks=[
            Task(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, objective="r",
                 input=RetrieveImagesInput(query="x"), depends_on=[]),
            Task(task_id="t2", task_type=TaskType.FILTER_METADATA, objective="m",
                 input=FilterMetadataInput(), depends_on=["t1"]),
            Task(task_id="t3", task_type=TaskType.GROUP_IMAGES, objective="g",
                 input=GroupImagesInput(group_by="location"), depends_on=["t1"]),
            Task(task_id="t4", task_type=TaskType.SYNTHESIZE, objective="s",
                 input=SynthesizeInput(), depends_on=["t3"]),
        ],
    )
    return workers, plan


class _FW:
    def __init__(self, ttype, fn):
        self.task_type = ttype
        self._fn = fn

    def run(self, task, context):
        return self._fn(task, context)


def demo_failure_injection():
    from backend.ai_func.recommendation.orchestration.executor import DAGExecutor

    banner("3/4) 失败注入（Fake Workers）")

    # Demo 3：Partial Failure — metadata 失败，group 只依赖 retrieve → group 仍执行
    workers, plan = _mk_fake_workers(meta_fail=True)
    ex = DAGExecutor(workers=workers)
    evs = list(ex.execute_stream(plan))
    print("  Partial Failure 场景:")
    for e in evs:
        if e["event"] in ("task.failed", "task.skipped", "task.completed"):
            print(f"    {e['event']} {e.get('task_id')} status={e.get('status')} "
                  f"reason={e.get('reason', '')} code={e.get('error', {}).get('code', '')}")
    g_status = ex.context.results["t3"].status.value
    print(f"    → metadata FAILED；group 仍执行 → {g_status}（Partial Failure Isolation ✅）")

    # Demo 4：Dependency Failure — retrieve 失败 → group SKIPPED
    workers, plan = _mk_fake_workers(retr_fail=True)
    ex = DAGExecutor(workers=workers)
    evs = list(ex.execute_stream(plan))
    print("\n  Dependency Failure 场景:")
    for e in evs:
        if e["event"] in ("task.failed", "task.skipped"):
            print(f"    {e['event']} {e.get('task_id')} reason={e.get('reason', '')} "
                  f"code={e.get('error', {}).get('code', '')}")
    g_status = ex.context.results["t3"].status.value
    print(f"    → retrieve FAILED；group → {g_status}（Dependency Failure Propagation ✅）")


if __name__ == "__main__":
    demo_simple_legacy()
    demo_real_complex_e2e()
    demo_failure_injection()
    print("\n演示完成。")
