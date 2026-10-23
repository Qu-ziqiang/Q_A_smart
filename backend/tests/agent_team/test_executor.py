"""Phase 1-B — DAGExecutor 单元测试（Fake Workers，无真实 API / DB）。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_executor.py
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.models import (
    GroupImagesInput,
    RetrieveImagesInput,
    SynthesizeInput,
    Task,
    TaskPlan,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.workers import (
    GroupOutput,
    ImageCandidate,
    RetrievalOutput,
    SynthesisOutput,
    TaskError,
    WorkerResult,
)


def _task(tid, ttype, input_model, depends=None):
    return Task(task_id=tid, task_type=ttype, objective="o",
                input=input_model, depends_on=depends or [])


def _plan(synth_id, tasks):
    return TaskPlan(plan_version="1.0", plan_id="plan_t", goal="g", tasks=tasks, synthesis_task_id=synth_id)


def _ok(tid, ttype, output=None):
    return WorkerResult(task_id=tid, task_type=ttype, status=TaskStatus.SUCCEEDED, output=output)


def _fail(tid, ttype, code="X"):
    return WorkerResult(task_id=tid, task_type=ttype, status=TaskStatus.FAILED,
                        error=TaskError(code=code, message=code))


def _cand(id):
    return ImageCandidate(id=id, score=0.5, public_url=f"/s/{id}.jpg")


def _retr(tid, *ids):
    return _ok(tid, TaskType.RETRIEVE_IMAGES,
               RetrievalOutput(candidates=[_cand(i) for i in ids], query="q", total=len(ids)))


class FakeWorker:
    """可编程 Fake Worker。fn(task, context) -> WorkerResult；记录执行顺序。"""

    def __init__(self, ttype, fn):
        self.task_type = ttype
        self._fn = fn
        self.order = []

    def run(self, task, context):
        self.order.append(task.task_id)
        return self._fn(task, context)


def _exec(workers, plan, max_executed=None):
    ex = DAGExecutor(workers=workers, max_executed_tasks=max_executed or 16)
    events = list(ex.execute_stream(plan))
    return ex, events


# ---------------------------------------------------------------------------
# 基本执行
# ---------------------------------------------------------------------------
def test_linear_dag_all_succeed():
    w1 = FakeWorker(TaskType.RETRIEVE_IMAGES, lambda t, c: _retr(t.task_id, 1, 2))
    w2 = FakeWorker(TaskType.SYNTHESIZE, lambda t, c: _ok(t.task_id, TaskType.SYNTHESIZE,
                                                          SynthesisOutput(image_ids=[1, 2], candidates=[])))
    plan = _plan("t2", [
        _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="x")),
        _task("t2", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t1"]),
    ])
    ex, events = _exec({TaskType.RETRIEVE_IMAGES: w1, TaskType.SYNTHESIZE: w2}, plan)
    assert w1.order == ["t1"]
    assert w2.order == ["t2"]
    assert ex.context.results["t1"].status == TaskStatus.SUCCEEDED
    assert ex.context.results["t2"].status == TaskStatus.SUCCEEDED
    names = [e["event"] for e in events]
    assert names == ["task.started", "task.completed", "task.started", "task.completed"]
    assert ex.result.succeeded_count == 2
    assert ex.result.executed_count == 2


def test_multi_node_dag_order():
    order = []

    def synth(t, c):
        order.append("synth")
        return _ok(t.task_id, TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[1, 2], candidates=[]))

    plan = _plan("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="a")),
        _task("t2", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="b")),
        _task("t3", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t1", "t2"]),
    ])
    workers = {
        TaskType.RETRIEVE_IMAGES: FakeWorker(TaskType.RETRIEVE_IMAGES, lambda t, c: _retr(t.task_id, 1)),
        TaskType.SYNTHESIZE: FakeWorker(TaskType.SYNTHESIZE, synth),
    }
    ex, _ = _exec(workers, plan)
    assert order[-1] == "synth"
    assert ex.result.succeeded_count == 3


def test_dependency_order_no_simple_array_iteration():
    """验证 Executor 不按数组顺序：数组 [t2, t1]，依赖 t2←t1，t1 必须先执行。"""
    plan = _plan("t2", [
        _task("t2", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t1"]),
        _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="x")),
    ])
    workers = {
        TaskType.RETRIEVE_IMAGES: FakeWorker(TaskType.RETRIEVE_IMAGES, lambda t, c: _retr(t.task_id, 1)),
        TaskType.SYNTHESIZE: FakeWorker(TaskType.SYNTHESIZE, lambda t, c: _ok(t.task_id, TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[1]))),
    }
    ex, events = _exec(workers, plan)
    started = [e["task_id"] for e in events if e["event"] == "task.started"]
    assert started == ["t1", "t2"]


# ---------------------------------------------------------------------------
# 失败 / 跳过 / 部分失败
# ---------------------------------------------------------------------------
def test_failure_propagation_skipped():
    w1 = FakeWorker(TaskType.RETRIEVE_IMAGES, lambda t, c: _fail(t.task_id, TaskType.RETRIEVE_IMAGES, "RETRIEVAL_ERROR"))
    w2 = FakeWorker(TaskType.GROUP_IMAGES, lambda t, c: _ok(t.task_id, TaskType.GROUP_IMAGES, GroupOutput(group_by="x", groups=[])))
    w3 = FakeWorker(TaskType.SYNTHESIZE, lambda t, c: _ok(t.task_id, TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[])))
    plan = _plan("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="x")),
        _task("t2", TaskType.GROUP_IMAGES, GroupImagesInput(group_by="location"), depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t2"]),
    ])
    ex, events = _exec({TaskType.RETRIEVE_IMAGES: w1, TaskType.GROUP_IMAGES: w2, TaskType.SYNTHESIZE: w3}, plan)
    assert ex.context.results["t1"].status == TaskStatus.FAILED
    assert ex.context.results["t2"].status == TaskStatus.SKIPPED
    assert ex.context.results["t3"].status == TaskStatus.SKIPPED
    assert w2.order == []  # group worker 未被调用
    assert w3.order == []
    skipped_reasons = [e["reason"] for e in events if e["event"] == "task.skipped"]
    assert skipped_reasons == ["DEPENDENCY_UNAVAILABLE", "DEPENDENCY_UNAVAILABLE"]
    assert ex.result.skipped_count == 2


def test_partial_failure_isolation():
    """metadata 失败，但 group 只依赖 retrieve → group 仍执行。"""
    from backend.ai_func.recommendation.orchestration.models import FilterMetadataInput

    w_retr = FakeWorker(TaskType.RETRIEVE_IMAGES, lambda t, c: _retr(t.task_id, 1, 2))
    w_meta = FakeWorker(TaskType.FILTER_METADATA, lambda t, c: _fail(t.task_id, TaskType.FILTER_METADATA, "METADATA_READ_ERROR"))
    w_group = FakeWorker(TaskType.GROUP_IMAGES, lambda t, c: _ok(t.task_id, TaskType.GROUP_IMAGES,
                                                                 GroupOutput(group_by="location", groups=[])))
    w_synth = FakeWorker(TaskType.SYNTHESIZE, lambda t, c: _ok(t.task_id, TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[])))
    plan = _plan("t4", [
        _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="x")),
        _task("t2", TaskType.FILTER_METADATA, FilterMetadataInput(), depends=["t1"]),
        _task("t3", TaskType.GROUP_IMAGES, GroupImagesInput(group_by="location"), depends=["t1"]),
        _task("t4", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t3"]),
    ])
    workers = {
        TaskType.RETRIEVE_IMAGES: w_retr,
        TaskType.FILTER_METADATA: w_meta,
        TaskType.GROUP_IMAGES: w_group,
        TaskType.SYNTHESIZE: w_synth,
    }
    ex, _ = _exec(workers, plan)
    assert ex.context.results["t1"].status == TaskStatus.SUCCEEDED
    assert ex.context.results["t2"].status == TaskStatus.FAILED
    assert ex.context.results["t3"].status == TaskStatus.SUCCEEDED  # 只依赖 t1
    assert ex.context.results["t4"].status == TaskStatus.SUCCEEDED
    assert w_group.order == ["t3"]
    assert ex.result.failed_count == 1
    assert ex.result.partial_failures and ex.result.partial_failures[0].code == "METADATA_READ_ERROR"


def test_unknown_worker_fails():
    plan = _plan("t2", [
        _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="x")),
        _task("t2", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t1"]),
    ])
    # 只注册 RETRIEVE，不注册 SYNTHESIZE
    workers = {TaskType.RETRIEVE_IMAGES: FakeWorker(TaskType.RETRIEVE_IMAGES, lambda t, c: _retr(t.task_id, 1))}
    ex, events = _exec(workers, plan)
    assert ex.context.results["t1"].status == TaskStatus.SUCCEEDED
    assert ex.context.results["t2"].status == TaskStatus.FAILED
    assert ex.context.results["t2"].error.code == "WORKER_UNAVAILABLE"
    assert any(e["event"] == "task.failed" and e["error"]["code"] == "WORKER_UNAVAILABLE" for e in events)


def test_max_executed_tasks_guard():
    plan = _plan("t2", [
        _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="x")),
        _task("t2", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t1"]),
    ])
    workers = {
        TaskType.RETRIEVE_IMAGES: FakeWorker(TaskType.RETRIEVE_IMAGES, lambda t, c: _retr(t.task_id, 1)),
        TaskType.SYNTHESIZE: FakeWorker(TaskType.SYNTHESIZE, lambda t, c: _ok(t.task_id, TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[1]))),
    }
    ex, events = _exec(workers, plan, max_executed=1)
    assert ex.context.results["t1"].status == TaskStatus.SUCCEEDED
    assert ex.context.results["t2"].status == TaskStatus.SKIPPED
    assert ex.context.results["t2"].error.code == "EXECUTION_LIMIT"
    assert any(e["event"] == "task.skipped" and e["reason"] == "EXECUTION_LIMIT" for e in events)


def test_no_infinite_loop_on_long_chain():
    """3 节点链：恰好执行 3 个任务，不出现重复执行。"""
    plan = _plan("t3", [
        _task("t1", TaskType.RETRIEVE_IMAGES, RetrieveImagesInput(query="x")),
        _task("t2", TaskType.GROUP_IMAGES, GroupImagesInput(group_by="location"), depends=["t1"]),
        _task("t3", TaskType.SYNTHESIZE, SynthesizeInput(), depends=["t2"]),
    ])
    workers = {
        TaskType.RETRIEVE_IMAGES: FakeWorker(TaskType.RETRIEVE_IMAGES, lambda t, c: _retr(t.task_id, 1)),
        TaskType.GROUP_IMAGES: FakeWorker(TaskType.GROUP_IMAGES, lambda t, c: _ok(t.task_id, TaskType.GROUP_IMAGES, GroupOutput(group_by="x", groups=[]))),
        TaskType.SYNTHESIZE: FakeWorker(TaskType.SYNTHESIZE, lambda t, c: _ok(t.task_id, TaskType.SYNTHESIZE, SynthesisOutput(image_ids=[1]))),
    }
    ex, events = _exec(workers, plan)
    assert sum(1 for e in events if e["event"] == "task.started") == 3
    assert ex.result.executed_count == 3


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

