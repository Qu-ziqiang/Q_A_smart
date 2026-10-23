"""Phase 1-B — Bounded DAG Executor（规划 §9 / §10 / §13 / §14）。

- 串行按依赖执行（本阶段不做并行；ExecutionContext 为未来并发预留）
- 依赖失败传播：所有依赖 FAILED/SKIPPED → 当前任务 SKIPPED(DEPENDENCY_UNAVAILABLE)
- Partial Failure Isolation：至少一个依赖 SUCCEEDED → 继续执行
- MAX_EXECUTED_TASKS 执行上限保护，杜绝无限执行
- 每任务事件：task.started / task.completed / task.failed / task.skipped
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, Field

from .models import VISION_GATED_TYPES, Task, TaskPlan, TaskStatus, TaskType
from .workers import (
    ExecutionContext,
    GroupOutput,
    MetadataOutput,
    RetrievalOutput,
    SynthesisOutput,
    TaskError,
    Worker,
    WorkerResult,
    default_workers,
)
from .vision import VisionTaskOutput
from .opt_flags import dag_parallel_enabled

# Executor 自身执行保护：即使绕过 Validator，也绝不允许执行超过该数量
MAX_EXECUTED_TASKS = 16
# B1：层内非视觉并行上限（flag 关时不使用）
MAX_LAYER_CONCURRENCY = 2
_VISION_TYPES = set(VISION_GATED_TYPES)


class ExecutionResult(BaseModel):
    plan_id: str
    executed_count: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    task_statuses: Dict[str, str] = Field(default_factory=dict)
    partial_failures: List[TaskError] = Field(default_factory=list)


class DAGExecutor:
    """串行 DAG 执行器。

    execute_stream(plan) 逐事件产出（dict），并把结果写入 self.context.results。
    所有依赖判断基于 ExecutionContext，不读取聊天历史 / 系统提示。
    """

    def __init__(
        self,
        workers: Optional[Dict[TaskType, Worker]] = None,
        max_executed_tasks: int = MAX_EXECUTED_TASKS,
    ):
        self.workers = workers if workers is not None else default_workers()
        self.max_executed_tasks = max_executed_tasks
        self.context: Optional[ExecutionContext] = None
        self.result: Optional[ExecutionResult] = None

    # -- 主入口 ------------------------------------------------------------
    def execute_stream(self, plan: TaskPlan) -> Iterator[Dict[str, Any]]:
        self.context = ExecutionContext(plan)
        self.result = None

        tasks_by_id = {t.task_id: t for t in plan.tasks}
        dependents: Dict[str, List[str]] = {t.task_id: [] for t in plan.tasks}
        indeg = {t.task_id: len(t.depends_on) for t in plan.tasks}
        for t in plan.tasks:
            for d in t.depends_on:
                if d in dependents:
                    dependents[d].append(t.task_id)

        ready: List[Task] = [t for t in plan.tasks if indeg[t.task_id] == 0]
        executed = 0
        guard_hit = False
        parallel = dag_parallel_enabled()

        while ready:
            if parallel:
                wave = list(ready)
                ready = []
                for ev in self._execute_wave(
                    wave, tasks_by_id, dependents, indeg, ready, executed,
                ):
                    kind = ev.get("_kind")
                    if kind == "executed":
                        executed += 1
                    elif kind == "guard":
                        guard_hit = True
                    else:
                        yield ev
                continue

            task = ready.pop(0)

            # 执行上限保护（防御 Validator 被绕过 / 异常 Plan）
            if executed >= self.max_executed_tasks:
                guard_hit = True
                remaining = [task] + [tasks_by_id[i] for i in ready if i in tasks_by_id]
                for t in remaining:
                    self.context.results[t.task_id] = WorkerResult(
                        task_id=t.task_id, task_type=t.task_type,
                        status=TaskStatus.SKIPPED,
                        error=TaskError(code="EXECUTION_LIMIT", message="超过 MAX_EXECUTED_TASKS"),
                    )
                    yield self._event_skipped(t, "EXECUTION_LIMIT")
                break

            # 依赖可用性：至少一个依赖 SUCCEEDED 才执行
            deps_ok = any(
                self.context.results[d].status == TaskStatus.SUCCEEDED
                for d in task.depends_on if d in self.context.results
            )
            if task.depends_on and not deps_ok:
                self.context.results[task.task_id] = WorkerResult(
                    task_id=task.task_id, task_type=task.task_type,
                    status=TaskStatus.SKIPPED,
                    error=TaskError(
                        code="DEPENDENCY_UNAVAILABLE",
                        message="所有依赖均失败或跳过，无法执行",
                    ),
                )
                yield self._event_skipped(task, "DEPENDENCY_UNAVAILABLE")
            else:
                executed += 1
                started = time.time()
                yield self._event_started(task)
                result = self._run_task(task)
                result.duration_ms = int((time.time() - started) * 1000)
                self.context.results[task.task_id] = result
                if result.status == TaskStatus.SUCCEEDED:
                    yield self._event_completed(task, result)
                else:
                    yield self._event_failed(task, result)

            # 解除后继任务依赖
            for j in dependents.get(task.task_id, []):
                indeg[j] -= 1
                if indeg[j] == 0 and tasks_by_id.get(j):
                    ready.append(tasks_by_id[j])

        self._finalize(plan, guard_hit)

    def _deps_ok(self, task: Task) -> bool:
        return any(
            self.context.results[d].status == TaskStatus.SUCCEEDED
            for d in task.depends_on if d in self.context.results
        )

    def _unlock(self, task: Task, dependents: Dict[str, List[str]],
                indeg: Dict[str, int], tasks_by_id: Dict[str, Task],
                ready: List[Task]) -> None:
        for j in dependents.get(task.task_id, []):
            indeg[j] -= 1
            if indeg[j] == 0 and tasks_by_id.get(j):
                ready.append(tasks_by_id[j])

    def _skip(self, task: Task, code: str, message: str) -> WorkerResult:
        result = WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SKIPPED,
            error=TaskError(code=code, message=message),
        )
        self.context.results[task.task_id] = result
        return result

    def _execute_wave(
        self,
        wave: List[Task],
        tasks_by_id: Dict[str, Task],
        dependents: Dict[str, List[str]],
        indeg: Dict[str, int],
        ready: List[Task],
        executed: int,
    ) -> Iterator[Dict[str, Any]]:
        """同层：非视觉最多 2 路线程；Vision 始终串行。事件穿插 started/completed。"""
        runnable: List[Task] = []
        for task in wave:
            if task.depends_on and not self._deps_ok(task):
                self._skip(task, "DEPENDENCY_UNAVAILABLE", "所有依赖均失败或跳过，无法执行")
                yield self._event_skipped(task, "DEPENDENCY_UNAVAILABLE")
                self._unlock(task, dependents, indeg, tasks_by_id, ready)
                continue
            runnable.append(task)

        others = [t for t in runnable if t.task_type not in _VISION_TYPES]
        visions = [t for t in runnable if t.task_type in _VISION_TYPES]

        for i in range(0, len(others), MAX_LAYER_CONCURRENCY):
            chunk = others[i:i + MAX_LAYER_CONCURRENCY]
            if executed + len(chunk) > self.max_executed_tasks:
                room = max(0, self.max_executed_tasks - executed)
                overflow = chunk[room:]
                chunk = chunk[:room]
                yield {"_kind": "guard"}
                for t in overflow:
                    self._skip(t, "EXECUTION_LIMIT", "超过 MAX_EXECUTED_TASKS")
                    yield self._event_skipped(t, "EXECUTION_LIMIT")
                    self._unlock(t, dependents, indeg, tasks_by_id, ready)
            if not chunk:
                continue
            for ev in self._run_chunk_parallel(chunk):
                yield ev
                if ev.get("_kind") == "executed":
                    executed += 1
            for t in chunk:
                self._unlock(t, dependents, indeg, tasks_by_id, ready)

        for task in visions:
            if executed >= self.max_executed_tasks:
                yield {"_kind": "guard"}
                self._skip(task, "EXECUTION_LIMIT", "超过 MAX_EXECUTED_TASKS")
                yield self._event_skipped(task, "EXECUTION_LIMIT")
                self._unlock(task, dependents, indeg, tasks_by_id, ready)
                continue
            yield {"_kind": "executed"}
            started = time.time()
            yield self._event_started(task)
            result = self._run_task(task)
            result.duration_ms = int((time.time() - started) * 1000)
            self.context.results[task.task_id] = result
            if result.status == TaskStatus.SUCCEEDED:
                yield self._event_completed(task, result)
            else:
                yield self._event_failed(task, result)
            self._unlock(task, dependents, indeg, tasks_by_id, ready)

    def _run_chunk_parallel(self, chunk: List[Task]) -> Iterator[Dict[str, Any]]:
        for task in chunk:
            yield self._event_started(task)
            yield {"_kind": "executed"}
        if len(chunk) == 1:
            started = time.time()
            result = self._run_task(chunk[0])
            result.duration_ms = int((time.time() - started) * 1000)
            self.context.results[chunk[0].task_id] = result
            if result.status == TaskStatus.SUCCEEDED:
                yield self._event_completed(chunk[0], result)
            else:
                yield self._event_failed(chunk[0], result)
            return

        def _job(task: Task) -> Tuple[str, WorkerResult, int]:
            t0 = time.time()
            res = self._run_task(task)
            return task.task_id, res, int((time.time() - t0) * 1000)

        with ThreadPoolExecutor(max_workers=MAX_LAYER_CONCURRENCY) as pool:
            futs = {pool.submit(_job, t): t for t in chunk}
            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    tid, result, dur = fut.result()
                except Exception as e:
                    result = WorkerResult(
                        task_id=task.task_id, task_type=task.task_type,
                        status=TaskStatus.FAILED,
                        error=TaskError(code="WORKER_CRASH", message=str(e)[:200]),
                    )
                    dur = 0
                    tid = task.task_id
                result.duration_ms = dur
                self.context.results[tid] = result
                if result.status == TaskStatus.SUCCEEDED:
                    yield self._event_completed(task, result)
                else:
                    yield self._event_failed(task, result)

    # -- 内部 ----------------------------------------------------------------
    def _run_task(self, task: Task) -> WorkerResult:
        worker = self.workers.get(task.task_type)
        if worker is None:
            return WorkerResult(
                task_id=task.task_id, task_type=task.task_type,
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="WORKER_UNAVAILABLE",
                    message=f"未注册的 Worker: {task.task_type.value}",
                ),
            )
        try:
            return worker.run(task, self.context)
        except Exception as e:
            return WorkerResult(
                task_id=task.task_id, task_type=task.task_type,
                status=TaskStatus.FAILED,
                error=TaskError(code="WORKER_CRASH", message=str(e)[:200]),
            )

    def _finalize(self, plan: TaskPlan, guard_hit: bool):
        results = self.context.results
        partial = [r.error for r in results.values() if r.error and r.status == TaskStatus.FAILED]
        if guard_hit:
            partial.append(TaskError(code="EXECUTION_LIMIT", message="命中执行上限保护"))
        self.result = ExecutionResult(
            plan_id=plan.plan_id,
            executed_count=sum(1 for r in results.values() if r.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED)),
            succeeded_count=sum(1 for r in results.values() if r.status == TaskStatus.SUCCEEDED),
            failed_count=sum(1 for r in results.values() if r.status == TaskStatus.FAILED),
            skipped_count=sum(1 for r in results.values() if r.status == TaskStatus.SKIPPED),
            task_statuses={tid: r.status.value for tid, r in results.items()},
            partial_failures=partial,
        )

    # -- 事件构造（只含摘要，不含完整候选 / filepath / prompt）----------------
    @staticmethod
    def _event_started(task: Task) -> Dict[str, Any]:
        return {
            "event": "task.started",
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "status": "running",
        }

    @staticmethod
    def _event_completed(task: Task, result: WorkerResult) -> Dict[str, Any]:
        return {
            "event": "task.completed",
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "status": "succeeded",
            "duration_ms": result.duration_ms,
            "output_summary": _output_summary(result),
        }

    @staticmethod
    def _event_failed(task: Task, result: WorkerResult) -> Dict[str, Any]:
        return {
            "event": "task.failed",
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "status": "failed",
            "error": {
                "code": result.error.code if result.error else "UNKNOWN",
                "message": (result.error.message if result.error else "")[:160],
            },
        }

    @staticmethod
    def _event_skipped(task: Task, reason: str) -> Dict[str, Any]:
        return {
            "event": "task.skipped",
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "status": "skipped",
            "reason": reason,
        }


def _output_summary(result: WorkerResult) -> Dict[str, Any]:
    out = result.output
    if isinstance(out, RetrievalOutput):
        return {"candidate_count": len(out.candidates), "image_ids": [c.id for c in out.candidates]}
    if isinstance(out, MetadataOutput):
        return {"candidate_count": len(out.candidates), "rejected_ids": out.rejected_ids}
    if isinstance(out, GroupOutput):
        return {"group_count": len(out.groups), "group_keys": [g.key for g in out.groups]}
    if isinstance(out, VisionTaskOutput):
        # SSE observability（Stage 7）：只输出安全统计，不输出任何视觉内容
        return {
            "vision": {
                "candidate_count": out.total,
                "success_count": out.success_count,
                "failed_count": out.failed_count,
                "systemic_failed": out.systemic_failed,
            }
        }
    if isinstance(out, SynthesisOutput):
        summary: Dict[str, Any] = {"image_ids": out.image_ids, "group_summary": out.group_summary}
        # Phase 1-C：合成模式可观测（复用 task.completed，不新增事件）
        summary["synthesis"] = {"mode": out.synthesis_mode, "degraded": out.synthesis_degraded}
        return summary
    return {"candidate_count": 0}
