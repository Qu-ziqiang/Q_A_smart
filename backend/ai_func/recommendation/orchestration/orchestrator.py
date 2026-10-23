"""Phase 1-B/C — AdaptiveTeamOrchestrator（规划 §16 / §19 / §21）。

Phase 1 Specialist 集成（Requirement 5）：
默认 Executor 的 workers 表不再“TaskType → Worker 直连”，而是
    Task → AgentRegistry.get_agent_for_task(task_type)
        → SpecialistAgent.execute(task, context)
        → Worker.run(task, context)
DAGExecutor / Task 依赖 / Validator / Planner 行为完全不变（纯分发包变更）。

run_stream(messages) 产出事件流：
  plan.created / plan.failed
  → task.started / task.completed / task.failed / task.skipped（由 Executor 产出）
  → assistant_delta（LLM Synthesizer 回答；失败时为确定性 fallback 文本）
  → selection（结构化候选）

SYNTHESIZE 任务由 LLM SynthesizerAgent 执行；其内部失败自动回退
DeterministicSynthesisWorker（synthesis_degraded=true）。
计划失败时 last_plan_ok=False，由调用层（_RecommendationService）回退 Legacy。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .agent_registry import AgentRegistry, build_default_agent_registry, build_dispatched_workers
from .executor import DAGExecutor
from .models import PlanValidator, PlannerFailure, TaskPlan
from .planner import PlannerAgent
from .workers import (
    ExecutionContext,
    GroupOutput,
    ImageCandidate,
    MetadataOutput,
    RetrievalOutput,
    SynthesisOutput,
    TaskStatus,
    default_workers,
    status_summary,
)


class AdaptiveTeamOrchestrator:
    def __init__(
        self,
        planner: Optional[PlannerAgent] = None,
        validator: Optional[PlanValidator] = None,
        executor: Optional[DAGExecutor] = None,
        memory_service: Optional[Any] = None,
        agent_registry: Optional[AgentRegistry] = None,
    ):
        self.planner = planner if planner is not None else PlannerAgent()
        self.validator = validator if validator is not None else PlanValidator()
        # Phase 1 Specialist：注册表缺省为默认三个 Specialist（Retrieval/Organizer/Vision）
        self.agent_registry = agent_registry if agent_registry is not None else self._default_agent_registry()
        self.executor = executor if executor is not None else self._default_executor()
        self.memory_service = memory_service
        self.last_plan_ok: bool = False
        self.last_plan: Optional[TaskPlan] = None

    @staticmethod
    def _default_agent_registry() -> AgentRegistry:
        return build_default_agent_registry()

    def _default_executor(self) -> DAGExecutor:
        """默认 Executor：其余任务经 AgentRegistry 分发给 Specialist Agent，
        SYNTHESIZE 用 LLM SynthesizerAgent（与既有行为一致）。"""
        from .synthesizer import SynthesizerAgent

        workers = build_dispatched_workers(
            default_workers(),
            self.agent_registry,
            synthesize_worker=SynthesizerAgent(),
        )
        return DAGExecutor(workers=workers)

    # -- 公共入口 ------------------------------------------------------------
    def run_stream(
        self,
        messages: List[Dict[str, str]],
        conversation_id: Optional[str] = None,
        memory_service: Optional[Any] = None,
        goal_override: Optional[str] = None,
    ) -> Iterable[str]:
        """产出 JSON 事件字符串（与 Legacy run_stream 相同的 Iterable[str] 契约）。

        conversation_id：episodic memory scope（None 时完全不启用 memory）。
        memory_service：注入 EpisodicMemoryService；缺省回退构造时传入的实例。
        goal_override：Intent Rewrite 提供的规范化 goal；None 时退回最后一条 user query。
        """
        goal = goal_override if (goal_override and str(goal_override).strip()) else self._last_user_query(messages)
        self.last_plan_ok = False
        self.last_plan = None
        svc = memory_service if memory_service is not None else self.memory_service

        # 0) Memory recall（仅 TEAM 路径；bounded hints，失败不影响规划）
        memory_hints = None
        if svc is not None and conversation_id:
            try:
                episodes = svc.recall_episodes(conversation_id, goal)
                memory_hints = svc.render_memory_hints(
                    episodes, conversation_id=conversation_id
                ) or None
            except Exception as e:
                print(f"[Memory] recall failed: {type(e).__name__}: {e}")

        # 1) Planner + 空计划硬底线（重试一次后 Legacy 单步检索）
        from .plan_floor import ensure_executable_plan, has_executable_tasks

        def _create(repair):
            return self.planner.create_plan(
                goal=goal,
                recent_messages=messages,
                memory_hints=memory_hints,
                repair_errors=repair,
            )

        vres = ensure_executable_plan(
            create_draft=_create,
            validate=self.validator.validate,
            goal=goal,
        )
        if not vres.valid or vres.plan is None or not has_executable_tasks(vres.plan):
            print(f"[PlanValidator] validation failed: {vres.errors}")
            yield self._plan_failed("validation_failed", (vres.errors or [])[:5])
            return

        plan = vres.plan
        self.last_plan_ok = True
        self.last_plan = plan
        print(f"[Planner] plan generated (plan_id={plan.plan_id})")
        print("[PlanValidator] validation passed")
        yield self._plan_created(plan)

        # 3) Bounded DAG Executor（串行；Worker 异常已隔离为 FAILED）
        for ev in self.executor.execute_stream(plan):
            yield json.dumps(ev, ensure_ascii=False) + "\n\n"

        # 4) 最终结果：LLM Synthesizer 回答（若成功）或确定性 fallback 文本
        text, ids, images = self._build_final(plan, self.executor.context)
        # Memory write：仅成功完成的 Team run（synthesis/selection 路径完成）才写 episode
        if svc is not None and conversation_id:
            try:
                svc.remember(
                    self.executor.context, route="team", conversation_id=conversation_id
                )
            except Exception as e:
                print(f"[Memory] write failed: {type(e).__name__}: {e}")
        yield json.dumps({"event": "assistant_delta", "delta": text}, ensure_ascii=False) + "\n\n"
        yield json.dumps({"event": "selection", "ids": ids, "images": images}, ensure_ascii=False) + "\n\n"

    # -- 内部 ----------------------------------------------------------------
    @staticmethod
    def _last_user_query(messages: Optional[List[Dict[str, str]]]) -> str:
        if not messages:
            return ""
        for m in reversed(messages):
            if m.get("role") == "user":
                return (m.get("content") or "").strip()
        return ""

    @staticmethod
    def _plan_created(plan: TaskPlan) -> str:
        tasks = [
            {
                "task_id": t.task_id,
                "task_type": t.task_type.value,
                "objective": t.objective,
                "depends_on": list(t.depends_on),
            }
            for t in plan.tasks
        ]
        return json.dumps(
            {
                "event": "plan.created",
                "plan_id": plan.plan_id,
                "plan_version": plan.plan_version,
                "goal": plan.goal,
                "synthesis_task_id": plan.synthesis_task_id,
                "tasks": tasks,
            },
            ensure_ascii=False,
        ) + "\n\n"

    @staticmethod
    def _plan_failed(reason: str, errors: List[str]) -> str:
        return json.dumps(
            {"event": "plan.failed", "reason": reason, "errors": errors},
            ensure_ascii=False,
        ) + "\n\n"

    def _build_final(
        self, plan: TaskPlan, context: Optional[ExecutionContext]
    ) -> tuple[str, List[int], List[Dict[str, Any]]]:
        """最终输出：优先使用 LLM Synthesizer 的 answer；否则用确定性 fallback。"""
        results = context.results if context is not None else {}
        synth = results.get(plan.synthesis_task_id)
        if (
            synth is not None
            and synth.status == TaskStatus.SUCCEEDED
            and isinstance(synth.output, SynthesisOutput)
            and synth.output.answer
        ):
            return (
                synth.output.answer,
                synth.output.image_ids,
                self._public_images(synth.output.candidates),
            )
        return self._build_fallback(plan, context)

    @staticmethod
    def _public_images(candidates: List[ImageCandidate]) -> List[Dict[str, Any]]:
        return [
            {
                "id": c.id,
                "score": c.score,
                "title": c.title,
                "tags": list(c.tags),
                "public_url": c.public_url,
            }
            for c in candidates
        ]

    def _build_fallback(
        self, plan: TaskPlan, context: Optional[ExecutionContext]
    ) -> tuple[str, List[int], List[Dict[str, Any]]]:
        """确定性 fallback：结构化执行摘要 + 最终候选（不含 LLM 生成）。"""
        results = context.results if context is not None else {}
        synth = results.get(plan.synthesis_task_id)

        candidates: List[ImageCandidate] = []
        ids: List[int] = []
        group_summary: Dict[str, int] = {}
        if synth is not None and synth.status == TaskStatus.SUCCEEDED and isinstance(synth.output, SynthesisOutput):
            candidates = synth.output.candidates
            ids = synth.output.image_ids
            group_summary = synth.output.group_summary

        lines = ["已按任务计划完成图片检索与整理（Team 执行）。"]
        for t in plan.tasks:
            res = results.get(t.task_id)
            if res is None:
                continue
            if res.status == TaskStatus.SUCCEEDED:
                lines.append(f"- 任务 {t.task_id}（{t.task_type.value}）：{status_summary(res)}")
            elif res.status == TaskStatus.FAILED:
                code = res.error.code if res.error else "UNKNOWN"
                lines.append(f"- 任务 {t.task_id}（{t.task_type.value}）：失败（{code}）")
            elif res.status == TaskStatus.SKIPPED:
                lines.append(f"- 任务 {t.task_id}（{t.task_type.value}）：已跳过")
        if group_summary:
            summary = "，".join(f"{k}: {n}张" for k, n in group_summary.items())
            lines.append(f"分组情况：{summary}")
        if ids:
            lines.append(f"共推荐 {len(ids)} 张图片。")
        else:
            lines.append("未找到满足条件的图片。")

        return "\n".join(lines), ids, self._public_images(candidates)
