"""Recommendation package exports.

直接暴露基于 RecommendAgent 的轻量 recommendation_service，兼容既有
chat + streaming 调用。

Phase 1-B 接入点（规划 §19 / §16）：
  stream_core()
    → ComplexityGate.decide(messages)
        ├─ LEGACY → RecommendAgent.run_stream（原样保留）
        └─ TEAM   → AdaptiveTeamOrchestrator.run_stream
                     Planner → Validator → DAGExecutor → Workers
                     → plan.created + task.* 事件 + 确定性 fallback + selection
                     仅当计划失败（plan.failed）才回退 Legacy Fast Path
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from .agent import RecommendAgent
from .orchestration import ComplexityGate, RouteType

# Phase 3.1 Production Integration：Supervisor Runtime feature flag（默认开启）
# 显式关闭：ENABLE_SUPERVISOR_RUNTIME ∈ {0,false,no,off}
ENABLE_SUPERVISOR_RUNTIME = "ENABLE_SUPERVISOR_RUNTIME"


class _RecommendationService:
    def __init__(self):
        self._agent = RecommendAgent()
        self._gate = None
        self._orchestrator = None
        self._memory_service = None
        self._intent_rewrite = None
        self._supervisor_runtime = None

    @staticmethod
    def _supervisor_runtime_enabled() -> bool:
        """TEAM 路径默认走 Supervisor；ENABLE_SUPERVISOR_RUNTIME 显式关闭时回退 Orchestrator。"""
        value = (os.environ.get(ENABLE_SUPERVISOR_RUNTIME) or "").strip().lower()
        if not value:
            return True
        if value in ("0", "false", "no", "off"):
            return False
        return value in ("1", "true", "yes", "on")

    # -- orchestration（惰性初始化；已注入的组件不被覆盖）-------------------
    def _get_orchestrator(self):
        if self._orchestrator is None:
            from .orchestration.orchestrator import AdaptiveTeamOrchestrator

            self._orchestrator = AdaptiveTeamOrchestrator()
        return self._orchestrator

    def _get_supervisor_runtime(self):
        """flag=True 时的生产 TEAM 控制器：SupervisorAgent(enhanced=True)。"""
        if self._supervisor_runtime is None:
            from .orchestration.supervisor import SupervisorAgent

            self._supervisor_runtime = SupervisorAgent(enhanced=True)
        return self._supervisor_runtime

    def _get_memory_service(self):
        if self._memory_service is None:
            from .orchestration.memory import EpisodicMemoryService

            self._memory_service = EpisodicMemoryService()
        return self._memory_service

    def _get_intent_rewrite(self):
        if self._intent_rewrite is None:
            from .orchestration.intent import IntentRewrite

            self._intent_rewrite = IntentRewrite()
        return self._intent_rewrite

    @staticmethod
    def _last_user_text(messages: List[Dict[str, str]]) -> str:
        if not messages:
            return ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return (msg.get("content") or "").strip()
        return ""

    def stream_core(
        self,
        messages: List[Dict[str, str]],
        state: Optional[Dict[str, Any]] = None,
        vector_targets: Optional[List[str]] = None,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> Iterable[str]:
        """Phase 1-B 入口：复杂度路由 + Team 执行 / Legacy。

        conversation_id：episodic memory scope。传入时 TEAM 路径启用
        memory recall（Planner bounded hints）与成功写入；Legacy 路径不变。

        Phase B Intent Rewrite（确定性规则）：User Query → NormalizedIntent →
        ComplexityGate（用 normalized_query 判定）→ Planner（Team goal_override）。
        Intent 解析失败 / greeting 不影响原行为（回退 raw）。
        """
        # 0) Intent Rewrite（轻量确定性规则；失败不影响主链路）
        intent = None
        gate_query = None
        try:
            user_text = self._last_user_text(messages)
            if user_text:
                intent = self._get_intent_rewrite().rewrite(user_text)
                gate_query = intent.normalized_query or None
        except Exception as e:
            print(f"[IntentRewrite] error (fallback raw): {type(e).__name__}: {e}")

        decision = None
        try:
            if self._gate is None:
                self._gate = ComplexityGate()
            decision = self._gate.decide(messages, query=gate_query)
        except Exception as e:
            print(f"[ComplexityGate] route=legacy (gate error: {type(e).__name__}: {e})")

        if decision is not None and decision.route == RouteType.TEAM:
            print("[ComplexityGate] route=team")
            # Phase 3.1：TEAM 默认 SupervisorAgent(enhanced=True)；
            #   ENABLE_SUPERVISOR_RUNTIME=false → AdaptiveTeamOrchestrator
            supervisor_mode = self._supervisor_runtime_enabled()
            controller = (self._get_supervisor_runtime() if supervisor_mode
                          else self._get_orchestrator())
            mem = self._get_memory_service() if conversation_id else None
            # Intent 规范化 goal 仅喂给 Planner（greeting/无图片需求时不覆盖，保持原行为）
            goal_override = None
            if intent is not None and intent.needs_images and intent.normalized_query:
                goal_override = intent.normalized_query
            controller_kwargs: Dict[str, Any] = {
                "goal_override": goal_override,
                "conversation_id": conversation_id,
                "memory_service": mem,
            }
            if supervisor_mode:
                print("[SupervisorRuntime] TEAM path via SupervisorAgent (enhanced=True)")
            for ev in controller.run_stream(messages, **controller_kwargs):
                yield ev
            if not controller.last_plan_ok:
                # 计划失败（planner / validation）→ 回退 Legacy Fast Path
                print("[ComplexityGate] plan failed → legacy fallback")
                for raw in self._agent.run_stream(
                    messages=messages,
                    vector_targets=vector_targets,
                    filters=filters,
                    limit=limit,
                ):
                    yield raw
        else:
            print("[ComplexityGate] route=legacy")
            for raw in self._agent.run_stream(
                messages=messages,
                vector_targets=vector_targets,
                filters=filters,
                limit=limit,
            ):
                yield raw

    def chat_recommend(
        self,
        messages: List[Dict[str, str]],
        state: Optional[Dict[str, Any]] = None,
        vector_targets: Optional[List[str]] = None,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 简单消费一次 stream 拿到最终 selection
        images: List[Dict[str, Any]] = []
        ordered_ids: List[int] = []
        for raw in self.stream_core(
            messages, state, vector_targets, limit, filters,
            conversation_id=conversation_id,
        ):
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if obj.get("event") == "selection":
                images = obj.get("images") or []
                ordered_ids = obj.get("ids") or []
        return {
            "success": True,
            "images": images,
            "image_ids": ordered_ids,
            "limit": limit,
        }


recommendation_service = _RecommendationService()

__all__ = [
    "recommendation_service",
    "_RecommendationService",
    "RecommendAgent",
    "ENABLE_SUPERVISOR_RUNTIME",
]
