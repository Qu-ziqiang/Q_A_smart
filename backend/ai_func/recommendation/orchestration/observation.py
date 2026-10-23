"""Supervisor 结构化 Observation：Specialist 证据 → 触发分类 → Planner 回灌。

不调用 LLM。阈值 LOW_RELEVANCE_THRESHOLD 为 exploratory，需真库标定。
"""

from __future__ import annotations

import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .models import VISION_GATED_TYPES, TaskPlan, TaskStatus, TaskType, vision_available

# exploratory：CLIP score = 1 - cosine distance。未做千级分布标定前禁止当生产结论。
EXPLORATORY_CLIP_TOP1_THRESHOLD = 0.4
TAG_HIT_FLOOR = 0.15
TOPK_FOR_MEAN = 5
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class TriggerType(str, Enum):
    NONE = "NONE"
    EMPTY_RETRIEVAL = "EMPTY_RETRIEVAL"
    LOW_RELEVANCE = "LOW_RELEVANCE"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    SYNTHESIS_FAILURE = "SYNTHESIS_FAILURE"
    INSUFFICIENT_CANDIDATES = "INSUFFICIENT_CANDIDATES"


class ReplanAction(str, Enum):
    NONE = "NONE"
    RETRY_REWRITE = "RETRY_REWRITE"
    ADD_VISION = "ADD_VISION"
    RETRY_SAME = "RETRY_SAME"
    REPLAN_STRUCT = "REPLAN_STRUCT"
    REFILL = "REFILL"
    ABORT = "ABORT"


def low_relevance_threshold() -> float:
    raw = (os.environ.get("LOW_RELEVANCE_THRESHOLD") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return EXPLORATORY_CLIP_TOP1_THRESHOLD


def _tokens(text: str) -> set:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if t.strip()}


def _keyword_overlap(query: str, doc: str) -> float:
    tq, td = _tokens(query), _tokens(doc)
    if not tq:
        return 0.0
    return len(tq & td) / len(tq)


class RetrievalSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "absent"
    empty: bool = False
    candidate_count: int = 0
    top1_score: Optional[float] = None
    topk_mean_score: Optional[float] = None
    tag_overlap: float = 0.0
    tag_hit: bool = False
    query: Optional[str] = None
    limit: int = 0
    candidate_ids: List[int] = Field(default_factory=list)


class OrganizerSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "absent"
    input_count: int = 0
    output_count: int = 0
    filter_empty: bool = False
    group_status: str = "absent"


class VisionSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "absent"
    inspected_count: int = 0
    passed_count: int = 0
    confidence_summary: Optional[float] = None
    new_tags: List[str] = Field(default_factory=list)


class FailureSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = ""
    code: str = ""
    message: str = ""


class RoundObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = ""
    iteration: int = 1
    original_goal: str = ""
    trigger: TriggerType = TriggerType.NONE
    suggested_action: ReplanAction = ReplanAction.NONE
    retrieval: RetrievalSlice = Field(default_factory=RetrievalSlice)
    organizer: OrganizerSlice = Field(default_factory=OrganizerSlice)
    vision: VisionSlice = Field(default_factory=VisionSlice)
    failure: FailureSlice = Field(default_factory=FailureSlice)
    failed_task_ids: List[str] = Field(default_factory=list)
    empty_retrieve_task_ids: List[str] = Field(default_factory=list)
    vision_available: bool = False
    plan_had_vision: bool = False
    synthesis_has_selection: bool = False
    requested_count: int = 0
    matched_count: int = 0
    pool_narrowed: bool = False
    plan_task_types: List[str] = Field(default_factory=list)
    current_plan_summary: str = ""
    top_titles: List[str] = Field(default_factory=list)

    def render_for_planner(self) -> str:
        lines = [
            f"iteration={self.iteration}",
            f"trigger={self.trigger.value}",
            f"suggested_action={self.suggested_action.value}",
            f"original_goal={self.original_goal[:120]}",
            f"retrieve_query={self.retrieval.query or ''}",
            f"retrieval_empty={self.retrieval.empty} count={self.retrieval.candidate_count}",
            f"top1_score={self.retrieval.top1_score} topk_mean={self.retrieval.topk_mean_score}",
            f"tag_overlap={self.retrieval.tag_overlap:.3f} tag_hit={self.retrieval.tag_hit}",
            f"organizer={self.organizer.status} filter_empty={self.organizer.filter_empty}",
            f"requested={self.requested_count} matched={self.matched_count} narrowed={self.pool_narrowed}",
            f"vision_available={self.vision_available} plan_had_vision={self.plan_had_vision}",
            f"synthesis_has_selection={self.synthesis_has_selection}",
            f"current_plan_summary={self.current_plan_summary}",
        ]
        if self.failure.type:
            lines.append(
                f"failure={self.failure.type} code={self.failure.code} msg={self.failure.message[:80]}"
            )
        if self.top_titles:
            lines.append("top_titles=" + " | ".join(self.top_titles))
        if self.trigger in (TriggerType.EMPTY_RETRIEVAL, TriggerType.LOW_RELEVANCE):
            lines.append(
                "要求：监督器会插入 expand_query 再检索；不要只增加 limit。"
            )
        if self.suggested_action == ReplanAction.ADD_VISION:
            lines.append("要求：第二轮计划必须包含 analyze_visual。")
        if self.suggested_action == ReplanAction.REFILL:
            lines.append(
                "要求：匹配数少于用户要求。下一轮 retrieve.limit 必须大于上一轮，或改写 query；"
                "筛选条件必须保留，禁止拿掉约束后用无关图凑数。"
            )
        text = "上一轮执行观察（结构化，据此改策略；当前用户目标优先）：\n" + "\n".join(lines)
        return text[:900]


def _status_value(result: Any) -> str:
    st = getattr(result, "status", None)
    if st is None:
        return "absent"
    return getattr(st, "value", str(st))


def _candidates_from_result(result: Any) -> List[Any]:
    out = getattr(result, "output", None)
    if out is None:
        return []
    cands = getattr(out, "candidates", None)
    return list(cands) if isinstance(cands, list) else []


def retrieve_queries(plan: Optional[TaskPlan]) -> List[str]:
    if plan is None:
        return []
    out: List[str] = []
    for t in plan.tasks:
        if t.task_type in (
            TaskType.RETRIEVE_IMAGES,
            TaskType.EXPAND_QUERY,
            TaskType.REWRITE_QUERY,
        ):
            q = getattr(t.input, "query", None)
            if q:
                out.append(str(q).strip())
    return out


def plan_task_type_values(plan: Optional[TaskPlan]) -> List[str]:
    if plan is None:
        return []
    return [t.task_type.value for t in plan.tasks]


def rewrite_query_changed(previous_queries: Sequence[str], new_plan: TaskPlan) -> bool:
    old = {q.strip().lower() for q in previous_queries if q and q.strip()}
    new = {q.strip().lower() for q in retrieve_queries(new_plan)}
    if not old:
        return bool(new)
    return new != old


def plan_has_expand(plan: Optional[TaskPlan]) -> bool:
    if plan is None:
        return False
    return any(t.task_type == TaskType.EXPAND_QUERY for t in plan.tasks)


def plan_has_vision(plan: Optional[TaskPlan]) -> bool:
    if plan is None:
        return False
    return any(t.task_type in VISION_GATED_TYPES for t in plan.tasks)


def assert_replan_changed(observation: RoundObservation, new_plan: TaskPlan) -> Optional[str]:
    """第二轮计划未满足建议动作时返回错误码，否则 None。"""
    action = observation.suggested_action
    if action == ReplanAction.RETRY_REWRITE:
        changed = rewrite_query_changed(retrieve_queries_from_obs(observation), new_plan)
        if not (changed or plan_has_expand(new_plan)):
            return "REPLAN_QUERY_UNCHANGED"
    if action == ReplanAction.ADD_VISION and not plan_has_vision(new_plan):
        return "REPLAN_VISION_MISSING"
    if action == ReplanAction.REFILL and not refill_changed(observation, new_plan):
        return "REFILL_UNCHANGED"
    return None


def refill_changed(observation: RoundObservation, new_plan: TaskPlan) -> bool:
    """补量：提高检索条数，或改写 query。两者都没变则视为没补。"""
    limits: List[int] = []
    for t in new_plan.tasks:
        if t.task_type == TaskType.RETRIEVE_IMAGES:
            limits.append(int(getattr(t.input, "limit", 0) or 0))
    limit_up = bool(limits) and max(limits) > int(observation.retrieval.limit or 0)
    query_changed = rewrite_query_changed(retrieve_queries_from_obs(observation), new_plan)
    return limit_up or query_changed


def retrieve_queries_from_obs(observation: RoundObservation) -> List[str]:
    q = observation.retrieval.query
    return [q] if q else []


def build_observation(
    plan: TaskPlan,
    results: Dict[str, Any],
    *,
    iteration: int,
    request_id: str = "",
    goal: str = "",
    min_candidates: int = 1,
) -> RoundObservation:
    failed_ids: List[str] = []
    empty_ids: List[str] = []
    error_code = ""
    error_msg = ""
    all_cands: List[Any] = []
    retrieve_query = None
    retrieve_status = "absent"
    retrieve_limit = 0

    for t in plan.tasks:
        r = results.get(t.task_id)
        if r is None:
            continue
        if getattr(r, "status", None) == TaskStatus.FAILED:
            failed_ids.append(t.task_id)
            err = getattr(r, "error", None)
            if err is not None:
                error_code = getattr(err, "code", "") or error_code
                error_msg = getattr(err, "message", "") or error_msg
        if t.task_type == TaskType.RETRIEVE_IMAGES:
            retrieve_status = _status_value(r)
            retrieve_query = getattr(getattr(t, "input", None), "query", None) or retrieve_query
            retrieve_limit = max(
                retrieve_limit,
                int(getattr(getattr(t, "input", None), "limit", 0) or 0),
            )
            if getattr(r, "status", None) == TaskStatus.SUCCEEDED:
                cands = _candidates_from_result(r)
                all_cands.extend(cands)
                if not cands:
                    empty_ids.append(t.task_id)
            out = getattr(r, "output", None)
            if retrieve_query is None and out is not None:
                retrieve_query = getattr(out, "query", None)

    scores = []
    ids: List[int] = []
    titles: List[str] = []
    tag_docs: List[str] = []
    for c in all_cands:
        sid = getattr(c, "id", None)
        if sid is not None:
            ids.append(int(sid))
        sc = getattr(c, "score", None)
        if sc is not None:
            try:
                scores.append(float(sc))
            except (TypeError, ValueError):
                pass
        title = (getattr(c, "title", None) or "")[:40]
        if title:
            titles.append(title)
        tags = getattr(c, "tags", None) or []
        tag_docs.append(" ".join(str(x) for x in tags) + " " + title)

    scores_sorted = sorted(scores, reverse=True)
    top1 = scores_sorted[0] if scores_sorted else None
    k = min(TOPK_FOR_MEAN, len(scores_sorted))
    topk_mean = (sum(scores_sorted[:k]) / k) if k else None
    qtext = str(retrieve_query or goal or "")
    overlap = 0.0
    if qtext and tag_docs:
        overlap = max(_keyword_overlap(qtext, d) for d in tag_docs)
    elif qtext and not tag_docs:
        overlap = 0.0
    tag_hit = overlap >= TAG_HIT_FLOOR

    org = OrganizerSlice()
    for t in plan.tasks:
        r = results.get(t.task_id)
        if r is None:
            continue
        if t.task_type == TaskType.FILTER_METADATA:
            cands = _candidates_from_result(r)
            pred = getattr(getattr(r, "output", None), "rejected_ids", None) or []
            org = OrganizerSlice(
                status=_status_value(r),
                input_count=len(cands) + (len(pred) if isinstance(pred, list) else 0),
                output_count=len(cands),
                filter_empty=getattr(r, "status", None) == TaskStatus.SUCCEEDED and len(cands) == 0,
                group_status=org.group_status,
            )
        if t.task_type == TaskType.GROUP_IMAGES:
            out = getattr(r, "output", None)
            groups = getattr(out, "groups", None) if out is not None else None
            org = org.model_copy(update={
                "group_status": _status_value(r),
                "output_count": org.output_count or (
                    sum(len(getattr(g, "image_ids", []) or []) for g in groups) if groups else 0
                ),
            })

    vis = VisionSlice()
    for t in plan.tasks:
        if t.task_type not in VISION_GATED_TYPES:
            continue
        r = results.get(t.task_id)
        if r is None:
            continue
        out = getattr(r, "output", None)
        if t.task_type != TaskType.ANALYZE_VISUAL:
            items = getattr(out, "items", None) if out is not None else None
            inspected = getattr(out, "inspected_count", None)
            if inspected is None:
                inspected = len(items) if isinstance(items, list) else 0
            passed = getattr(out, "passed_count", None)
            if passed is None:
                passed = 0
                if isinstance(items, list):
                    passed = sum(1 for it in items if getattr(it, "passed", False))
            confs: List[float] = []
            tags: List[str] = []
            if isinstance(items, list):
                for it in items:
                    cf = getattr(it, "confidence", None)
                    if cf is not None:
                        try:
                            confs.append(float(cf))
                        except (TypeError, ValueError):
                            pass
                    for tg in getattr(it, "tags", None) or []:
                        tags.append(str(tg))
            vis = VisionSlice(
                status=_status_value(r),
                inspected_count=int(inspected),
                passed_count=int(passed),
                confidence_summary=(sum(confs) / len(confs)) if confs else None,
                new_tags=tags[:8],
            )
            continue
        items = getattr(out, "items", None) if out is not None else None
        inspected = len(items) if isinstance(items, list) else 0
        passed = 0
        confs: List[float] = []
        tags: List[str] = []
        if isinstance(items, list):
            for it in items:
                st = getattr(it, "status", None)
                st_val = getattr(st, "value", st)
                if st_val in ("success", None) and not getattr(it, "error_code", None):
                    passed += 1
                cf = getattr(it, "confidence", None)
                if cf is not None:
                    try:
                        confs.append(float(cf))
                    except (TypeError, ValueError):
                        pass
                for tg in getattr(it, "tags", None) or []:
                    tags.append(str(tg))
        vis = VisionSlice(
            status=_status_value(r),
            inspected_count=inspected,
            passed_count=passed,
            confidence_summary=(sum(confs) / len(confs)) if confs else None,
            new_tags=tags[:8],
        )

    synth = results.get(plan.synthesis_task_id)
    synth_ids = []
    if synth is not None:
        out = getattr(synth, "output", None)
        synth_ids = list(getattr(out, "image_ids", None) or []) if out is not None else []

    retrieval = RetrievalSlice(
        status=retrieve_status,
        empty=bool(empty_ids) or (retrieve_status == "succeeded" and len(all_cands) == 0),
        candidate_count=len(all_cands),
        top1_score=top1,
        topk_mean_score=topk_mean,
        tag_overlap=overlap,
        tag_hit=tag_hit,
        query=str(retrieve_query) if retrieve_query else None,
        limit=retrieve_limit,
        candidate_ids=ids[:20],
    )

    requested = 0
    for t in plan.tasks:
        if t.task_type == TaskType.SYNTHESIZE:
            requested = int(getattr(t.input, "recommendation_limit", 0) or 0)

    narrowed = (
        org.status == "succeeded" and org.input_count > org.output_count
    ) or (
        vis.status == "succeeded" and vis.inspected_count > vis.passed_count
    )
    if synth_ids:
        matched = len(synth_ids)
    elif org.status == "succeeded":
        matched = org.output_count
    elif vis.status == "succeeded":
        matched = vis.passed_count
    else:
        matched = len(all_cands)

    obs = RoundObservation(
        request_id=request_id,
        iteration=iteration,
        original_goal=goal,
        retrieval=retrieval,
        organizer=org,
        vision=vis,
        failed_task_ids=failed_ids,
        empty_retrieve_task_ids=empty_ids,
        vision_available=bool(vision_available()),
        plan_had_vision=plan_has_vision(plan),
        synthesis_has_selection=bool(synth_ids),
        requested_count=requested,
        matched_count=matched,
        pool_narrowed=narrowed,
        plan_task_types=plan_task_type_values(plan),
        current_plan_summary=" → ".join(plan_task_type_values(plan)),
        top_titles=titles[:3],
    )
    trigger, action, fail = classify_trigger(
        obs, failed_ids=failed_ids, error_code=error_code, error_msg=error_msg,
        min_candidates=min_candidates, synth_ids=synth_ids,
    )
    obs.trigger = trigger
    obs.suggested_action = action
    obs.failure = fail
    return obs


def classify_trigger(
    obs: RoundObservation,
    *,
    failed_ids: List[str],
    error_code: str,
    error_msg: str,
    min_candidates: int,
    synth_ids: List[Any],
) -> tuple:
    if failed_ids:
        return (
            TriggerType.EXECUTION_FAILURE,
            ReplanAction.RETRY_SAME,
            FailureSlice(type="EXECUTION_FAILURE", code=error_code, message=error_msg[:160]),
        )
    if obs.retrieval.empty or (
        obs.retrieval.status == "succeeded" and obs.retrieval.candidate_count == 0
    ):
        return (
            TriggerType.EMPTY_RETRIEVAL,
            ReplanAction.RETRY_REWRITE,
            FailureSlice(type="EMPTY_RETRIEVAL", code="EMPTY", message="retrieve returned 0 candidates"),
        )
    threshold = low_relevance_threshold()
    top1 = obs.retrieval.top1_score
    if (
        obs.retrieval.candidate_count > 0
        and top1 is not None
        and top1 < threshold
        and not obs.retrieval.tag_hit
    ):
        action = ReplanAction.RETRY_REWRITE
        if obs.vision_available and not obs.plan_had_vision:
            action = ReplanAction.ADD_VISION
        return (
            TriggerType.LOW_RELEVANCE,
            action,
            FailureSlice(
                type="LOW_RELEVANCE",
                code="LOW_SCORE_AND_TAG_MISS",
                message=f"top1={top1} < {threshold} and tag_hit=False (exploratory)",
            ),
        )
    if (
        obs.pool_narrowed
        and obs.requested_count > 0
        and obs.matched_count < obs.requested_count
    ):
        return (
            TriggerType.INSUFFICIENT_CANDIDATES,
            ReplanAction.REFILL,
            FailureSlice(
                type="INSUFFICIENT_CANDIDATES",
                code="BELOW_REQUESTED",
                message=f"matched={obs.matched_count} < requested={obs.requested_count}",
            ),
        )
    if not synth_ids or obs.retrieval.candidate_count < min_candidates:
        return (
            TriggerType.SYNTHESIS_FAILURE,
            ReplanAction.REPLAN_STRUCT,
            FailureSlice(type="SYNTHESIS_FAILURE", code="NO_SELECTION", message="synthesis has no image_ids"),
        )
    return (TriggerType.NONE, ReplanAction.NONE, FailureSlice())
