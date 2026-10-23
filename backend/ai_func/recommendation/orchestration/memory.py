"""Lightweight Episodic Memory Service（设计 §5–§8）。

边界：Planner → Memory Service → Episode Repository。
- Working Memory 仍为 ExecutionContext（不新建表）。
- 不新增 Memory Agent / Memory Task / embedding；不把写记忆做成 DAG 节点。
- 只写成功 Team，且至少有一次 retrieve_images 或 group_images 成功。
- 结构化条件仍确定性抽取；叙事摘要复用 Synthesizer 已生成的 LLM rationale/answer。
- 只 recall bounded hints（Top-N ≤ 3，总渲染 ≤ 800 字符）。
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ....db_func.repositories.agent_episodes import (
    AgentEpisodeRepository,
    ISO,
    _now_iso,
)

# ---------------------------------------------------------------------------
# 常量（bounded）
# ---------------------------------------------------------------------------
EPISODE_TTL_DAYS = 30                 # 默认 TTL
EPISODE_CAP_PER_CONVERSATION = 20     # per-conversation 上限
RECALL_TOP_N = 3                      # recall 硬上限
RECALL_BUDGET_CHARS = 800             # 渲染后总字符硬上限
EPISODE_RENDER_CHAR_LIMIT = 280       # 单条 episode 渲染上限
DUPLICATE_WINDOW_HOURS = 24           # 去重时间窗
DUPLICATE_SIMILARITY_THRESHOLD = 0.85  # 去重相似度阈值（Dice 2-gram）
QUERY_SUMMARY_LIMIT = 120
STRATEGY_SUMMARY_LIMIT = 200
RESULT_SUMMARY_LIMIT = 200
PROFILE_MAX_FILTERS = 4
PROFILE_MAX_GROUPS = 2
CONSTRAINT_VALUE_LIMIT = 40
_RECORDABLE_TASK_TYPES = frozenset({"retrieve_images", "group_images"})

# 禁止持久化/渲染的模式（安全边界）
_FORBIDDEN_PATTERNS = (
    "sk-", "api_key", "authorization", "bearer ", "base64", "data:image",
    "filepath", "chain_of_thought", "raw_response", "raw prompt",
)
_REDACTED = "[REDACTED]"

_LATIN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _sanitize(text: str) -> str:
    """遮盖禁止字段；任何匹配都整体替换为 [REDACTED]。"""
    if not text:
        return ""
    lowered = text.lower()
    for p in _FORBIDDEN_PATTERNS:
        if p in lowered:
            return _REDACTED
    return text


def _clip_narrative(text: str, limit: int) -> str:
    """裁切 Synthesizer 已生成的摘要；优先在句末断开，不是另起一次 LLM。"""
    text = _sanitize((text or "").replace("\n", " ").strip())
    if not text or text == _REDACTED or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("。", "！", "？", ". ", "! ", "? "):
        idx = cut.rfind(sep)
        if idx >= max(12, limit // 3):
            return cut[: idx + len(sep)].rstrip()
    return cut.rstrip()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in (ISO, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _bigrams(text: str) -> set:
    s = text.lower().replace(" ", "")
    if not s:
        return set()
    grams = {s[i : i + 2] for i in range(len(s) - 1)}
    grams.add(s)
    return grams


def _similarity(a: str, b: str) -> float:
    """Dice coefficient over character bigrams（deterministic，无 embedding）。"""
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    return 2.0 * len(ba & bb) / (len(ba) + len(bb))


def _tokens(text: str) -> set:
    t = set(_LATIN_RE.findall(text.lower()))
    cjk = _CJK_RE.findall(text)
    t.update(a + b for a, b in zip(cjk[:-1], cjk[1:]))
    return t


class EpisodicMemoryService:
    """minimal episodic memory service（无 Memory Agent / 无 Memory Task / 无新 LLM）。"""

    def __init__(self, repo: Optional[AgentEpisodeRepository] = None):
        self._repo = repo if repo is not None else AgentEpisodeRepository()

    # ------------------------------------------------------------------
    # 写路径
    # ------------------------------------------------------------------
    def build_episode_from_context(
        self,
        context: Any,
        *,
        route: str = "team",
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """从 ExecutionContext 提取确定性 episode；非成功完成返回 None。

        成功定义（§8）：Gate→TEAM + Planner/Validator 成功 + DAG 到达 synthesis
        + 最终 synthesis/selection 路径成功完成（synthesis SUCCEEDED 且非空 selection）
        + 至少一次 retrieve_images / group_images 成功。
        """
        if not conversation_id or context is None or context.plan is None:
            return None
        plan = context.plan
        results = getattr(context, "results", {}) or {}
        synth = results.get(plan.synthesis_task_id)
        # worker fatal（任何 task FAILED）→ 不生成成功 episode（§8）
        if any(r.status.value == "failed" for r in results.values()):
            return None
        if not self._has_recordable_success(plan, results):
            return None
        if (
            synth is None
            or synth.status.value != "succeeded"
            or synth.output is None
            or not getattr(synth.output, "image_ids", None)
        ):
            return None

        task_types = [t.task_type.value for t in plan.tasks]
        query_summary = _sanitize((context.goal or "")[:QUERY_SUMMARY_LIMIT])
        strategy = self._build_strategy_summary(plan, results)
        result = self._build_result_summary(plan, results)
        hints = self._build_preference_hints(plan)
        now = now or datetime.utcnow()
        created = now.strftime(ISO)
        return {
            "conversation_id": conversation_id,
            "request_id": request_id,
            "user_id": user_id,
            "route": route,
            "task_types": task_types,
            "query_summary": query_summary,
            "strategy_summary": strategy,
            "result_summary": result,
            "preference_hints": hints,
            "selected_count": len(synth.output.image_ids),
            "success": True,
            "plan_valid": True,
            "created_at": created,
            "expires_at": (now + timedelta(days=EPISODE_TTL_DAYS)).strftime(ISO),
        }

    def should_write_episode(
        self,
        episode: Optional[Dict[str, Any]],
        recent_episodes: Optional[List[Dict[str, Any]]] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """写门控：TEAM only + conversation_id + success，且 24h 内无重复。"""
        if not episode or not episode.get("conversation_id") or not episode.get("success"):
            return False
        if episode.get("route") != "team":
            return False
        now = now or datetime.utcnow()
        window_start = now - timedelta(hours=DUPLICATE_WINDOW_HOURS)
        if recent_episodes is None:
            recent_episodes = self._repo.list_by_conversation(
                episode["conversation_id"],
                limit=EPISODE_CAP_PER_CONVERSATION,
                now=_now_iso(),
            )
        for old in recent_episodes:
            created = _parse_iso(old.get("created_at"))
            if created is None or created < window_start:
                continue
            if list(old.get("task_types") or []) != list(episode.get("task_types") or []):
                continue
            if _similarity(old.get("query_summary", ""), episode.get("query_summary", "")) \
                    >= DUPLICATE_SIMILARITY_THRESHOLD:
                return False
        return True

    def insert_episode(self, episode: Dict[str, Any]) -> int:
        """insert + per-conversation cap prune。"""
        new_id = self._repo.insert(episode)
        try:
            self._repo.prune_conversation(
                episode["conversation_id"], cap=EPISODE_CAP_PER_CONVERSATION
            )
        except Exception as e:  # 裁剪失败不影响写入
            print(f"[Memory] prune failed: {type(e).__name__}: {e}")
        return new_id

    def remember(
        self,
        context: Any,
        *,
        route: str = "team",
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[int]:
        """完整写路径：build → should_write → insert。返回 episode id 或 None。"""
        ep = self.build_episode_from_context(
            context, route=route, conversation_id=conversation_id,
            request_id=request_id, user_id=user_id,
        )
        if ep is None or not self.should_write_episode(ep):
            return None
        return self.insert_episode(ep)


    # ------------------------------------------------------------------
    # Recall 路径
    # ------------------------------------------------------------------
    def recall_episodes(
        self,
        conversation_id: Optional[str],
        query: str,
        top_n: int = RECALL_TOP_N,
        budget: int = RECALL_BUDGET_CHARS,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """conversation-scoped bounded recall：keyword overlap + recency + success。"""
        if not conversation_id or not query:
            return []
        recent = self._repo.list_by_conversation(
            conversation_id, limit=EPISODE_CAP_PER_CONVERSATION, now=_now_iso()
        )
        now = now or datetime.utcnow()
        scored = sorted(
            recent,
            key=lambda ep: self._score_episode(ep, query, now),
            reverse=True,
        )
        selected: List[Dict[str, Any]] = []
        total = 0
        for ep in scored[:top_n]:
            rendered = self._render_episode(ep)
            if total + len(rendered) > budget:
                continue
            selected.append(ep)
            total += len(rendered)
        return selected

    def render_memory_hints(
        self,
        episodes: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
        budget: int = RECALL_BUDGET_CHARS,
    ) -> List[str]:
        """渲染 bounded hints：会话画像（若有）+ 最近相关 episode。

        conversation_id 有值时，画像从该会话全部未过期 episode 聚合，
        不依赖本次 keyword recall 是否命中。
        """
        lines: List[str] = []
        total = 0
        profile_src = (
            self.list_recent(conversation_id) if conversation_id else list(episodes or [])
        )
        profile_line = self._render_profile(self.build_profile(profile_src))
        if profile_line:
            chunk = profile_line[:EPISODE_RENDER_CHAR_LIMIT]
            if total + len(chunk) <= budget:
                lines.append(chunk)
                total += len(chunk)
        for ep in episodes or []:
            text = self._render_episode(ep)
            if total + len(text) > budget:
                continue
            lines.append(text)
            total += len(text)
        return lines

    def list_recent(
        self, conversation_id: str, limit: int = EPISODE_CAP_PER_CONVERSATION
    ) -> List[Dict[str, Any]]:
        return self._repo.list_by_conversation(conversation_id, limit=limit)

    def build_profile(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从成功 episode 确定性聚合会话画像。无 LLM。"""
        filters: Counter = Counter()
        groups: Counter = Counter()
        sorts: Counter = Counter()
        selects: Counter = Counter()
        presentations: Counter = Counter()
        vision = 0
        for ep in episodes or []:
            seen_group = False
            for raw in ep.get("preference_hints") or []:
                h = str(raw)
                if h.startswith("filter:"):
                    filters[h[len("filter:"):]] += 1
                elif h.startswith("group_by:") and not seen_group:
                    groups[h[len("group_by:"):]] += 1
                    seen_group = True
                elif h.startswith("uses_group_by=") and not seen_group:
                    groups[h.split("=", 1)[1]] += 1
                    seen_group = True
                elif h.startswith("sort:"):
                    sorts[h[len("sort:"):]] += 1
                elif h.startswith("select:"):
                    selects[h[len("select:"):]] += 1
                elif h.startswith("presentation:"):
                    presentations[h[len("presentation:"):]] += 1
                elif h in ("uses_vision_verification", "uses_vision"):
                    vision += 1
        return {
            "filters": filters.most_common(PROFILE_MAX_FILTERS),
            "group_bys": groups.most_common(PROFILE_MAX_GROUPS),
            "sorts": sorts.most_common(1),
            "selects": selects.most_common(1),
            "presentations": presentations.most_common(1),
            "vision_count": vision,
            "episode_count": len(episodes or []),
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _has_recordable_success(plan, results) -> bool:
        """只有检索或分组真正成功时才值得写入会话记录。"""
        from .workers import TaskStatus

        for t in plan.tasks:
            if t.task_type.value not in _RECORDABLE_TASK_TYPES:
                continue
            res = results.get(t.task_id)
            if res is not None and res.status == TaskStatus.SUCCEEDED:
                return True
        return False

    @staticmethod
    def _build_strategy_summary(plan, results) -> str:
        from .workers import TaskStatus
        from .vision import VisionTaskOutput

        seq = "→".join(t.task_type.value for t in plan.tasks)
        parts = [seq]
        for t in plan.tasks:
            if t.task_type.value == "analyze_visual":
                res = results.get(t.task_id)
                if res is not None and res.status == TaskStatus.SUCCEEDED \
                        and isinstance(res.output, VisionTaskOutput):
                    parts.append(f"Vision {res.output.success_count}/{res.output.total} 成功")
        return _sanitize("; ".join(parts)[:STRATEGY_SUMMARY_LIMIT])

    @staticmethod
    def _build_result_summary(plan, results) -> str:
        from .workers import GroupOutput, RetrievalOutput, SynthesisOutput, TaskStatus

        stats: List[str] = []
        narrative = ""
        synth = results.get(plan.synthesis_task_id)
        if synth is not None and synth.status == TaskStatus.SUCCEEDED \
                and isinstance(synth.output, SynthesisOutput):
            out = synth.output
            stats.append(f"selected_count={len(out.image_ids)}")
            if out.synthesis_mode == "llm":
                narrative = (out.rationale or "").strip() or (out.answer or "").strip()
            if out.synthesis_degraded:
                stats.append("synthesis_degraded=true")
            if out.synthesis_mode == "deterministic":
                stats.append("mode=deterministic")

        for t in plan.tasks:
            res = results.get(t.task_id)
            if res is None or res.status != TaskStatus.SUCCEEDED:
                continue
            if t.task_type.value == "retrieve_images" and isinstance(res.output, RetrievalOutput):
                stats.append(f"retrieved={res.output.total or len(res.output.candidates)}")
            elif t.task_type.value == "group_images" and isinstance(res.output, GroupOutput):
                groups = res.output.groups or []
                stats.append(f"groups={len(groups)}")
                keys = []
                for g in groups[:4]:
                    key = _sanitize(str(getattr(g, "key", "") or "").replace("\n", " ").strip())
                    if key and key != _REDACTED:
                        keys.append(key[:CONSTRAINT_VALUE_LIMIT])
                if keys:
                    stats.append("group_keys=" + ",".join(keys))

        fails = sorted(
            {r.error.code for r in results.values()
             if r.status == TaskStatus.FAILED and r.error}
        )
        if fails:
            stats.append(f"failures={','.join(fails)}")
        stats_text = _sanitize("; ".join(stats))
        if not narrative:
            return stats_text[:RESULT_SUMMARY_LIMIT]
        if not stats_text:
            return _clip_narrative(narrative, RESULT_SUMMARY_LIMIT)
        room = RESULT_SUMMARY_LIMIT - len(stats_text) - 2
        if room < 24:
            return stats_text[:RESULT_SUMMARY_LIMIT]
        return f"{_clip_narrative(narrative, room)}; {stats_text}"

    @staticmethod
    def _build_preference_hints(plan) -> List[str]:
        """结构化偏好：真实谓词/分组键，而不是只写 uses_metadata_filter。"""
        hints: List[str] = []
        for t in plan.tasks:
            tt = getattr(t.task_type, "value", t.task_type)
            inp = getattr(t, "input", None)
            if tt == "filter_metadata":
                preds = list(getattr(inp, "predicates", None) or [])
                if preds:
                    hints.append("uses_metadata_filter")
                for p in preds[:6]:
                    rec = _format_predicate(p)
                    if rec:
                        hints.append(f"filter:{rec}")
            elif tt == "group_images":
                group_by = getattr(inp, "group_by", None)
                if group_by:
                    hints.append(f"uses_group_by={group_by}")
                    hints.append(f"group_by:{group_by}")
                else:
                    hints.append("uses_grouping")
            elif tt == "sort_images":
                sort_by = getattr(inp, "sort_by", None)
                order = getattr(inp, "order", "desc")
                if sort_by:
                    hints.append(f"sort:{sort_by}:{order}")
            elif tt == "select_images":
                strategy = getattr(inp, "strategy", None)
                limit = getattr(inp, "limit", None)
                if strategy:
                    hints.append(f"select:{strategy}:{limit}" if limit is not None else f"select:{strategy}")
            elif tt in ("analyze_visual", "verify_visual_condition"):
                hints.append("uses_vision_verification")
                cond = getattr(inp, "condition", None)
                if cond:
                    safe = _sanitize(str(cond)[:CONSTRAINT_VALUE_LIMIT])
                    if safe and safe != _REDACTED:
                        hints.append(f"verify:{safe}")
            elif tt == "synthesize":
                if getattr(inp, "presentation", "flat") == "grouped":
                    hints.append("prefers_grouped")
                    hints.append("presentation:grouped")
        return hints[:12]

    @staticmethod
    def _score_episode(ep, query, now) -> float:
        score = 0.0
        created = _parse_iso(ep.get("created_at"))
        if created:
            age_days = (now - created).total_seconds() / 86400.0
            if age_days <= 7:
                score += 3.0
            elif age_days <= 30:
                score += 1.0
        hints = " ".join(str(h) for h in (ep.get("preference_hints") or []))
        doc = f"{ep.get('query_summary', '')} {ep.get('strategy_summary', '')} {hints}"
        score += 2.0 * _keyword_overlap(query, doc)
        if ep.get("success"):
            score += 1.0
        if ep.get("route") == "team":
            score += 0.5
        return score

    @staticmethod
    def _render_episode(ep, char_limit: int = EPISODE_RENDER_CHAR_LIMIT) -> str:
        created = (ep.get("created_at") or "")[:10]
        constraints = _render_constraint_brief(ep.get("preference_hints") or [])
        text = (
            f"[{created}] 查询:{ep.get('query_summary', '')} "
            f"{constraints}"
            f"策略:{ep.get('strategy_summary', '')} 结果:{ep.get('result_summary', '')}"
        )
        return text[:char_limit]

    @staticmethod
    def _render_profile(profile: Dict[str, Any]) -> str:
        if not profile or not profile.get("episode_count"):
            return ""
        parts: List[str] = []
        filters = profile.get("filters") or []
        if filters:
            parts.append("筛选 " + "；".join(f"{k}×{n}" for k, n in filters))
        groups = profile.get("group_bys") or []
        if groups:
            parts.append("分组 " + "；".join(f"{k}×{n}" for k, n in groups))
        sorts = profile.get("sorts") or []
        if sorts:
            parts.append("排序 " + "；".join(f"{k}×{n}" for k, n in sorts))
        selects = profile.get("selects") or []
        if selects:
            parts.append("选择 " + "；".join(f"{k}×{n}" for k, n in selects))
        presentations = profile.get("presentations") or []
        if presentations:
            parts.append("呈现 " + "；".join(f"{k}×{n}" for k, n in presentations))
        vision = int(profile.get("vision_count") or 0)
        if vision:
            parts.append(f"视觉核验×{vision}")
        if not parts:
            return ""
        return (
            f"会话画像(n={profile['episode_count']}): "
            + "；".join(parts)
            + "。当前请求未改口时可延续这些条件，冲突以当前目标为准。"
        )


def _keyword_overlap(query: str, doc: str) -> float:
    tq, td = _tokens(query), _tokens(doc)
    if not tq:
        return 0.0
    return len(tq & td) / len(tq)


def _format_predicate(pred: Any) -> str:
    field = getattr(pred, "field", None)
    operator = getattr(pred, "operator", None)
    value = getattr(pred, "value", None)
    if not field or operator is None:
        return ""
    if isinstance(value, list):
        raw = ",".join(str(x) for x in value[:4])
    else:
        raw = "" if value is None else str(value)
    safe = _sanitize(raw.strip().replace("\n", " ")[:CONSTRAINT_VALUE_LIMIT])
    if not safe or safe == _REDACTED:
        return ""
    return f"{field}={operator}:{safe}"


def _render_constraint_brief(hints: List[Any]) -> str:
    filters = []
    group = ""
    for h in hints:
        s = str(h)
        if s.startswith("filter:") and len(filters) < 3:
            filters.append(s[len("filter:"):])
        elif s.startswith("group_by:") and not group:
            group = s[len("group_by:"):]
        elif s.startswith("uses_group_by=") and not group:
            group = s.split("=", 1)[1]
    bits = []
    if filters:
        bits.append("筛选:" + ",".join(filters))
    if group:
        bits.append(f"分组:{group}")
    return ((" ".join(bits) + " ") if bits else "")
