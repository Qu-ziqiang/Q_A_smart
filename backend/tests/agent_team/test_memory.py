"""Lightweight Agent Memory — 单元测试（test_memory.py）。

覆盖：schema / write policy / recall / TTL / duplicate / security。

运行（repo 根）：
    venv\\Scripts\\python.exe -X utf8 -m pytest backend/tests/agent_team/test_memory.py -q
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.memory import (
    EPISODE_TTL_DAYS,
    RECALL_BUDGET_CHARS,
    RECALL_TOP_N,
    EpisodicMemoryService,
)
from backend.ai_func.recommendation.orchestration.models import TaskType
from backend.ai_func.recommendation.orchestration.vision import VisionTaskOutput
from backend.ai_func.recommendation.orchestration.workers import (
    ExecutionContext,
    GroupOutput,
    ImageGroup,
    RetrievalOutput,
    SynthesisOutput,
    TaskError,
    TaskStatus,
    WorkerResult,
)
from backend.db_func.repositories.agent_episodes import (
    AGENT_EPISODES_INDEXES_SQL,
    AGENT_EPISODES_TABLE_SQL,
    AgentEpisodeRepository,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(AGENT_EPISODES_TABLE_SQL)
    for s in AGENT_EPISODES_INDEXES_SQL:
        conn.execute(s)
    conn.commit()
    return conn


def _svc(conn=None):
    return EpisodicMemoryService(repo=AgentEpisodeRepository(conn=conn or _make_db()))


def _plan(goal="找旅行照片按地点整理", with_vision=False):
    tasks = [
        SimpleNamespace(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES,
                        input=SimpleNamespace()),
        SimpleNamespace(task_id="t2", task_type=TaskType.GROUP_IMAGES,
                        input=SimpleNamespace(group_by="location")),
    ]
    if with_vision:
        tasks.append(SimpleNamespace(task_id="t3", task_type=TaskType.ANALYZE_VISUAL,
                                     input=SimpleNamespace(criteria=["是否有生日蛋糕"])))
        synth_id = "t4"
    else:
        synth_id = "t3"
    tasks.append(SimpleNamespace(task_id=synth_id, task_type=TaskType.SYNTHESIZE,
                                 input=SimpleNamespace(presentation="flat")))
    return SimpleNamespace(goal=goal, tasks=tasks, synthesis_task_id=synth_id)


def _ctx(goal="找旅行照片按地点整理", synth_ids=(1, 2), synth_ok=True, worker_fatal=False,
         with_vision=False, vision=(2, 4), synth_mode="deterministic", degraded=False,
         retrieve_ok=True, group_ok=True, answer="", rationale="", retrieved=6):
    plan = _plan(goal=goal, with_vision=with_vision)
    ctx = ExecutionContext(plan)
    ctx.results["t1"] = WorkerResult(
        task_id="t1", task_type=TaskType.RETRIEVE_IMAGES,
        status=TaskStatus.FAILED if worker_fatal or not retrieve_ok else TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=[], query=goal, total=retrieved),
        error=None if not (worker_fatal or not retrieve_ok) else TaskError(
            code="RETRIEVAL_FATAL", message="fatal"
        ),
    )
    ctx.results["t2"] = WorkerResult(
        task_id="t2", task_type=TaskType.GROUP_IMAGES,
        status=TaskStatus.SUCCEEDED if group_ok else TaskStatus.FAILED,
        output=GroupOutput(
            group_by="location",
            groups=[ImageGroup(key="巴黎", image_ids=[1, 2])],
        ) if group_ok else None,
        error=None if group_ok else TaskError(code="GROUP_FATAL", message="fatal"),
    )
    ctx.results[plan.synthesis_task_id] = WorkerResult(
        task_id=plan.synthesis_task_id, task_type=TaskType.SYNTHESIZE,
        status=TaskStatus.SUCCEEDED if synth_ok else TaskStatus.FAILED,
        output=SynthesisOutput(
            image_ids=list(synth_ids), candidates=[],
            synthesis_mode=synth_mode, synthesis_degraded=degraded,
            answer=answer, rationale=rationale,
        ) if synth_ok else None,
    )
    if with_vision:
        succ, total = vision
        ctx.results["t3"] = WorkerResult(
            task_id="t3", task_type=TaskType.ANALYZE_VISUAL,
            status=TaskStatus.SUCCEEDED,
            output=VisionTaskOutput(
                items=[], total=total, success_count=succ, failed_count=total - succ,
                systemic_failed=False,
            ),
        )
    return ctx


def _insert(svc, conversation_id="A", query="找旅行照片按地点整理", created=None):
    ep = svc.build_episode_from_context(
        _ctx(goal=query), route="team", conversation_id=conversation_id
    )
    assert ep is not None
    if created:
        ep["created_at"] = created
        ep["expires_at"] = (
            datetime.strptime(created, "%Y-%m-%dT%H:%M:%S.%fZ")
            + timedelta(days=EPISODE_TTL_DAYS)
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return svc.insert_episode(ep)


# ---------------------------------------------------------------------------
# Write policy
# ---------------------------------------------------------------------------
def test_write_successful_team_run():
    svc = _svc()
    new_id = svc.remember(_ctx(), route="team", conversation_id="A")
    assert new_id is not None
    rows = svc.list_recent("A")
    assert len(rows) == 1
    assert rows[0]["query_summary"] == "找旅行照片按地点整理"
    assert "retrieve_images→group_images→synthesize" in rows[0]["strategy_summary"]
    assert "selected_count=2" in rows[0]["result_summary"]
    assert "retrieved=6" in rows[0]["result_summary"]
    assert "groups=1" in rows[0]["result_summary"]
    assert "group_keys=巴黎" in rows[0]["result_summary"]
    hints = rows[0]["preference_hints"]
    assert "uses_group_by=location" in hints
    assert "group_by:location" in hints


def test_write_reuses_synthesizer_llm_summary():
    svc = _svc()
    rationale = "已按地点分成巴黎一组，选出2张代表性旅行照。"
    new_id = svc.remember(
        _ctx(synth_mode="llm", rationale=rationale, retrieved=6),
        route="team",
        conversation_id="A",
    )
    assert new_id is not None
    summary = svc.list_recent("A")[0]["result_summary"]
    assert rationale in summary
    assert "selected_count=2" in summary
    assert "mode=deterministic" not in summary


def test_write_requires_retrieve_or_group_success():
    svc = _svc()
    plan = SimpleNamespace(
        goal="随便说两句",
        synthesis_task_id="t1",
        tasks=[SimpleNamespace(task_id="t1", task_type=TaskType.SYNTHESIZE,
                               input=SimpleNamespace())],
    )
    ctx = ExecutionContext(plan)
    ctx.results["t1"] = WorkerResult(
        task_id="t1", task_type=TaskType.SYNTHESIZE,
        status=TaskStatus.SUCCEEDED,
        output=SynthesisOutput(image_ids=[1], candidates=[], synthesis_mode="llm",
                               rationale="没有检索也没有分组"),
    )
    assert svc.remember(ctx, route="team", conversation_id="A") is None
    assert svc.list_recent("A") == []


def test_write_planner_failure_no_write():
    svc = _svc()
    assert svc.remember(None, conversation_id="A") is None
    assert svc.list_recent("A") == []


def test_write_synth_failure_no_write():
    svc = _svc()
    assert svc.remember(_ctx(synth_ok=False), conversation_id="A") is None
    assert svc.list_recent("A") == []


def test_write_worker_fatal_no_write():
    svc = _svc()
    assert svc.remember(_ctx(worker_fatal=True), conversation_id="A") is None
    assert svc.list_recent("A") == []


def test_write_empty_selection_no_write():
    svc = _svc()
    assert svc.remember(_ctx(synth_ids=()), conversation_id="A") is None
    assert svc.list_recent("A") == []


def test_write_no_conversation_id_no_write():
    svc = _svc()
    assert svc.remember(_ctx(), conversation_id=None) is None


def test_duplicate_suppression():
    svc = _svc()
    svc.remember(_ctx(), conversation_id="A")
    svc.remember(_ctx(), conversation_id="A")  # 相同 query + strategy + 24h 内
    assert len(svc.list_recent("A")) == 1


def test_duplicate_different_query_allowed():
    svc = _svc()
    svc.remember(_ctx(), conversation_id="A")
    svc.remember(_ctx(goal="找夜景照片并按城市分组"), conversation_id="A")
    assert len(svc.list_recent("A")) == 2


def test_legacy_route_no_recall():
    svc = _svc()
    assert svc.remember(_ctx(), route="legacy", conversation_id="A") is None
    assert svc.list_recent("A") == []

# ---------------------------------------------------------------------------
def test_schema_insert_and_list():
    svc = _svc()
    _insert(svc, "A")
    rows = svc.list_recent("A")
    assert len(rows) == 1
    ep = rows[0]
    assert ep["conversation_id"] == "A"
    assert isinstance(ep["task_types"], list)
    assert ep["task_types"] == ["retrieve_images", "group_images", "synthesize"]
    assert ep["success"] == 1
    assert ep["plan_valid"] == 1
    assert ep["expires_at"] > ep["created_at"]


def test_schema_missing_conversation_id_rejected():
    svc = _svc()
    ep = svc.build_episode_from_context(_ctx(), conversation_id=None)
    assert ep is None


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------
def test_recall_top_n_limit():
    svc = _svc()
    for i in range(5):
        _insert(svc, "A", query=f"第{i}组旅行照片任务",
                created=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    eps = svc.recall_episodes("A", "旅行照片")
    assert len(eps) <= RECALL_TOP_N


def test_recall_budget_800():
    svc = _svc()
    for i in range(5):
        _insert(svc, "A", query=f"第{i}组旅行照片任务" + "重复词" * 10)
    hints = svc.render_memory_hints(svc.recall_episodes("A", "旅行照片"))
    assert sum(len(h) for h in hints) <= RECALL_BUDGET_CHARS


def test_recall_expired_excluded():
    svc = _svc()
    old = (datetime.utcnow() - timedelta(days=31)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    _insert(svc, "A", created=old)
    eps = svc.recall_episodes("A", "旅行照片")
    assert eps == []


def test_recall_relevant_ranks_higher():
    svc = _svc()
    _insert(svc, "A", query="找宠物猫的照片")
    _insert(svc, "A", query="找旅行风景照片并标注地点")
    eps = svc.recall_episodes("A", "找旅行风景照片")
    assert eps, "recall 不应为空"
    assert eps[0]["query_summary"] == "找旅行风景照片并标注地点"


def test_recall_conversation_isolation():
    svc = _svc()
    _insert(svc, "A")
    assert svc.recall_episodes("B", "旅行照片") == []


def test_recall_newest_first_ordering():
    svc = _svc()
    now = datetime.utcnow()
    for i, q in enumerate(["较新的旅行任务", "较早的旅行任务"]):
        created = (now - timedelta(hours=2 * i)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        _insert(svc, "A", query=q, created=created)
    eps = svc.recall_episodes("A", "旅行任务", top_n=2)
    assert eps[0]["query_summary"] == "较新的旅行任务"


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------
def test_ttl_30_days():
    svc = _svc()
    _insert(svc, "A")
    ep = svc.list_recent("A")[0]
    created = datetime.strptime(ep["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
    expires = datetime.strptime(ep["expires_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
    assert (expires - created).days == EPISODE_TTL_DAYS


def test_purge_expired():
    svc = _svc()
    old = (datetime.utcnow() - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    _insert(svc, "A", created=old)
    _insert(svc, "B", query="另一个会话的正常任务")
    deleted = svc._repo.purge_expired()
    assert deleted >= 1
    assert svc.list_recent("A") == []
    assert len(svc.list_recent("B")) == 1


def test_conversation_cap_20():
    svc = _svc()
    for i in range(25):
        _insert(svc, "A", query=f"任务编号{i}旅行照片")
    assert len(svc.list_recent("A")) <= 20


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
_FORBIDDEN = ("sk-", "api_key", "authorization", "base64", "data:image",
              "filepath", "chain_of_thought", "raw_response")


def test_security_no_forbidden_fields():
    svc = _svc()
    dirty = "找照片 data:image/png;base64,xx filepath=/etc 传 sk-abc chain_of_thought"
    _insert(svc, "A", query=dirty)
    rows = svc.list_recent("A")
    blob = json.dumps(rows, ensure_ascii=False).lower()
    for p in _FORBIDDEN:
        assert p not in blob, f"发现禁止字段 {p}"


def test_security_vision_only_counts():
    svc = _svc()
    ep = svc.build_episode_from_context(
        _ctx(with_vision=True, vision=(3, 4)), conversation_id="A"
    )
    assert ep is not None
    strat = json.dumps(ep["strategy_summary"], ensure_ascii=False).lower()
    assert "vision 3/4 成功" in strat
    for p in ("title", "description", "evidence", "confidence", "public_url"):
        assert p not in strat
    assert "base64" not in json.dumps(ep)


def _filter_ctx(goal="只要北京的jpg旅行照片", location="北京", file_type="jpg",
                presentation="grouped"):
    plan = SimpleNamespace(
        goal=goal,
        synthesis_task_id="t4",
        tasks=[
            SimpleNamespace(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES,
                            input=SimpleNamespace()),
            SimpleNamespace(
                task_id="t2", task_type=TaskType.FILTER_METADATA,
                input=SimpleNamespace(predicates=[
                    SimpleNamespace(field="file_type", operator="eq", value=file_type),
                    SimpleNamespace(field="location", operator="eq", value=location),
                ]),
            ),
            SimpleNamespace(task_id="t3", task_type=TaskType.GROUP_IMAGES,
                            input=SimpleNamespace(group_by="location")),
            SimpleNamespace(task_id="t4", task_type=TaskType.SYNTHESIZE,
                            input=SimpleNamespace(presentation=presentation)),
        ],
    )
    ctx = ExecutionContext(plan)
    ctx.goal = goal
    ctx.results["t1"] = WorkerResult(
        task_id="t1", task_type=TaskType.RETRIEVE_IMAGES,
        status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=[], query=goal, total=0),
    )
    ctx.results["t4"] = WorkerResult(
        task_id="t4", task_type=TaskType.SYNTHESIZE,
        status=TaskStatus.SUCCEEDED,
        output=SynthesisOutput(image_ids=[1, 2], candidates=[]),
    )
    return ctx


def test_records_structured_filter_and_group():
    svc = _svc()
    ep = svc.build_episode_from_context(_filter_ctx(), conversation_id="A")
    assert ep is not None
    hints = ep["preference_hints"]
    assert "uses_metadata_filter" in hints
    assert "filter:file_type=eq:jpg" in hints
    assert "filter:location=eq:北京" in hints
    assert "group_by:location" in hints
    assert "presentation:grouped" in hints
    text = svc._render_episode(ep)
    assert "筛选:" in text
    assert "分组:location" in text


def test_conversation_profile_aggregates_repeated_conditions():
    svc = _svc()
    for goal in ("白天只要北京的横向旅行照", "夜景街拍仍筛选北京jpg"):
        ep = svc.build_episode_from_context(_filter_ctx(goal=goal), conversation_id="A")
        assert ep is not None, goal
        svc.insert_episode(ep)
    rows = svc.list_recent("A")
    assert len(rows) == 2, [r.get("query_summary") for r in rows]
    profile = svc.build_profile(rows)
    filter_keys = dict(profile["filters"])
    assert filter_keys.get("location=eq:北京") == 2, filter_keys
    assert filter_keys.get("file_type=eq:jpg") == 2, filter_keys
    assert dict(profile["group_bys"]).get("location") == 2, profile["group_bys"]
    rendered = svc.render_memory_hints(rows, conversation_id="A")
    assert rendered
    assert any("会话画像" in line for line in rendered)
    assert any("location=eq:北京" in line for line in rendered)


def test_memory_hints_none_prompt_identical_to_legacy():
    """memory_hints=None 时 user prompt 与既有版本完全一致（capability routing 不受影响）。"""
    from backend.ai_func.recommendation.orchestration.planner import PlannerAgent

    goal = "找几张海边日落照片"
    msgs = [{"role": "user", "content": goal}, {"role": "assistant", "content": "好的"},
            {"role": "user", "content": "找几张海边日落照片"}]
    legacy = PlannerAgent._build_user_prompt(goal, msgs)
    with_hints_none = PlannerAgent._build_user_prompt(goal, msgs, None)
    with_hints = PlannerAgent._build_user_prompt(goal, msgs, ["[2026-09-01] 查询:海边 → 策略:retrieve"])
    assert legacy == with_hints_none
    assert "相关历史任务摘要" not in legacy
    assert "相关历史任务摘要" in with_hints
    assert "当前用户请求优先级最高" in with_hints
    assert "以当前目标为准" in with_hints


# ---------------------------------------------------------------------------
def _main():
    failed = passed = 0
    g = globals()
    for name in sorted(g):
        if name.startswith("test_"):
            try:
                g[name]()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n== memory unit: {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

