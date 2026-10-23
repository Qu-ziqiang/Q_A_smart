"""Phase 3 — SupervisorAgent 有限状态机单元测试。

覆盖：
  - normal success（增强模式：一次 EXECUTE → continue → DONE）
  - empty retrieval retry（空检索 → REVIEW revise → 重试成功 → DONE）
  - tool failure abort（工具层持续失败 → revise → 上限 → ABORT，ToolCallEvent 失败留痕）
  - max iteration limit（候选低于阈值 → replan → 上限 → ABORT，无无限循环）
  - baseline（enhanced=False：空检索也不重试，保留既有单次行为）

全部 Fake / Mock（工具经 ToolRegistry + adapter，真实 Worker/Executor/Validator 路径），
零真实 DB / LLM / 视觉 API。运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_supervisor_state_machine.py
"""

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    PlanValidator,
    PlannerAgent,
    SupervisorAgent,
    SupervisorState,
    ToolRegistry,
    build_default_agent_registry,
    build_registry_backed_workers,
    register_builtin_tools,
)
from backend.ai_func.recommendation.orchestration.workers import (
    DeterministicSynthesisWorker,
    TaskStatus,
)
from backend.tests.agent_team.loop_support import rewrite_aware_llm


def _rows():
    return [
        {"id": 201, "score": 0.91, "title": "城市建筑A", "description": "a", "tags": ["建筑"],
         "metadata": {"location": "北京"}, "width": 1920, "height": 1280,
         "file_type": "jpg", "created_at": "2025-06-01T10:00:00Z",
         "public_url": "/static/201.jpg", "filepath": ""},
        {"id": 202, "score": 0.84, "title": "城市建筑B", "description": "b", "tags": ["建筑"],
         "metadata": {"location": "上海"}, "width": 1920, "height": 1080,
         "file_type": "jpg", "created_at": "2025-06-02T10:00:00Z",
         "public_url": "/static/202.jpg", "filepath": ""},
    ]


def _plan_json():
    return {
        "goal": "找城市建筑图片",
        "synthesis_task_id": "t2",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索城市建筑图片",
             "input": {"query": "城市建筑", "limit": 20}, "depends_on": []},
            {"task_id": "t2", "task_type": "synthesize", "objective": "汇总推荐",
             "input": {"recommendation_limit": 10}, "depends_on": ["t1"]},
        ],
    }


class _FakeSearch:
    """按调用次序返回固定结果的 search_images 实现（经 ToolRegistry 注入）。"""

    def __init__(self, results_by_call, fail=False):
        self.results = list(results_by_call)
        self.calls = 0
        self.fail = fail

    def __call__(self, query=None, vector_targets=None, filters=None, limit=20):
        if self.fail:
            self.calls += 1
            raise RuntimeError("search tool down")
        idx = min(self.calls, len(self.results) - 1) if self.results else 0
        self.calls += 1
        return list(self.results[idx])


def _make_supervisor(search_impl, *, enhanced=True, max_iterations=2, min_candidates=1,
                     plan_json=None):
    tool_reg = ToolRegistry()
    register_builtin_tools(tool_reg, overrides={"search_images": search_impl})
    workers = build_registry_backed_workers(tool_reg, with_vision=False)
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=tool_reg)
    planner = PlannerAgent(llm=rewrite_aware_llm(
        plan_json if plan_json is not None else _plan_json()))
    return SupervisorAgent(
        planner=planner,
        validator=PlanValidator(),
        workers=workers,
        agent_registry=agent_reg,
        tool_registry=tool_reg,
        synthesize_worker=DeterministicSynthesisWorker(),
        enhanced=enhanced,
        max_iterations=max_iterations,
        min_candidates=min_candidates,
    )


def _run(supervisor):
    query = "找城市建筑图片"
    events = [json.loads(raw) for raw in supervisor.run_stream(
        [{"role": "user", "content": query}])]
    reviews = [e for e in events if e["event"] == "agent.review"]
    decisions = [r["decision"] for r in reviews]
    selections = [e for e in events if e["event"] == "selection"]
    return events, decisions, selections, reviews


def test_normal_success_continue_done():
    search = _FakeSearch([_rows()])
    supervisor = _make_supervisor(search, enhanced=True, max_iterations=2)
    events, decisions, selections, _ = _run(supervisor)
    assert decisions == ["continue"]
    assert supervisor.run_state.final_state == SupervisorState.DONE
    assert search.calls == 1  # 单次 EXECUTE
    assert selections and selections[-1]["ids"] == [201, 202]
    names = [e["event"] for e in events]
    assert "plan.created" in names and "assistant_delta" in names
    # AgentMessage：Supervisor→Specialist task；Specialist→Supervisor report
    types = [m.message_type for m in supervisor.run_state.agent_messages]
    assert "task" in types and "report" in types
    senders = {m.sender for m in supervisor.run_state.agent_messages}
    assert "supervisor" in senders and "retrieval" in senders


def test_empty_retrieval_retry_revise_then_done():
    search = _FakeSearch([[], _rows()])  # 第一次空、第二次有结果
    supervisor = _make_supervisor(search, enhanced=True, max_iterations=2)
    _, decisions, selections, _ = _run(supervisor)
    assert decisions == ["replan", "continue"]
    assert supervisor.run_state.final_state == SupervisorState.DONE
    assert search.calls == 2  # 触发一次 REPLAN 重跑
    assert selections and selections[-1]["ids"] == [201, 202]
    execute_attempts = [t["attempt"] for t in supervisor.run_state.supervisor_trace
                        if t["state"] == "execute"]
    assert execute_attempts == [1, 2]


def test_tool_failure_abort_after_max_iterations():
    search = _FakeSearch([_rows()], fail=True)  # 工具持续失败
    supervisor = _make_supervisor(search, enhanced=True, max_iterations=2)
    _, decisions, selections, _ = _run(supervisor)
    assert decisions == ["revise", "abort"]
    assert supervisor.run_state.final_state == SupervisorState.ABORT
    assert search.calls == 2
    assert selections and selections[-1]["ids"] == []  # 无有效选择
    # ToolCallEvent 失败留痕（供 LLM Judge / 调优）
    tool_failures = [e for e in supervisor.run_state.tool_events if not e.success]
    assert tool_failures, "应有失败的工具调用事件"
    assert tool_failures[0].tool_name == "search_images"
    # 执行结果中存在 failed 任务
    results = supervisor.run_state.execution_results
    assert any(r.status == TaskStatus.FAILED for r in results.values())


def test_max_iteration_limit_abort_no_infinite_loop():
    # 候选恒 1 条，低于 min_candidates=3 → 走 replan；达到上限后 ABORT
    search = _FakeSearch([_rows()[:1]])
    supervisor = _make_supervisor(search, enhanced=True, max_iterations=2, min_candidates=3)
    _, decisions, selections, reviews = _run(supervisor)
    assert decisions == ["replan", "abort"]
    assert supervisor.run_state.final_state == SupervisorState.ABORT
    assert search.calls == 2  # 两次 EXECUTE 后停止，未无限循环
    execute_attempts = [t["attempt"] for t in supervisor.run_state.supervisor_trace
                        if t["state"] == "execute"]
    assert execute_attempts == [1, 2]
    assert reviews[-1]["reason"] != ""


def test_baseline_enhanced_disabled_preserves_single_pass():
    # 默认增强关闭：空检索也不 revise/replan，保持既有行为
    search = _FakeSearch([[]])
    supervisor = _make_supervisor(search, enhanced=False, max_iterations=2)
    _, decisions, selections, _ = _run(supervisor)
    assert decisions == ["continue"]
    assert supervisor.run_state.final_state == SupervisorState.DONE
    assert search.calls == 1  # 不重试


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

