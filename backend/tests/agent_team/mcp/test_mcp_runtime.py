"""Phase 5 — Runtime Integration（Planner → DAG → Specialist → MCP Client → MCP Server）。

场景：
  A. image retrieval（supervisor + 远端 MCP search_images）
  B. retrieval + vision（远端 search_images + 远端 analyze_image）
  C. tool failure recovery（远端 search_images 首次失败 → REVISE → retry → done）

使用 official MCP SDK 的 in-process memory 会话（真实 tools/list、tools/call JSON-RPC）。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/mcp/test_mcp_runtime.py
"""

import json
import os
import re
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

os.environ["SMARTIMAGER_MCP_RETRIEVAL_SOURCE"] = "fake"
os.environ["SMARTIMAGER_MCP_VISION_SOURCE"] = "fake"

import backend.ai_func.recommendation.orchestration.models as models_mod
import backend.ai_func.recommendation.orchestration.planner as planner_mod
import backend.ai_func.recommendation.orchestration.workers as workers_mod
import backend.mcp_servers.retrieval_server.server as retrieval_server
import backend.mcp_servers.vision_server.server as vision_server

from backend.ai_func.recommendation.orchestration import (
    PlanValidator,
    PlannerAgent,
    SupervisorAgent,
    ToolRegistry,
    build_default_agent_registry,
    make_registry_retrieval_adapter,
    register_builtin_tools,
)
from backend.ai_func.recommendation.orchestration.tools.mcp_client import (
    MemoryMCPClient,
    register_remote_tools,
)
from backend.ai_func.recommendation.orchestration.workers import (
    DeterministicSynthesisWorker,
    RetrievalWorker,
    TaskType,
    VisionWorker,
)


class _VisionOn:
    def __enter__(self):
        self._orig = (models_mod.vision_available, planner_mod.vision_available,
                      workers_mod.vision_available)
        models_mod.vision_available = lambda: True
        planner_mod.vision_available = lambda: True
        workers_mod.vision_available = lambda: True

    def __exit__(self, *a):
        models_mod.vision_available, planner_mod.vision_available, workers_mod.vision_available = self._orig


class _MCPVisionAnalyzer:
    """VisionWorker analyzer 注入位：经 ToolRegistry 调用远端 MCP analyze_image。"""

    client = True

    def __init__(self, registry):
        self.registry = registry

    def analyze_image(self, image_url=None, local_image_path=None, **kwargs):
        source = local_image_path or image_url or ""
        m = re.search(r"(\d+)", source)
        image_id = m.group(1) if m else "0"
        payload = self.registry.call("analyze_image", {
            "image_id": image_id,
            "task": (kwargs.get("prompt") or "画面内容")[:120],
        })
        return {k: payload[k] for k in ("title", "description", "tags",
                                        "evidence", "confidence")}


def _plan(query, with_vision=False):
    tasks = [
        {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
         "input": {"query": query, "limit": 20}, "depends_on": []},
    ]
    if with_vision:
        tasks.append({"task_id": "t2", "task_type": "analyze_visual",
                      "objective": "视觉核验", "input": {"criteria": ["画面内容"],
                      "max_candidates": 2, "detail": "low"}, "depends_on": ["t1"]})
        synth = {"task_id": "t3", "task_type": "synthesize", "objective": "汇总",
                 "input": {"recommendation_limit": 5}, "depends_on": ["t1", "t2"]}
        tasks.append(synth)
        synth_id = "t3"
    else:
        tasks.append({"task_id": "t2", "task_type": "synthesize", "objective": "汇总",
                      "input": {"recommendation_limit": 5}, "depends_on": ["t1"]})
        synth_id = "t2"
    return {"goal": query, "synthesis_task_id": synth_id, "tasks": tasks}


def _setup(with_vision=False):
    reg = ToolRegistry()
    register_builtin_tools(reg)
    retrieval_client = MemoryMCPClient(retrieval_server.mcp)
    vision_client = MemoryMCPClient(vision_server.mcp)
    register_remote_tools(reg, retrieval_client, names=["search_images"])
    register_remote_tools(reg, vision_client, names=["analyze_image"])
    return reg, retrieval_client, vision_client


def _make_supervisor(reg, plan_json, with_vision=False):
    workers = {TaskType.RETRIEVE_IMAGES: RetrievalWorker(
        search_tool=make_registry_retrieval_adapter(reg))}
    if with_vision:
        workers[TaskType.ANALYZE_VISUAL] = VisionWorker(analyzer=_MCPVisionAnalyzer(reg))
    agent_reg = build_default_agent_registry(workers=workers, tool_registry=reg)
    planner = PlannerAgent(llm=lambda messages: json.dumps(plan_json, ensure_ascii=False))
    return SupervisorAgent(
        planner=planner, validator=PlanValidator(), workers=workers,
        agent_registry=agent_reg, tool_registry=reg,
        synthesize_worker=DeterministicSynthesisWorker(),
        enhanced=True, max_iterations=2,
    )


def _run(sup, query):
    events = [json.loads(raw) for raw in sup.run_stream(
        [{"role": "user", "content": query}])]
    reviews = [e for e in events if e["event"] == "agent.review"]
    decisions = [r["decision"] for r in reviews]
    selections = [e for e in events if e["event"] == "selection"]
    return events, decisions, selections


def test_scenario_a_image_retrieval():
    reg, retrieval_client, vision_client = _setup()
    sup = _make_supervisor(reg, _plan("城市建筑", with_vision=False))
    events, decisions, selections = _run(sup, "城市建筑")
    assert decisions == ["continue"]
    assert sup.run_state.final_state.value == "done"
    assert selections[-1]["ids"] == [901, 902]
    tool_names = {e.tool_name for e in sup.run_state.tool_events}
    assert "search_images" in tool_names
    assert retrieval_client.trace()[-1]["success"] is True


def test_scenario_b_retrieval_plus_vision():
    reg, retrieval_client, vision_client = _setup()
    sup = _make_supervisor(reg, _plan("城市建筑", with_vision=True), with_vision=True)
    with _VisionOn():
        events, decisions, selections = _run(sup, "城市建筑")
    assert decisions == ["continue"]
    assert selections[-1]["ids"] == [901, 902]
    tool_names = {e.tool_name for e in sup.run_state.tool_events}
    assert "search_images" in tool_names and "analyze_image" in tool_names
    vision_calls = [t for t in vision_client.trace() if t["tool_name"] == "analyze_image"]
    assert len(vision_calls) == 2
    assert all(t["success"] for t in vision_calls)


def test_scenario_c_tool_failure_recovery():
    reg, retrieval_client, vision_client = _setup()
    sup = _make_supervisor(reg, _plan("风景recover", with_vision=False))
    events, decisions, selections = _run(sup, "风景recover")
    assert decisions == ["revise", "continue"]
    assert sup.run_state.final_state.value == "done"
    assert selections[-1]["ids"] == [903]
    trace = retrieval_client.trace()
    assert len(trace) == 2
    assert trace[0]["success"] is False
    assert trace[1]["success"] is True
    assert any(e.success is False for e in sup.run_state.tool_events)


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

