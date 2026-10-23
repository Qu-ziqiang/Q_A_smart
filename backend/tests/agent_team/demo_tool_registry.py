"""Phase 2 MCP-style Tool Registry — Runtime Verification（离线演示，Requirement 9）。

演示内容（零真实 API / DB）：
  1) 默认 ToolRegistry 视图 + MCP schema 导出预览
  2) Agent Tool Binding：Retrieval / Vision / Organizer Agent 经 ToolRegistry 调用工具
     （search_images / rerank / analyze_image / filter_images / group_images 注入 Fake 实现）
  3) 既有执行链稳定路径核对：DAGExecutor → SpecialistAgent.execute → Worker.run

运行（repo 根目录）：
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/demo_tool_registry.py
"""

import json
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration import (
    AgentCard,
    SpecialistAgent,
    TaskType,
    ToolRegistry,
    default_tool_registry,
    register_builtin_tools,
)
from backend.ai_func.recommendation.orchestration.tools.mcp_schema import export_tool_schema


def banner(title):
    print("\n" + "=" * 74)
    print("== " + title)
    print("=" * 74)


def _fake_registry():
    """内置定义 + Fake 实现（避免真实 DB / 视觉 API）。"""
    reg = ToolRegistry()

    def search_images(query, vector_targets=None, filters=None, limit=20):
        rows = [
            {"id": 11, "score": 0.92, "title": "北京夜景", "metadata": {"location": "北京"},
             "tags": ["city"], "public_url": "/static/11.jpg"},
            {"id": 12, "score": 0.85, "title": "上海外滩", "metadata": {"location": "上海"},
             "tags": ["city"], "public_url": "/static/12.jpg"},
        ]
        return rows[: int(limit or 20)]

    def rerank(query, candidates, intent=None, weights=None):
        return sorted(candidates, key=lambda c: float(c.get("score", 0)), reverse=True)

    def analyze_image(image_url=None, local_image_path=None, **kwargs):
        return {"title": "海边日落", "description": "画面可见日落与海面", "tags": ["海边", "日落"],
                "evidence": "图中可见橙红色天空与水面", "confidence": 0.9}

    def filter_images(candidates, predicates):
        kept = [c for c in candidates if "北京" in str(c.get("metadata", {}).get("location", ""))]
        return {"kept": kept, "rejected": [c for c in candidates if c not in kept]}

    def group_images(candidates, group_by, unknown_group_name="未分类"):
        groups = {}
        for c in candidates:
            key = str(c.get("metadata", {}).get("location", unknown_group_name))
            groups.setdefault(key, []).append(c)
        return {"group_by": group_by,
                "groups": [{"key": k, "count": len(v)} for k, v in groups.items()]}

    register_builtin_tools(reg, overrides={
        "search_images": search_images,
        "rerank": rerank,
        "analyze_image": analyze_image,
        "filter_images": filter_images,
        "group_images": group_images,
    })
    return reg


class _FW:
    def __init__(self, ttype):
        self.task_type = ttype

    def run(self, task, context):
        raise AssertionError("binding demo 不应触发 Worker.run")


def _agents(reg):
    return {
        "retrieval": SpecialistAgent(
            AgentCard(agent_id="retrieval", name="RetrievalAgent", description="semantic retrieval",
                      supported_tasks=[TaskType.RETRIEVE_IMAGES],
                      available_tools=["search_images", "rerank"]),
            _FW(TaskType.RETRIEVE_IMAGES), registry=reg,
        ),
        "vision": SpecialistAgent(
            AgentCard(agent_id="vision", name="VisionAgent", description="visual verification",
                      supported_tasks=[TaskType.ANALYZE_VISUAL],
                      available_tools=["analyze_image"]),
            _FW(TaskType.ANALYZE_VISUAL), registry=reg,
        ),
        "organizer": SpecialistAgent(
            AgentCard(agent_id="organizer", name="OrganizerAgent", description="organization",
                      supported_tasks=[TaskType.FILTER_METADATA, TaskType.GROUP_IMAGES],
                      available_tools=["filter_images", "group_images"]),
            {TaskType.FILTER_METADATA: _FW(TaskType.FILTER_METADATA),
             TaskType.GROUP_IMAGES: _FW(TaskType.GROUP_IMAGES)},
            registry=reg,
        ),
    }


def demo_registry_view():
    banner("1) 默认 ToolRegistry 视图 + MCP schema 导出")
    reg = default_tool_registry()
    for t in reg.list_tools():
        print(f"  {t.name:<14} category={t.category:<12} cost={t.cost_level:<8} "
              f"enabled={t.enabled} desc={t.description[:34]}…")
    sample = export_tool_schema(reg.get("search_images"))
    print("\n  MCP export (search_images):")
    print("   ", json.dumps({"name": sample["name"],
                             "description": sample["description"][:40] + "…",
                             "inputSchema_type": sample["inputSchema"]["type"],
                             "required": sample["inputSchema"]["required"]}, ensure_ascii=False))


def demo_tool_calling():
    banner("2) Agent → ToolRegistry → Tool 统一调用（Fake 实现，离线）")
    reg = _fake_registry()
    agents = _agents(reg)

    retrieval = agents["retrieval"]
    rows = retrieval.call_tool("search_images", {"query": "城市夜景", "limit": 2})
    print(f"    retrieval_agent -> search_images => {len(rows)} 行候选")
    reranked = retrieval.call_tool("rerank", {"query": "城市夜景", "candidates": rows})
    print(f"    retrieval_agent -> rerank        => {len(reranked)} 行（同池）")

    vision = agents["vision"]
    vres = vision.call_tool("analyze_image", {"local_image_path": "/tmp/demo.jpg"})
    print(f"    vision_agent -> analyze_image    => title={vres['title']}")

    organizer = agents["organizer"]
    fres = organizer.call_tool("filter_images", {
        "candidates": rows,
        "predicates": [{"field": "location", "operator": "eq", "value": "北京"}],
    })
    print(f"    organizer_agent -> filter_images => kept={len(fres['kept'])}")
    gres = organizer.call_tool("group_images", {"candidates": rows, "group_by": "location"})
    print(f"    organizer_agent -> group_images  => {gres['groups']}")

    print("\n  ToolCallEvent Trace（供 LLM Judge / 调优）：")
    for ev in reg.trace():
        print(f"    [ToolRegistry] {ev.agent_id} -> {ev.tool_name} "
              f"success={ev.success} latency_ms={ev.latency_ms:.1f}")

    summary = reg.stats_summary()
    print("\n  调用统计：")
    for name, s in summary.items():
        print(f"    {name:<14} calls={s['calls']} ok={s['successes']} fail={s['failures']}")


def demo_execution_chain():
    banner("3) 既有执行链稳定路径核对（Planner → Validator → Executor → Specialist → Worker）")
    from backend.ai_func.recommendation.orchestration.orchestrator import AdaptiveTeamOrchestrator
    orch = AdaptiveTeamOrchestrator()
    for ttype, w in orch.executor.workers.items():
        who = getattr(getattr(w, "card", None), "name", type(w).__name__)
        print(f"    executor.workers[{ttype.value}] → {who}")
    print("    说明：DAG 内任务仍经 SpecialistAgent.execute → Worker.run 执行；"
          "ToolRegistry 是 Agent 工具的统一入口与 Trace 层，二者并存、行为不变。")


if __name__ == "__main__":
    demo_registry_view()
    demo_tool_calling()
    demo_execution_chain()
    print("\nRuntime Verification 完成：工具已统一为 MCP-style Tool Registry，Agent 绑定并通过注册中心调用。")

