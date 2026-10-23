"""T4 VLM 试点：进程内开 Vision，不改 config.yaml。

产品指标：Gate TEAM、结构化断言成功率、延迟。Judge 不作为决策。
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from openai import OpenAI

from scripts._load_deepseek_env import load_deepseek_env
from scripts._load_mm_rerank_env import load_mm_rerank_env

from backend.ai_func.image_analysis import ImageAnalysis
from backend.ai_func.recommendation.evaluation.latency import StageTimer, format_summary
from backend.ai_func.recommendation.orchestration.gate import ComplexityGate
from backend.ai_func.recommendation.orchestration.models import (
    AnalyzeVisualInput,
    RouteType,
    Task,
    TaskStatus,
    TaskType,
    VerifyVisualInput,
)
from backend.ai_func.recommendation.tools import tool_search_images
from backend.ai_func.recommendation.orchestration.workers import (
    RetrievalOutput,
    RetrievalWorker,
    VerifyVisualWorker,
    VisionWorker,
    WorkerResult,
)

_DS = os.path.join(
    _REPO, "backend", "ai_func", "recommendation",
    "evaluation", "datasets", "vlm_pilot_queries.json",
)
_OUT = os.path.join(
    _REPO, "backend", "ai_func", "recommendation",
    "evaluation", "results", "vlm_pilot",
)


def _write(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _resolve_vision() -> Dict[str, str]:
    """优先 GLM-4V；否则 DeepSeek 视觉模型。不改 yaml。"""
    if load_mm_rerank_env():
        return {
            "api_key": os.environ["MM_RERANK_API_KEY"],
            "base_url": os.environ["MM_RERANK_BASE_URL"],
            "model": os.environ.get("MM_RERANK_MODEL") or "glm-4v-flash",
            "provider": "glm",
        }
    prefix = load_deepseek_env()
    # DeepSeek 当前拒 null；进程内覆盖，不改 yaml
    model = os.environ.get("VISION_PILOT_MODEL") or "deepseek-v4-pro"
    return {
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.environ["OPENAI_API_BASE"],
        "model": model,
        "provider": "deepseek",
        "key_prefix": prefix,
    }


def _ctx(goal: str, cands) -> Any:
    class Ctx:
        pass

    ctx = Ctx()
    ctx.goal = goal
    ctx.results = {
        "retrieve_1": WorkerResult(
            task_id="retrieve_1",
            task_type=TaskType.RETRIEVE_IMAGES,
            status=TaskStatus.SUCCEEDED,
            output=RetrievalOutput(candidates=cands, query=goal, total=len(cands)),
        )
    }
    return ctx


def main() -> int:
    os.makedirs(_OUT, exist_ok=True)
    with open(_DS, encoding="utf-8") as f:
        spec = json.load(f)
    queries = spec["queries"]
    limit = int(os.environ.get("VLM_PILOT_LIMIT") or 0)
    if limit > 0:
        queries = queries[:limit]

    gate = ComplexityGate()
    gate_rows = []
    for q in queries:
        d = gate.decide(query=q["query"])
        got = d.route.value if isinstance(d.route, RouteType) else str(d.route)
        gate_rows.append({
            "id": q["id"],
            "query": q["query"],
            "got": got,
            "ok": got == "team",
        })
    gate_ok = sum(1 for r in gate_rows if r["ok"]) / len(gate_rows) if gate_rows else 0

    try:
        creds = _resolve_vision()
    except SystemExit as e:
        report = {
            "blocked": True,
            "reason": str(e),
            "n": len(queries),
            "gate_team_rate": round(gate_ok, 4),
            "gate": gate_rows,
        }
        _write(os.path.join(_OUT, "vlm_pilot.json"), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    analyzer = ImageAnalysis()
    analyzer.client = OpenAI(
        api_key=creds["api_key"],
        base_url=creds["base_url"].rstrip("/"),
        timeout=45.0,
    )
    model = creds["model"]

    class _Pinned:
        def __init__(self, inner, model_name: str):
            self.client = inner.client
            self._inner = inner
            self._model = model_name

        def analyze_image(self, **kwargs):
            kwargs["model"] = self._model
            return self._inner.analyze_image(**kwargs)

    pinned = _Pinned(analyzer, model)
    vision = VisionWorker(analyzer=pinned, model=model, item_timeout=30)
    verify = VerifyVisualWorker(analyzer=pinned, item_timeout=30)

    timer = StageTimer()
    records = []
    assert_ok = 0
    for idx, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        rows = tool_search_images(
            query=q["query"],
            targets=["title", "description", "image"],
            filters={},
            limit=8,
        ) or []
        cands = [RetrievalWorker._to_candidate(r) for r in rows][:2]
        ctx = _ctx(q["query"], cands)
        structured_ok = False
        detail: Dict[str, Any] = {"retrieved": len(cands)}
        try:
            if q.get("kind") == "analyze":
                task = Task(
                    task_id="analyze_1",
                    task_type=TaskType.ANALYZE_VISUAL,
                    objective=q["condition"],
                    input=AnalyzeVisualInput(criteria=[q["condition"]], max_candidates=2),
                    depends_on=["retrieve_1"],
                )
                with timer.span("vlm"):
                    res = vision.run(task, ctx)
                out = res.output
                items = list(getattr(out, "items", None) or [])
                structured_ok = bool(items) and all(
                    getattr(it, "status", None) is not None for it in items
                )
                detail["items"] = [
                    {
                        "id": getattr(it, "image_id", None),
                        "status": str(getattr(it, "status", "")),
                        "error": str(getattr(it, "error_code", "") or ""),
                    }
                    for it in items
                ]
            else:
                task = Task(
                    task_id="verify_1",
                    task_type=TaskType.VERIFY_VISUAL_CONDITION,
                    objective=q["condition"],
                    input=VerifyVisualInput(condition=q["condition"], max_candidates=2),
                    depends_on=["retrieve_1"],
                )
                with timer.span("vlm"):
                    res = verify.run(task, ctx)
                out = res.output
                items = list(getattr(out, "items", None) or [])
                structured_ok = bool(cands) and bool(items) and all(
                    isinstance(getattr(it, "passed", None), bool)
                    and getattr(it, "status", "") != "failed"
                    for it in items
                )
                # 无图时不算断言成功
                if not cands:
                    structured_ok = False
                detail["passed_count"] = getattr(out, "passed_count", 0)
                detail["inspected"] = getattr(out, "inspected_count", 0)
                detail["items"] = [
                    {
                        "id": getattr(it, "candidate_id", None),
                        "passed": getattr(it, "passed", None),
                        "status": getattr(it, "status", None),
                        "error": getattr(it, "error", None),
                    }
                    for it in items
                ]
        except Exception as e:
            detail["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            structured_ok = False
        ms = (time.perf_counter() - t0) * 1000.0
        if structured_ok:
            assert_ok += 1
        rec = {
            "id": q["id"],
            "query": q["query"],
            "kind": q.get("kind"),
            "gate_team": next(r["ok"] for r in gate_rows if r["id"] == q["id"]),
            "assertion_ok": structured_ok,
            "ms": round(ms, 1),
            **detail,
        }
        records.append(rec)
        print(f"  [t4] {idx}/{len(queries)} {q['id']} ok={structured_ok} n={detail.get('retrieved')}", flush=True)

    n = len(records) or 1
    report = {
        "blocked": False,
        "provider": creds["provider"],
        "model": model,
        "config_yaml_untouched": True,
        "n": len(records),
        "gate_team_rate": round(gate_ok, 4),
        "assertion_success_rate": round(assert_ok / n, 4),
        "stage_latency": timer.summary(),
        "mean_query_ms": round(sum(r["ms"] for r in records) / n, 1),
        "note": "决策看 assertion_success_rate 与延迟，不看 Judge。",
        "records": records,
        "gate": gate_rows,
    }
    path = os.path.join(_OUT, "vlm_pilot.json")
    _write(path, report)
    with open(os.path.join(_OUT, "vlm_pilot.md"), "w", encoding="utf-8") as f:
        f.write("# VLM 试点\n\n")
        f.write(f"- n={report['n']} gate TEAM={report['gate_team_rate']} "
                f"assertion={report['assertion_success_rate']}\n")
        f.write(f"- model={model} provider={creds['provider']}\n\n")
        f.write(format_summary(timer.summary()))
        f.write("\n")
    print(json.dumps({
        "out": path,
        "gate_team": report["gate_team_rate"],
        "assertion": report["assertion_success_rate"],
        "latency": report["stage_latency"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
