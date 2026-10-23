"""Phase 2 Stage 9 — Real E2E demo script。

真实组件：DeepSeek Planner / CLIP + sqlite-vec 检索 / DeepSeek Vision / DeepSeek Synthesizer /
真实 _RecommendationService.stream_core（SSE service path）/ ComplexityGate / PlanValidator / DAGExecutor。

Fake 组件：受控 failure injection（Fake ImageAnalysis）、Fake Legacy agent（仅用于 fallback 检测）、
前端浏览器（NOT REQUIRED）。

VISION_MODEL 采用 runtime override（模型注入 + 运行时 capability patch），不改 config.yaml。

运行（repo 根目录，务必 UTF-8）：
    $env:PYTHONIOENCODING='utf-8'
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/demo_phase2_e2e.py
"""

import json
import os
import sys
import time

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

import backend.ai_func.recommendation.orchestration.models as models_mod
from backend.ai_func.image_analysis import ImageAnalysis, ImageAnalysisError, ImageAnalysisErrorCode
from backend.ai_func.recommendation import _RecommendationService
from backend.ai_func.recommendation.orchestration import (
    AdaptiveTeamOrchestrator,
    ComplexityGate,
    PlanValidator,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.models import (
    AnalyzeVisualInput,
    RetrieveImagesInput,
    SynthesizeInput,
    Task,
    TaskPlan,
)
from backend.ai_func.recommendation.orchestration.planner import PlannerAgent
from backend.ai_func.recommendation.orchestration.synthesizer import SynthesizerAgent
from backend.ai_func.recommendation.orchestration.vision import VisionStatus
from backend.ai_func.recommendation.orchestration.workers import (
    DeterministicSynthesisWorker,
    GroupImagesWorker,
    ImageCandidate,
    RetrievalOutput,
    RetrievalWorker,
    VisionWorker,
    WorkerResult,
)
from backend.config import settings
from backend.config.init_service import get_openai_client

VISION_MODEL = "deepseek-v4-flash-vision-exp"
TEXT_MODEL = "deepseek-chat"

_OUT = []


def log(msg):
    _OUT.append(str(msg))
    print("[E2E]", str(msg))


# ---------------------------------------------------------------------------
# 运行时 instrumentation / override
# ---------------------------------------------------------------------------
class _LLMCounter:
    """包装共享 OpenAI client 的 chat.completions.create，统计 LLM 调用数。"""

    def __init__(self):
        self.n = 0
        self._orig = None

    def install(self):
        client = get_openai_client()
        self._orig = client.chat.completions.create

        def wrapped(*a, **k):
            self.n += 1
            return self._orig(*a, **k)

        client.chat.completions.create = wrapped

    def restore(self):
        if self._orig is not None:
            get_openai_client().chat.completions.create = self._orig


class _CountingAnalyzer:
    """包一层真实 ImageAnalysis，只统计 analyze_image 调用数。"""

    def __init__(self, inner):
        self._inner = inner
        self.client = inner.client
        self.calls = 0

    def analyze_image(self, **kwargs):
        self.calls += 1
        return self._inner.analyze_image(**kwargs)


class _VisionOn:
    """运行时 capability override（进程内，不改 config）。"""

    def __enter__(self):
        self._orig = models_mod.vision_available
        models_mod.vision_available = lambda: True
        return self

    def __exit__(self, *a):
        models_mod.vision_available = self._orig


class _FakeLegacyAgent:
    """Fake Legacy agent：仅用于检测 fallback（不调用真实 Legacy）。"""

    def __init__(self):
        self.called = False

    def run_stream(self, **kwargs):
        self.called = True
        yield json.dumps({"event": "assistant_delta", "delta": "[legacy-fallback]"}, ensure_ascii=False)
        yield json.dumps({"event": "selection", "ids": [], "images": []}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 真实 E2E 主流程
# ---------------------------------------------------------------------------
def build_real_orchestrator(analyzer):
    planner = PlannerAgent()  # 真实 DeepSeek Planner（CHAT_MODEL）
    workers = {
        TaskType.RETRIEVE_IMAGES: RetrievalWorker(),          # 真实 CLIP + sqlite-vec
        TaskType.GROUP_IMAGES: GroupImagesWorker(),           # 真实确定性分组
        TaskType.ANALYZE_VISUAL: VisionWorker(
            analyzer=analyzer, model=VISION_MODEL, item_timeout=30),  # 真实 DeepSeek Vision
        TaskType.SYNTHESIZE: SynthesizerAgent(),              # 真实 DeepSeek Synthesizer
    }
    executor = DAGExecutor(workers=workers)
    return AdaptiveTeamOrchestrator(planner=planner, executor=executor)


def run_real_e2e(query, analyzer, counter):
    svc = _RecommendationService()
    svc._agent = _FakeLegacyAgent()
    svc._gate = ComplexityGate()
    svc._orchestrator = build_real_orchestrator(analyzer)
    events = []
    t0 = time.time()
    for raw in svc.stream_core(messages=[{"role": "user", "content": query}]):
        events.append(json.loads(raw))
    total_s = round(time.time() - t0, 2)

    summary = {
        "query": query,
        "events": events,
        "total_latency_s": total_s,
        "fallback_to_legacy": svc._agent.called,
        "last_plan_ok": svc._orchestrator.last_plan_ok,
        "plan_failed": any(e.get("event") == "plan.failed" for e in events),
    }
    plan_ev = next((e for e in events if e.get("event") == "plan.created"), None)
    summary["plan_tasks"] = [t["task_type"] for t in (plan_ev or {}).get("tasks", [])]
    summary["has_analyze_visual"] = "analyze_visual" in summary["plan_tasks"]
    summary["task_latency_ms"] = {
        e["task_type"]: e.get("duration_ms")
        for e in events if e.get("event") == "task.completed" and e.get("duration_ms") is not None
    }
    vis_ev = next((e for e in events if e.get("event") == "task.completed"
                   and e.get("task_type") == "analyze_visual"), None)
    summary["vision_summary"] = (vis_ev or {}).get("output_summary", {}).get("vision")
    sel = next((e for e in events if e.get("event") == "selection"), None)
    summary["selection_ids"] = (sel or {}).get("ids", [])
    deltas = [e.get("delta", "") for e in events if e.get("event") == "assistant_delta"]
    summary["answer"] = "".join(deltas).strip()
    # Vision evidence 数据边界验证（executor context 的 synthesize 候选）
    summary["vision_evidence_count"] = 0
    summary["validated_image_ids"] = []
    summary["vision_sample"] = []
    synth_id = (plan_ev or {}).get("synthesis_task_id")
    ctx = svc._orchestrator.executor.context
    if ctx is not None and synth_id:
        res = ctx.results.get(synth_id)
        if res and res.status == TaskStatus.SUCCEEDED and hasattr(res.output, "candidates"):
            cands = res.output.candidates
            summary["vision_evidence_count"] = sum(1 for c in cands if c.vision is not None)
            summary["validated_image_ids"] = list(res.output.image_ids)
            for c in cands:
                if c.vision is not None:
                    v = c.vision
                    summary["vision_sample"].append({
                        "image_id": v.image_id,
                        "title": v.title,
                        "tags": list(v.tags),
                        "confidence": v.confidence,
                        "evidence": (v.evidence or "")[:80],
                    })
    retr_ev = next((e for e in events if e.get("event") == "task.completed"
                    and e.get("task_type") == "retrieve_images"), None)
    summary["retrieval_candidate_count"] = ((retr_ev or {}).get("output_summary") or {}).get("candidate_count")
    summary["retrieval_candidate_ids"] = ((retr_ev or {}).get("output_summary") or {}).get("image_ids", [])
    summary["validated_plan"] = svc._orchestrator.last_plan
    return summary, svc


def run_vision_supplement(plan, analyzer, counter):
    """Vision 补充验证：用真实 Planner 的合法计划 + 追加 analyze_visual 任务（文档化补充），
    经真实 Validator 复验后，走真实 Executor + 真实 Vision + 真实 Synthesizer。"""
    from backend.ai_func.recommendation.orchestration.models import (
        TaskDraft, TaskPlanDraft,
    )

    # 序列化真实计划 → draft，追加 analyze_visual 任务（依赖 retrieve；synth 增加依赖）
    tasks = [t.model_dump() for t in plan.tasks]
    synth_idx = next(i for i, t in enumerate(tasks) if t["task_type"].value == "synthesize")
    retr_ids = [t["task_id"] for t in tasks if t["task_type"].value == "retrieve_images"]
    vis_task = {
        "task_id": "t_vis", "task_type": TaskType.ANALYZE_VISUAL, "objective": "视觉分析",
        "input": {"criteria": ["旅行记录"], "max_candidates": 4}, "depends_on": list(retr_ids),
    }
    tasks.append(vis_task)
    tasks[synth_idx] = dict(tasks[synth_idx])
    tasks[synth_idx]["depends_on"] = list(tasks[synth_idx]["depends_on"]) + ["t_vis"]
    draft = TaskPlanDraft(goal=plan.goal, synthesis_task_id=plan.synthesis_task_id, tasks=[
        TaskDraft(task_id=t["task_id"], task_type=t["task_type"], objective=t.get("objective", "o"),
                  input=t.get("input", {}), depends_on=list(t.get("depends_on", [])))
        for t in tasks
    ])
    vres = PlanValidator().validate(draft, vision_available_flag=True)
    if not vres.valid or vres.plan is None:
        return {"plan_valid": False, "errors": vres.errors}

    workers = {
        TaskType.RETRIEVE_IMAGES: RetrievalWorker(),          # 真实检索
        TaskType.GROUP_IMAGES: GroupImagesWorker(),
        TaskType.ANALYZE_VISUAL: VisionWorker(
            analyzer=analyzer, model=VISION_MODEL, item_timeout=30),  # 真实 Vision
        TaskType.SYNTHESIZE: SynthesizerAgent(),              # 真实 Synthesizer
    }
    ex = DAGExecutor(workers=workers)
    events = []
    t0 = time.time()
    for ev in ex.execute_stream(vres.plan):
        events.append(ev)
    total_s = round(time.time() - t0, 2)
    res = ex.context.results
    vis = res.get("t_vis")
    synth = res.get(plan.synthesis_task_id)
    summary = {
        "plan_valid": True,
        "plan_tasks": [t["task_type"].value for t in tasks],
        "events": events,
        "total_latency_s": total_s,
        "vision_task_status": vis.status.value if vis else None,
        "vision_summary": (vis.output.model_dump() if vis and vis.output else None),
        "synth_status": synth.status.value if synth else None,
        "vision_analyzer_calls": analyzer.calls,
        "vision_evidence_count": 0,
        "validated_image_ids": [],
        "vision_sample": [],
        "answer": "",
    }
    if synth and synth.status == TaskStatus.SUCCEEDED and synth.output:
        summary["validated_image_ids"] = list(synth.output.image_ids)
        for c in synth.output.candidates:
            if c.vision is not None:
                summary["vision_evidence_count"] += 1
                v = c.vision
                summary["vision_sample"].append({
                    "image_id": v.image_id, "title": v.title, "tags": list(v.tags),
                    "confidence": v.confidence, "evidence": (v.evidence or "")[:80],
                })
    deltas = [e.get("delta", "") for e in events if e.get("event") == "assistant_delta"]
    return summary, ex


# ---------------------------------------------------------------------------
# 受控 failure injection（Fake）
# ---------------------------------------------------------------------------
class _FailSecondAnalyzer:
    """Fake：第二次调用抛 TIMEOUT（受控 fake failure，不调用真实 API）。"""

    def __init__(self):
        self.client = object()
        self.calls = 0

    def analyze_image(self, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise ImageAnalysisError(ImageAnalysisErrorCode.TIMEOUT, "controlled fake timeout")
        return {"title": "ok", "description": "desc", "tags": ["t"], "evidence": "e", "confidence": 0.9}


def run_failure_injection():
    """受控 failure E2E：Fake analyzer 让第 2 个 candidate 失败，验证 synth 仍执行。"""
    from backend.ai_func.recommendation.orchestration.models import GroupImagesInput

    rows = [
        {"id": 201, "score": 0.9, "title": "a", "tags": [], "metadata": {},
         "public_url": "/s/201.jpg", "filepath": None},
        {"id": 202, "score": 0.8, "title": "b", "tags": [], "metadata": {},
         "public_url": "/s/202.jpg", "filepath": None},
        {"id": 203, "score": 0.7, "title": "c", "tags": [], "metadata": {},
         "public_url": "/s/203.jpg", "filepath": None},
    ]
    plan = TaskPlan(
        plan_version="1.0", plan_id="plan_fail", goal="g", synthesis_task_id="t4",
        tasks=[
            Task(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, objective="r",
                 input=RetrieveImagesInput(query="q"), depends_on=[]),
            Task(task_id="t2", task_type=TaskType.GROUP_IMAGES, objective="g",
                 input=GroupImagesInput(group_by="location"), depends_on=["t1"]),
            Task(task_id="t3", task_type=TaskType.ANALYZE_VISUAL, objective="v",
                 input=AnalyzeVisualInput(criteria=["c"], max_candidates=8), depends_on=["t1"]),
            Task(task_id="t4", task_type=TaskType.SYNTHESIZE, objective="s",
                 input=SynthesizeInput(recommendation_limit=10), depends_on=["t2", "t3"]),
        ],
    )
    workers = {
        TaskType.RETRIEVE_IMAGES: RetrievalWorker(search_tool=lambda **kw: list(rows)),
        TaskType.GROUP_IMAGES: GroupImagesWorker(),
        TaskType.ANALYZE_VISUAL: VisionWorker(analyzer=_FailSecondAnalyzer()),
        TaskType.SYNTHESIZE: DeterministicSynthesisWorker(),  # 确定性合成，不耗真实 API
    }
    ex = DAGExecutor(workers=workers)
    list(ex.execute_stream(plan))
    vis = ex.context.results["t3"]
    synth = ex.context.results["t4"]
    return {
        "vision_status": vis.status.value,
        "vision_total": vis.output.total,
        "vision_success": vis.output.success_count,
        "vision_failed": vis.output.failed_count,
        "synth_status": synth.status.value,
        "synth_ids": list(synth.output.image_ids) if synth.output else [],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _credential_present():
    cfg = settings.get_config()
    return bool(cfg and getattr(cfg, "OPENAI_API_KEY", None))


def main():
    log("=== Phase 2 Stage 9 — Real E2E ===")
    cfg = settings.get_config()
    log(f"credential_present = {_credential_present()}")
    log(f"config VISION_MODEL = {getattr(cfg, 'VISION_MODEL', None)!r}（E2E 使用 runtime override）")
    log(f"config CHAT_MODEL = {getattr(cfg, 'CHAT_MODEL', None)!r}")
    log(f"config OPENAI_API_BASE = {getattr(cfg, 'OPENAI_API_BASE', None)}")

    counter = _LLMCounter()
    counter.install()
    try:
        with _VisionOn():
            log(f"vision_available()（runtime override）= {models_mod.vision_available()}")

            log("---- Real E2E #1（真实 Planner + 检索 + 分组 + 合成；planner 非确定性，最多重试 3 次）----")
            analyzer = _CountingAnalyzer(ImageAnalysis())
            s1 = svc1 = None
            for attempt in range(1, 4):
                s1, svc1 = run_real_e2e("找旅行照片，按地点分类并整理结果", analyzer, counter)
                log(f"attempt {attempt}: plan_failed={s1['plan_failed']} "
                    f"plan_tasks={s1['plan_tasks']}")
                if s1["validated_plan"] is not None:
                    break
            if s1 is None or s1["validated_plan"] is None:
                log("E2E #1 失败：真实 Planner 连续 3 次未产生通过 Validator 的计划（F-002 方差）")
            else:
                log(f"plan_tasks = {s1['plan_tasks']}")
                log(f"has_analyze_visual = {s1['has_analyze_visual']}")
                log(f"retrieval_candidates = {s1['retrieval_candidate_count']} (ids={s1['retrieval_candidate_ids'][:8]})")
                log(f"vision_summary = {s1['vision_summary']}")
                log(f"vision_analyzer_calls = {analyzer.calls}")
                log(f"task_latency_ms = {s1['task_latency_ms']}")
                log(f"selection_ids = {s1['selection_ids']}")
                log(f"validated_image_ids = {s1['validated_image_ids']}")
                log(f"vision_evidence_count = {s1['vision_evidence_count']}")
                log(f"answer_head = {s1['answer'][:120]!r}")
                log(f"fallback_to_legacy = {s1['fallback_to_legacy']} / plan_failed = {s1['plan_failed']}")
            s2 = None

            # Vision 补充：真实 Planner 未产出 analyze_visual（F-002 实证），
            # 用真实 Planner 的合法计划 + 追加 analyze_visual 任务（文档化补充）验证真实 Vision 链
            if s1["validated_plan"] is not None and not s1["has_analyze_visual"]:
                log("---- Real Vision Supplement (documented: planner 不产出 vision 任务, 计划补充 analyze_visual) ----")
                analyzer2 = _CountingAnalyzer(ImageAnalysis())
                s2, ex2 = run_vision_supplement(s1["validated_plan"], analyzer2, counter)
                log(f"supplement_plan_valid = {s2['plan_valid']}")
                if s2["plan_valid"]:
                    log(f"plan_tasks = {s2['plan_tasks']}")
                    log(f"vision_task_status = {s2['vision_task_status']}")
                    vs = s2["vision_summary"]
                    if vs:
                        log(f"vision_total = {vs.get('total')} success = {vs.get('success_count')} "
                            f"failed = {vs.get('failed_count')} systemic = {vs.get('systemic_failed')}")
                    log(f"vision_analyzer_calls = {s2['vision_analyzer_calls']}")
                    log(f"synth_status = {s2['synth_status']}")
                    log(f"validated_image_ids = {s2['validated_image_ids']}")
                    log(f"vision_evidence_count = {s2['vision_evidence_count']}")
                    log(f"supplement_latency_s = {s2['total_latency_s']}")
                    log("real_vision_sample:")
                    for it in s2["vision_sample"][:3]:
                        log(f"  {it}")
            else:
                s2 = None

        log("---- total LLM calls (Planner + Vision + Synthesizer) ----")
        log(f"client.chat.completions.create calls = {counter.n}")
        vision_calls = (analyzer.calls + (s2 and analyzer2.calls or 0)) if s2 else analyzer.calls
        log(f"vision analyzer calls = {vision_calls}")

        log("---- Controlled Failure Injection (FAKE) ----")
        fail_summary = run_failure_injection()
        log(f"vision_status={fail_summary['vision_status']} "
            f"success={fail_summary['vision_success']} failed={fail_summary['vision_failed']} "
            f"total={fail_summary['vision_total']}")
        log(f"synth_status={fail_summary['synth_status']} ids={fail_summary['synth_ids']}")
    finally:
        counter.restore()

    log("---- Rollback / Config Check ----")
    cfg_after = settings.get_config()
    log(f"config VISION_MODEL after E2E = {getattr(cfg_after, 'VISION_MODEL', None)!r}（未修改生产配置）")
    log(f"vision_available() after restore = {models_mod.vision_available()}")

    # 干净 UTF-8 结果记录（python 侧写文件，避免终端重定向编码损坏）
    try:
        out_path = os.path.join(_REPO_ROOT, "stage9_e2e_result.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(_OUT))
        log(f"result written to {out_path}")
    except Exception as e:
        log(f"result file write failed: {e}")
    return _OUT


if __name__ == "__main__":
    for line in main():
        pass
    print("\n[E2E] done")

