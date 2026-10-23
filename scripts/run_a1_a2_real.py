"""T2：A1/A2 真 GLM + 分阶段延迟。无 key 则写 blocked，不调 API。

质量只报旧 66 图 / 18 query 池内指标，不报万级 Hit@1。
mock 80ms/503ms 不写。
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

from scripts._load_mm_rerank_env import key_prefix, load_mm_rerank_env
from scripts._load_deepseek_env import load_deepseek_env

from backend.ai_func.recommendation.evaluation.latency import StageTimer, format_summary
from backend.ai_func.recommendation.orchestration.model_reranker import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PROVIDER,
    RelevanceError,
    build_multimodal_reranker,
    parse_rank_json,
)
from backend.ai_func.recommendation.orchestration.opt_flags import (
    STATS,
    clip_top_gap,
    reset_opt_stats,
    should_trigger_mm_rerank,
)
from backend.ai_func.recommendation.orchestration.rerank import rerank_candidates

_AGENT = os.path.join(_REPO, "backend", "tests", "agent_team")
if _AGENT not in sys.path:
    sys.path.insert(0, _AGENT)

from eval_r2_gold import GOLD, fetch_pool, rank_metrics  # noqa: E402


_OUT_DIR = os.path.join(
    _REPO, "backend", "ai_func", "recommendation",
    "evaluation", "results", "a1_a2_real",
)
_SIMPLE = [
    "找红色花朵照片",
    "找几张海边日落照片",
    "推荐几张可爱的猫照片",
    "找城市夜景照片",
    "找黑白风格人像",
    "找赛博朋克风格街景",
    "找清新日系风格照片",
    "找海滩度假照片",
]


def _pct(xs: List[float], p: float) -> float:
    from backend.ai_func.recommendation.evaluation.latency import percentile
    return round(percentile(xs, p), 1) if xs else 0.0


def _write(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_gold_queries() -> List[Dict[str, Any]]:
    with open(GOLD, encoding="utf-8") as f:
        return json.load(f)["queries"]


def _ids(pool, cands) -> List[str]:
    id_of = {c.id: k for k, _, c in pool}
    return [id_of[c.id] for c in cands]


def main() -> int:
    os.makedirs(_OUT_DIR, exist_ok=True)
    ready = load_mm_rerank_env()
    blocked_path = os.path.join(_OUT_DIR, "blocked.json")
    if not ready:
        payload = {
            "blocked": True,
            "reason": "MM_RERANK_* 不齐，T2 阻塞。未发起 GLM 调用。",
            "have_provider": bool(os.environ.get(ENV_PROVIDER)),
            "have_key": bool(os.environ.get(ENV_API_KEY)),
            "have_base": bool(os.environ.get(ENV_BASE_URL)),
            "have_model": bool(os.environ.get(ENV_MODEL)),
        }
        _write(blocked_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(
        f"[t2] glm ready prefix={key_prefix()} model={os.environ.get(ENV_MODEL)}",
        flush=True,
    )

    queries = _load_gold_queries()
    pools = []
    for q in queries:
        try:
            pool = fetch_pool(q["query"])
        except SystemExit as e:
            _write(os.path.join(_OUT_DIR, "blocked.json"), {
                "blocked": True,
                "reason": f"旧 18-query 池拉取失败: {e}",
            })
            print(f"[t2] pool fail: {e}", flush=True)
            return 0
        pools.append((q, pool))

    # ----- A1 sequential -----
    os.environ.pop("SM_RERANK_BATCH", None)
    reset_opt_stats()
    mm_seq = build_multimodal_reranker()
    seq_ms: List[float] = []
    seq_parse_fail = 0
    seq_fallback = 0
    seq_rows = []
    timer = StageTimer()

    orig_rel = mm_seq.provider.relevance

    def rel_wrap(query, image):
        try:
            return orig_rel(query, image)
        except RelevanceError:
            nonlocal seq_parse_fail
            seq_parse_fail += 1
            raise

    mm_seq.provider.relevance = rel_wrap

    for q, pool in pools:
        cands = [c for _, _, c in pool]
        gt = set(q["gt_unique"])
        with timer.span("r2_sequential"):
            t0 = time.perf_counter()
            try:
                out = mm_seq.rerank(q["query"], list(cands))
            except Exception:
                seq_fallback += 1
                out = list(cands)
            seq_ms.append((time.perf_counter() - t0) * 1000.0)
        ids = _ids(pool, out)
        seq_rows.append({
            "query": q["query"],
            "class": q["class"],
            "metrics": rank_metrics(ids, gt),
            "ms": round(seq_ms[-1], 1),
        })

    # ----- A1 batch -----
    os.environ["SM_RERANK_BATCH"] = "1"
    reset_opt_stats()
    mm_batch = build_multimodal_reranker()
    batch_ms: List[float] = []
    batch_json_fail = 0
    orig_chat = getattr(mm_batch.provider, "_chat_batch", None)
    if orig_chat is not None:
        def chat_wrap(query, candidates):
            raw = orig_chat(query, candidates)
            try:
                parse_rank_json(raw, len(candidates))
            except Exception:
                nonlocal batch_json_fail
                batch_json_fail += 1
            return raw

        mm_batch.provider._chat_batch = chat_wrap

    batch_rows = []
    for q, pool in pools:
        cands = [c for _, _, c in pool]
        gt = set(q["gt_unique"])
        with timer.span("r2_batch"):
            t0 = time.perf_counter()
            out = mm_batch.rerank(q["query"], list(cands))
            batch_ms.append((time.perf_counter() - t0) * 1000.0)
        ids = _ids(pool, out)
        batch_rows.append({
            "query": q["query"],
            "class": q["class"],
            "metrics": rank_metrics(ids, gt),
            "ms": round(batch_ms[-1], 1),
            "fallback_r1": STATS.batch_fallback > 0,
        })

    # ----- A2 trigger on gold + simple -----
    os.environ["SM_RERANK_GAP_TRIGGER"] = "1"
    gold_trig = []
    extra_ms = []
    for q, pool in pools:
        cands = [c for _, _, c in pool]
        gap = clip_top_gap(cands)
        fire = should_trigger_mm_rerank(cands)
        gold_trig.append({"query": q["query"], "gap": gap, "trigger": fire})
        if fire:
            extra_ms.append(seq_ms[len(extra_ms)] if extra_ms else seq_ms[0] if seq_ms else 0)

    simple_trig = []
    from backend.ai_func.recommendation.orchestration.workers import RetrievalWorker
    from backend.ai_func.recommendation.tools import tool_search_images

    for q in _SIMPLE:
        rows = tool_search_images(query=q, targets=["title", "description", "image"],
                                  filters={}, limit=10) or []
        cands = [RetrievalWorker._to_candidate(r) for r in rows]
        gap = clip_top_gap(cands)
        fire = should_trigger_mm_rerank(cands)
        simple_trig.append({"query": q, "gap": gap, "trigger": fire, "n": len(cands)})

    gold_rate = (sum(1 for r in gold_trig if r["trigger"]) / len(gold_trig)) if gold_trig else 0
    simple_fp = (sum(1 for r in simple_trig if r["trigger"]) / len(simple_trig)) if simple_trig else 0

    # ----- Synth 真打（DeepSeek Synthesizer，小样本）-----
    synth_ok = False
    try:
        load_deepseek_env()
        from backend.ai_func.recommendation.orchestration.models import (
            SynthesizeInput,
            Task,
            TaskType,
        )
        from backend.ai_func.recommendation.orchestration.synthesizer import SynthesizerAgent
        from backend.ai_func.recommendation.orchestration.workers import (
            RetrievalOutput,
            WorkerResult,
        )
        from backend.ai_func.recommendation.orchestration.models import TaskStatus

        class _Ctx:
            def __init__(self, goal, cands):
                self.goal = goal
                self.plan = None
                self.results = {
                    "retrieve_1": WorkerResult(
                        task_id="retrieve_1",
                        task_type=TaskType.RETRIEVE_IMAGES,
                        status=TaskStatus.SUCCEEDED,
                        output=RetrievalOutput(candidates=cands, query=goal, total=len(cands)),
                    )
                }

        agent = SynthesizerAgent()
        for q, pool in pools[:8]:
            cands = [c for _, _, c in pool][:8]
            task = Task(
                task_id="synth_1",
                task_type=TaskType.SYNTHESIZE,
                objective="推荐",
                input=SynthesizeInput(recommendation_limit=5),
                depends_on=["retrieve_1"],
            )
            with timer.span("synthesize"):
                agent.run(task, _Ctx(q["query"], cands))
        synth_ok = True
    except Exception as e:
        timer.record("synthesize_error", 0.0)
        print(f"[t2] synth skip: {type(e).__name__}: {str(e)[:120]}", flush=True)

    def _macro(rows):
        n = len(rows) or 1
        keys = ("hit1", "hit3", "hit5", "mrr")
        return {k: round(sum(r["metrics"][k] for r in rows) / n, 4) for k in keys}

    report = {
        "blocked": False,
        "model": os.environ.get(ENV_MODEL),
        "n_queries": len(pools),
        "note": "池内质量，不是万级 Hit@1。resume 不写 mock 80/503ms。",
        "a1": {
            "sequential": {
                "p50_ms": _pct(seq_ms, 50),
                "p95_ms": _pct(seq_ms, 95),
                "mean_ms": round(sum(seq_ms) / len(seq_ms), 1) if seq_ms else 0,
                "json_or_relevance_fail": seq_parse_fail,
                "fallback_r1_queries": seq_fallback,
                "macro_pool": _macro(seq_rows),
            },
            "batch": {
                "p50_ms": _pct(batch_ms, 50),
                "p95_ms": _pct(batch_ms, 95),
                "mean_ms": round(sum(batch_ms) / len(batch_ms), 1) if batch_ms else 0,
                "json_parse_fail": batch_json_fail,
                "batch_ok": STATS.batch_ok,
                "batch_fallback": STATS.batch_fallback,
                "macro_pool": _macro(batch_rows),
            },
        },
        "a2": {
            "gold_trigger_rate": round(gold_rate, 4),
            "gold_trigger_target": "0.20-0.40",
            "simple_false_trigger_rate": round(simple_fp, 4),
            "mean_extra_latency_ms": round(sum(seq_ms) / len(seq_ms), 1) if seq_ms else 0,
            "gold": gold_trig,
            "simple": simple_trig,
        },
        "stage_latency": timer.summary(),
        "p95_2_7s_verdict": None,
        "synth_real": synth_ok,
        "quality_guard": {
            "seq_pool_hit1": _macro(seq_rows).get("hit1"),
            "batch_pool_hit1": _macro(batch_rows).get("hit1"),
            "note": "只作旧 18-query 防退化，不进万级 Hit@1 简历。",
        },
        "rows": {"sequential": seq_rows, "batch": batch_rows},
    }
    stages = timer.summary()
    p95s = [s.get("p95_ms", 0) for s in stages.values()]
    e2e_guess = (stages.get("planner") or {}).get("p95_ms") or 0
    e2e_guess += (stages.get("r2_sequential") or {}).get("p95_ms") or 0
    e2e_guess += (stages.get("synthesize") or {}).get("p95_ms") or 0
    report["p95_2_7s_verdict"] = {
        "sum_p95_planner_r2seq_synth_ms": round(e2e_guess, 1),
        "can_write_2_7s": False,
        "reason": "2.7s 是历史 e2e 口径；此处给分阶段真值。仅当本文件有 planner+r2+synth 真 P95 才裁决。",
        "stages": stages,
        "max_stage_p95_ms": max(p95s) if p95s else 0,
    }
    path = os.path.join(_OUT_DIR, "a1_a2_real.json")
    _write(path, report)
    md = os.path.join(_OUT_DIR, "a1_a2_real.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# A1/A2 真 GLM\n\n")
        f.write(f"- sequential p50/p95: {report['a1']['sequential']['p50_ms']}/{report['a1']['sequential']['p95_ms']} ms\n")
        f.write(f"- batch p50/p95: {report['a1']['batch']['p50_ms']}/{report['a1']['batch']['p95_ms']} ms\n")
        f.write(f"- A2 gold trigger: {report['a2']['gold_trigger_rate']}\n")
        f.write(f"- A2 simple FP: {report['a2']['simple_false_trigger_rate']}\n\n")
        f.write(format_summary(stages))
        f.write("\n")
    print(json.dumps({
        "out": path,
        "a1_seq_p95": report["a1"]["sequential"]["p95_ms"],
        "a1_batch_p95": report["a1"]["batch"]["p95_ms"],
        "a2_trigger": report["a2"]["gold_trigger_rate"],
        "simple_fp": report["a2"]["simple_false_trigger_rate"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
