#!/usr/bin/env python
"""A2：CLIP top1−top5 分差 < τ 才走 GLM-4V 重排。400 张子集评测。

阶段：
  1) CLIP 探针（无 VLM）：gap 分布、Hit@1/5、触发率、简单 query 误触发
  2) 真 GLM-4V batch：Baseline vs 全量 R2 vs A2 分差触发

检索限制在 gold.gallery_ids（约 400 张）。不覆盖旧 a1_a2_real/ 与冻结 18-query。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts._load_mm_rerank_env import key_prefix, load_mm_rerank_env

from backend.ai_func.recommendation.orchestration.model_reranker import (
    ENV_MODEL,
    build_multimodal_reranker,
)
from backend.ai_func.recommendation.orchestration.opt_flags import (
    STATS,
    clip_top_gap,
    reset_opt_stats,
    should_trigger_mm_rerank,
)
from backend.ai_func.generate_vector import encode_text
from backend.ai_func.recommendation.orchestration.rerank import rerank_candidates
from backend.ai_func.recommendation.orchestration.workers import RetrievalWorker
from backend.db_func.core.connection import get_db_connection
from backend.db_func.repositories.search import SearchRepository
from backend.db_func.utils.common import json_from_db_to_python

_AGENT = os.path.join(_REPO, "backend", "tests", "agent_team")
if _AGENT not in sys.path:
    sys.path.insert(0, _AGENT)

from eval_r2_gold import FETCH_LIMIT, POOL_SIZE, content_id, rank_metrics  # noqa: E402

GOLD = os.path.join(
    _REPO, "backend", "ai_func", "recommendation",
    "evaluation", "datasets", "gold_voc_400_rerank.json",
)
OUT_DIR = os.path.join(
    _REPO, "backend", "ai_func", "recommendation",
    "evaluation", "results", "a2_gap_400",
)
_GALLERY_IDS: List[int] = []
_SUBSET: Optional["SubsetIndex"] = None
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
_TAU_SWEEP = (0.04, 0.06, 0.08, 0.10, 0.12)


def _write(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return round(ys[k], 1)


def _macro(rows: List[dict], key: str = "baseline") -> Dict[str, float]:
    usable = [r for r in rows if r.get("gt_in_pool", 0) > 0]
    n = len(usable) or 1
    out = {}
    for m in ("hit1", "hit3", "hit5", "mrr", "p5"):
        out[m] = round(sum(r[key][m] for r in usable) / n, 4)
    out["n"] = len(usable)
    return out


def _ids(pool, cands) -> List[str]:
    id_of = {c.id: k for k, _, c in pool}
    return [id_of[c.id] for c in cands if c.id in id_of]


def _parse_vec(raw: Any):
    import numpy as np
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=np.float32)
    if hasattr(raw, "tolist"):
        try:
            return np.asarray(raw.tolist(), dtype=np.float32)
        except Exception:
            pass
    s = str(raw).strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    return np.fromstring(s, sep=",", dtype=np.float32)


class SubsetIndex:
    """400 子集上精确三路 CLIP（不用 HNSW+ANY，避免 400/10k 近邻漏检）。"""

    def __init__(self, image_ids: List[int]):
        import numpy as np
        self.ids = [int(x) for x in image_ids]
        self.mats: Dict[str, Any] = {}
        self.ok: Dict[str, Any] = {}
        with get_db_connection() as conn:
            cur = conn.cursor()
            for table, key in (
                ("image_vectors", "image"),
                ("title_vectors", "title"),
                ("description_vectors", "description"),
            ):
                cur.execute(
                    f"SELECT image_id, embedding FROM {table} WHERE image_id = ANY(%s)",
                    (self.ids,),
                )
                by_id = {}
                for r in cur.fetchall():
                    rec = dict(r)
                    vec = _parse_vec(rec["embedding"])
                    if vec is None or vec.size == 0:
                        continue
                    by_id[int(rec["image_id"])] = vec
                rows = []
                mask = []
                dim = 512
                for iid in self.ids:
                    v = by_id.get(iid)
                    if v is None:
                        rows.append(np.zeros(dim, dtype=np.float32))
                        mask.append(False)
                    else:
                        dim = int(v.size)
                        n = float(np.linalg.norm(v)) or 1.0
                        rows.append((v / n).astype(np.float32))
                        mask.append(True)
                self.mats[key] = np.stack(rows, axis=0)
                self.ok[key] = np.array(mask, dtype=bool)
                print(f"[subset] {key} vectors {int(self.ok[key].sum())}/{len(self.ids)}", flush=True)
        repo = SearchRepository()
        images = repo._get_images_by_ids(self.ids)
        images = [json_from_db_to_python(dict(x)) if not isinstance(x, dict) else json_from_db_to_python(x) for x in images]
        self.images = {int(img["id"]): img for img in images}

    def search(self, query: str, limit: int) -> List[dict]:
        import numpy as np
        q = encode_text(query)
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(q)) or 1.0
        q = q / n
        acc = np.zeros(len(self.ids), dtype=np.float32)
        wsum = np.zeros(len(self.ids), dtype=np.float32)
        for key, mat in self.mats.items():
            sim = mat @ q
            ok = self.ok[key]
            acc = acc + np.where(ok, sim, 0.0)
            wsum = wsum + ok.astype(np.float32)
        scores = np.divide(acc, np.maximum(wsum, 1e-6))
        order = np.argsort(-scores)[:limit]
        out = []
        for idx in order:
            iid = self.ids[int(idx)]
            img = dict(self.images.get(iid) or {})
            if not img:
                continue
            img["score"] = float(scores[int(idx)])
            out.append(img)
        return SearchRepository()._attach_public_url(out)


def fetch_pool(query: str) -> List[Tuple[str, dict, Any]]:
    if _SUBSET is None:
        raise RuntimeError("subset index 未初始化")
    rows = _SUBSET.search(query, FETCH_LIMIT) or []
    cands = [RetrievalWorker._to_candidate(r) for r in rows]
    seen, out = set(), []
    for row, cand in zip(rows, cands):
        key = content_id(row.get("filename") or "")
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append((key, row, cand))
    pool = out[:POOL_SIZE]
    if len(pool) != POOL_SIZE:
        raise RuntimeError(
            f"pool<{POOL_SIZE} query={query!r} rows={len(rows)} unique={len(out)}"
        )
    return pool


def _gap_trigger(cands, tau: float) -> Tuple[Optional[float], bool]:
    gap = clip_top_gap(cands)
    os.environ["SM_RERANK_GAP_TRIGGER"] = "1"
    os.environ["SM_RERANK_GAP_TAU"] = str(tau)
    fire = should_trigger_mm_rerank(cands, tau=tau)
    return gap, fire


def probe_clip(queries: List[dict], tau: float) -> Dict[str, Any]:
    rows = []
    skipped = []
    for i, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        try:
            pool = fetch_pool(q["query"])
        except Exception as e:
            skipped.append({"query": q["query"], "error": str(e)[:160]})
            print(f"[clip {i}/{len(queries)}] SKIP {q['query']}: {e}", flush=True)
            continue
        cands = [c for _, _, c in pool]
        base_ids = [k for k, _, _ in pool]
        gt = set(q["gt_unique"])
        gt_in = [x for x in base_ids if x in gt]
        gap, fire = _gap_trigger(cands, tau)
        b = rank_metrics(base_ids, gt)
        rec = {
            "id": q.get("id"),
            "query": q["query"],
            "kind": q.get("kind"),
            "class": q.get("class"),
            "gt_total": q.get("gt_total"),
            "gt_in_pool": len(gt_in),
            "gap": None if gap is None else round(float(gap), 6),
            "trigger_tau": fire,
            "baseline": b,
            "baseline_ids": base_ids,
            "retrieval_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "pool": [(k, c.id) for k, _, c in pool],
        }
        rows.append(rec)
        print(
            f"[clip {i}/{len(queries)}] {q['kind']:6s} hit1={b['hit1']} hit5={b['hit5']} "
            f"gap={rec['gap']} fire={int(fire)} {q['query']}",
            flush=True,
        )

    def rate(kind: Optional[str] = None) -> float:
        xs = [r for r in rows if kind is None or r["kind"] == kind]
        if not xs:
            return 0.0
        return round(sum(1 for r in xs if r["trigger_tau"]) / len(xs), 4)

    sweep = {}
    for t in _TAU_SWEEP:
        fires = []
        for r in rows:
            cands = []  # reconstruct from stored gap only
            gap = r["gap"]
            fire = True if gap is None else (gap < t)
            fires.append(fire)
            r.setdefault("sweep", {})[str(t)] = fire
        sweep[str(t)] = round(sum(fires) / len(fires), 4) if fires else 0.0

    simple = []
    os.environ["SM_RERANK_GAP_TRIGGER"] = "1"
    os.environ["SM_RERANK_GAP_TAU"] = str(tau)
    for q in _SIMPLE:
        srows = _SUBSET.search(q, 10) if _SUBSET is not None else []
        cands = [RetrievalWorker._to_candidate(r) for r in srows]
        gap = clip_top_gap(cands)
        fire = should_trigger_mm_rerank(cands, tau=tau)
        simple.append({
            "query": q,
            "gap": None if gap is None else round(float(gap), 6),
            "trigger": fire,
            "n": len(cands),
        })

    return {
        "tau": tau,
        "n_ok": len(rows),
        "n_skip": len(skipped),
        "skipped": skipped,
        "trigger_rate_all": rate(),
        "trigger_rate_single": rate("single"),
        "trigger_rate_pair": rate("pair"),
        "tau_sweep_trigger_rate": sweep,
        "macro_baseline": _macro(rows, "baseline"),
        "macro_baseline_triggered": _macro(
            [r for r in rows if r["trigger_tau"]], "baseline"
        ),
        "macro_baseline_skipped": _macro(
            [r for r in rows if not r["trigger_tau"]], "baseline"
        ),
        "simple_false_trigger_rate": round(
            sum(1 for r in simple if r["trigger"]) / len(simple), 4
        ) if simple else 0.0,
        "simple": simple,
        "rows": rows,
    }


def _rerank_batch(mm, query: str, pool) -> Tuple[List[str], float, bool]:
    cands = [c for _, _, c in pool]
    t0 = time.perf_counter()
    fb = STATS.batch_fallback
    out = mm.rerank(query, list(cands), fallback=rerank_candidates)
    ms = (time.perf_counter() - t0) * 1000.0
    fallback = STATS.batch_fallback > fb
    return _ids(pool, out), ms, fallback


def run_glm(clip: Dict[str, Any], gold_by_q: Dict[str, dict], tau: float) -> Dict[str, Any]:
    # glm-4v-flash 多图 batch 会 1210「输入图片数量超过限制」，评测走逐张。
    os.environ.pop("SM_RERANK_BATCH", None)
    os.environ.pop("SM_RERANK_GAP_TRIGGER", None)
    reset_opt_stats()
    mm = build_multimodal_reranker()
    if mm is None or mm.name != "openai_chat":
        raise SystemExit("GLM reranker 未就绪")
    if getattr(mm.provider, "_reason", ""):
        raise SystemExit(f"provider unavailable: {mm.provider._reason}")

    cache_path = os.path.join(OUT_DIR, "glm_cache.json")
    cache: Dict[str, Any] = {}
    if os.path.isfile(cache_path):
        cache = json.loads(open(cache_path, encoding="utf-8").read())

    rows_out = []
    n = len(clip["rows"])
    for i, r in enumerate(clip["rows"], 1):
        qtext = r["query"]
        q = gold_by_q[qtext]
        pool_keys = [k for k, _cid in r["pool"]]
        # 重新拉池，保证 candidate 对象带图路径
        pool = fetch_pool(qtext)
        got = [k for k, _, _ in pool]
        if got != pool_keys:
            print(f"[glm {i}/{n}] warn pool drift {qtext}", flush=True)
        gt = set(q["gt_unique"])
        cands = [c for _, _, c in pool]
        fire = bool(r["trigger_tau"])
        gap = r["gap"]

        if qtext in cache and not cache[qtext].get("fallback", False):
            r2_ids = cache[qtext]["r2_ids"]
            r2_ms = cache[qtext]["r2_ms"]
            fb = False
        else:
            r2_ids, r2_ms, fb = _rerank_batch(mm, qtext, pool)
            if not fb:
                cache[qtext] = {"r2_ids": r2_ids, "r2_ms": round(r2_ms, 1), "fallback": False}
                _write(cache_path, cache)

        a2_ids = r2_ids if fire else [k for k, _, _ in pool]
        rec = {
            "id": r.get("id"),
            "query": qtext,
            "kind": r.get("kind"),
            "class": r.get("class"),
            "gt_in_pool": r["gt_in_pool"],
            "gap": gap,
            "trigger": fire,
            "baseline": r["baseline"],
            "always_r2": rank_metrics(r2_ids, gt),
            "a2": rank_metrics(a2_ids, gt),
            "r2_ms": round(r2_ms, 1),
            "a2_ms": round(r2_ms, 1) if fire else 0.0,
            "fallback": fb,
        }
        rows_out.append(rec)
        print(
            f"[glm {i}/{n}] fire={int(fire)} "
            f"B={rec['baseline']['hit1']}/{rec['baseline']['mrr']:.2f} "
            f"R2={rec['always_r2']['hit1']}/{rec['always_r2']['mrr']:.2f} "
            f"A2={rec['a2']['hit1']}/{rec['a2']['mrr']:.2f} "
            f"{r2_ms:.0f}ms {qtext}",
            flush=True,
        )

    triggered = [x for x in rows_out if x["trigger"]]
    skipped = [x for x in rows_out if not x["trigger"]]
    always_ms = [x["r2_ms"] for x in rows_out]
    a2_ms = [x["a2_ms"] for x in rows_out]
    return {
        "model": os.environ.get(ENV_MODEL),
        "tau": tau,
        "n": len(rows_out),
        "trigger_rate": round(len(triggered) / len(rows_out), 4) if rows_out else 0.0,
        "fallback_n": sum(1 for x in rows_out if x["fallback"]),
        "macro": {
            "baseline": _macro(rows_out, "baseline"),
            "always_r2": _macro(rows_out, "always_r2"),
            "a2": _macro(rows_out, "a2"),
        },
        "macro_triggered_only": {
            "n": len(triggered),
            "baseline": _macro(triggered, "baseline"),
            "always_r2": _macro(triggered, "always_r2"),
            "a2": _macro(triggered, "a2"),
        },
        "macro_skipped_only": {
            "n": len(skipped),
            "baseline": _macro(skipped, "baseline"),
            "always_r2": _macro(skipped, "always_r2"),
            "a2": _macro(skipped, "a2"),
        },
        "latency": {
            "always_r2_mean_ms": round(sum(always_ms) / len(always_ms), 1) if always_ms else 0,
            "always_r2_p50_ms": _pct(always_ms, 50),
            "always_r2_p95_ms": _pct(always_ms, 95),
            "a2_mean_ms": round(sum(a2_ms) / len(a2_ms), 1) if a2_ms else 0,
            "a2_p50_ms": _pct(a2_ms, 50),
            "a2_p95_ms": _pct(a2_ms, 95),
            "note": "A2 未触发记 0ms（保持 CLIP 序）；触发则逐张 GLM-4V。glm-4v-flash 不支持 10 图 batch。",
        },
        "rows": rows_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-only", action="store_true")
    ap.add_argument("--tau", type=float, default=0.08)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.chdir(_REPO)

    gold = json.loads(open(GOLD, encoding="utf-8").read())
    queries = gold["queries"]
    global _GALLERY_IDS, _SUBSET
    _GALLERY_IDS = [int(x) for x in (gold.get("gallery_ids") or [])]
    _SUBSET = SubsetIndex(_GALLERY_IDS)
    print(
        f"[a2] gold n={len(queries)} subset={gold.get('gallery_size')} "
        f"ids={len(_GALLERY_IDS)} tau={args.tau}",
        flush=True,
    )

    probe_path = os.path.join(OUT_DIR, "clip_probe.json")
    clip = probe_clip(queries, args.tau)
    clip["gold_file"] = GOLD
    clip["gallery_size"] = gold.get("gallery_size")
    _write(probe_path, clip)
    print(json.dumps({
        "clip": probe_path,
        "n_ok": clip["n_ok"],
        "trigger": clip["trigger_rate_all"],
        "single": clip["trigger_rate_single"],
        "pair": clip["trigger_rate_pair"],
        "hit1": clip["macro_baseline"].get("hit1"),
        "simple_fp": clip["simple_false_trigger_rate"],
        "tau_sweep": clip["tau_sweep_trigger_rate"],
    }, ensure_ascii=False, indent=2), flush=True)

    if args.clip_only:
        return 0

    ready = load_mm_rerank_env()
    if not ready:
        blocked = {
            "blocked": True,
            "reason": "MM_RERANK_* 仍不齐，CLIP 探针已完成，未打 GLM。",
            "clip": probe_path,
        }
        _write(os.path.join(OUT_DIR, "blocked.json"), blocked)
        print(json.dumps(blocked, ensure_ascii=False, indent=2), flush=True)
        return 0

    print(f"[a2] glm ready prefix={key_prefix()} model={os.environ.get(ENV_MODEL)}", flush=True)
    gold_by_q = {q["query"]: q for q in queries}
    glm = run_glm(clip, gold_by_q, args.tau)
    glm["clip_probe"] = probe_path
    glm["simple_false_trigger_rate"] = clip["simple_false_trigger_rate"]
    glm["tau_sweep_trigger_rate"] = clip["tau_sweep_trigger_rate"]
    glm_path = os.path.join(OUT_DIR, "a2_eval.json")
    _write(glm_path, glm)

    md = os.path.join(OUT_DIR, "a2_eval.md")
    m = glm["macro"]
    with open(md, "w", encoding="utf-8") as f:
        f.write("# A2 分差触发评测（400 张子集）\n\n")
        f.write(f"- n={glm['n']}  tau={glm['tau']}  model={glm['model']}\n")
        f.write(f"- 触发率 {glm['trigger_rate']}（目标参考 20–40%）\n")
        f.write(f"- 简单 query 误触发 {glm['simple_false_trigger_rate']}\n")
        f.write(f"- Hit@1 baseline {m['baseline']['hit1']} → always R2 {m['always_r2']['hit1']} → A2 {m['a2']['hit1']}\n")
        f.write(f"- MRR baseline {m['baseline']['mrr']} → always R2 {m['always_r2']['mrr']} → A2 {m['a2']['mrr']}\n")
        f.write(f"- 触发子集 n={glm['macro_triggered_only']['n']} "
                f"Hit@1 {glm['macro_triggered_only']['baseline']['hit1']} → "
                f"{glm['macro_triggered_only']['always_r2']['hit1']}\n")
        f.write(f"- always R2 p50/p95 {glm['latency']['always_r2_p50_ms']}/{glm['latency']['always_r2_p95_ms']} ms\n")
        f.write(f"- A2 均摊 p50/p95 {glm['latency']['a2_p50_ms']}/{glm['latency']['a2_p95_ms']} ms\n")
    print("saved", glm_path, md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
