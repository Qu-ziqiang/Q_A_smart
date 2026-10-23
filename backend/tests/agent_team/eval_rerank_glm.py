"""R2 Evaluation Runner（手动/显式运行，非 CI；默认只出计划与成本估算，不调 API）。

目标对比（同一 candidate pool、同 Query、同 DB、同 Retrieval 参数，唯一变化=Rerank 策略）：
    Baseline = CLIP Vector Retrieval（tool_search_images 原始顺序）
    R1       = Baseline + deterministic rerank（rerank_candidates）
    R2       = Baseline + GLM multimodal rerank（MultimodalReranker，openai_chat adapter）

严格复用同一候选池：R1/R2 都只重排 Baseline 返回的同一批对象，不改数量/集合。
R2 会逐张调用真实 GLM API，成本 = N_queries × N_candidates。

用法（repo 根）：
    # 只打印计划与预计 API 调用次数/耗时/成本量级（0 次真实调用）
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/eval_rerank_glm.py --queries 3 --k 6

    # 配置好 MM_RERANK_* 后真正执行小规模评估
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/eval_rerank_glm.py --queries 3 --k 6 --yes
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.tools import tool_search_images
from backend.ai_func.recommendation.orchestration.model_reranker import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PROVIDER,
    build_multimodal_reranker,
)
from backend.ai_func.recommendation.orchestration.rerank import rerank_candidates
from backend.ai_func.recommendation.orchestration.workers import RetrievalWorker

DEFAULT_QUERIES = [
    "海边日落的照片",
    "一辆行驶在公路上的汽车",
    "一只猫或狗的照片",
]
AVG_SECONDS_PER_CALL = 1.4  # 基于真实 smoke 观察（glm-4v-flash，单图逐张）


def _env(k):
    return (os.environ.get(k) or "").strip()


def _ids(cands):
    return [c.id for c in cands]


def _retrieve_pool(query, k):
    """与生产 RetrievalWorker 完全一致的检索路径（RetrievalRepository/sqlite-vec 未改动）。"""
    rows = tool_search_images(query=query, targets=["title", "description", "image"],
                              filters={}, limit=k) or []
    return rows, [RetrievalWorker._to_candidate(r) for r in rows]


def _format_top(k, ids):
    return ", ".join(str(i) for i in ids[:k])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=3)
    ap.add_argument("--k", type=int, default=6, help="每个 query 的候选数（R2 API 调用数/query）")
    ap.add_argument("--yes", action="store_true", help="确认执行真实 R2 API 调用")
    args = ap.parse_args(argv)

    queries = DEFAULT_QUERIES[: args.queries]
    total_calls = args.queries * args.k
    est_seconds = total_calls * AVG_SECONDS_PER_CALL
    print("== R2 Evaluation Plan ==")
    print(f"queries={len(queries)} k={args.k} → 预计真实 API 调用次数 = {args.queries} × {args.k} = "
          f"{total_calls} 次")
    print(f"预计顺序耗时（单图平均 {AVG_SECONDS_PER_CALL}s，无并发）≈ {est_seconds:.0f}s")
    print("成本量级：glm-4v-flash 当前免费档可用；glm-4v-plus 等按智谱 token 计价，"
          "正式评估前请以智谱控制台为准。")
    for q in queries:
        print(f"  - {q!r}")
    if not args.yes:
        print("\n未传 --yes，未发起任何真实 API 调用。确认成本后加 --yes 执行。")
        return 0

    if not (_env(ENV_PROVIDER) and _env(ENV_API_KEY) and _env(ENV_BASE_URL) and _env(ENV_MODEL)):
        print("FAIL: MM_RERANK_PROVIDER/API_KEY/BASE_URL/MODEL 未完整配置。未发起任何 API 调用。")
        return 2
    mm = build_multimodal_reranker()
    if mm is None or mm.name != "openai_chat":
        print(f"FAIL: 期望 openai_chat provider，实际={None if mm is None else mm.name}")
        return 2
    print(f"R2 model={mm.provider._model} endpoint={mm.provider._base_url}")

    print("\n== Run（同 pool、逐张顺序、无并发）==")
    for qi, query in enumerate(queries, 1):
        rows, baseline = _retrieve_pool(query, args.k)
        print(f"\n[{qi}/{len(queries)}] query={query!r} pool_size={len(baseline)}")
        r1 = rerank_candidates(query, list(baseline))
        # R2：逐张真实打分（生产 adapter），任一失败 → 整轮回退 R1（与 MultimodalReranker 语义一致）
        scored, r2_fallback = [], False
        for idx, cand in enumerate(baseline):
            try:
                scored.append((mm.score(query, cand), idx, cand))
            except Exception as e:
                r2_fallback = True
                print(f"  [R2] id={cand.id} score error → {type(e).__name__}: {str(e)[:100]}")
                break
        r2 = [c for _, _, c in sorted(scored, key=lambda t: (-t[0], t[1]))] if not r2_fallback else list(r1)
        assert set(_ids(r2)) == set(_ids(baseline)) and len(r2) == len(baseline), "R2 破坏 pool contract"
        print(f"  Baseline top-{args.k}: {_format_top(args.k, _ids(baseline))}")
        print(f"  R1      top-{args.k}: {_format_top(args.k, _ids(r1))}")
        if r2_fallback:
            print(f"  R2      top-{args.k}: {_format_top(args.k, _ids(r2))}  ← 本次整体 fallback 到 R1")
        else:
            print(f"  R2      top-{args.k}: {_format_top(args.k, _ids(r2))}")
            print("  R2 GLM scores: "
                  + ", ".join(f"{cid}:{_round(score)}" for score, _, cand in scored))
    print("\n注意：本 runner 只报告排序与 top-K 差异，不做正确性结论。"
          "在人工标注/评测集建立前，禁止宣称任何策略更优。")
    return 0


def _round(x):
    return f"{x:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
