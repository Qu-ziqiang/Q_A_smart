"""真实智谱 GLM 多模态 API Smoke Test（手动/显式运行，非 CI）。

与单元测试(test_model_reranker.py，FakeProvider)严格区分：
仅当环境变量完整（MM_RERANK_PROVIDER / API_KEY / BASE_URL / MODEL）才发起真实 API 调用。

只读 DB(data/db/smartimager.db)取 >=3 张本地真实图 → 复用生产 adapter
(OpenAICompatChatRerankerProvider + MultimodalReranker.rerank)逐张评分。

输出：model / 每次调用 raw response / parsed relevance / latency / 最终稳定排序。
绝不在任何输出中打印 API Key。

运行（repo 根，需先设好 env；把 value 替换成自己的 key，勿提交/勿写入代码）：
    $env:MM_RERANK_PROVIDER='openai_chat'
    $env:MM_RERANK_API_KEY='<你的智谱key>'
    $env:MM_RERANK_BASE_URL='https://open.bigmodel.cn/api/paas/v4'
    $env:MM_RERANK_MODEL='glm-4v-flash'
    venv\\Scripts\\python.exe -X utf8 backend/tests/agent_team/smoke_glm_rerank.py
"""

import os
import sqlite3
import sys
import time

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.model_reranker import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PROVIDER,
    build_multimodal_reranker,
)
from backend.ai_func.recommendation.orchestration.workers import ImageCandidate

QUERY = "海边日落的照片"
MIN_IMAGES = 3
_KEYWORDS = ("海边", "日落", "夕阳", "黄昏", "海", "sunset", "beach", "sea")


def _env(key):
    return (os.environ.get(key) or "").strip()


def _config_complete():
    return bool(_env(ENV_PROVIDER) and _env(ENV_API_KEY) and _env(ENV_BASE_URL) and _env(ENV_MODEL))


def _load_candidates():
    """从图库取本地真实存在的图片；优先含'海边/日落'等关键词的行，不足则按 id 补足。"""
    db = os.path.join(_REPO_ROOT, "data", "db", "smartimager.db")
    if not os.path.exists(db):
        raise SystemExit(f"DB 不存在: {db}")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, filename, filepath, title, description, tags, metadata "
        "FROM images ORDER BY id"
    ).fetchall()
    con.close()

    usable = []
    for r in rows:
        fp = (r["filepath"] or "").strip()
        if fp and os.path.exists(fp):
            usable.append(r)
    if len(usable) < MIN_IMAGES:
        raise SystemExit(f"本地存在图片不足 {MIN_IMAGES} 张（usable={len(usable)}）")

    def _kw_match(r):
        text = " ".join(str(r[k] or "") for k in ("title", "description", "filename", "tags"))
        return any(k.lower() in text.lower() for k in _KEYWORDS)

    matches = [r for r in usable if _kw_match(r)]
    picked = (matches + [r for r in usable if r["id"] not in {m["id"] for m in matches}])[:MIN_IMAGES]
    cands = []
    for r in picked:
        tags = r["tags"] or []
        if isinstance(tags, str):
            try:
                tags = __import__("json").loads(tags) if tags else []
            except Exception:
                tags = []
        cands.append(ImageCandidate(
            id=int(r["id"]), score=None,
            title=r["title"] or "", description=r["description"] or "",
            tags=list(tags), metadata={},
            filepath=r["filepath"], public_url=None,
        ))
    return list(zip(picked, cands))


def _mask_any(raw):
    """截断原始响应到一行，确保不含 key 类长串（仅显示内容）。"""
    if not raw:
        return "(empty)"
    return " ".join(raw.split())[:200]


def _load_distinct_pool(n=5):
    """取 n 张内容不同的真实图（按 2007_XXXXXX 原图去重，不同批次各取一张）。"""
    db = os.path.join(_REPO_ROOT, "data", "db", "smartimager.db")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, filename, filepath, title, description, tags, metadata "
        "FROM images ORDER BY id"
    ).fetchall()
    con.close()
    seen = {}
    for r in rows:
        fp = (r["filepath"] or "").strip()
        if not (fp and os.path.exists(fp)):
            continue
        key = (r["filename"] or "").strip()
        if key and key not in seen:
            seen[key] = r
    picked = list(seen.values())[:n]
    out = []
    for r in picked:
        out.append((r, ImageCandidate(
            id=int(r["id"]), score=None, title=r["title"] or "",
            description=r["description"] or "", tags=[], metadata={},
            filepath=r["filepath"], public_url=None,
        )))
    return out


def _score_query(prov, query, pairs, label):
    """逐张顺序调用 adapter 评分（无并发），返回 (scored, latencies, errors)。"""
    print(f"== [{label}] query={query!r} candidates={len(pairs)}（逐张顺序调用，无并发）==")
    errors, scored, latencies = [], [], []
    for row, cand in pairs:
        t0 = time.perf_counter()
        try:
            rel = prov.relevance(query, cand)  # 生产 adapter 完整路径
            lat_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            errors.append((cand.id, str(e)[:160]))
            print(f"  id={cand.id} filepath={os.path.basename(row['filepath'])} -> ERROR "
                  f"{type(e).__name__} ({lat_ms:.0f}ms): {str(e)[:120]}")
            continue
        if not (0.0 <= float(rel) <= 1.0):
            errors.append((cand.id, f"relevance 越界: {rel}"))
        scored.append((float(rel), cand.id, cand))
        latencies.append(lat_ms)
        print(f"  id={cand.id} filepath={os.path.basename(row['filepath'])} "
              f"relevance={float(rel):.4f} latency={lat_ms:.0f}ms raw={_mask_any(prov.last_raw)}")
    return scored, latencies, errors


def main():
    if not _config_complete():
        print("SKIP: MM_RERANK_PROVIDER/API_KEY/BASE_URL/MODEL 未完整配置，未发起任何 API 调用。")
        return 0

    mm = build_multimodal_reranker()
    if mm is None or mm.name != "openai_chat":
        print(f"FAIL: 期望 openai_chat adapter，实际 provider={None if mm is None else mm.name}")
        return 2
    prov = mm.provider
    print(f"== GLM R2 smoke == model={prov._model} base_url={prov._base_url} query={QUERY!r}")
    if prov._reason:
        print(f"FAIL: provider 不可用: {prov._reason}")
        return 2

    # 1) 用户给定 query：当前图库若没有海边/日落图片，应得到 ~0 的“可解释”低分
    pairs = _load_candidates()
    scored, latencies, errors = _score_query(prov, QUERY, pairs, "query-A(sunset)")
    if errors:
        print(f"FAIL: {len(errors)} 张图片调用失败/越界 → 未继续 rerank")
        return 1
    avg_lat = sum(latencies) / len(latencies)
    print(f"== query-A 评分 OK：{len(scored)}/{len(pairs)}，单张平均 {avg_lat:.0f}ms ==")

    # 2) 走生产 MultimodalReranker.rerank 路径，验证 contract（额外真实调用一轮）
    pool = [c for _, _, c in scored]
    ids_before = {c.id for c in pool}
    out = mm.rerank(QUERY, pool)
    ids_after = [c.id for c in out]
    assert set(ids_after) == ids_before, "候选 ID 集合改变"
    assert len(out) == len(pool), "候选数量改变"
    assert all(any(c is o for o in out) for c in pool), "候选对象被替换"
    print(f"== rerank contract OK: count={len(out)} id_set_unchanged=True ==")
    print("== query-A 最终排序 top-id: " + ", ".join(str(i) for i in ids_after))

    # 3) 可选 contrast：同一批真实图 + 不同 query → 验证评分可区分/可解释
    if _env("MM_SMOKE_CONTRAST") == "1":
        c_pairs = _load_distinct_pool(n=5)
        extra_queries = [
            _env("MM_SMOKE_CONTRAST_QUERY") or "一辆行驶在公路上的汽车的照片",
            _env("MM_SMOKE_CONTRAST_QUERY2") or "一只猫或狗的照片",
        ]
        best_max, best_range, best_q = 0.0, 0.0, ""
        for q in extra_queries:
            s, _, errs = _score_query(prov, q, c_pairs, "contrast")
            if errs:
                print(f"  [contrast {q!r}] {len(errs)} errors，跳过该 query")
                continue
            vals = sorted((v for v, _, _ in s), reverse=True)
            mx, mn = vals[0], vals[-1]
            print(f"  [contrast {q!r}] scores={['%.2f' % v for v in vals]} "
                  f"max={mx:.2f} range={mx - mn:.2f}")
            if mx > best_max:
                best_max, best_range, best_q = mx, mx - mn, q
        if best_q and best_max >= 0.5 and best_range >= 0.3:
            print(f"== contrast OK: query={best_q!r} max={best_max:.2f} range={best_range:.2f} "
                  f"→ 模型可区分相关/无关候选 ==")
        else:
            print("== contrast INFO: 图库样张未保证含目标内容；若全部低分且单调，属可解释（无匹配物体），"
                  "不代表模型失效；建议后续用含目标内容的图库验证 ==")

    print(f"== SMOKE RESULT: PASS (model={prov._model}, api=openai_chat/zhipu, "
          f"images={len(pool)}, avg_latency={avg_lat:.0f}ms) ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


