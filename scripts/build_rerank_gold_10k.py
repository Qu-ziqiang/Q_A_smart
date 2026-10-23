#!/usr/bin/env python
"""从万级库分层抽 ~400 张带 VOC 标签的图，生成 A2 重排评测 Gold。

检索与 Hit 只在这 400 张子集上算。不覆盖冻结 18-query gold。
"""
from __future__ import annotations

import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.db_func.core.connection import get_db_connection, initialize_connection_pool
from backend.db_func.utils.common import json_from_db_to_python, python_to_json_for_db

OUT = ROOT / "backend" / "ai_func" / "recommendation" / "evaluation" / "datasets" / "gold_voc_400_rerank.json"
TARGET = int(os.environ.get("GALLERY_EVAL_N", "400"))
SEED = int(os.environ.get("GALLERY_EVAL_SEED", "42"))
GOLD2007 = ROOT / "backend" / "tests" / "agent_team" / "gold_voc2007_healthy.json"

VOC_ZH = {
    "aeroplane": "飞机",
    "bicycle": "自行车",
    "bird": "鸟",
    "boat": "船",
    "bottle": "瓶子",
    "bus": "公交车",
    "car": "汽车",
    "cat": "猫",
    "chair": "椅子",
    "cow": "牛",
    "diningtable": "餐桌",
    "dog": "狗",
    "horse": "马",
    "motorbike": "摩托车",
    "person": "人",
    "pottedplant": "盆栽",
    "sheep": "羊",
    "sofa": "沙发",
    "train": "火车",
    "tvmonitor": "电视",
}

PAIR_ZH = {
    ("person", "bicycle"): "人和自行车",
    ("person", "motorbike"): "骑摩托车的人",
    ("person", "horse"): "骑马的人",
    ("person", "car"): "人和汽车",
    ("person", "bus"): "公交车和人",
    ("person", "train"): "火车和人",
    ("person", "boat"): "船上的人",
    ("person", "aeroplane"): "飞机和人",
    ("person", "dog"): "人和狗",
    ("person", "cat"): "人和猫",
    ("person", "cow"): "人和牛",
    ("person", "sheep"): "人和羊",
    ("person", "bird"): "人和鸟",
    ("person", "chair"): "人和椅子",
    ("person", "sofa"): "人和沙发",
    ("person", "diningtable"): "人和餐桌",
    ("person", "bottle"): "拿着瓶子的人",
    ("person", "tvmonitor"): "人和电视",
    ("chair", "diningtable"): "餐桌和椅子",
    ("sofa", "chair"): "沙发和椅子",
    ("sofa", "tvmonitor"): "沙发和电视",
    ("bottle", "diningtable"): "餐桌上的瓶子",
    ("pottedplant", "chair"): "盆栽和椅子",
    ("car", "bicycle"): "汽车和自行车",
    ("bus", "car"): "公交车和汽车",
    ("train", "car"): "火车和汽车",
    ("horse", "person"): "骑马的人",
    ("dog", "sofa"): "沙发上的狗",
    ("cat", "chair"): "椅子上的猫",
    ("boat", "person"): "船上的人",
}

MIN_SINGLE = 5
MIN_PAIR = 5
MAX_PAIRS = 30


def _parse_tags(xml_path: Path) -> list[str]:
    if not xml_path.exists():
        return []
    try:
        root = ET.parse(xml_path).getroot()
        names = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name and name not in names:
                names.append(name)
        return names
    except Exception:
        return []


def _xml_for_row(rec: dict) -> Path | None:
    stem = Path(rec.get("filename") or "").stem
    if not stem:
        return None
    fp = rec.get("filepath") or ""
    if fp:
        p = Path(fp)
        for up in (p.parent, p.parent.parent):
            cand = up / "Annotations" / f"{stem}.xml"
            if cand.exists():
                return cand
            cand = up.parent / "Annotations" / f"{stem}.xml"
            if cand.exists():
                return cand
    return None


def _norm_tags(raw) -> list[str]:
    tags = raw or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []
    out = []
    for t in tags:
        s = str(t).strip()
        if s and s not in out:
            out.append(s)
    return out


def _backfill(cur) -> int:
    cur.execute("SELECT id, filename, filepath, tags FROM images ORDER BY id")
    updated = 0
    for r in cur.fetchall():
        rec = json_from_db_to_python(dict(r))
        if _norm_tags(rec.get("tags")):
            continue
        xml_path = _xml_for_row(rec)
        if xml_path is None:
            continue
        parsed = _parse_tags(xml_path)
        if not parsed:
            continue
        cur.execute(
            "UPDATE images SET tags = %s WHERE id = %s",
            (python_to_json_for_db(parsed, []), rec["id"]),
        )
        updated += 1
    return updated


def _add_query(queries: list, seen: set, item: dict) -> None:
    q = item["query"].strip()
    if not q or q in seen:
        return
    if item.get("gt_total", 0) <= 0:
        return
    seen.add(q)
    item["id"] = f"q{len(queries)+1:03d}"
    queries.append(item)


def _stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """按 VOC 类 round-robin 抽 n 张，避免 person 占满。只抽带标签图。"""
    rng = random.Random(seed)
    voc = set(VOC_ZH)
    buckets: dict[str, list[dict]] = defaultdict(list)
    tagged = []
    for r in rows:
        tags = _norm_tags(r.get("tags"))
        if not tags:
            continue
        tagged.append(r)
        for t in tags:
            if t in voc:
                buckets[t].append(r)
    for cls in buckets:
        rng.shuffle(buckets[cls])
    picked: list[dict] = []
    seen: set[int] = set()
    classes = sorted(buckets.keys())
    progressed = True
    while len(picked) < n and progressed:
        progressed = False
        for cls in classes:
            if len(picked) >= n:
                break
            while buckets[cls]:
                rec = buckets[cls].pop()
                if rec["id"] in seen:
                    continue
                seen.add(rec["id"])
                picked.append(rec)
                progressed = True
                break
    if len(picked) < n:
        rng.shuffle(tagged)
        for rec in tagged:
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            picked.append(rec)
            if len(picked) >= n:
                break
    return picked[:n]


def main() -> int:
    os.chdir(ROOT)
    initialize_connection_pool(max_connections=4)
    with get_db_connection() as conn:
        cur = conn.cursor()
        filled = _backfill(cur)
        conn.commit()
        print(f"backfilled tags for {filled} images", flush=True)
        cur.execute("SELECT id, filename, tags FROM images ORDER BY id")
        all_rows = [json_from_db_to_python(dict(r)) for r in cur.fetchall()]

    rows = _stratified_sample(all_rows, TARGET, SEED)
    print(f"sampled {len(rows)} / {len(all_rows)} (target {TARGET})", flush=True)

    by_class: dict[str, list[dict]] = {}
    tag_sets: list[tuple[str, frozenset]] = []
    prefixes: Counter = Counter()
    tagged = 0
    for r in rows:
        fn = (r.get("filename") or "").strip()
        stem = Path(fn).stem
        if "_" in stem:
            prefixes[stem.split("_", 1)[0]] += 1
        tags = _norm_tags(r.get("tags"))
        if tags:
            tagged += 1
        tag_sets.append((stem, frozenset(tags)))
        for t in tags:
            by_class.setdefault(t, []).append({"id": r["id"], "stem": stem, "filename": fn})

    old_queries = []
    if GOLD2007.exists():
        old_queries = json.loads(GOLD2007.read_text(encoding="utf-8"))["queries"]

    queries: list[dict] = []
    seen: set[str] = set()

    for q in old_queries:
        cls = q["class"]
        hits = by_class.get(cls, [])
        _add_query(queries, seen, {
            "query": q["query"],
            "query_en": q.get("query_en") or f"photos of {cls}",
            "kind": "single",
            "class": cls,
            "classes": [cls],
            "gt_unique": [h["stem"] for h in hits],
            "gt_total": len(hits),
            "source": "legacy 18-query wording on 400-image subset",
        })

    for cls, zh in VOC_ZH.items():
        hits = by_class.get(cls, [])
        if len(hits) < MIN_SINGLE:
            continue
        for text, en in (
            (f"{zh}的照片", f"photos of {cls}"),
            (f"找几张{zh}", f"find some {cls}"),
        ):
            _add_query(queries, seen, {
                "query": text,
                "query_en": en,
                "kind": "single",
                "class": cls,
                "classes": [cls],
                "gt_unique": [h["stem"] for h in hits],
                "gt_total": len(hits),
                "source": "VOC single-class on 400-image subset",
            })

    pair_counts: Counter = Counter()
    pair_stems: dict[tuple[str, str], list[str]] = {}
    voc = set(VOC_ZH)
    for stem, tags in tag_sets:
        present = sorted(t for t in tags if t in voc)
        for a, b in combinations(present, 2):
            key = (a, b)
            pair_counts[key] += 1
            pair_stems.setdefault(key, []).append(stem)

    ranked_pairs = [k for k, n in pair_counts.most_common() if n >= MIN_PAIR][:MAX_PAIRS]
    for a, b in ranked_pairs:
        stems = pair_stems[(a, b)]
        label = PAIR_ZH.get((a, b)) or PAIR_ZH.get((b, a)) or f"{VOC_ZH[a]}和{VOC_ZH[b]}"
        _add_query(queries, seen, {
            "query": f"{label}的照片",
            "query_en": f"photos of {a} and {b}",
            "kind": "pair",
            "class": f"{a}+{b}",
            "classes": [a, b],
            "gt_unique": stems,
            "gt_total": len(stems),
            "source": "VOC co-occurrence on 400-image subset",
        })

    payload = {
        "dataset": "400-image stratified subset of PostgreSQL gallery + VOC tags",
        "unit": "filename stem",
        "parent_gallery_size": len(all_rows),
        "gallery_size": len(rows),
        "gallery_ids": [r["id"] for r in rows],
        "sample_seed": SEED,
        "tagged_images": tagged,
        "filename_year_prefix_counts": dict(prefixes),
        "class_counts": {k: len(v) for k, v in sorted(by_class.items(), key=lambda x: -len(x[1]))},
        "n_queries": len(queries),
        "n_single": sum(1 for q in queries if q["kind"] == "single"),
        "n_pair": sum(1 for q in queries if q["kind"] == "pair"),
        "note": "检索与 Hit 只在这 400 张上算。不覆盖冻结 18-query / 不报万级 Hit@1。",
        "queries": queries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("gallery", len(rows), "tagged", tagged, "queries", len(queries),
          "single", payload["n_single"], "pair", payload["n_pair"], flush=True)
    print("saved", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
