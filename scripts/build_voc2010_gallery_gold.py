#!/usr/bin/env python
"""根据当前 PostgreSQL 图库 + VOC 官方类别标签，生成千级图库 Gold。

协议：
- Query 文本 / 类别与 gold_voc2007_healthy.json 对齐（18 条健康单类 query）
- GT = 库中 tags 含该类的全部图片，content id = 文件名去扩展名（2007_000033 / 2010_xxxx）
- 不沿用 VOC2007 的 6 位 id（会与 2010 碰撞，且旧 Gold 不在千级库上可评）
"""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.db_func.core.connection import get_db_connection, initialize_connection_pool
from backend.db_func.utils.common import json_from_db_to_python, python_to_json_for_db

GOLD2007 = ROOT / "backend" / "tests" / "agent_team" / "gold_voc2007_healthy.json"
OUT = ROOT / "backend" / "tests" / "agent_team" / "gold_voc2010_gallery.json"
DEFAULT_ANNO = Path(r"D:\document\py_prj\NACLIP\mmsegmentation\data\VOCdevkit\VOC2010\Annotations")


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


def _backfill_empty_tags(cur) -> int:
    cur.execute("SELECT id, filename, filepath, tags FROM images ORDER BY id")
    updated = 0
    for r in cur.fetchall():
        rec = json_from_db_to_python(dict(r))
        tags = rec.get("tags") or []
        if tags:
            continue
        stem = Path(rec.get("filename") or "").stem
        xml_path = DEFAULT_ANNO / f"{stem}.xml"
        fp = rec.get("filepath") or ""
        if fp:
            alt = Path(fp).resolve().parents[1] / "Annotations" / f"{stem}.xml"
            if alt.exists():
                xml_path = alt
        parsed = _parse_tags(xml_path)
        if not parsed:
            continue
        cur.execute(
            "UPDATE images SET tags = %s WHERE id = %s",
            (python_to_json_for_db(parsed, []), rec["id"]),
        )
        updated += 1
    return updated


def main() -> int:
    os.chdir(ROOT)
    initialize_connection_pool(max_connections=4)
    with get_db_connection() as conn:
        cur = conn.cursor()
        filled = _backfill_empty_tags(cur)
        conn.commit()
        print(f"backfilled tags for {filled} images")
        cur.execute("SELECT id, filename, tags FROM images ORDER BY id")
        rows = [json_from_db_to_python(dict(r)) for r in cur.fetchall()]

    by_class: dict[str, list[dict]] = {}
    prefixes: dict[str, int] = {}
    for r in rows:
        fn = (r.get("filename") or "").strip()
        stem = Path(fn).stem
        if "_" in stem:
            prefixes[stem.split("_", 1)[0]] = prefixes.get(stem.split("_", 1)[0], 0) + 1
        tags = r.get("tags") or []
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        for t in tags:
            by_class.setdefault(t, []).append({"id": r["id"], "stem": stem, "filename": fn})

    with open(GOLD2007, encoding="utf-8") as f:
        old = json.load(f)

    queries = []
    for q in old["queries"]:
        cls = q["class"]
        hits = by_class.get(cls, [])
        queries.append(
            {
                "query": q["query"],
                "class": cls,
                "query_en": q["query_en"],
                "gt_unique": [h["stem"] for h in hits],
                "gt_db_rows": [h["id"] for h in hits],
                "gt_total": len(hits),
                "source": "PostgreSQL gallery tags from VOC Annotations (VOC2010 JPEGImages ingest)",
            }
        )

    empty = [q["class"] for q in queries if q["gt_total"] == 0]
    payload = {
        "dataset": "current PostgreSQL gallery (VOC JPEG + official class tags)",
        "unit": "filename stem (year_sixdigit)",
        "gallery_size": len(rows),
        "filename_year_prefix_counts": prefixes,
        "class_counts": {k: len(v) for k, v in sorted(by_class.items(), key=lambda x: -len(x[1]))},
        "empty_classes": empty,
        "queries": queries,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("gallery", len(rows), "prefixes", prefixes)
    print("empty_classes", empty)
    for q in queries:
        print(f"  {q['class']:14s} gt={q['gt_total']:4d}  {q['query']}")
    print("saved", OUT)
    return 0 if not empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
