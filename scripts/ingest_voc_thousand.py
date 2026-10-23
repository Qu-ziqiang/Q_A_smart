#!/usr/bin/env python
"""从本地 VOC JPEGImages 摄入约 1000 张图到 PostgreSQL + pgvector。"""
from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.db_func.core.connection import get_db_connection, initialize_connection_pool
from backend.db_func.core.database import init_db
from backend.db_func.repositories.images import ImageRepository
from backend.db_func.repositories.vectors import VectorRepository

DEFAULT_JPEG = Path(r"D:\document\py_prj\NACLIP\mmsegmentation\data\VOCdevkit\VOC2010\JPEGImages")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpeg-dir", default=str(DEFAULT_JPEG))
    parser.add_argument("--anno-dir", default=str(DEFAULT_ANNO))
    parser.add_argument("--target", type=int, default=1000, help="库内目标图片总数")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    jpeg_dir = Path(args.jpeg_dir)
    anno_dir = Path(args.anno_dir)
    if not jpeg_dir.is_dir():
        raise SystemExit(f"找不到 VOC 图片目录: {jpeg_dir}")

    init_db()
    initialize_connection_pool(max_connections=8)

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename FROM images")
        existing = {row["filename"] for row in cur.fetchall()}
        cur.execute("SELECT COUNT(*) AS n FROM images")
        current = int(cur.fetchone()["n"])

    print(f"当前库内图片: {current}，目标: {args.target}")
    need = max(0, args.target - current)
    if need == 0:
        print("已达到目标规模，跳过摄入")
        return

    candidates = sorted(p for p in jpeg_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
    to_add = [p for p in candidates if p.name not in existing][:need]
    print(f"将新摄入: {len(to_add)}")

    image_repo = ImageRepository()
    vector_repo = VectorRepository()
    t0 = time.perf_counter()
    inserted = 0

    for i in range(0, len(to_add), args.batch):
        chunk = to_add[i : i + args.batch]
        rows = []
        for p in chunk:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                w, h = 0, 0
            tags = _parse_tags(anno_dir / f"{p.stem}.xml")
            rows.append(
                {
                    "filename": p.name,
                    "filepath": str(p.resolve()),
                    "title": p.stem,
                    "description": "",
                    "file_size": p.stat().st_size,
                    "file_type": "image/jpeg",
                    "width": w,
                    "height": h,
                    "created_at": datetime.now().isoformat(timespec="microseconds"),
                    "metadata": {"source": "VOC2010", "path": str(p)},
                    "tags": tags,
                }
            )
        ids = image_repo.batch_create(rows)
        payload = [
            {
                "image_id": image_id,
                "filepath": rec["filepath"],
                "title": rec["title"],
                "description": rec["description"],
            }
            for image_id, rec in zip(ids, rows)
        ]
        if payload:
            vector_repo.batch_add_vectors(payload)
        inserted += len(ids)
        elapsed = time.perf_counter() - t0
        print(f"进度 {inserted}/{len(to_add)}  elapsed={elapsed:.1f}s")

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM images")
        n_img = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) AS n FROM image_vectors")
        n_vec = cur.fetchone()["n"]
    print(f"完成: images={n_img} image_vectors={n_vec} seconds={time.perf_counter()-t0:.1f}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
