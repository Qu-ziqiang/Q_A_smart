#!/usr/bin/env python
"""用当前 CLIP 模型为 PostgreSQL 中的图片重建三路向量。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.db_func.core.connection import initialize_connection_pool, get_db_connection
from backend.db_func.core.database import init_db
from backend.db_func.repositories.vectors import VectorRepository


def main():
    init_db()
    initialize_connection_pool(max_connections=4)
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, filepath, title, description FROM images ORDER BY id")
        rows = cur.fetchall()
    print(f"待重建: {len(rows)} 张")
    repo = VectorRepository()
    ok = 0
    for r in rows:
        image_id = r["id"]
        fp = r["filepath"]
        if repo.add_image_vector(image_id, fp):
            ok += 1
        if r.get("title"):
            repo.add_title_vector(image_id, r["title"])
        if r.get("description"):
            repo.add_description_vector(image_id, r["description"])
        if ok % 10 == 0:
            print(f"进度: {ok}/{len(rows)}")
    stats = repo.get_vector_stats()
    print("完成:", stats)


if __name__ == "__main__":
    main()
