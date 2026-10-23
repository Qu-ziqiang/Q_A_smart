"""Smoke test: PostgreSQL + pgvector search path."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.db_func.core.connection import initialize_connection_pool, get_db_connection
from backend.db_func.core.database import init_db
from backend.db_func.repositories.search import SearchRepository


def main():
    init_db()
    initialize_connection_pool(max_connections=4)

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM images")
        print("images", cur.fetchone()["n"])
        for t in ("image_vectors", "title_vectors", "description_vectors"):
            cur.execute(f"SELECT COUNT(*) AS n FROM {t}")
            print(t, cur.fetchone()["n"])
        cur.execute(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename IN ('image_vectors','title_vectors','description_vectors')
            ORDER BY indexname
            """
        )
        for row in cur.fetchall():
            print("index:", row["indexname"], "->", row["indexdef"][:120])

    repo = SearchRepository()
    t0 = time.perf_counter()
    results = repo.unified_search(
        query_type="text",
        query_content="海边日落",
        search_targets=["image", "title"],
        filters={},
        limit=5,
    )
    dt = (time.perf_counter() - t0) * 1000
    print(f"search hits={len(results)} latency_ms={dt:.1f}")
    for r in results[:5]:
        print(
            f"  id={r.get('id')} score={r.get('score'):.4f} "
            f"title={r.get('title')!r} file={r.get('filename')}"
        )

    # metadata filter on DB side
    results2 = repo.unified_search(
        query_type="text",
        query_content="海边日落",
        search_targets=["image"],
        filters={"min_width": 1},
        limit=3,
    )
    print(f"filtered hits={len(results2)}")


if __name__ == "__main__":
    main()
