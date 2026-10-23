from pathlib import Path
import sys

ROOT = Path(r"D:\document\py_prj\ImageTextSystem\Smartlmager-sqlite-vec")
sys.path.insert(0, str(ROOT))

from backend.db_func.core.connection import get_db_connection, initialize_connection_pool

initialize_connection_pool(max_connections=2)
with get_db_connection() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tablename, indexname, indexdef
        FROM pg_indexes
        WHERE tablename IN ('image_vectors','title_vectors','description_vectors')
        ORDER BY tablename, indexname
        """
    )
    print("=== indexdef ===")
    for r in cur.fetchall():
        row = dict(r)
        print(row["tablename"], row["indexname"])
        print(row["indexdef"])
        print()

    cur.execute(
        """
        SELECT c.relname AS index_name, c.reloptions
        FROM pg_class c
        JOIN pg_index i ON i.indexrelid = c.oid
        JOIN pg_class t ON t.oid = i.indrelid
        WHERE t.relname IN ('image_vectors','title_vectors','description_vectors')
        ORDER BY c.relname
        """
    )
    print("=== reloptions ===")
    for r in cur.fetchall():
        row = dict(r)
        print(row["index_name"], row["reloptions"])

    print("=== session settings ===")
    for name in ("hnsw.ef_search", "hnsw.iterative_scan"):
        try:
            cur.execute(f"SHOW {name}")
            val = cur.fetchone()
            print(name, "=", val)
        except Exception as e2:
            print(name, "ERR", type(e2).__name__, e2)
            conn.rollback()
