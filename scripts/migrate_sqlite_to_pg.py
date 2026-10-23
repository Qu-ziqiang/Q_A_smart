#!/usr/bin/env python
"""把旧 SQLite + sqlite-vec 数据迁移到 PostgreSQL + pgvector。

用法（在项目根目录）:
  python scripts/migrate_sqlite_to_pg.py
  python scripts/migrate_sqlite_to_pg.py --sqlite ./data/db/smartimager.db
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psycopg2
import psycopg2.extras

from backend.config import settings
from backend.db_func.core.connection import _database_url
from backend.db_func.core.database import init_db
from backend.db_func.repositories.vectors import _as_vector_literal


VECTOR_TABLES = ("title_vectors", "description_vectors", "image_vectors")


def _parse_embedding(raw) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, memoryview):
        raw = bytes(raw)
    if isinstance(raw, (bytes, bytearray)):
        # sqlite-vec 可能存二进制；尽力解析失败则跳过，后续可重建
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            return _as_vector_literal(json.loads(text))
        return None
    if isinstance(raw, (list, tuple)):
        return _as_vector_literal(raw)
    return None


def _load_sqlite_vec(conn: sqlite3.Connection):
    """尽量加载 sqlite-vec，便于从旧虚表导出 embedding。"""
    candidates = []
    cfg = settings.get_config()
    driver = getattr(cfg, "VECTOR_DB_DRIVER", None)
    if driver:
        candidates.append(driver)
    root = ROOT / "backend" / "config" / "files" / "vector_db_driver"
    for p in root.rglob("vec0.dll"):
        candidates.append(str(p))
    for p in root.rglob("vec0.so"):
        candidates.append(str(p))
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            conn.enable_load_extension(True)
            conn.load_extension(path)
            print(f"已加载 sqlite-vec: {path}")
            return True
        except Exception as e:
            print(f"加载 sqlite-vec 失败 ({path}): {e}")
    print("警告: 未能加载 sqlite-vec，将跳过向量导出（可稍后重建）")
    return False


def migrate(sqlite_path: str):
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite 文件不存在: {sqlite_path}")

    print(f"源: {sqlite_path}")
    print(f"目标: {_database_url()}")
    init_db()

    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    _load_sqlite_vec(src)
    dst = psycopg2.connect(_database_url(), cursor_factory=psycopg2.extras.RealDictCursor)
    dcur = dst.cursor()

    # images
    rows = src.execute("SELECT * FROM images").fetchall()
    print(f"迁移 images: {len(rows)}")
    for r in rows:
        dcur.execute(
            """
            INSERT INTO images (
                id, filename, filepath, title, description, file_size, file_type,
                width, height, created_at, updated_at, metadata, tags
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
            )
            ON CONFLICT (id) DO UPDATE SET
                filename=EXCLUDED.filename,
                filepath=EXCLUDED.filepath,
                title=EXCLUDED.title,
                description=EXCLUDED.description,
                file_size=EXCLUDED.file_size,
                file_type=EXCLUDED.file_type,
                width=EXCLUDED.width,
                height=EXCLUDED.height,
                created_at=EXCLUDED.created_at,
                updated_at=EXCLUDED.updated_at,
                metadata=EXCLUDED.metadata,
                tags=EXCLUDED.tags
            """,
            (
                r["id"],
                r["filename"],
                r["filepath"],
                r["title"],
                r["description"],
                r["file_size"],
                r["file_type"],
                r["width"],
                r["height"],
                r["created_at"],
                r["updated_at"],
                r["metadata"],
                r["tags"],
            ),
        )
    if rows:
        dcur.execute("SELECT setval(pg_get_serial_sequence('images','id'), (SELECT COALESCE(MAX(id),1) FROM images))")
    dst.commit()

    # vectors
    for table in VECTOR_TABLES:
        try:
            vrows = src.execute(
                f"SELECT image_id, vec_to_json(embedding) AS embedding FROM {table}"
            ).fetchall()
        except sqlite3.Error as e:
            print(f"跳过 {table}: {e}")
            continue
        ok = 0
        skip = 0
        for r in vrows:
            vec_lit = _parse_embedding(r["embedding"])
            if not vec_lit:
                skip += 1
                continue
            dcur.execute(
                f"""
                INSERT INTO {table}(image_id, embedding)
                VALUES (%s, %s::vector)
                ON CONFLICT (image_id) DO UPDATE SET embedding = EXCLUDED.embedding
                """,
                (r["image_id"], vec_lit),
            )
            ok += 1
        dst.commit()
        print(f"迁移 {table}: ok={ok}, skip={skip}")

    # soft-copy text tables if present
    for table, cols in (
        (
            "request_sessions",
            [
                "id",
                "request_id",
                "conversation_id",
                "user_id",
                "endpoint",
                "messages",
                "state",
                "vector_targets",
                "filters",
                "selected_ids",
                "status",
                "error",
                "created_at",
                "updated_at",
            ],
        ),
        (
            "conversation_messages",
            ["id", "conversation_id", "role", "content", "image_ids", "metadata", "created_at"],
        ),
        (
            "agent_episodes",
            [
                "id",
                "conversation_id",
                "request_id",
                "user_id",
                "route",
                "task_types",
                "query_summary",
                "strategy_summary",
                "result_summary",
                "preference_hints",
                "selected_count",
                "success",
                "plan_valid",
                "created_at",
                "expires_at",
            ],
        ),
    ):
        try:
            trows = src.execute(f"SELECT * FROM {table}").fetchall()
        except sqlite3.Error:
            continue
        if not trows:
            print(f"迁移 {table}: 0")
            continue
        col_list = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        updates = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "id")
        for r in trows:
            values = [r[c] if c in r.keys() else None for c in cols]
            dcur.execute(
                f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                f"ON CONFLICT (id) DO UPDATE SET {updates}",
                values,
            )
        dcur.execute(
            f"SELECT setval(pg_get_serial_sequence('{table}','id'), "
            f"(SELECT COALESCE(MAX(id),1) FROM {table}))"
        )
        dst.commit()
        print(f"迁移 {table}: {len(trows)}")

    src.close()
    dst.close()
    print("迁移完成。若向量 skip 较多，请运行向量重建脚本。")


def main():
    parser = argparse.ArgumentParser()
    cfg = settings.get_config()
    default_sqlite = getattr(cfg, "DB_PATH", None) or "./data/db/smartimager.db"
    parser.add_argument("--sqlite", default=default_sqlite)
    args = parser.parse_args()
    migrate(args.sqlite)


if __name__ == "__main__":
    main()
