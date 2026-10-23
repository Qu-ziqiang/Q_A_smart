"""
PostgreSQL 连接管理

提供线程安全的连接池。向量能力依赖服务器端 pgvector 扩展，
不再加载本地 sqlite-vec DLL。
"""
from __future__ import annotations

import threading
import queue
from contextlib import contextmanager
from typing import Any, Optional
from urllib.parse import quote_plus

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection

from ...config import settings


def _database_url() -> str:
    cfg = settings.get_config()
    url = getattr(cfg, "DATABASE_URL", None)
    if url:
        return url
    user = quote_plus(getattr(cfg, "PG_USER", "smartimager") or "smartimager")
    password = quote_plus(getattr(cfg, "PG_PASSWORD", "smartimager") or "smartimager")
    host = getattr(cfg, "PG_HOST", "127.0.0.1") or "127.0.0.1"
    port = getattr(cfg, "PG_PORT", 5432) or 5432
    db = getattr(cfg, "PG_DATABASE", "smartimager") or "smartimager"
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


# HNSW 参数：与 pgvector 默认值对齐并显式写出，便于运维/面试口径一致。
# m：每层最大边数；ef_construction：建图候选宽度；ef_search：查询候选宽度（会话级）。
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 40


def _configure_connection(conn: PgConnection) -> PgConnection:
    conn.autocommit = False
    psycopg2.extras.register_default_json(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {int(HNSW_EF_SEARCH)}")
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    return conn


class DatabaseConnectionPool:
    """PostgreSQL 连接池。"""

    def __init__(self, dsn: str, max_connections: int = 10):
        self.dsn = dsn
        self.max_connections = max_connections
        self.connections: queue.Queue = queue.Queue(maxsize=max_connections)
        self.connection_count = 0
        self._lock = threading.Lock()

    def _create_connection(self) -> PgConnection:
        conn = psycopg2.connect(self.dsn, cursor_factory=psycopg2.extras.RealDictCursor)
        return _configure_connection(conn)

    def get_connection(self) -> PgConnection:
        try:
            return self.connections.get(block=False)
        except queue.Empty:
            with self._lock:
                if self.connection_count < self.max_connections:
                    self.connection_count += 1
                    return self._create_connection()
                return self.connections.get()

    def release_connection(self, connection: PgConnection):
        if connection.closed:
            with self._lock:
                self.connection_count = max(0, self.connection_count - 1)
            return
        try:
            connection.rollback()
        except Exception:
            pass
        self.connections.put(connection)

    def close_all(self):
        while not self.connections.empty():
            conn = self.connections.get()
            try:
                conn.close()
            except Exception:
                pass
        with self._lock:
            self.connection_count = 0


_connection_pool: Optional[DatabaseConnectionPool] = None


def initialize_connection_pool(
    database_path: Optional[str] = None,
    max_connections: int = 10,
    dsn: Optional[str] = None,
) -> DatabaseConnectionPool:
    """初始化连接池。

    ``database_path`` 仅为兼容旧调用签名保留，实际忽略；
    连接串来自 ``DATABASE_URL`` / ``PG_*`` 配置。
    """
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = DatabaseConnectionPool(dsn or _database_url(), max_connections)
        print(f"PostgreSQL 连接池已初始化 (max={max_connections})")
    return _connection_pool


def get_connection_pool() -> DatabaseConnectionPool:
    global _connection_pool
    if _connection_pool is None:
        raise RuntimeError("Connection pool not initialized. Call initialize_connection_pool first.")
    return _connection_pool


@contextmanager
def get_db_connection_from_pool():
    pool = get_connection_pool()
    conn = pool.get_connection()
    try:
        yield conn
    finally:
        pool.release_connection(conn)


@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(_database_url(), cursor_factory=psycopg2.extras.RealDictCursor)
        _configure_connection(conn)
        yield conn
    finally:
        if conn and not conn.closed:
            conn.close()


def get_db():
    """FastAPI 依赖项。"""
    try:
        with get_db_connection_from_pool() as conn:
            yield conn
    except RuntimeError:
        with get_db_connection() as conn:
            yield conn
