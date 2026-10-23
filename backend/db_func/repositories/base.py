"""
数据仓库基类，提供通用的数据库操作模式（PostgreSQL）。
"""
from __future__ import annotations

import sqlite3
from abc import ABC
from contextlib import contextmanager
from typing import Any, List, Optional, Sequence, Tuple, Union

from ..core.connection import get_db_connection, get_db_connection_from_pool
from ..utils.common import row_to_dict, rows_to_dicts


Params = Union[Sequence[Any], Tuple[Any, ...]]


def adapt_sql(query: str) -> str:
    """将 SQLite 风格的 ``?`` 占位符转换为 psycopg2 的 ``%s``。"""
    if "?" not in query:
        return query
    out: List[str] = []
    i = 0
    n = len(query)
    in_single = False
    in_double = False
    while i < n:
        ch = query[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            i += 1
            continue
        if ch == "?" and not in_single and not in_double:
            out.append("%s")
            i += 1
            continue
        if ch == "%" and not in_single and not in_double:
            # 避免与 psycopg2 的 % 格式化冲突（LIKE '%x%' 仍安全，因为走参数）
            nxt = query[i + 1] if i + 1 < n else ""
            if nxt not in ("s", "%", "(", ")", "d", "f"):
                out.append("%%")
            else:
                out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


class BaseRepository(ABC):
    """基础数据仓库类

    支持两种使用方式:
    1. 无参构造: 每次操作自动从连接池 / 单连接获取连接
    2. 传入外部已打开的连接: 复用该连接（不负责关闭）
    """

    def __init__(self, conn: Any = None):
        self._external_conn = conn

    @contextmanager
    def get_connection(self):
        if self._external_conn is not None:
            yield self._external_conn
            return
        try:
            with get_db_connection_from_pool() as conn:
                yield conn
        except RuntimeError:
            with get_db_connection() as conn:
                yield conn

    @staticmethod
    def _sql_for_connection(query: str, conn: Any) -> str:
        """生产连接适配 psycopg2；测试注入的 sqlite3 连接保留 ``?``。"""
        return query if isinstance(conn, sqlite3.Connection) else adapt_sql(query)

    def execute_query(self, query: str, params: Params = ()) -> List[dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._sql_for_connection(query, conn), params)
            rows = cursor.fetchall()
            return rows_to_dicts(rows)

    def execute_query_one(self, query: str, params: Params = ()) -> Optional[dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._sql_for_connection(query, conn), params)
            row = cursor.fetchone()
            return row_to_dict(row) if row else None

    def execute_insert(self, query: str, params: Params = ()) -> int:
        with self.get_connection() as conn:
            sql = self._sql_for_connection(query, conn).rstrip().rstrip(";")
            is_sqlite = isinstance(conn, sqlite3.Connection)
            if not is_sqlite and " returning " not in sql.lower():
                sql = f"{sql} RETURNING id"
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if is_sqlite:
                conn.commit()
                return int(cursor.lastrowid or 0)
            row = cursor.fetchone()
            conn.commit()
            if row is None:
                return 0
            if isinstance(row, dict):
                return int(row.get("id") or next(iter(row.values())))
            return int(row[0])

    def execute_update(self, query: str, params: Params = ()) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._sql_for_connection(query, conn), params)
            conn.commit()
            return cursor.rowcount

    def execute_delete(self, query: str, params: Params = ()) -> int:
        return self.execute_update(query, params)

    def execute_batch(self, queries: List[tuple]) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                for query, params in queries:
                    cursor.execute(self._sql_for_connection(query, conn), params)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
