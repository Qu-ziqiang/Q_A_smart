"""pgvector 扩展校验（服务器端扩展，无需本地 DLL）。"""
from typing import Any, Optional, Tuple


def verify_vector_extension(connection: Any) -> Tuple[bool, Optional[str]]:
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cursor.fetchone()
        if not row:
            return False, "pgvector extension not installed"
        ver = row["extversion"] if isinstance(row, dict) else row[0]
        return True, ver
    except Exception as e:
        return False, str(e)


def setup_connection(connection: Any, silent: bool = False) -> bool:
    """兼容旧接口：确认 pgvector 可用。"""
    ok, info = verify_vector_extension(connection)
    if ok:
        if not silent:
            print(f"pgvector 已就绪，版本: {info}")
        return True
    if not silent:
        print(f"pgvector 未就绪: {info}")
    return False


def load_vector_extension(connection: Any, silent: bool = False) -> Tuple[bool, Optional[str]]:
    return verify_vector_extension(connection)
