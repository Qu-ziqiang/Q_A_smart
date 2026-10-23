"""
数据库状态管理模块 - 负责获取和管理数据库状态信息（PostgreSQL + pgvector）
"""

import traceback
from typing import Any, Dict, Optional, Tuple

from ..config import settings
from ..db_func.core.connection import _database_url
from ..db_func.utils import row_to_dict


def check_table_exists(conn, table_name: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name=%s",
        (table_name,),
    )
    return cursor.fetchone() is not None


def get_image_statistics(conn) -> Tuple[int, int]:
    cursor = conn.cursor()
    image_count = 0
    total_size = 0
    if check_table_exists(conn, "images"):
        cursor.execute("SELECT COUNT(*) AS count FROM images")
        image_count_row = cursor.fetchone()
        if image_count_row:
            image_count = row_to_dict(image_count_row)["count"]
        cursor.execute("SELECT SUM(file_size) AS total_size FROM images")
        total_size_row = cursor.fetchone()
        if total_size_row:
            size_val = row_to_dict(total_size_row)["total_size"]
            total_size = size_val if size_val is not None else 0
    return image_count, total_size


def get_tag_count(conn) -> int:
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM images WHERE tags IS NOT NULL AND tags != '[]'"
        )
        row = cursor.fetchone()
        return (row_to_dict(row).get("cnt") or 0) if row else 0
    except Exception as e:
        print(f"获取标签数量失败: {e}")
        return 0


def check_vector_db_status(conn) -> bool:
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
        return cursor.fetchone() is not None
    except Exception:
        return False


def get_db_version(conn) -> str:
    cursor = conn.cursor()
    cursor.execute("SELECT version() AS version")
    result = cursor.fetchone()
    result_dict = row_to_dict(result)
    return result_dict["version"]


def get_database_info(conn: Optional[Any] = None) -> Dict[str, Any]:
    try:
        print("[db_info] start")
        image_count, total_size = get_image_statistics(conn)
        tag_count = get_tag_count(conn)
        vector_status = check_vector_db_status(conn)
        db_version = get_db_version(conn)
        return {
            "status": "connected",
            "type": "postgresql+pgvector",
            "path": _database_url(),
            "image_count": image_count,
            "total_size": total_size,
            "tag_count": tag_count,
            "vector_status": vector_status,
            "db_version": db_version,
            "error": None,
        }
    except Exception as e:
        error_msg = f"数据库连接错误: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            "status": "error",
            "type": "postgresql+pgvector",
            "path": _database_url(),
            "image_count": 0,
            "total_size": 0,
            "tag_count": 0,
            "vector_status": False,
            "db_version": "未知",
            "tables_info": {},
            "error": str(e),
        }


def get_storage_info(conn: Optional[Any] = None) -> Dict[str, Any]:
    db_info = get_database_info(conn)
    config = settings.get_config()
    return {
        "total_images": db_info["image_count"],
        "total_size_mb": round(db_info["total_size"] / (1024 * 1024), 2),
        "total_tags": db_info["tag_count"],
        "upload_dir": config.UPLOAD_DIR,
    }
