"""
向量数据仓库（PostgreSQL + pgvector）
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseRepository


def _as_vector_literal(vector) -> str:
    """将 numpy/list 转为 pgvector 文本字面量，如 '[0.1,0.2,...]'。"""
    if hasattr(vector, "tolist"):
        values = vector.tolist()
    else:
        values = list(vector)
    return "[" + ",".join(str(float(x)) for x in values) + "]"


class VectorRepository(BaseRepository):
    """向量数据仓库类"""

    def add_image_vector(self, image_id: int, image_path: str) -> bool:
        if not image_path or not Path(image_path).exists():
            print(f"图像文件不存在: {image_path}")
            return False

        try:
            from ...ai_func.generate_vector import encode_image

            vector = encode_image(image_path)
            vec_lit = _as_vector_literal(vector)

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO image_vectors(image_id, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (image_id) DO UPDATE SET embedding = EXCLUDED.embedding
                    """,
                    (image_id, vec_lit),
                )
                conn.commit()
                print(f"写入图像向量成功: {image_id}")
                return True
        except Exception as e:
            print(f"添加图像向量失败: {e}")
            traceback.print_exc()
            return False

    def add_text_vector(self, image_id: int, text: str, vector_type: str) -> bool:
        if not text or not text.strip():
            return False
        if vector_type not in ["title", "description"]:
            raise ValueError("vector_type 必须是 'title' 或 'description'")

        table_name = f"{vector_type}_vectors"
        try:
            from ...ai_func.generate_vector import encode_text

            vector = encode_text(text)
            vec_lit = _as_vector_literal(vector)

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    INSERT INTO {table_name}(image_id, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (image_id) DO UPDATE SET embedding = EXCLUDED.embedding
                    """,
                    (image_id, vec_lit),
                )
                conn.commit()
                print(f"写入{vector_type}向量成功: {image_id}")
                return True
        except Exception as e:
            print(f"添加{vector_type}向量失败: {e}")
            traceback.print_exc()
            return False

    def add_title_vector(self, image_id: int, title: str) -> bool:
        return self.add_text_vector(image_id, title, "title")

    def add_description_vector(self, image_id: int, description: str) -> bool:
        return self.add_text_vector(image_id, description, "description")

    def delete_vectors(self, image_id: int) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM title_vectors WHERE image_id = %s", (image_id,))
                cursor.execute("DELETE FROM description_vectors WHERE image_id = %s", (image_id,))
                cursor.execute("DELETE FROM image_vectors WHERE image_id = %s", (image_id,))
                conn.commit()
                print(f"删除向量成功: {image_id}")
                return True
        except Exception as e:
            print(f"删除向量失败: {e}")
            traceback.print_exc()
            return False

    def delete_vector_by_type(self, image_id: int, vector_type: str) -> bool:
        if vector_type not in ["title", "description", "image"]:
            raise ValueError("vector_type 必须是 'title', 'description' 或 'image'")
        table_name = f"{vector_type}_vectors"
        try:
            affected = self.execute_delete(
                f"DELETE FROM {table_name} WHERE image_id = ?", (image_id,)
            )
            if affected > 0:
                print(f"删除{vector_type}向量成功: {image_id}")
                return True
            print(f"未找到图片 {image_id} 的{vector_type}向量")
            return False
        except Exception as e:
            print(f"删除{vector_type}向量失败: {e}")
            traceback.print_exc()
            return False

    def batch_add_vectors(self, image_data_list: List[Dict[str, Any]]) -> List[int]:
        if not image_data_list:
            return []
        try:
            from .batch_vector_manager import BatchVectorManager

            with self.get_connection() as conn:
                manager = BatchVectorManager(conn)
                return manager.add_batch_vectors(image_data_list)
        except Exception as e:
            print(f"批量添加向量失败: {e}")
            return self._fallback_single_processing(image_data_list)

    def _fallback_single_processing(self, image_data_list: List[Dict[str, Any]]) -> List[int]:
        print("批量处理失败，使用逐个处理模式")
        success_ids = []
        for data in image_data_list:
            try:
                image_id = data["image_id"]
                self.add_image_vector(image_id, data["filepath"])
                self.add_title_vector(image_id, data.get("title", ""))
                self.add_description_vector(image_id, data.get("description", ""))
                success_ids.append(image_id)
            except Exception as e:
                print(f"处理图片 ID {data.get('image_id')} 失败: {e}")
        return success_ids

    def vector_search(
        self,
        query_embedding: List[float],
        vector_type: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if vector_type not in ["title", "description", "image"]:
            raise ValueError("vector_type 必须是 'title', 'description' 或 'image'")
        table_name = f"{vector_type}_vectors"
        try:
            vec_lit = _as_vector_literal(query_embedding)
            sql = f"""
            SELECT
                image_id,
                (1 - (embedding <=> %s::vector)) AS score
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s OFFSET %s
            """
            return self.execute_query(sql, (vec_lit, vec_lit, limit, offset))
        except Exception as e:
            print(f"向量搜索失败: {e}")
            traceback.print_exc()
            return []

    def multi_vector_search(
        self,
        query_embedding: List[float],
        search_targets: List[str],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
        exclude_image_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """兼容旧调用：转交 SearchRepository。"""
        from .search import SearchRepository

        repo = SearchRepository(self._external_conn)
        results = repo.vector_search_direct(
            query_embedding=query_embedding,
            search_targets=search_targets,
            filters=filters,
            limit=limit + (1 if exclude_image_id else 0),
            offset=offset,
        )
        if exclude_image_id is not None:
            results = [r for r in results if r.get("id") != exclude_image_id][:limit]
        return results

    def get_vector_stats(self) -> Dict[str, int]:
        stats = {}
        for vector_type in ["title", "description", "image"]:
            table_name = f"{vector_type}_vectors"
            result = self.execute_query_one(f"SELECT COUNT(*) as count FROM {table_name}")
            stats[vector_type] = result["count"] if result else 0
        return stats

    def check_vector_exists(self, image_id: int, vector_type: str) -> bool:
        if vector_type not in ["title", "description", "image"]:
            raise ValueError("vector_type 必须是 'title', 'description' 或 'image'")
        table_name = f"{vector_type}_vectors"
        result = self.execute_query_one(
            f"SELECT id FROM {table_name} WHERE image_id = ?", (image_id,)
        )
        return result is not None
