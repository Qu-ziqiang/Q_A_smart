"""
批量向量管理器（PostgreSQL + pgvector）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .vectors import _as_vector_literal


class BatchVectorManager:
    """批量向量管理器，用于高效生成和保存向量"""

    def __init__(self, conn: Any):
        self.conn = conn

    def add_batch_vectors(self, image_data_list: List[Dict[str, Any]]) -> List[int]:
        if not image_data_list:
            return []

        image_ids = []
        image_paths = []
        titles = []
        descriptions = []

        for data in image_data_list:
            image_ids.append(data["image_id"])
            image_paths.append(data["filepath"])
            titles.append(data.get("title", ""))
            descriptions.append(data.get("description", ""))

        print(f"开始批量生成向量，共 {len(image_data_list)} 张图片")

        try:
            self._batch_upsert("image_vectors", image_ids, image_paths, kind="image")
            self._batch_upsert("title_vectors", image_ids, titles, kind="text")
            self._batch_upsert("description_vectors", image_ids, descriptions, kind="text")
            print(f"批量向量生成完成，共处理 {len(image_data_list)} 张图片")
            return image_ids
        except Exception as e:
            print(f"批量向量生成失败: {e}")
            return self._fallback_single_processing(image_data_list)

    def _batch_upsert(self, table_name: str, image_ids: List[int], payloads: List[Any], kind: str):
        valid_ids = []
        valid_payloads = []
        if kind == "image":
            for image_id, path in zip(image_ids, payloads):
                if path and Path(path).exists():
                    valid_ids.append(image_id)
                    valid_payloads.append(path)
            if not valid_ids:
                print("没有有效的图片文件需要处理")
                return
            from ...ai_func.generate_vector import encode_image

            print(f"批量生成图像向量，共 {len(valid_payloads)} 张图片")
            vectors = encode_image(list(valid_payloads))
        else:
            for image_id, text in zip(image_ids, payloads):
                if text and str(text).strip():
                    valid_ids.append(image_id)
                    valid_payloads.append(str(text).strip())
            if not valid_ids:
                print(f"没有有效的文本需要写入 {table_name}")
                return
            from ...ai_func.generate_vector import encode_text

            print(f"批量生成文本向量，共 {len(valid_payloads)} 条 -> {table_name}")
            vectors = encode_text(list(valid_payloads))

        cursor = self.conn.cursor()
        for image_id, vector in zip(valid_ids, vectors):
            vec_lit = _as_vector_literal(vector)
            cursor.execute(
                f"""
                INSERT INTO {table_name}(image_id, embedding)
                VALUES (%s, %s::vector)
                ON CONFLICT (image_id) DO UPDATE SET embedding = EXCLUDED.embedding
                """,
                (image_id, vec_lit),
            )
        self.conn.commit()
        print(f"批量保存 {table_name} 完成，共保存 {len(valid_ids)} 个向量")

    def _fallback_single_processing(self, image_data_list: List[Dict[str, Any]]) -> List[int]:
        print("批量处理失败，使用逐个处理模式")
        success_ids = []
        from .vectors import VectorRepository

        vector_repo = VectorRepository(self.conn)
        for data in image_data_list:
            try:
                image_id = data["image_id"]
                vector_repo.add_image_vector(image_id, data["filepath"])
                vector_repo.add_title_vector(image_id, data.get("title", ""))
                vector_repo.add_description_vector(image_id, data.get("description", ""))
                success_ids.append(image_id)
            except Exception as e:
                print(f"处理图片 ID {data.get('image_id')} 失败: {e}")
        return success_ids
