"""
搜索数据仓库
处理搜索相关的所有数据库操作（PostgreSQL + pgvector）
"""
import logging
from typing import Dict, List, Any, Optional, Tuple

from .base import BaseRepository
from .vectors import _as_vector_literal
from ...utils.image_utils import build_public_url

logger = logging.getLogger(__name__)


class SearchRepository(BaseRepository):
    """搜索数据仓库类"""

    _ALLOWED_VECTOR_TARGETS = {"title", "description", "image"}

    @classmethod
    def _sanitize_targets(cls, targets: List[str]) -> List[str]:
        valid = [t for t in targets if isinstance(t, str) and t in cls._ALLOWED_VECTOR_TARGETS]
        return valid or ["title", "description", "image"]

    def _build_image_filter_sql(
        self, filters: Optional[Dict[str, Any]], alias: str = "i"
    ) -> Tuple[str, List[Any]]:
        """在 images 表上构建库侧过滤条件（元数据同库）。"""
        filters = filters or {}
        conditions: List[str] = []
        params: List[Any] = []
        prefix = f"{alias}." if alias else ""

        if filters.get("start_date"):
            conditions.append(f"{prefix}created_at >= %s")
            params.append(filters["start_date"])
        if filters.get("end_date"):
            conditions.append(f"{prefix}created_at <= %s")
            params.append(filters["end_date"])
        if filters.get("dataset_id") is not None:
            conditions.append(f"{prefix}dataset_id = %s")
            params.append(int(filters["dataset_id"]))
        if filters.get("labels"):
            labels = [str(x).strip() for x in filters["labels"] if str(x).strip()]
            if labels:
                match = str(filters.get("label_match") or "all").lower()
                if match == "any":
                    conditions.append(
                        f"{prefix}id IN ("
                        f"SELECT image_id FROM image_labels WHERE label = ANY(%s)"
                        f")"
                    )
                    params.append(labels)
                else:
                    conditions.append(
                        f"{prefix}id IN ("
                        f"SELECT image_id FROM image_labels WHERE label = ANY(%s) "
                        f"GROUP BY image_id HAVING COUNT(DISTINCT label) >= %s"
                        f")"
                    )
                    params.append(labels)
                    params.append(len(labels))
        if filters.get("tags"):
            tag_conditions = []
            for tag in filters["tags"]:
                tag_conditions.append(f"{prefix}tags LIKE %s")
                params.append(f'%"{tag}"%')
            conditions.append("(" + " OR ".join(tag_conditions) + ")")
        if filters.get("keyword"):
            keyword_conditions = []
            for field in ["title", "description", "filename"]:
                keyword_conditions.append(f"{prefix}{field} LIKE %s")
                params.append(f"%{filters['keyword']}%")
            conditions.append("(" + " OR ".join(keyword_conditions) + ")")
        if filters.get("file_type"):
            conditions.append(f"{prefix}file_type = %s")
            params.append(filters["file_type"])
        if filters.get("min_width") is not None:
            conditions.append(f"{prefix}width >= %s")
            params.append(filters["min_width"])
        if filters.get("max_width") is not None:
            conditions.append(f"{prefix}width <= %s")
            params.append(filters["max_width"])
        if filters.get("min_height") is not None:
            conditions.append(f"{prefix}height >= %s")
            params.append(filters["min_height"])
        if filters.get("max_height") is not None:
            conditions.append(f"{prefix}height <= %s")
            params.append(filters["max_height"])
        if filters.get("ids"):
            conditions.append(f"{prefix}id = ANY(%s)")
            params.append([int(x) for x in filters["ids"]])

        where = (" AND ".join(conditions)) if conditions else ""
        return where, params

    def search_with_filters(
        self, filters: Dict[str, Any], limit: int = 20, offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        where, params = self._build_image_filter_sql(filters, alias="")
        query = "SELECT * FROM images"
        count_query = "SELECT COUNT(*) as count FROM images"
        if where:
            query += " WHERE " + where
            count_query += " WHERE " + where

        sort_by = filters.get("sort_by", "created_at")
        allowed_sort = {"created_at", "updated_at", "filename", "file_size", "width", "height"}
        if sort_by not in allowed_sort:
            sort_by = "created_at"
        order = "ASC" if str(filters.get("order", "desc")).lower() == "asc" else "DESC"
        query += f" ORDER BY {sort_by} {order} LIMIT %s OFFSET %s"

        results = self.execute_query(query, tuple(params) + (limit, offset))
        count_result = self.execute_query_one(count_query, tuple(params))
        total_count = count_result["count"] if count_result else 0
        return results, total_count

    def fuzzy_search(
        self,
        text: str,
        fields: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []
        fields = fields or ["title", "description"]
        allowed = {"title", "description", "filename"}
        use_fields = [f for f in fields if f in allowed] or ["title", "description"]

        query = "SELECT * FROM images"
        conditions: List[str] = []
        params: List[Any] = []

        like_exprs = []
        kw = f"%{text.strip()}%"
        for f in use_fields:
            like_exprs.append(f"{f} LIKE %s")
            params.append(kw)
        conditions.append("(" + " OR ".join(like_exprs) + ")")

        filters = filters or {}
        if filters.get("tags"):
            tag_conditions = []
            for tag in filters["tags"]:
                tag_conditions.append("tags LIKE %s")
                params.append(f'%"{tag}"%')
            conditions.append("(" + " OR ".join(tag_conditions) + ")")
        if filters.get("start_date"):
            conditions.append("created_at >= %s")
            params.append(filters["start_date"])
        if filters.get("end_date"):
            conditions.append("created_at <= %s")
            params.append(filters["end_date"])

        query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        results = self.execute_query(query, tuple(params))
        return self._attach_public_url(results)

    def get_filtered_image_ids(self, filters: Dict[str, Any]) -> List[int]:
        where, params = self._build_image_filter_sql(filters, alias="")
        query = "SELECT id FROM images"
        if where:
            query += " WHERE " + where
        results = self.execute_query(query, tuple(params))
        return [row["id"] for row in results]

    def vector_search(
        self, vector_type: str, query_vector: list, top_k: int = 50
    ) -> List[Tuple[int, float]]:
        try:
            table_name = f"{vector_type}_vectors"
            vec_lit = _as_vector_literal(query_vector)
            query = f"""
            SELECT image_id, (embedding <=> %s::vector) AS distance
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            results = self.execute_query(query, (vec_lit, vec_lit, top_k))
            return [(row["image_id"], float(row["distance"])) for row in results]
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []

    def _vector_search_with_filter(
        self,
        vector_type: str,
        query_vector: list,
        filtered_ids: Optional[List[int]],
        top_k: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[int, float]]:
        try:
            if filtered_ids is not None and len(filtered_ids) == 0:
                return []

            table_name = f"{vector_type}_vectors"
            vec_lit = _as_vector_literal(query_vector)
            where_parts = []
            params: List[Any] = [vec_lit]

            if filtered_ids is not None:
                where_parts.append("v.image_id = ANY(%s)")
                params.append(list(filtered_ids))

            meta_where, meta_params = self._build_image_filter_sql(filters or {}, alias="i")
            join_sql = ""
            if meta_where:
                join_sql = "JOIN images i ON i.id = v.image_id"
                where_parts.append(meta_where)
                params.extend(meta_params)

            where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
            limit_n = min(top_k, len(filtered_ids)) if filtered_ids is not None else top_k
            params.append(vec_lit)
            params.append(limit_n)

            query = f"""
            SELECT v.image_id, (v.embedding <=> %s::vector) AS distance
            FROM {table_name} v
            {join_sql}
            {where_sql}
            ORDER BY v.embedding <=> %s::vector
            LIMIT %s
            """
            results = self.execute_query(query, tuple(params))
            return [(row["image_id"], float(row["distance"])) for row in results]
        except Exception as e:
            logger.error(f"过滤向量搜索失败: {str(e)}")
            return []

    def search_by_image_vector(self, query_vector: list, top_k: int = 50) -> List[Tuple[int, float]]:
        return self.vector_search("image", query_vector, top_k)

    def search_by_title_vector(self, query_vector: list, top_k: int = 50) -> List[Tuple[int, float]]:
        return self.vector_search("title", query_vector, top_k)

    def search_by_description_vector(
        self, query_vector: list, top_k: int = 50
    ) -> List[Tuple[int, float]]:
        return self.vector_search("description", query_vector, top_k)

    def image_search(
        self,
        image_path: str,
        search_targets: List[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = 20,
        offset: int = 0,
        weights: Optional[Dict[str, float]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        from ...ai_func.generate_vector import encode_image

        search_targets = search_targets or ["image"]
        query_embedding = encode_image(image_path).tolist()
        return self._multi_vector_search(
            query_embedding=query_embedding,
            search_targets=search_targets,
            filters=filters,
            limit=limit,
            offset=offset,
            weights=weights,
            min_score=min_score,
        )

    def _get_images_by_ids(self, image_ids: List[int]) -> List[Dict[str, Any]]:
        if not image_ids:
            return []
        return self.execute_query("SELECT * FROM images WHERE id = ANY(%s)", (list(image_ids),))

    def text_search(
        self,
        text_query: str,
        search_targets: List[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = 20,
        offset: int = 0,
        weights: Optional[Dict[str, float]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        from ...ai_func.generate_vector import encode_text

        search_targets = search_targets or ["title", "description", "image"]
        query_embedding = encode_text(text_query).tolist()
        return self._multi_vector_search(
            query_embedding=query_embedding,
            search_targets=self._sanitize_targets(search_targets),
            filters=filters,
            limit=limit,
            offset=offset,
            weights=weights,
            min_score=min_score,
        )

    def unified_search(
        self,
        query_type: str,
        query_content: str,
        search_targets: List[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = 20,
        offset: int = 0,
        weights: Optional[Dict[str, float]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        search_targets = search_targets or ["title", "description", "image"]
        if query_type == "text":
            return self.text_search(
                query_content,
                self._sanitize_targets(search_targets),
                filters,
                limit,
                offset,
                weights=weights,
                min_score=min_score,
            )
        if query_type == "image":
            return self.image_search(
                query_content,
                search_targets,
                filters,
                limit,
                offset,
                weights=weights,
                min_score=min_score,
            )
        raise ValueError(f"不支持的查询类型: {query_type}，支持的类型：'text', 'image'")

    def vector_search_direct(
        self,
        query_embedding: list,
        search_targets: List[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = 20,
        offset: int = 0,
        weights: Optional[Dict[str, float]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        search_targets = search_targets or ["title", "description", "image"]
        return self._multi_vector_search(
            query_embedding=query_embedding,
            search_targets=search_targets,
            filters=filters,
            limit=limit,
            offset=offset,
            weights=weights,
            min_score=min_score,
        )

    def _multi_vector_search(
        self,
        query_embedding: list,
        search_targets: List[str],
        filters: Dict[str, Any] = None,
        limit: int = 20,
        offset: int = 0,
        weights: Optional[Dict[str, float]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        search_targets = self._sanitize_targets(search_targets)
        filters = filters or {}

        has_filter = bool(
            filters.get("start_date")
            or filters.get("end_date")
            or filters.get("tags")
            or filters.get("keyword")
            or filters.get("file_type")
            or filters.get("ids")
            or filters.get("min_width") is not None
            or filters.get("max_width") is not None
            or filters.get("min_height") is not None
            or filters.get("max_height") is not None
        )
        filtered_ids = self.get_filtered_image_ids(filters) if has_filter else None
        if has_filter and not filtered_ids:
            return []

        if len(search_targets) == 1:
            vector_results = self._vector_search_with_filter(
                search_targets[0],
                query_embedding,
                filtered_ids,
                limit + offset,
                filters=filters if has_filter else None,
            )
            if not vector_results:
                return []
            paginated = vector_results[offset : offset + limit]
            ids = [r[0] for r in paginated]
            if not ids:
                return []
            images = self._get_images_by_ids(ids)
            id2score = {iid: 1 - dist for iid, dist in paginated}
            id2dist = {iid: dist for iid, dist in paginated}
            id_order = {iid: idx for idx, iid in enumerate(ids)}
            out = []
            for img in images:
                score = id2score.get(img["id"], 0.0)
                if min_score is not None and score < min_score:
                    continue
                img["score"] = score
                img["distance"] = id2dist.get(img["id"])
                out.append(img)
            out.sort(key=lambda x: id_order.get(x["id"], 10**9))
            return self._attach_public_url(out)

        weights = weights or {}
        norm_weights: Dict[str, float] = {}
        for t in search_targets:
            w = float(weights.get(t, 1.0))
            if w > 0:
                norm_weights[t] = w
        if not norm_weights:
            norm_weights = {t: 1.0 for t in search_targets}

        per_target_scores: Dict[str, Dict[int, float]] = {}
        candidate_ids: set = set()
        fetch_k = max(limit * 2, limit + offset)
        for target in search_targets:
            vres = self._vector_search_with_filter(
                target,
                query_embedding,
                filtered_ids,
                fetch_k,
                filters=filters if has_filter else None,
            )
            target_map: Dict[int, float] = {}
            for iid, dist in vres:
                score = 1 - dist
                target_map[iid] = max(target_map.get(iid, 0.0), score)
                candidate_ids.add(iid)
            per_target_scores[target] = target_map

        scored: List[Tuple[int, float]] = []
        for iid in candidate_ids:
            weighted_sum = 0.0
            weight_sum = 0.0
            for t, w in norm_weights.items():
                s = per_target_scores.get(t, {}).get(iid)
                if s is not None:
                    weighted_sum += s * w
                    weight_sum += w
            if weight_sum == 0:
                continue
            final_score = weighted_sum / weight_sum
            if min_score is not None and final_score < min_score:
                continue
            scored.append((iid, final_score))

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        page = scored[offset : offset + limit]
        if not page:
            return []
        ids = [i for i, _ in page]
        images = self._get_images_by_ids(ids)
        id2score = {i: s for i, s in page}
        min_dist_ref: Dict[int, float] = {}
        for tmap in per_target_scores.values():
            for iid, s in tmap.items():
                dist = 1 - s
                if iid not in min_dist_ref or dist < min_dist_ref[iid]:
                    min_dist_ref[iid] = dist
        for img in images:
            img["score"] = id2score.get(img["id"], 0.0)
            img["distance"] = min_dist_ref.get(img["id"])
        images.sort(key=lambda x: x["score"], reverse=True)
        return self._attach_public_url(images)

    def search_by_image_id(
        self,
        image_id: int,
        vector_type: str = "image",
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        exclude_self: bool = True,
    ) -> List[Dict[str, Any]]:
        if not self.execute_query_one("SELECT id FROM images WHERE id = %s", (image_id,)):
            logger.warning(f"图像ID {image_id} 不存在于数据库中")
            return []

        vector_table = f"{vector_type}_vectors"
        vector_count_result = self.execute_query_one(
            f"SELECT COUNT(*) as count FROM {vector_table} WHERE image_id = %s",
            (image_id,),
        )
        if not vector_count_result or vector_count_result.get("count", 0) == 0:
            logger.warning(f"图像ID {image_id} 在 {vector_table} 表中不存在向量")
            return []

        filters = filters or {}
        where_parts = ["v.image_id <> %s"] if exclude_self else []
        params: List[Any] = []
        if exclude_self:
            params.append(image_id)

        meta_where, meta_params = self._build_image_filter_sql(filters, alias="img")
        if meta_where:
            where_parts.append(meta_where)
            params.extend(meta_params)

        where_sql = (" AND " + " AND ".join(where_parts)) if where_parts else ""
        sql_query = f"""
        WITH query_vector AS (
            SELECT embedding FROM {vector_table} WHERE image_id = %s
        )
        SELECT
            img.id,
            img.filename,
            img.filepath,
            img.title,
            img.description,
            img.file_size,
            img.file_type,
            img.width,
            img.height,
            img.created_at,
            img.updated_at,
            img.metadata,
            img.tags,
            (v.embedding <=> (SELECT embedding FROM query_vector)) AS distance,
            (1 - (v.embedding <=> (SELECT embedding FROM query_vector))) AS score
        FROM {vector_table} v
        JOIN images img ON img.id = v.image_id
        WHERE TRUE
        {where_sql}
        ORDER BY v.embedding <=> (SELECT embedding FROM query_vector)
        LIMIT %s
        """
        try:
            results = self.execute_query(sql_query, tuple([image_id] + params + [k]))
            results = self._attach_public_url(results)
            logger.info(
                f"使用图像ID {image_id} 的 {vector_type} 向量找到 {len(results)} 个相似结果"
            )
            return results
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []

    def search_by_labels(
        self,
        labels: List[str],
        *,
        dataset_id: Optional[int] = None,
        match: str = "all",
        limit: int = 40,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """按精确类别表筛选（默认多类 AND），不走 CLIP。"""
        clean = [str(x).strip() for x in (labels or []) if str(x).strip()]
        if not clean:
            return []
        filters: Dict[str, Any] = {
            "labels": clean,
            "label_match": "any" if str(match).lower() == "any" else "all",
        }
        if dataset_id is not None:
            filters["dataset_id"] = int(dataset_id)
        rows, _ = self.search_with_filters(filters, limit=limit, offset=offset)
        for r in rows:
            r.setdefault("score", 1.0)
        return self._attach_public_url(rows)

    def _attach_public_url(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for img in images:
            if not img.get("public_url"):
                fp = img.get("filepath") or ""
                if fp:
                    img["public_url"] = build_public_url(fp)
        return images
