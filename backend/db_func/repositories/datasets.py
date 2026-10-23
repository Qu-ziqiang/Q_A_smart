"""数据集与类别标签仓库（实验室资产域）。

- datasets：限定域 + 词表
- image_labels：精确类别（COCO 式多类 AND / OR）
- images.dataset_id：图片归属
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .base import BaseRepository
from ..utils.common import python_to_json_for_db

# VOC 2010 官方 20 类（英文词表；中文别名由上层映射）
VOC2010_LABELS = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class DatasetRepository(BaseRepository):
    """数据集 CRUD 与按类别筛选。"""

    def ensure_schema_seed(self) -> None:
        """幂等：保证表存在后写入 default / voc2010（若尚无）。"""
        self._ensure_dataset("default", "默认图库", [])
        self._ensure_dataset("voc2010", "VOC2010", VOC2010_LABELS)

    def _ensure_dataset(self, name: str, title: str, labels: Sequence[str]) -> int:
        row = self.execute_query_one(
            "SELECT id FROM datasets WHERE name = %s", (name,)
        )
        if row:
            return int(row["id"])
        now = datetime.now().isoformat(timespec="seconds")
        self.execute_update(
            """
            INSERT INTO datasets (name, title, label_vocab, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                name,
                title,
                python_to_json_for_db(list(labels), []),
                now,
                now,
            ),
        )
        row = self.execute_query_one(
            "SELECT id FROM datasets WHERE name = %s", (name,)
        )
        return int(row["id"]) if row else 0

    def get_by_id(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        row = self.execute_query_one(
            "SELECT * FROM datasets WHERE id = %s", (int(dataset_id),)
        )
        return self._normalize(row) if row else None

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        key = (name or "").strip()
        if not key:
            return None
        row = self.execute_query_one(
            "SELECT * FROM datasets WHERE lower(name) = lower(%s) OR lower(title) = lower(%s)",
            (key, key),
        )
        if row:
            return self._normalize(row)
        # 常见别名
        aliases = {
            "voc": "voc2010",
            "voc2010": "voc2010",
            "pascal": "voc2010",
            "pascal voc": "voc2010",
            "默认": "default",
            "default": "default",
        }
        mapped = aliases.get(key.lower())
        if mapped:
            row = self.execute_query_one(
                "SELECT * FROM datasets WHERE name = %s", (mapped,)
            )
            return self._normalize(row) if row else None
        return None

    def list_datasets(self) -> List[Dict[str, Any]]:
        rows = self.execute_query(
            "SELECT * FROM datasets ORDER BY id ASC"
        ) or []
        return [self._normalize(r) for r in rows]

    def label_vocab(self, dataset_id: int) -> List[str]:
        ds = self.get_by_id(dataset_id)
        if not ds:
            return []
        return list(ds.get("label_vocab") or [])

    def set_image_dataset(self, image_id: int, dataset_id: int) -> bool:
        n = self.execute_update(
            "UPDATE images SET dataset_id = %s WHERE id = %s",
            (int(dataset_id), int(image_id)),
        )
        return n > 0

    def replace_image_labels(self, image_id: int, labels: Sequence[str]) -> List[str]:
        clean = []
        seen = set()
        for raw in labels:
            lab = str(raw).strip()
            if not lab or lab in seen:
                continue
            seen.add(lab)
            clean.append(lab)
        self.execute_update(
            "DELETE FROM image_labels WHERE image_id = %s", (int(image_id),)
        )
        for lab in clean:
            self.execute_update(
                "INSERT INTO image_labels (image_id, label) VALUES (%s, %s) "
                "ON CONFLICT (image_id, label) DO NOTHING",
                (int(image_id), lab),
            )
        return clean

    def sync_labels_from_tags_json(
        self, dataset_id: Optional[int] = None, limit: int = 0
    ) -> int:
        """把 images.tags JSON 同步进 image_labels（幂等追加）。"""
        sql = "SELECT id, tags, dataset_id FROM images"
        params: List[Any] = []
        if dataset_id is not None:
            sql += " WHERE dataset_id = %s"
            params.append(int(dataset_id))
        if limit and limit > 0:
            sql += " ORDER BY id LIMIT %s"
            params.append(int(limit))
        rows = self.execute_query(sql, tuple(params)) or []
        synced = 0
        for row in rows:
            tags = row.get("tags") or []
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags) if tags else []
                except json.JSONDecodeError:
                    tags = []
            if not isinstance(tags, list) or not tags:
                continue
            before = self.execute_query(
                "SELECT label FROM image_labels WHERE image_id = %s",
                (int(row["id"]),),
            ) or []
            existing = {r["label"] for r in before}
            added = False
            for t in tags:
                lab = str(t).strip()
                if not lab or lab in existing:
                    continue
                self.execute_update(
                    "INSERT INTO image_labels (image_id, label) VALUES (%s, %s) "
                    "ON CONFLICT (image_id, label) DO NOTHING",
                    (int(row["id"]), lab),
                )
                existing.add(lab)
                added = True
            if added:
                synced += 1
        return synced

    @staticmethod
    def _normalize(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        out = dict(row)
        vocab = out.get("label_vocab")
        if isinstance(vocab, str):
            try:
                vocab = json.loads(vocab) if vocab else []
            except json.JSONDecodeError:
                vocab = []
        out["label_vocab"] = list(vocab) if isinstance(vocab, list) else []
        return out
