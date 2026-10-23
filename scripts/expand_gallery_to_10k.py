"""从 NACLIP/mmsegmentation/data 收集小 JPEG，写入 PostgreSQL 至约 10000 张并批量 CLIP 向量化。

优先 VOCdevkit JPEGImages（体积小）。已存在的 filepath 跳过。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC_ROOT = Path(os.environ.get(
    "GALLERY_SRC", r"D:\document\py_prj\NACLIP\mmsegmentation\data"
))
_TARGET = int(os.environ.get("GALLERY_TARGET", "10000"))
_MAX_BYTES = int(os.environ.get("GALLERY_MAX_BYTES", str(400 * 1024)))
_BATCH = int(os.environ.get("GALLERY_BATCH", "24"))


def _iter_small_jpegs(root: Path):
    preferred = [
        root / "VOCdevkit",
        root / "coco_object",
        root / "coco_stuff164k",
        root / "ADEChanllengeData2016",
        root / "cityscapes",
    ]
    seen = set()
    for base in preferred:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            try:
                if p.stat().st_size > _MAX_BYTES:
                    continue
            except OSError:
                continue
            key = p.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield p


def _existing_filepaths(conn) -> set:
    cur = conn.cursor()
    cur.execute("SELECT filepath FROM images")
    return {str(r["filepath"] if isinstance(r, dict) else r[0]) for r in cur.fetchall()}


def _count(conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM images")
    row = cur.fetchone()
    return int(row["n"] if isinstance(row, dict) else row[0])


def main() -> int:
    from backend.db_func.core.connection import get_db_connection
    from backend.db_func.core.database import init_db
    from backend.db_func.repositories.images import ImageRepository
    from backend.db_func.repositories.batch_vector_manager import BatchVectorManager

    if not _SRC_ROOT.exists():
        print(f"源目录不存在: {_SRC_ROOT}")
        return 2
    init_db()
    with get_db_connection() as conn:
        n0 = _count(conn)
        print(f"[gallery] 当前 images = {n0}，目标 {_TARGET}")
        if n0 >= _TARGET:
            print("[gallery] 已达目标，跳过导入")
            return 0
        existing = _existing_filepaths(conn)
        repo = ImageRepository()
        need = _TARGET - n0
        pending = []
        scanned = 0
        for src in _iter_small_jpegs(_SRC_ROOT):
            scanned += 1
            fp = str(src)
            if fp in existing:
                continue
            size = src.stat().st_size
            pending.append({
                "filename": src.name,
                "filepath": fp,
                "title": src.stem,
                "description": f"imported from {src.parent.name}",
                "file_size": size,
                "file_type": src.suffix.lower().lstrip("."),
                "width": 0,
                "height": 0,
                "metadata": {"source": "naclip_mmseg", "rel": str(src)},
                "tags": [src.parent.name[:40]],
            })
            existing.add(fp)
            if len(pending) >= need:
                break
        print(f"[gallery] scanned={scanned} new={len(pending)}")
        if not pending:
            print("[gallery] 没有新的小 JPEG")
            return 1
        ids = repo.batch_create(pending)
        print(f"[gallery] inserted {len(ids)} rows")
        payloads = []
        for iid, meta in zip(ids, pending):
            payloads.append({
                "image_id": iid,
                "filepath": meta["filepath"],
                "title": meta["title"],
                "description": meta["description"],
            })
        mgr = BatchVectorManager(conn)
        for i in range(0, len(payloads), _BATCH):
            chunk = payloads[i:i + _BATCH]
            mgr.add_batch_vectors(chunk)
            print(f"[gallery] vectors {i + len(chunk)}/{len(payloads)}")
        conn.commit()
        n1 = _count(conn)
        print(f"[gallery] done images={n1}")
        out = _REPO / "backend" / "ai_func" / "recommendation" / "evaluation" / "results" / "gallery_expand.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        import json
        out.write_text(json.dumps({
            "before": n0, "inserted": len(ids), "after": n1, "target": _TARGET,
            "src": str(_SRC_ROOT),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
