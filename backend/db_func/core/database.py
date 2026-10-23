"""
数据库初始化：PostgreSQL + pgvector。

仅负责 init_db：扩展 / 表 / HNSW 索引。
运行期获取连接统一使用 core.connection.get_db。
"""
from ...ai_func.generate_vector import get_embedding_dimension
from .connection import (
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    get_db_connection,
)

_INIT_DONE = False


def init_db():
    """初始化 PostgreSQL 表结构与 pgvector 索引。"""
    global _INIT_DONE
    if _INIT_DONE:
        print("init_db 已执行，跳过")
        return

    from ...config import settings

    cfg = settings.get_config()
    embedding_dim = getattr(cfg, "EMBEDDING_DIMENSION", None) or get_embedding_dimension()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cursor.fetchone()
        ver = row["extversion"] if row else "unknown"
        print(f"pgvector 扩展就绪: {ver}")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id BIGSERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                title TEXT,
                description TEXT,
                file_size BIGINT NOT NULL,
                file_type TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                tags TEXT
            )
            """
        )
        # 数据集域（实验室资产）：图片归属 + 精确类别表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                title TEXT,
                label_vocab TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            "ALTER TABLE images ADD COLUMN IF NOT EXISTS dataset_id BIGINT "
            "REFERENCES datasets(id) ON DELETE SET NULL"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_dataset_id ON images(dataset_id)"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS image_labels (
                image_id BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                PRIMARY KEY (image_id, label)
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_image_labels_label ON image_labels(label)"
        )

        for table in ("title_vectors", "description_vectors", "image_vectors"):
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id BIGSERIAL PRIMARY KEY,
                    image_id BIGINT UNIQUE NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                    embedding vector({embedding_dim}) NOT NULL
                )
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_embedding_hnsw
                ON {table}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {int(HNSW_M)}, ef_construction = {int(HNSW_EF_CONSTRUCTION)})
                """
            )

        print(
            f"创建向量表成功，向量维度: {embedding_dim}，"
            f"索引: HNSW(cosine, m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})"
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_width ON images(width)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_height ON images(height)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_file_type ON images(file_type)"
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS request_sessions (
                id BIGSERIAL PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL,
                user_id TEXT,
                endpoint TEXT NOT NULL,
                messages TEXT,
                state TEXT,
                vector_targets TEXT,
                filters TEXT,
                selected_ids TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_request_sessions_conv ON request_sessions(conversation_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_request_sessions_created ON request_sessions(created_at)"
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id BIGSERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                image_ids TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_messages_conv_created "
            "ON conversation_messages(conversation_id, created_at)"
        )

        from ..repositories.agent_episodes import (
            AGENT_EPISODES_INDEXES_SQL,
            AGENT_EPISODES_TABLE_SQL,
        )

        cursor.execute(AGENT_EPISODES_TABLE_SQL)
        for idx_sql in AGENT_EPISODES_INDEXES_SQL:
            cursor.execute(idx_sql)

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_migrations (
                id BIGSERIAL PRIMARY KEY,
                old_model_path TEXT,
                new_model_path TEXT NOT NULL,
                new_model_hash TEXT NOT NULL,
                new_embedding_dim INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                total_images INTEGER,
                processed_images INTEGER,
                last_image_id BIGINT,
                note TEXT
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_migrations_status ON model_migrations(status)"
        )

        conn.commit()

    try:
        from ..repositories.datasets import DatasetRepository

        DatasetRepository().ensure_schema_seed()
    except Exception as e:
        print(f"datasets seed 跳过: {type(e).__name__}: {e}")

    _INIT_DONE = True
    print("数据库初始化完成 (PostgreSQL + pgvector)")
