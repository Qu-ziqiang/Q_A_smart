"""Agent Episode 仓库：conversation-scoped episodic memory（设计 §9）。

表结构（additive，单一来源 `AGENT_EPISODES_TABLE_SQL`）：
- conversation_id 是主 scope；user_id 仅当客户端传入时冗余保存，不参与 recall。
- 只存 deterministic task-level summary；不存 raw prompt/response/CoT/path/base64。
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .base import BaseRepository

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

#: agent_episodes DDL —— 单一来源（database.py 与测试共用）
AGENT_EPISODES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_episodes (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    request_id TEXT,
    user_id TEXT,
    route TEXT NOT NULL,
    task_types TEXT NOT NULL,
    query_summary TEXT NOT NULL,
    strategy_summary TEXT NOT NULL,
    result_summary TEXT,
    preference_hints TEXT,
    selected_count INTEGER DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1,
    plan_valid INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
)
"""

AGENT_EPISODES_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_agent_episodes_conv_created "
    "ON agent_episodes(conversation_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_agent_episodes_expires ON agent_episodes(expires_at)",
]


def _now_iso() -> str:
    return datetime.utcnow().strftime(ISO)


def _expires_at(created: str, ttl_days: int = 30) -> str:
    try:
        dt = datetime.strptime(created, ISO)
    except ValueError:
        dt = datetime.utcnow()
    return (dt + timedelta(days=ttl_days)).strftime(ISO)


class AgentEpisodeRepository(BaseRepository):
    """agent_episodes 最小 repository（参数化 SQL，conversation_id 主 scope）。"""

    def insert(self, episode: Dict[str, Any], ttl_days: int = 30) -> int:
        """插入一条 episode，返回新记录 id。"""
        created = episode.get("created_at") or _now_iso()
        q = (
            "INSERT INTO agent_episodes "
            "(conversation_id, request_id, user_id, route, task_types, query_summary, "
            " strategy_summary, result_summary, preference_hints, selected_count, "
            " success, plan_valid, created_at, expires_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        params = (
            episode["conversation_id"],
            episode.get("request_id"),
            episode.get("user_id"),
            episode.get("route") or "team",
            json.dumps(episode.get("task_types") or [], ensure_ascii=False),
            (episode.get("query_summary") or "")[:120],
            (episode.get("strategy_summary") or "")[:200],
            (episode.get("result_summary") or "")[:200],
            json.dumps(episode.get("preference_hints") or [], ensure_ascii=False),
            int(episode.get("selected_count") or 0),
            int(bool(episode.get("success"))),
            int(bool(episode.get("plan_valid", True))),
            created,
            episode.get("expires_at") or _expires_at(created, ttl_days),
        )
        return self.execute_insert(q, params)

    def list_by_conversation(
        self, conversation_id: str, limit: int = 20, now: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """该会话未过期 episodes，按 created_at 倒序；自动排除 expires_at <= now。"""
        now = now or _now_iso()
        rows = self.execute_query(
            "SELECT * FROM agent_episodes WHERE conversation_id = ? AND expires_at > ? "
            "ORDER BY created_at DESC LIMIT ?",
            (conversation_id, now, limit),
        )
        return self._parse_rows(rows)

    def recall(
        self, conversation_id: str, limit: int = 3, now: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Top-N 未过期 episodes（repository 层 recall；相关性排序在 Memory Service）。"""
        return self.list_by_conversation(conversation_id, limit=limit, now=now)

    def purge_expired(self, now: Optional[str] = None) -> int:
        """删除全部过期 episodes，返回删除行数。"""
        return self.execute_delete(
            "DELETE FROM agent_episodes WHERE expires_at < ?", (now or _now_iso(),)
        )

    def delete_by_conversation(self, conversation_id: str) -> int:
        """删除某 conversation 的全部 episodes（DELETE conversation 级联）。"""
        return self.execute_delete(
            "DELETE FROM agent_episodes WHERE conversation_id = ?", (conversation_id,)
        )

    def prune_conversation(self, conversation_id: str, cap: int = 20) -> int:
        """per-conversation cap：仅保留最新 cap 条，删除其余。"""
        return self.execute_delete(
            "DELETE FROM agent_episodes WHERE conversation_id = ? AND id NOT IN ("
            "  SELECT id FROM agent_episodes WHERE conversation_id = ? "
            "  ORDER BY created_at DESC LIMIT ?)",
            (conversation_id, conversation_id, cap),
        )

    def count(self, conversation_id: Optional[str] = None) -> int:
        if conversation_id:
            row = self.execute_query_one(
                "SELECT COUNT(*) AS n FROM agent_episodes WHERE conversation_id = ?",
                (conversation_id,),
            )
        else:
            row = self.execute_query_one("SELECT COUNT(*) AS n FROM agent_episodes")
        return int((row or {}).get("n", 0))

    # -- 内部 ----------------------------------------------------------------
    @staticmethod
    def _parse_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for r in rows:
            for key in ("task_types", "preference_hints"):
                try:
                    r[key] = json.loads(r.get(key) or "[]")
                except Exception:
                    r[key] = []
        return rows
