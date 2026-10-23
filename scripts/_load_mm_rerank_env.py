"""从 .env / 环境灌 MM_RERANK_*，不打印 Key。"""

from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ENV = _REPO / ".env"


def _read_dotenv(path: Path) -> dict:
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, _, v = raw.partition("=")
        rows[k.strip()] = v.strip().strip('"').strip("'")
    return rows


def mm_rerank_ready() -> bool:
    keys = (
        "MM_RERANK_PROVIDER",
        "MM_RERANK_API_KEY",
        "MM_RERANK_BASE_URL",
        "MM_RERANK_MODEL",
    )
    return all((os.environ.get(k) or "").strip() for k in keys)


def load_mm_rerank_env() -> bool:
    """补齐 MM_RERANK_*。已有环境变量不覆盖。返回是否齐全。"""
    rows = _read_dotenv(_ENV)
    for k, v in rows.items():
        if k and k not in os.environ and v:
            os.environ[k] = v

    key = (
        os.environ.get("MM_RERANK_API_KEY")
        or os.environ.get("ZHIPU_API_KEY")
        or os.environ.get("GLM_API_KEY")
        or ""
    ).strip()
    base_hint = (
        os.environ.get("MM_RERANK_BASE_URL")
        or os.environ.get("LLM_JUDGE_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    ).lower()
    # .env 里智谱 key 常写在 OPENAI_* / SMARTIMAGER_*，只要 base 是 bigmodel 就能给 R2 用。
    # 不把 DeepSeek key 借给 GLM-4V。
    zhipu_base = "bigmodel" in base_hint or "zhipuai" in base_hint
    if not key and zhipu_base:
        key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("SMARTIMAGER_LLM_API_KEY")
            or os.environ.get("LLM_JUDGE_API_KEY")
            or ""
        ).strip()
    if key and not os.environ.get("MM_RERANK_API_KEY"):
        os.environ["MM_RERANK_API_KEY"] = key
    if not os.environ.get("MM_RERANK_PROVIDER"):
        if key or os.environ.get("MM_RERANK_BASE_URL"):
            os.environ["MM_RERANK_PROVIDER"] = "openai_chat"
    if not os.environ.get("MM_RERANK_BASE_URL") and key:
        os.environ["MM_RERANK_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"
    if not os.environ.get("MM_RERANK_MODEL") and key:
        os.environ["MM_RERANK_MODEL"] = "glm-4v-flash"
    if not os.environ.get("MM_RERANK_TIMEOUT_SECONDS"):
        os.environ["MM_RERANK_TIMEOUT_SECONDS"] = "90"
    return mm_rerank_ready()


def key_prefix() -> str:
    raw = (os.environ.get("MM_RERANK_API_KEY") or "").strip()
    return raw[:4] if raw else ""
