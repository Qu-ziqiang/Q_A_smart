"""
服务初始化模块
只负责初始化和获取 OpenAI 客户端。
"""

from openai import OpenAI
from .config import settings

_openai_client = None
_vision_client = None


def _dotenv_rows() -> dict:
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / ".env"
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        rows[key.strip()] = value.strip().strip('"').strip("'")
    return rows


def get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    config = settings.get_config()
    if not config or not config.OPENAI_API_KEY:
        print("警告: OPENAI_API_KEY 未配置，OpenAI 客户端初始化失败")
        return None
    client_kwargs = {"api_key": config.OPENAI_API_KEY}
    if config.OPENAI_API_BASE:
        client_kwargs["base_url"] = config.OPENAI_API_BASE
    _openai_client = OpenAI(**client_kwargs)
    print(f"OpenAI 客户端初始化成功，API Base: {config.OPENAI_API_BASE or 'default'}")
    return _openai_client


def get_vision_client():
    """看图客户端。不复用 DeepSeek 的对话密钥。"""
    global _vision_client
    if _vision_client is not None:
        return _vision_client
    config = settings.get_config()
    key = (getattr(config, "VISION_API_KEY", None) or "").strip()
    base = (getattr(config, "VISION_API_BASE", None) or "").strip()
    if not key or not base:
        rows = _dotenv_rows()
        base_hint = (rows.get("OPENAI_API_BASE") or rows.get("LLM_JUDGE_BASE_URL") or "").lower()
        if "bigmodel" in base_hint or "zhipuai" in base_hint:
            key = key or (rows.get("OPENAI_API_KEY") or rows.get("LLM_JUDGE_API_KEY") or "").strip()
            base = base or (rows.get("OPENAI_API_BASE") or rows.get("LLM_JUDGE_BASE_URL") or "").strip()
    if not key or not base:
        return None
    chat_key = (getattr(config, "OPENAI_API_KEY", None) or "").strip()
    chat_base = (getattr(config, "OPENAI_API_BASE", None) or "").strip()
    if key == chat_key and base == chat_base and "deepseek" in chat_base.lower():
        return None
    _vision_client = OpenAI(api_key=key, base_url=base)
    print(f"视觉客户端已连接，API Base: {base}")
    return _vision_client


