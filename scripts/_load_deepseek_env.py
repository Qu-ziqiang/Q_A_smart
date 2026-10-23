"""从 config.yaml 灌 DeepSeek 环境变量，不打印完整 Key。供评测脚本调用。"""
from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CFG = _REPO / "backend" / "config" / "files" / "config.yaml"


def load_deepseek_env() -> str:
    key = ""
    for line in _CFG.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY:"):
            key = line.split(":", 1)[1].strip().strip('"').strip("'")
            break
    if not key:
        raise SystemExit("config.yaml 缺少 OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = key
    os.environ["LLM_JUDGE_API_KEY"] = key
    os.environ["SMARTIMAGER_LLM_API_KEY"] = key
    os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"
    os.environ["LLM_JUDGE_BASE_URL"] = "https://api.deepseek.com"
    os.environ["CHAT_MODEL"] = "deepseek-chat"
    os.environ["LLM_JUDGE_MODEL"] = "deepseek-chat"
    return key[:6]
