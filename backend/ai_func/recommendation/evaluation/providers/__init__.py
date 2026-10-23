"""Phase 4.3 — Evaluation LLM providers（不绑定厂商）。

实现真实 LLMClient：OpenAICompatibleClient（OpenAI / Azure OpenAI 兼容 / vLLM /
本地 gateway 等，凡提供 /v1/chat/completions 均可）。
"""

from .openai_client import OpenAICompatibleClient

__all__ = ["OpenAICompatibleClient"]
