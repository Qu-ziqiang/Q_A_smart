"""Phase 4.3 — OpenAICompatibleClient（真实外部 LLM Judge 客户端）。

支持任何 OpenAI-compatible 的 /chat/completions 端点：OpenAI、Azure 兼容网关、
vLLM、本地 gateway（凡提供 /v1/chat/completions 均可）。

配置（只读环境变量，禁止写进代码/配置文件/git）：
  LLM_JUDGE_API_KEY    API Key
  LLM_JUDGE_BASE_URL   base_url，如 https://api.openai.com/v1
  LLM_JUDGE_MODEL      模型名，如 gpt-4o-mini / glm-4.6 / qwen-plus

接口：
  - chat(messages: list[{role, content}]) -> str   （消息式请求）
  - generate(prompt: str) -> str                    （实现既有 LLMClient ABC）

能力：timeout / 指数退避重试（408/429/5xx/网络错误）；错误聚合 LLMClientError；
response logging 只记录 model/status/latency/content 长度，绝不打印 API Key。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ..llm_client import LLMClient, LLMClientError

logger = logging.getLogger("smartimager.evaluation.llm")

ENV_API_KEY = "LLM_JUDGE_API_KEY"
ENV_BASE_URL = "LLM_JUDGE_BASE_URL"
ENV_MODEL = "LLM_JUDGE_MODEL"

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class _Retryable(Exception):
    """可重试错误（网络错误 / 限流 / 5xx）。"""


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible chat 客户端（Judge LLM）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
        http_client: Optional[httpx.Client] = None,
    ):
        self.api_key = api_key or os.environ.get(ENV_API_KEY) or ""
        base = (base_url or os.environ.get(ENV_BASE_URL) or _DEFAULT_BASE_URL).strip()
        self.base_url = base.rstrip("/")
        self.model = model or os.environ.get(ENV_MODEL) or ""
        if not self.api_key:
            raise LLMClientError(
                f"缺少 Judge LLM API Key：请设置环境变量 {ENV_API_KEY}")
        if not self.model:
            raise LLMClientError(
                f"缺少 Judge LLM 模型：请设置环境变量 {ENV_MODEL}")

        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self._http = http_client or httpx.Client(timeout=float(timeout_seconds))
        self.last_meta: Dict[str, Any] = {}   # response 摘要（无敏感信息）

    # -- 请求路径 ----------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """发送 messages 到 /chat/completions，返回 assistant content。"""
        if not messages or not isinstance(messages, list):
            raise LLMClientError("messages 必须是非空列表")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._request_once(payload, attempt)
            except _Retryable as e:
                last = e
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (2 ** attempt))
        raise LLMClientError(f"Judge LLM 请求失败（重试 {self.max_retries} 次后）：{last}")

    def generate(self, prompt: str) -> str:
        """LLMClient ABC：兼容既有 LLMJudge 的单串 prompt 调用。"""
        return self.chat([{"role": "user", "content": prompt}])

    # -- 内部 --------------------------------------------------------------
    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def _request_once(self, payload: Dict[str, Any], attempt: int) -> str:
        started = time.perf_counter()
        try:
            response = self._post(payload, attempt)
            status = int(getattr(response, "status_code", 0) or 0)
            if status in _RETRY_STATUS:
                raise _Retryable(f"http {status}")
            if status >= 400:
                raise LLMClientError(f"Judge LLM HTTP {status}")
            data = response.json()
        except LLMClientError:
            raise
        except httpx.HTTPError as e:
            raise _Retryable(f"{type(e).__name__}: {e}") from e
        except Exception as e:  # json 解析等
            raise LLMClientError(f"Judge LLM 响应解析失败：{type(e).__name__}") from e

        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            choices = (data or {}).get("choices") or []
            content = (choices[0] or {}).get("message") or {}
            text = content.get("content")
        except Exception as e:  # pragma: no cover - 防御
            raise LLMClientError("Judge LLM 响应缺少 choices/message/content") from e
        if not isinstance(text, str) or not text.strip():
            raise LLMClientError("Judge LLM 返回空 content")

        usage = (data or {}).get("usage") or {}
        self.last_meta = {
            "status_code": status,
            "model": (data or {}).get("model") or self.model,
            "latency_ms": round(latency_ms, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "content_len": len(text),
        }
        # 只记元信息，绝不打印 payload / API Key
        logger.info(
            "[LLMJudgeClient] model=%s status=%s latency_ms=%.1f content_len=%s",
            self.last_meta["model"], status, latency_ms, len(text),
        )
        return text

    def _post(self, payload: Dict[str, Any], attempt: int):
        """发 POST；子类/测试可覆写。错误信息中不包含 API Key。"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return self._http.post(self._endpoint(), json=payload, headers=headers)

    def __repr__(self) -> str:  # 安全表示：不暴露 Key
        return (f"OpenAICompatibleClient(model={self.model!r}, "
                f"base_url={self.base_url!r}, api_key=<hidden>)")

