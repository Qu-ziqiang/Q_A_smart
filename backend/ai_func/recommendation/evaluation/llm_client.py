"""Phase 4.2-B — LLMClient 抽象（不绑定厂商）。

LLMJudge 通过 LLMClient.generate(prompt) -> str 调用任意后端；
MockLLMClient 返回固定 JSON，用于测试链路。

未来可接入：OpenAI / Azure / vLLM / 本地模型（各实现一个 LLMClient）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMClientError(RuntimeError):
    """LLM 客户端错误（配置缺失 / HTTP / 解析 / 重试耗尽等）。"""


class LLMClient(ABC):
    """统一 LLM 客户端抽象。"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """输入完整 prompt（system+user），返回模型输出文本。"""


class MockLLMClient(LLMClient):
    """测试用：返回固定响应并记录 prompts。"""

    def __init__(self, response: Optional[str] = None):
        self.response = response if response is not None else _PASS_JSON
        self.prompts: list = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


_PASS_JSON = (
    '{"verdict": "pass", '
    '"criterion_scores": {"planning_quality": 5.0, "agent_routing": 5.0, '
    '"tool_usage": 5.0, "recovery_behavior": 5.0, "final_result_quality": 5.0}, '
    '"reasons": ["mock pass"]}'
)
