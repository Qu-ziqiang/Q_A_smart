"""Phase 4.2-B — LLMJudge（真实 LLM-as-a-Judge 实现，可替换 MockJudge）。

链路：
  JudgeInput
    → JudgePromptBuilder（精简上下文）
    → LLMClient.generate(prompt)
    → parse_judge_response（严格 JSON，含 markdown 围栏剥离）
    → JudgeResult

不绑定厂商：LLMClient 抽象由调用方注入（MockLLMClient 测试 / 未来 OpenAI/Azure/vLLM）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .judge import Judge, JudgeInput, JudgeResult
from .llm_client import LLMClient
from .prompts.judge_prompt import JudgePromptBuilder
from .rubric import criterion_keys

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL)


def parse_judge_response(raw: str) -> Dict[str, Any]:
    """解析 LLM JSON 输出。

    支持：纯 JSON / ```json 代码块围栏。
    非法 JSON 或非对象 → 抛 ValueError（由 LLMJudge 收敛为 flag + invalid judge response）。
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty judge response")
    match = _JSON_FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except Exception as e:
        raise ValueError(f"invalid judge JSON: {type(e).__name__}: {e}")
    if not isinstance(data, dict):
        raise ValueError("judge response is not a JSON object")
    return data


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(5.0, score)), 2)


class LLMJudge(Judge):
    """真实 LLM Judge（可替换 MockJudge，接口不变）。"""

    name = "llm"
    criteria = criterion_keys()

    def __init__(self, client: LLMClient,
                 prompt_builder: Optional[JudgePromptBuilder] = None):
        if not isinstance(client, LLMClient):
            raise TypeError("LLMJudge 需要 LLMClient 实例")
        self.client = client
        self.builder = prompt_builder if prompt_builder is not None else JudgePromptBuilder()
        self.last_raw: Optional[str] = None

    def evaluate(self, judge_input: JudgeInput) -> JudgeResult:
        prompt = self.builder.build_full_prompt(judge_input)
        raw = self.client.generate(prompt)
        self.last_raw = raw
        try:
            data = parse_judge_response(raw)
        except Exception:
            return JudgeResult(
                case_id=judge_input.case_id,
                request_id=judge_input.request_id,
                verdict="flag",
                error="invalid judge response",
            )

        verdict = data.get("verdict")
        if verdict not in ("pass", "flag", "fail"):
            return JudgeResult(
                case_id=judge_input.case_id,
                request_id=judge_input.request_id,
                verdict="flag",
                error="invalid judge response",
            )
        raw_scores = data.get("criterion_scores")
        if not isinstance(raw_scores, dict):
            return JudgeResult(
                case_id=judge_input.case_id,
                request_id=judge_input.request_id,
                verdict="flag",
                error="invalid judge response",
            )
        scores: Dict[str, float] = {k: _clamp_score(raw_scores.get(k)) for k in self.criteria}
        reasons: List[str] = []
        raw_reasons = data.get("reasons")
        if isinstance(raw_reasons, list):
            reasons = [str(r)[:300] for r in raw_reasons]

        return JudgeResult(
            case_id=judge_input.case_id,
            request_id=judge_input.request_id,
            verdict=verdict,
            criterion_scores=scores,
            reasons=reasons,
            error=None,
        )
