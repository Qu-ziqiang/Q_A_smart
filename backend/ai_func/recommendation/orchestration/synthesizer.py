"""Phase 1-C — LLM Synthesizer Agent（规划 §11 / §9 / §21）。

职责：
  - 只接收 Orchestrator 显式传入的数据（goal + TaskPlan + 结构化 Worker Results）
  - 一次 LLM 调用，输出结构化 SynthesisResult（answer + image_ids + rationale）
  - validate_selection()：LLM 返回的 image_ids 必须属于真实候选集，非法 ID 一律剔除
  - 任何失败（timeout / 非法 JSON / 非法 ID / 空选择 / 异常）→ 确定性 fallback
    （复用 DeterministicSynthesisWorker）

不做：不创建 Task / 不调用 Tool / 不重新搜索 / 不修改数据库 / 无 Loop。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    SYNTHESIZER_TIMEOUT_SECONDS,
    SynthesizeInput,
    Task,
    TaskStatus,
    TaskType,
)
from .workers import (
    DeterministicSynthesisWorker,
    GroupOutput,
    ImageCandidate,
    SynthesisOutput,
    WorkerResult,
    collect_candidates,
    dedup_candidates,
)


# ---------------------------------------------------------------------------
# Synthesizer 结构化输出（LLM 原始输出 → Pydantic 校验）
# ---------------------------------------------------------------------------
class SynthesisResult(BaseModel):
    """LLM 必须返回的结构化结果。image_ids 之后仍需本地校验。"""
    model_config = ConfigDict(extra="forbid")
    answer: str
    image_ids: List[int] = Field(default_factory=list)
    rationale: str = ""


class SynthesizerFailure(Exception):
    """Synthesizer LLM 路径失败（触发确定性 fallback）。"""


# ---------------------------------------------------------------------------
# Selection 安全校验（§九）：LLM 不能凭空制造不存在的 ID
# ---------------------------------------------------------------------------
def validate_selection(
    image_ids: List[int], candidates: List[ImageCandidate], limit: int
) -> List[int]:
    """过滤非法/重复 ID，保持 LLM 顺序，截断到 limit。"""
    valid_ids = {c.id for c in candidates}
    out: List[int] = []
    seen = set()
    for raw in image_ids or []:
        try:
            i = int(raw)
        except (TypeError, ValueError):
            continue
        if i in valid_ids and i not in seen:
            seen.add(i)
            out.append(i)
            if len(out) >= limit:
                break
    return out


class SynthesizerAgent:
    """作为 SYNTHESIZE worker 挂入 DAG 的 LLM Synthesizer。"""

    task_type = TaskType.SYNTHESIZE

    def __init__(
        self,
        llm: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        model: Optional[str] = None,
        client: Any = None,
        timeout: int = SYNTHESIZER_TIMEOUT_SECONDS,
        fallback_worker: Optional[DeterministicSynthesisWorker] = None,
    ):
        self._llm = llm          # 可注入 Fake（测试）
        self._model = model
        self._client = client
        self._timeout = timeout
        self._fallback = fallback_worker or DeterministicSynthesisWorker()

    # -- Worker 契约 --------------------------------------------------------
    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, SynthesizeInput):
            return WorkerResult(
                task_id=task.task_id, task_type=task.task_type,
                status=TaskStatus.FAILED,
                error=None,
            )
        return self._run_inner(task, context, inp)

    def _run_inner(self, task: Task, context: Any, inp: SynthesizeInput) -> WorkerResult:
        try:
            cands = dedup_candidates(collect_candidates(task, context.results))
        except Exception as e:
            return self._failed(task, "SYNTHESIS_ERROR", str(e)[:200])

        # 空候选：不浪费 LLM 调用，直接走确定性结果（“未找到”文本由 Orchestrator 生成）
        if not cands:
            return self._degraded(task, context, inp, reason="no candidates")

        # LLM 主路径（一次调用）
        try:
            result = self._synthesize(task, context, cands, inp)
            selected = validate_selection(result.image_ids, cands, inp.recommendation_limit)
            if not selected:
                raise SynthesizerFailure("empty valid selection")
            by_id = {c.id: c for c in cands}
            final = [by_id[i] for i in selected if i in by_id]
            output = SynthesisOutput(
                image_ids=selected,
                candidates=final,
                presentation=inp.presentation,
                group_summary=self._group_summary(task, context),
                answer=result.answer,
                rationale=result.rationale,
                synthesis_mode="llm",
                synthesis_degraded=False,
            )
            print(f"[SynthesizerAgent] LLM synthesis success ({len(selected)} ids)")
            return WorkerResult(
                task_id=task.task_id, task_type=task.task_type,
                status=TaskStatus.SUCCEEDED, output=output,
            )
        except Exception as e:
            print(f"[SynthesizerAgent] LLM synthesis failed → deterministic fallback "
                  f"({type(e).__name__}: {str(e)[:120]})")
            return self._degraded(task, context, inp, reason=str(e)[:160])

    # -- LLM 主路径 ---------------------------------------------------------
    def _synthesize(self, task, context, cands, inp) -> SynthesisResult:
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(task, context, cands, inp)
        raw = self._call_llm(system_prompt, user_prompt)
        return self._parse(raw)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if self._llm is not None:
            return self._llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        from ....config import settings
        from ....config.init_service import get_openai_client

        client = self._client or get_openai_client()
        if not client:
            raise SynthesizerFailure("OpenAI 客户端未配置，无法合成")
        config = settings.get_config()
        model = self._model or (getattr(config, "CHAT_MODEL", None) or "deepseek-chat")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            stream=False,
            timeout=self._timeout,
        )
        return (response.choices[0].message.content or "")

    def _parse(self, raw: str) -> SynthesisResult:
        text = _strip_fences(raw).strip()
        if not text:
            raise SynthesizerFailure("empty LLM output")
        try:
            data = json.loads(text)
        except Exception as e:
            raise SynthesizerFailure(f"invalid JSON: {e}")
        if not isinstance(data, dict):
            raise SynthesizerFailure("LLM 输出不是 JSON 对象")
        try:
            return SynthesisResult.model_validate(data)
        except Exception as e:
            raise SynthesizerFailure(f"schema validation failed: {e}")

    # -- Prompt 构建（§十三：只含显式传入数据，不含历史/DB/filepath）--------
    def _system_prompt(self) -> str:
        from ..prompts import SYNTHESIZER_SYSTEM_PROMPT_TEMPLATE

        return SYNTHESIZER_SYSTEM_PROMPT_TEMPLATE

    def _user_prompt(self, task, context, cands, inp) -> str:
        payload = {
            "goal": context.goal or "",
            "plan_summary": self._plan_summary(context),
            "failures": self._failures(context),
            "groups": self._groups(task, context),
            "candidates": [
                {
                    "id": c.id,
                    "score": c.score,
                    "title": c.title,
                    "tags": list(c.tags),
                    **({
                        "vision": {
                            "image_id": c.vision.image_id,
                            "title": c.vision.title,
                            "description": c.vision.description,
                            "tags": list(c.vision.tags),
                            "objects": list(c.vision.objects),
                            "scene": c.vision.scene,
                            "colors": list(c.vision.colors),
                            "ocr_text": c.vision.ocr_text,
                            "evidence": c.vision.evidence,
                            "confidence": c.vision.confidence,
                        }
                    } if c.vision is not None else {}),
                }
                for c in cands
            ],
            "recommendation_limit": inp.recommendation_limit,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _plan_summary(self, context) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        plan = context.plan
        if plan is None:
            return out
        for t in plan.tasks:
            res = context.results.get(t.task_id)
            out.append({
                "task_id": t.task_id,
                "task_type": t.task_type.value,
                "status": res.status.value if res else "pending",
            })
        return out

    def _failures(self, context) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        plan = context.plan
        if plan is None:
            return out
        for t in plan.tasks:
            res = context.results.get(t.task_id)
            if res and res.status in (TaskStatus.FAILED, TaskStatus.SKIPPED):
                out.append({
                    "task_id": t.task_id,
                    "code": res.error.code if res.error else "SKIPPED",
                })
        return out

    def _groups(self, task, context) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for dep in task.depends_on:
            res = context.results.get(dep)
            if res and res.status == TaskStatus.SUCCEEDED and isinstance(res.output, GroupOutput):
                for g in res.output.groups:
                    groups.append({"key": g.key, "image_ids": g.image_ids, "count": len(g.image_ids)})
        return groups

    @staticmethod
    def _group_summary(task, context) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for dep in task.depends_on:
            res = context.results.get(dep)
            if res and res.status == TaskStatus.SUCCEEDED and isinstance(res.output, GroupOutput):
                for g in res.output.groups:
                    summary[g.key] = len(g.image_ids)
        return summary

    # -- 确定性 fallback ----------------------------------------------------
    def _degraded(self, task, context, inp, reason: str) -> WorkerResult:
        fallback_res = self._fallback.run(task, context)
        if fallback_res.status == TaskStatus.SUCCEEDED and isinstance(fallback_res.output, SynthesisOutput):
            out = fallback_res.output.model_copy(
                update={"synthesis_mode": "deterministic", "synthesis_degraded": True}
            )
            print(f"[SynthesizerAgent] deterministic fallback (reason={reason[:80]})")
            return WorkerResult(
                task_id=task.task_id, task_type=task.task_type,
                status=TaskStatus.SUCCEEDED, output=out,
            )
        return fallback_res

    @staticmethod
    def _failed(task, code: str, message: str) -> WorkerResult:
        from .workers import TaskError

        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.FAILED,
            error=TaskError(code=code, message=message),
        )


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    return m.group(1) if m else text

