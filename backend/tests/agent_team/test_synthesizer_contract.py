"""Phase 2 Stage 6 — Synthesizer Contract / Vision Evidence Controlled Input 单元测试。

覆盖：
  - vision evidence success → synth prompt 含 bounded evidence（6 白名单字段）
  - prompt 数据边界：无 public_url / filepath / base64 / raw / API key
  - vision partial failure → synth 仍成功（失败 item 无证据）
  - vision systemic + 空候选 → deterministic fallback（不调 LLM）
  - vision systemic + 有候选 → synth 仍成功（无证据注入）
  - VisionEvidence extra=forbid 拒绝未知字段 / 白名单字段集

全部 Fake（Fake LLM / Fake analyzer），零真实 API；不修改配置。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_synthesizer_contract.py
"""

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from pydantic import ValidationError

from backend.ai_func.recommendation.orchestration.models import (
    AnalyzeVisualInput,
    RetrieveImagesInput,
    SynthesizeInput,
    Task,
    TaskPlan,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.synthesizer import SynthesizerAgent
from backend.ai_func.recommendation.orchestration.vision import (
    VisionErrorCode,
    VisionEvidence,
    build_vision_output,
    vision_item_failed,
    vision_item_success,
    vision_systemic_failure,
)
from backend.ai_func.recommendation.orchestration.workers import (
    ExecutionContext,
    ImageCandidate,
    RetrievalOutput,
    WorkerResult,
)


class _SpyLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.response


def _cand(iid, title="t", score=0.9):
    return ImageCandidate(id=iid, title=title, score=score, public_url=f"/s/{iid}.jpg")


def _make_ctx(cands, vision_out=None):
    plan = TaskPlan(
        plan_version="1.0", plan_id="p", goal="找旅行照片", synthesis_task_id="t2",
        tasks=[
            Task(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, objective="r",
                 input=RetrieveImagesInput(query="q"), depends_on=[]),
            Task(task_id="t3", task_type=TaskType.ANALYZE_VISUAL, objective="v",
                 input=AnalyzeVisualInput(criteria=["c"]), depends_on=["t1"]),
            Task(task_id="t2", task_type=TaskType.SYNTHESIZE, objective="s",
                 input=SynthesizeInput(recommendation_limit=10), depends_on=["t1", "t3"]),
        ],
    )
    ctx = ExecutionContext(plan)
    ctx.results["t1"] = WorkerResult(
        task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=list(cands), query="q", total=len(cands)),
    )
    ctx.results["t3"] = WorkerResult(
        task_id="t3", task_type=TaskType.ANALYZE_VISUAL,
        status=TaskStatus.SUCCEEDED, output=vision_out,
    )
    return ctx


def _synth_task():
    return Task(task_id="t2", task_type=TaskType.SYNTHESIZE, objective="s",
                input=SynthesizeInput(recommendation_limit=10), depends_on=["t1", "t3"])


# ---------------------------------------------------------------------------
# 1) vision evidence success → prompt 含 bounded evidence
# ---------------------------------------------------------------------------
def test_synth_prompt_contains_bounded_vision_evidence():
    cands = [_cand(1)]
    vo = build_vision_output([
        vision_item_success(1, title="v-title", description="v-desc", tags=["t1", "t2"],
                            evidence="v-evidence", confidence=0.85),
    ])
    spy = _SpyLLM(json.dumps({"answer": "a", "image_ids": [1]}))
    agent = SynthesizerAgent(llm=spy)
    res = agent.run(_synth_task(), _make_ctx(cands, vo))
    assert res.status == TaskStatus.SUCCEEDED
    assert res.output.synthesis_mode == "llm"
    payload = json.loads(spy.calls[0][1]["content"])
    c = [x for x in payload["candidates"] if x["id"] == 1][0]
    assert c["vision"]["title"] == "v-title"
    assert c["vision"]["description"] == "v-desc"
    assert c["vision"]["tags"] == ["t1", "t2"]
    assert c["vision"]["evidence"] == "v-evidence"
    assert c["vision"]["confidence"] == 0.85
    # 系统提示包含 vision 受控使用规则
    sys_msg = spy.calls[0][0]["content"]
    assert "vision" in sys_msg and "未知" in sys_msg


# ---------------------------------------------------------------------------
# 2) prompt 数据边界：无 public_url / filepath / base64 / raw / API key
# ---------------------------------------------------------------------------
def test_synth_prompt_no_sensitive_fields():
    cands = [_cand(1, title="a"), _cand(2, title="b")]
    vo = build_vision_output([
        vision_item_success(1, title="v1", description="d1", tags=["t"], evidence="e1", confidence=0.8),
        vision_item_success(2, title="v2", description="d2", tags=["t"], evidence="e2", confidence=0.7),
    ])
    spy = _SpyLLM(json.dumps({"answer": "a", "image_ids": [1, 2]}))
    agent = SynthesizerAgent(llm=spy)
    agent.run(_synth_task(), _make_ctx(cands, vo))
    text = json.dumps(spy.calls[0][1]["content"], ensure_ascii=False).lower()
    for bad in ("public_url", "filepath", "base64", "raw", "api_key", "authorization", "sk-", "data:image/"):
        assert bad not in text, f"payload 不应包含 {bad}"
    payload = json.loads(spy.calls[0][1]["content"])
    for c in payload["candidates"]:
        if "vision" in c:
            assert set(c["vision"].keys()) == {
                "image_id", "title", "description", "tags", "evidence", "confidence",
            }


# ---------------------------------------------------------------------------
# 3) vision partial failure → synth 仍成功（失败 item 无伪证据）
# ---------------------------------------------------------------------------
def test_synth_vision_partial_failure_still_succeeds():
    cands = [_cand(1), _cand(2), _cand(3)]
    vo = build_vision_output([
        vision_item_success(1, title="v1", description="d1", tags=["t"], evidence="e1", confidence=0.8),
        vision_item_failed(2, VisionErrorCode.TIMEOUT, "timeout"),
        vision_item_success(3, title="v3", description="d3", tags=["t"], evidence="e3", confidence=0.7),
    ])
    spy = _SpyLLM(json.dumps({"answer": "a", "image_ids": [1, 2, 3]}))
    agent = SynthesizerAgent(llm=spy)
    res = agent.run(_synth_task(), _make_ctx(cands, vo))
    assert res.status == TaskStatus.SUCCEEDED
    payload = json.loads(spy.calls[0][1]["content"])
    by_id = {c["id"]: c for c in payload["candidates"]}
    assert by_id[1]["vision"]["title"] == "v1"
    assert "vision" not in by_id[2]   # 失败 item 不注入伪证据
    assert by_id[3]["vision"]["title"] == "v3"


# ---------------------------------------------------------------------------
# 4) vision systemic + 空候选 → deterministic fallback（不调 LLM）
# ---------------------------------------------------------------------------
def test_synth_systemic_vision_fallback():
    vo = vision_systemic_failure(VisionErrorCode.CLIENT_UNAVAILABLE, "no client")
    spy = _SpyLLM(json.dumps({"answer": "a", "image_ids": [1]}))
    agent = SynthesizerAgent(llm=spy)
    res = agent.run(_synth_task(), _make_ctx([], vo))
    assert res.status == TaskStatus.SUCCEEDED
    assert res.output.synthesis_degraded is True
    assert res.output.synthesis_mode == "deterministic"
    assert spy.calls == []   # 空候选不调用 LLM


# ---------------------------------------------------------------------------
# 5) vision systemic + 有候选 → synth 仍成功（无证据注入）
# ---------------------------------------------------------------------------
def test_synth_systemic_vision_not_blocking():
    cands = [_cand(1), _cand(2)]
    vo = vision_systemic_failure(VisionErrorCode.PROVIDER_ERROR, "provider down")
    spy = _SpyLLM(json.dumps({"answer": "a", "image_ids": [1, 2]}))
    agent = SynthesizerAgent(llm=spy)
    res = agent.run(_synth_task(), _make_ctx(cands, vo))
    assert res.status == TaskStatus.SUCCEEDED
    assert res.output.synthesis_mode == "llm"
    payload = json.loads(spy.calls[0][1]["content"])
    for c in payload["candidates"]:
        assert "vision" not in c   # systemic 不注入证据


# ---------------------------------------------------------------------------
# 6) VisionEvidence 契约：extra=forbid / 白名单字段集
# ---------------------------------------------------------------------------
def test_vision_evidence_extra_fields_rejected():
    try:
        VisionEvidence(image_id=1, title="t", description="d", junk=True)
        assert False, "extra=forbid 应拒绝未知字段"
    except ValidationError:
        pass


def test_vision_evidence_whitelist_fields():
    ev = VisionEvidence(image_id=1, title="t", description="d", tags=["a"],
                        evidence="e", confidence=0.5)
    assert set(ev.model_dump().keys()) == {
        "image_id", "title", "description", "tags", "evidence", "confidence",
    }


# ---------------------------------------------------------------------------
# 纯 python 运行器
# ---------------------------------------------------------------------------
def _main():
    failed = 0
    passed = 0
    mod_globals = globals()
    for name in sorted(mod_globals):
        if name.startswith("test_"):
            fn = mod_globals[name]
            try:
                fn()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print(f"\n== {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
