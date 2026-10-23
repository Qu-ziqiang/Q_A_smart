"""Phase 1-C — SynthesizerAgent 单元测试（全部 Fake LLM，零真实 API）。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_synthesizer.py
"""

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.recommendation.orchestration.executor import DAGExecutor
from backend.ai_func.recommendation.orchestration.models import (
    FilterMetadataInput,
    GroupImagesInput,
    RetrieveImagesInput,
    SynthesizeInput,
    Task,
    TaskPlan,
    TaskStatus,
    TaskType,
)
from backend.ai_func.recommendation.orchestration.orchestrator import AdaptiveTeamOrchestrator
from backend.ai_func.recommendation.orchestration.planner import PlannerAgent
from backend.ai_func.recommendation.orchestration.synthesizer import (
    SynthesizerAgent,
    validate_selection,
)
from backend.ai_func.recommendation.orchestration.workers import (
    ExecutionContext,
    GroupOutput,
    ImageCandidate,
    ImageGroup,
    MetadataOutput,
    RetrievalOutput,
    SynthesisOutput,
    TaskError,
    WorkerResult,
)

_CANDS = [
    ImageCandidate(id=1, score=0.9, title="北京", metadata={}),
    ImageCandidate(id=2, score=0.8, title="上海", metadata={}),
    ImageCandidate(id=3, score=0.7, title="广州", metadata={}),
]


def _plan():
    return TaskPlan(
        plan_version="1.0", plan_id="plan_t", goal="找旅行照片按地点分类",
        synthesis_task_id="t4",
        tasks=[
            Task(task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, objective="r",
                 input=RetrieveImagesInput(query="旅行"), depends_on=[]),
            Task(task_id="t2", task_type=TaskType.FILTER_METADATA, objective="m",
                 input=FilterMetadataInput(), depends_on=["t1"]),
            Task(task_id="t3", task_type=TaskType.GROUP_IMAGES, objective="g",
                 input=GroupImagesInput(group_by="location"), depends_on=["t1"]),
            Task(task_id="t4", task_type=TaskType.SYNTHESIZE, objective="s",
                 input=SynthesizeInput(recommendation_limit=5), depends_on=["t3"]),
        ],
    )


def _ctx(meta_failed=False, empty_cands=False):
    plan = _plan()
    ctx = ExecutionContext(plan)
    cands = [] if empty_cands else _CANDS
    ctx.results["t1"] = WorkerResult(
        task_id="t1", task_type=TaskType.RETRIEVE_IMAGES, status=TaskStatus.SUCCEEDED,
        output=RetrievalOutput(candidates=cands, query="旅行", total=len(cands)),
    )
    if meta_failed:
        ctx.results["t2"] = WorkerResult(
            task_id="t2", task_type=TaskType.FILTER_METADATA, status=TaskStatus.FAILED,
            error=TaskError(code="METADATA_READ_ERROR", message="boom"),
        )
    else:
        ctx.results["t2"] = WorkerResult(
            task_id="t2", task_type=TaskType.FILTER_METADATA, status=TaskStatus.SUCCEEDED,
            output=MetadataOutput(candidates=cands, rejected_ids=[], applied_predicates=[]),
        )
    group = ImageGroup(key="未知地点", image_ids=[c.id for c in cands], candidates=cands)
    ctx.results["t3"] = WorkerResult(
        task_id="t3", task_type=TaskType.GROUP_IMAGES, status=TaskStatus.SUCCEEDED,
        output=GroupOutput(group_by="location", groups=[group]),
    )
    return ctx


def _synth_task():
    return Task(task_id="t4", task_type=TaskType.SYNTHESIZE, objective="s",
                input=SynthesizeInput(recommendation_limit=5), depends_on=["t3"])


class FakeLLM:
    """记录收到的 messages；可返回预设文本或抛异常。"""

    def __init__(self, response="", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


def _valid_json():
    return json.dumps({"answer": "推荐北京与上海。", "image_ids": [2, 1], "rationale": "分高"})


# ---------------------------------------------------------------------------
# 1) 合法 synthesis
# ---------------------------------------------------------------------------
def test_valid_synthesis_uses_llm_output():
    fake = FakeLLM(response=_valid_json())
    agent = SynthesizerAgent(llm=fake)
    res = agent.run(_synth_task(), _ctx())
    assert res.status == TaskStatus.SUCCEEDED
    out = res.output
    assert isinstance(out, SynthesisOutput)
    assert out.synthesis_mode == "llm"
    assert out.synthesis_degraded is False
    assert out.answer == "推荐北京与上海。"
    assert out.image_ids == [2, 1]  # 保留 LLM 顺序
    assert out.candidates[0].id == 2


# ---------------------------------------------------------------------------
# 2) 非法 JSON → fallback
# ---------------------------------------------------------------------------
def test_invalid_json_falls_back_deterministic():
    fake = FakeLLM(response="这不是 JSON")
    agent = SynthesizerAgent(llm=fake)
    res = agent.run(_synth_task(), _ctx())
    assert res.status == TaskStatus.SUCCEEDED
    out = res.output
    assert out.synthesis_degraded is True
    assert out.synthesis_mode == "deterministic"
    # 确定性：按 score 降序 → [1,2,3]
    assert out.image_ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# 3) 非法 image_id → 剔除 / fallback
# ---------------------------------------------------------------------------
def test_illegal_image_ids_filtered():
    fake = FakeLLM(response=json.dumps({"answer": "a", "image_ids": [1, 999, 1, 3]}))
    agent = SynthesizerAgent(llm=fake)
    res = agent.run(_synth_task(), _ctx())
    out = res.output
    assert out.synthesis_mode == "llm"
    assert out.image_ids == [1, 3]  # 999 非法剔除、重复去重


def test_all_illegal_image_ids_falls_back():
    fake = FakeLLM(response=json.dumps({"answer": "a", "image_ids": [999, 888]}))
    agent = SynthesizerAgent(llm=fake)
    res = agent.run(_synth_task(), _ctx())
    out = res.output
    assert out.synthesis_degraded is True
    assert out.image_ids == [1, 2, 3]  # 确定性顺序


def test_validate_selection_unit():
    assert validate_selection([1, 999, 1, 2], _CANDS, 10) == [1, 2]
    assert validate_selection([], _CANDS, 10) == []
    assert validate_selection([1, 2, 3, 4], _CANDS, 2) == [1, 2]  # 截断 limit


# ---------------------------------------------------------------------------
# 4) 空候选 → 确定性“未找到”（且不调用 LLM）
# ---------------------------------------------------------------------------
def test_empty_candidates_no_llm_call():
    fake = FakeLLM(response=_valid_json())
    agent = SynthesizerAgent(llm=fake)
    res = agent.run(_synth_task(), _ctx(empty_cands=True))
    assert res.status == TaskStatus.SUCCEEDED
    assert res.output.image_ids == []
    assert res.output.synthesis_degraded is True
    assert fake.calls == []  # 空候选不调用 LLM


# ---------------------------------------------------------------------------
# 5) Worker failure 必须在 payload 中可见
# ---------------------------------------------------------------------------
def test_worker_failure_visible_in_payload():
    captured = {}

    def spy_llm(messages):
        captured["user"] = messages[1]["content"]
        return _valid_json()

    agent = SynthesizerAgent(llm=spy_llm)
    agent.run(_synth_task(), _ctx(meta_failed=True))
    payload = json.loads(captured["user"])
    codes = [f["code"] for f in payload["failures"]]
    assert "METADATA_READ_ERROR" in codes


# ---------------------------------------------------------------------------
# 6) 未知 metadata 不得出现在候选 payload 中（LLM 无 source 可编造）
# ---------------------------------------------------------------------------
def test_unknown_metadata_not_passed_to_llm():
    captured = {}

    def spy_llm(messages):
        captured["user"] = messages[1]["content"]
        return _valid_json()

    agent = SynthesizerAgent(llm=spy_llm)
    agent.run(_synth_task(), _ctx())
    payload = json.loads(captured["user"])
    for cand in payload["candidates"]:
        assert "metadata" not in cand
        assert "location" not in cand
        assert "filepath" not in cand


# ---------------------------------------------------------------------------
# 7) Prompt 输入隔离
# ---------------------------------------------------------------------------
def test_prompt_input_isolation():
    captured = {}

    def spy_llm(messages):
        captured["messages"] = messages
        return _valid_json()

    agent = SynthesizerAgent(llm=spy_llm)
    agent.run(_synth_task(), _ctx())
    sys_msg, user_msg = captured["messages"]
    assert "filepath" not in user_msg["content"]
    payload = json.loads(user_msg["content"])
    assert set(payload.keys()) == {"goal", "plan_summary", "failures", "groups", "candidates", "recommendation_limit"}
    assert len(captured["messages"]) == 2  # system + user，无历史


# ---------------------------------------------------------------------------
# 8) LLM 异常 → fallback
# ---------------------------------------------------------------------------
def test_llm_exception_falls_back():
    fake = FakeLLM(error=RuntimeError("network down"))
    agent = SynthesizerAgent(llm=fake)
    res = agent.run(_synth_task(), _ctx())
    assert res.status == TaskStatus.SUCCEEDED
    assert res.output.synthesis_degraded is True
    assert res.output.image_ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# 9) Legacy 回归：简单查询完全不进入 Synthesizer
# ---------------------------------------------------------------------------
def test_legacy_regression_no_synthesizer():
    from backend.ai_func.recommendation import _RecommendationService
    from backend.ai_func.recommendation.orchestration import ComplexityGate
    from backend.ai_func.recommendation.orchestration.models import RouteType

    assert ComplexityGate().decide(query="找猫的照片").route == RouteType.LEGACY

    class FakeAgent:
        def run_stream(self, messages, vector_targets=None, filters=None, limit=20):
            yield json.dumps({"event": "assistant_delta", "delta": "legacy"}, ensure_ascii=False) + "\n\n"
            yield json.dumps({"event": "selection", "ids": [7], "images": []}, ensure_ascii=False) + "\n\n"

    service = _RecommendationService()
    service._agent = FakeAgent()
    events = [json.loads(raw) for raw in service.stream_core([{"role": "user", "content": "找猫的照片"}])]
    names = [e["event"] for e in events]
    assert "plan.created" not in names
    assert names[-1] == "selection"


# ---------------------------------------------------------------------------
# 10) Complex E2E（Fake Planner + Fake Workers + Fake Synthesizer）
# ---------------------------------------------------------------------------
def test_complex_e2e_full_chain():
    plan_json = json.dumps({
        "goal": "找旅行照片按地点分类推荐",
        "synthesis_task_id": "t3",
        "tasks": [
            {"task_id": "t1", "task_type": "retrieve_images", "objective": "检索",
             "input": {"query": "旅行"}, "depends_on": []},
            {"task_id": "t2", "task_type": "group_images", "objective": "分组",
             "input": {"group_by": "location"}, "depends_on": ["t1"]},
            {"task_id": "t3", "task_type": "synthesize", "objective": "合成",
             "input": {"recommendation_limit": 5}, "depends_on": ["t2"]},
        ],
    })

    class _FW:
        def __init__(self, ttype, fn):
            self.task_type = ttype
            self._fn = fn

        def run(self, task, context):
            return self._fn(task, context)

    def retr(task, context):
        return WorkerResult(
            task_id=task.task_id, task_type=TaskType.RETRIEVE_IMAGES,
            status=TaskStatus.SUCCEEDED,
            output=RetrievalOutput(candidates=_CANDS, query="q", total=3),
        )

    def group(task, context):
        return WorkerResult(
            task_id=task.task_id, task_type=TaskType.GROUP_IMAGES,
            status=TaskStatus.SUCCEEDED,
            output=GroupOutput(group_by="location", groups=[
                ImageGroup(key="未知地点", image_ids=[1, 2, 3], candidates=_CANDS)]),
        )

    synth_llm = FakeLLM(response=json.dumps({"answer": "最终回答：推荐这些", "image_ids": [3, 1]}))
    orchestrator = AdaptiveTeamOrchestrator(
        planner=PlannerAgent(llm=lambda m: plan_json),
        executor=DAGExecutor(workers={
            TaskType.RETRIEVE_IMAGES: _FW(TaskType.RETRIEVE_IMAGES, retr),
            TaskType.GROUP_IMAGES: _FW(TaskType.GROUP_IMAGES, group),
            TaskType.SYNTHESIZE: SynthesizerAgent(llm=synth_llm),
        }),
    )
    events = [json.loads(raw) for raw in orchestrator.run_stream(
        [{"role": "user", "content": "找旅行照片，按地点分类并整理结果"}])]
    names = [e["event"] for e in events]
    assert "plan.created" in names
    assert "task.started" in names
    final_delta = [e for e in events if e["event"] == "assistant_delta"][-1]
    assert "最终回答：推荐这些" in final_delta["delta"]
    selection = events[-1]
    assert selection["event"] == "selection"
    assert selection["ids"] == [3, 1]
    assert orchestrator.last_plan_ok is True


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

