"""Phase 2 Stage 7 — SSE / event-level Observability 集成测试（FastAPI TestClient）。

覆盖：
  - Legacy 流程帧（rewrite_start → assistant_delta → complete）
  - Team 流程帧（plan.created → task.* 透传 → assistant_delta → complete）
  - task.failed / task.skipped 安全帧
  - Vision task.completed.output_summary 安全统计（candidate/success/failed/systemic）
  - 无泄露（filepath/base64/authorization/api_key/sk-/data:image/raw_response/provider_secret/traceback）
  - complete 唯一且不因 task 事件消失/重复
  - 事件顺序：所有 started task 都有 terminal 事件
  - executor _output_summary 的 Vision 分支单元测试

全部 Fake（Fake service stream_core / Fake repos / Fake worker output），零真实 API。
运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_sse_contract.py
"""

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routers.ai.recommendation as router_mod
from backend.ai_func.recommendation.orchestration.executor import _output_summary
from backend.ai_func.recommendation.orchestration.models import TaskStatus, TaskType
from backend.ai_func.recommendation.orchestration.vision import (
    VisionErrorCode,
    build_vision_output,
    vision_item_failed,
    vision_item_success,
    vision_systemic_failure,
)
from backend.ai_func.recommendation.orchestration.workers import WorkerResult


# ---------------------------------------------------------------------------
# Fakes（router 依赖打桩）
# ---------------------------------------------------------------------------
class _FakeMsgRepo:
    def __init__(self, *a, **k):
        pass

    def list_messages(self, conv_id, limit=40):
        return []

    def add_message(self, *a, **k):
        return None


class _FakeSessionRepo:
    def __init__(self, *a, **k):
        pass

    def update_by_request_id(self, *a, **k):
        return None


class _FakeService:
    """Fake recommendation_service.stream_core，产出预设事件 JSON。"""

    def __init__(self, events):
        self._events = list(events)

    def stream_core(self, **kwargs):
        for e in self._events:
            yield json.dumps(e, ensure_ascii=False)


def _run_router(events):
    router_mod.ConversationMessageRepository = _FakeMsgRepo
    router_mod.RequestSessionRepository = _FakeSessionRepo
    router_mod.recommendation_service = _FakeService(events)
    app = FastAPI()
    app.include_router(router_mod.router, prefix="/api/v1/ai")
    client = TestClient(app)
    resp = client.post("/api/v1/ai/recommend/chat/stream", json={"query": "找旅行照片"})
    assert resp.status_code == 200
    return _parse_sse(resp.text)


def _parse_sse(text):
    frames = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        ev = ""
        data_lines = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                ev = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
            else:
                data_lines.append(line.strip())
        data_text = "".join(data_lines)
        frames.append({"event": ev, "data": json.loads(data_text) if data_text else {}})
    return frames


_TEAM_EVENTS = [
    {"event": "plan.created", "plan_id": "plan_x", "plan_version": "1.0", "goal": "g",
     "synthesis_task_id": "t4",
     "tasks": [
         {"task_id": "t1", "task_type": "retrieve_images", "objective": "o", "depends_on": []},
         {"task_id": "t3", "task_type": "analyze_visual", "objective": "o", "depends_on": ["t1"]},
         {"task_id": "t4", "task_type": "synthesize", "objective": "o", "depends_on": ["t1", "t3"]},
     ]},
    {"event": "task.started", "task_id": "t1", "task_type": "retrieve_images", "status": "running"},
    {"event": "task.completed", "task_id": "t1", "task_type": "retrieve_images", "status": "succeeded",
     "duration_ms": 5, "output_summary": {"candidate_count": 3, "image_ids": [1, 2, 3]}},
    {"event": "task.started", "task_id": "t3", "task_type": "analyze_visual", "status": "running"},
    {"event": "task.completed", "task_id": "t3", "task_type": "analyze_visual", "status": "succeeded",
     "duration_ms": 10,
     "output_summary": {"vision": {"candidate_count": 3, "success_count": 2, "failed_count": 1, "systemic_failed": False}}},
    {"event": "task.started", "task_id": "t4", "task_type": "synthesize", "status": "running"},
    {"event": "task.completed", "task_id": "t4", "task_type": "synthesize", "status": "succeeded",
     "output_summary": {"image_ids": [1, 2], "synthesis": {"mode": "llm", "degraded": False}}},
    {"event": "assistant_delta", "delta": "推荐图片 1 和 2。"},
    {"event": "selection", "ids": [1, 2], "images": [
        {"id": 1, "score": 0.9, "title": "a", "tags": [], "public_url": "/s/1.jpg"},
        {"id": 2, "score": 0.8, "title": "b", "tags": [], "public_url": "/s/2.jpg"},
    ]},
]


# ---------------------------------------------------------------------------
# 1) Legacy 流程
# ---------------------------------------------------------------------------
def test_sse_legacy_flow():
    events = [
        {"event": "assistant_delta", "delta": "你好，请问您想找什么类型的图片？"},
        {"event": "selection", "ids": [1],
         "images": [{"id": 1, "score": 0.9, "title": "a", "tags": [], "public_url": "/s/1.jpg"}]},
    ]
    frames = _run_router(events)
    names = [f["event"] for f in frames]
    assert names == ["rewrite_start", "assistant_delta", "complete"]
    complete = frames[-1]["data"]
    assert complete["image_ids"] == [1]
    assert "你好" in complete["assistant_text"]
    assert not any(n.startswith("task.") for n in names)   # Legacy 无 task 事件


# ---------------------------------------------------------------------------
# 2) Team 流程 + Vision 安全统计
# ---------------------------------------------------------------------------
def test_sse_team_flow_and_vision_summary():
    frames = _run_router(_TEAM_EVENTS)
    names = [f["event"] for f in frames]
    assert names[0] == "rewrite_start"
    assert "plan.created" in names
    assert names.count("task.completed") == 3
    assert names[-2] == "assistant_delta"
    assert names[-1] == "complete"
    vis = [f for f in frames
           if f["event"] == "task.completed" and f["data"]["task_type"] == "analyze_visual"][0]
    assert vis["data"]["output_summary"]["vision"] == {
        "candidate_count": 3, "success_count": 2, "failed_count": 1, "systemic_failed": False,
    }
    vs_text = json.dumps(vis["data"]["output_summary"], ensure_ascii=False)
    for bad in ("title", "description", "tags", "evidence", "confidence"):
        assert bad not in vs_text   # 只输出统计，不输出视觉内容


# ---------------------------------------------------------------------------
# 3) task.failed 安全帧
# ---------------------------------------------------------------------------
def test_sse_task_failed_safe():
    events = [
        {"event": "task.started", "task_id": "t1", "task_type": "retrieve_images", "status": "running"},
        {"event": "task.failed", "task_id": "t1", "task_type": "retrieve_images", "status": "failed",
         "error": {"code": "RETRIEVAL_ERROR", "message": "检索服务异常"}},
        {"event": "assistant_delta", "delta": "抱歉"},
        {"event": "selection", "ids": [], "images": []},
    ]
    frames = _run_router(events)
    failed = [f for f in frames if f["event"] == "task.failed"][0]
    assert failed["data"]["error"]["code"] == "RETRIEVAL_ERROR"
    assert failed["data"]["error"]["message"] == "检索服务异常"
    text = json.dumps(frames, ensure_ascii=False)
    assert "Traceback" not in text
    assert "api_key" not in text.lower()
    assert "sk-" not in text


# ---------------------------------------------------------------------------
# 4) task.skipped 帧
# ---------------------------------------------------------------------------
def test_sse_task_skipped_reason():
    events = [
        {"event": "task.started", "task_id": "t1", "task_type": "retrieve_images", "status": "running"},
        {"event": "task.completed", "task_id": "t1", "task_type": "retrieve_images",
         "status": "succeeded", "output_summary": {"candidate_count": 0}},
        {"event": "task.skipped", "task_id": "t2", "task_type": "synthesize",
         "status": "skipped", "reason": "DEPENDENCY_UNAVAILABLE"},
        {"event": "assistant_delta", "delta": "无结果"},
        {"event": "selection", "ids": [], "images": []},
    ]
    frames = _run_router(events)
    skipped = [f for f in frames if f["event"] == "task.skipped"][0]
    assert skipped["data"]["reason"] == "DEPENDENCY_UNAVAILABLE"
    assert skipped["data"]["task_id"] == "t2"


# ---------------------------------------------------------------------------
# 5) 无泄露（全帧扫描）
# ---------------------------------------------------------------------------
def test_sse_no_leakage():
    events = _TEAM_EVENTS + [
        {"event": "task.started", "task_id": "t9", "task_type": "filter_metadata", "status": "running"},
        {"event": "task.failed", "task_id": "t9", "task_type": "filter_metadata", "status": "failed",
         "error": {"code": "METADATA_READ_ERROR", "message": "读取失败"}},
    ]
    frames = _run_router(events)
    text = json.dumps(frames, ensure_ascii=False).lower()
    for bad in ("filepath", "base64", "authorization", "api_key", "sk-",
                "data:image/", "raw_response", "provider_secret", "traceback",
                "http://internal", "/secret"):
        assert bad not in text, f"SSE 不应包含 {bad}"
    for f in frames:
        if f["event"] == "task.completed":
            os_ = f["data"].get("output_summary") or {}
            if "vision" in os_:
                assert set(os_["vision"].keys()) == {
                    "candidate_count", "success_count", "failed_count", "systemic_failed",
                }
                assert "title" not in json.dumps(os_, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 6) complete 唯一 + started/terminal 配对
# ---------------------------------------------------------------------------
def test_sse_complete_unique_and_terminal_pairing():
    frames = _run_router(_TEAM_EVENTS)
    assert sum(1 for f in frames if f["event"] == "complete") == 1
    started = [f for f in frames if f["event"] == "task.started"]
    terminals = [f for f in frames if f["event"] in ("task.completed", "task.failed", "task.skipped")]
    assert len(started) == len(terminals) == 3
    assert {f["data"]["task_id"] for f in started} == {f["data"]["task_id"] for f in terminals}


# ---------------------------------------------------------------------------
# 7) 事件顺序
# ---------------------------------------------------------------------------
def test_sse_event_ordering():
    frames = _run_router(_TEAM_EVENTS)
    names = [f["event"] for f in frames]
    assert names[0] == "rewrite_start"
    assert names[1] == "plan.created"
    assert names[-1] == "complete"
    first_delta = names.index("assistant_delta")
    for n in names:
        if n.startswith("task."):
            assert names.index(n) < first_delta   # 所有 task 事件在 assistant_delta 之前


# ---------------------------------------------------------------------------
# 8) executor _output_summary 的 Vision 分支（SSE 摘要来源）
# ---------------------------------------------------------------------------
def test_executor_output_summary_vision_counts():
    vo = build_vision_output([
        vision_item_success(1, title="a", description="d", tags=["t"], evidence="e", confidence=0.9),
        vision_item_success(2, title="b", description="d", tags=["t"], evidence="e", confidence=0.8),
        vision_item_success(3, title="c", description="d", tags=["t"], evidence="e", confidence=0.7),
        vision_item_failed(4, VisionErrorCode.TIMEOUT, "timeout"),
    ])
    res = WorkerResult(task_id="t3", task_type=TaskType.ANALYZE_VISUAL,
                       status=TaskStatus.SUCCEEDED, output=vo)
    s = _output_summary(res)
    assert s["vision"] == {
        "candidate_count": 4, "success_count": 3, "failed_count": 1, "systemic_failed": False,
    }
    text = json.dumps(s, ensure_ascii=False)
    for bad in ("title", "description", "tags", "evidence", "confidence",
                "filepath", "base64", "public_url"):
        assert bad not in text


def test_executor_output_summary_vision_systemic():
    vo = vision_systemic_failure(VisionErrorCode.CLIENT_UNAVAILABLE, "no client")
    res = WorkerResult(task_id="t3", task_type=TaskType.ANALYZE_VISUAL,
                       status=TaskStatus.SUCCEEDED, output=vo)
    s = _output_summary(res)
    assert s["vision"] == {
        "candidate_count": 0, "success_count": 0, "failed_count": 0, "systemic_failed": True,
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

