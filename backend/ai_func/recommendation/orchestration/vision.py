"""Phase 2 Stage 1 — Vision 结构化 Contract（Freeze Contracts）。

定义 Vision 能力的数据契约与 LLM 输出解析器，独立于 Worker/Executor/Synthesizer。
本模块不注册任何能力；ANALYZE_VISUAL 的启用由后续 Stage（4）统一走 vision_available()。

包含：
  - VisionStatus：success / item_failed / systemic_failed
  - VisionErrorCode：invalid_response / timeout / provider_error / parse_error /
                      invalid_image_id / unknown_error
  - VisionItemResult：单张图片的结构化视觉结果（extra=forbid）
  - VisionTaskOutput：一次 Vision 任务的聚合结果（extra=forbid）
  - parse_vision_response()：LLM 原始输出 → VisionItemResult（绝不抛异常）
  - build_vision_output() / vision_item_failed() / vision_systemic_failure()：构造辅助
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import DEFAULT_VISION_CANDIDATES, MAX_VISUAL_CANDIDATES, VISION_ITEM_TIMEOUT_SECONDS

# ---------------------------------------------------------------------------
# 常量（Vision 相关边界，集中定义；DEFAULT_VISION_CANDIDATES 与
# VISION_ITEM_TIMEOUT_SECONDS 定义在 models.py，此处仅再导出供本模块使用）
# ---------------------------------------------------------------------------
MAX_VISION_TITLE_LEN = 60              # 标题长度上限
MAX_VISION_DESC_LEN = 200              # 描述长度上限
MAX_VISION_EVIDENCE_LEN = 300          # evidence 文本长度上限
MAX_VISION_REASON_LEN = 200            # failure_reason 长度上限
MAX_VISION_TAGS = 8                    # tags 数量上限
MAX_VISION_TAG_LEN = 30                # 单个 tag 长度上限
MAX_VISION_TOTAL_EVIDENCE_CHARS = 4000 # Synthesizer payload 预算（Stage 6 使用）
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL)


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------
class VisionStatus(str, Enum):
    SUCCESS = "success"
    ITEM_FAILED = "item_failed"
    SYSTEMIC_FAILED = "systemic_failed"


class VisionErrorCode(str, Enum):
    INVALID_RESPONSE = "invalid_response"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    PARSE_ERROR = "parse_error"
    INVALID_IMAGE_ID = "invalid_image_id"
    UNSUPPORTED_MIME = "unsupported_mime"       # Phase 2 Stage 3：图片来源 MIME 不支持
    CLIENT_UNAVAILABLE = "client_unavailable"   # Phase 2 Stage 3：视觉客户端不可用（systemic）
    UNKNOWN_ERROR = "unknown_error"


# ---------------------------------------------------------------------------
# 单图结果（Pydantic，extra=forbid —— 不允许模型随意扩展 schema）
# ---------------------------------------------------------------------------
class VisionItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: int
    status: VisionStatus = VisionStatus.SUCCESS
    title: str = ""
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    evidence: str = ""          # 简短视觉证据（仅图片可见内容）
    objects: List[str] = Field(default_factory=list)
    scene: str = ""
    colors: List[str] = Field(default_factory=list)
    ocr_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    error_code: Optional[VisionErrorCode] = None
    failure_reason: str = ""

    @field_validator("image_id")
    @classmethod
    def _image_id_nonnegative(cls, v: int) -> int:
        # image_id == 0 为保留哨兵值：仅用于「非法 image_id」导致的 ITEM_FAILED 结果，
        # 真实候选图片的 id 恒为正整数。
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"invalid image_id: {v!r}")
        return v


# ---------------------------------------------------------------------------
# 任务聚合结果（extra=forbid）
# ---------------------------------------------------------------------------
class VisionTaskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[VisionItemResult] = Field(default_factory=list)
    total: int = 0
    success_count: int = 0
    failed_count: int = 0
    systemic_failed: bool = False
    systemic_error_code: Optional[VisionErrorCode] = None
    systemic_failure_reason: str = ""
    criteria: List[str] = Field(default_factory=list)


class VisionEvidence(BaseModel):
    """供 Synthesizer 消费的受控视觉证据（仅成功 item，Stage 5 propagation）。

    - 白名单字段，extra=forbid；由成功 VisionItemResult 派生
    - 不含 filepath / public_url / base64 / raw provider response
    - 字段长度 / confidence 在解析阶段已受 VisionItemResult 限制（bounded）
    """
    model_config = ConfigDict(extra="forbid")
    image_id: int
    title: str = ""
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    evidence: str = ""
    objects: List[str] = Field(default_factory=list)
    scene: str = ""
    colors: List[str] = Field(default_factory=list)
    ocr_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Parser：LLM 原始输出 → VisionItemResult（绝不抛异常；所有异常收敛为 ITEM_FAILED）
# 规则表：
#   空输出/非 JSON/非 dict/未知顶层字段/标题与描述均空/非法 confidence 类型
#       → ITEM_FAILED PARSE_ERROR
#   非法 image_id                             → ITEM_FAILED INVALID_IMAGE_ID
#   confidence 超出 [0,1]                     → 钳制到 [0,1]
#   超长字段                                  → 截断到对应上限
#   tags 超量 / 单个 tag 超长                  → 截断
# ---------------------------------------------------------------------------
def parse_vision_response(
    raw: str,
    image_id: int,
    *,
    title_max: int = MAX_VISION_TITLE_LEN,
    desc_max: int = MAX_VISION_DESC_LEN,
    evidence_max: int = MAX_VISION_EVIDENCE_LEN,
    reason_max: int = MAX_VISION_REASON_LEN,
    tags_max: int = MAX_VISION_TAGS,
    tag_max: int = MAX_VISION_TAG_LEN,
) -> VisionItemResult:
    # 1) 空输出 / 围栏剥离 / JSON 解析
    text = (raw or "").strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    if not text:
        return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, "empty vision output")
    try:
        data = json.loads(text)
    except Exception as e:
        return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, f"malformed JSON: {str(e)[:120]}")
    if not isinstance(data, dict):
        return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, "vision output is not a JSON object")

    # 2) 非法 image_id
    if not isinstance(image_id, int) or image_id <= 0:
        return vision_item_failed(0, VisionErrorCode.INVALID_IMAGE_ID, "invalid image_id")

    # 3) 未知顶层字段 → 拒绝（extra=forbid 语义）
    allowed = {
        "title", "description", "tags", "evidence", "confidence",
        "objects", "scene", "colors", "ocr_text",
    }
    unknown = [k for k in data if k not in allowed]
    if unknown:
        return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, f"unknown fields: {sorted(unknown)}")

    # 4) 类型与边界
    title = data.get("title")
    description = data.get("description")
    if not isinstance(title, str) or not isinstance(description, str):
        return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, "title/description must be strings")
    if not title.strip() and not description.strip():
        return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, "missing required fields (title/description)")

    tags = data.get("tags", [])
    if not isinstance(tags, list):
        return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, "tags must be a list")
    tags = [str(t) for t in tags if isinstance(t, (str, int, float))][:tags_max]
    tags = [t[:tag_max] for t in tags]
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, "objects must be a list")
    objects = [str(t)[:tag_max] for t in objects if isinstance(t, (str, int, float))][:tags_max]
    colors = data.get("colors", [])
    if not isinstance(colors, list):
        return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, "colors must be a list")
    colors = [str(t)[:tag_max] for t in colors if isinstance(t, (str, int, float))][:tags_max]
    scene = str(data.get("scene") or "")[:tag_max]
    ocr_text = str(data.get("ocr_text") or "")[:evidence_max]

    evidence = str(data.get("evidence") or "")[:evidence_max]

    confidence = 0.0
    conf_raw = data.get("confidence", 0.0)
    if isinstance(conf_raw, (int, float)) and not isinstance(conf_raw, bool):
        confidence = max(0.0, min(1.0, float(conf_raw)))  # 钳制
    elif isinstance(conf_raw, str) and conf_raw.strip():
        try:
            confidence = max(0.0, min(1.0, float(conf_raw)))
        except Exception:
            return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, "invalid confidence value")
    elif conf_raw is not None and not isinstance(conf_raw, str):
        return vision_item_failed(image_id, VisionErrorCode.PARSE_ERROR, "invalid confidence type")

    return VisionItemResult(
        image_id=image_id,
        status=VisionStatus.SUCCESS,
        title=title[:title_max],
        description=description[:desc_max],
        tags=tags,
        evidence=evidence,
        objects=objects,
        scene=scene,
        colors=colors,
        ocr_text=ocr_text,
        confidence=confidence,
        error_code=None,
        failure_reason="",
    )


# ---------------------------------------------------------------------------
# 构造辅助（Worker / 测试使用）
# ---------------------------------------------------------------------------
def vision_item_success(
    image_id: int,
    *,
    title: str,
    description: str,
    tags: Optional[List[str]] = None,
    evidence: str = "",
    confidence: float = 0.0,
) -> VisionItemResult:
    return VisionItemResult(
        image_id=image_id,
        status=VisionStatus.SUCCESS,
        title=str(title)[:MAX_VISION_TITLE_LEN],
        description=str(description)[:MAX_VISION_DESC_LEN],
        tags=[str(t)[:MAX_VISION_TAG_LEN] for t in (tags or [])][:MAX_VISION_TAGS],
        evidence=str(evidence)[:MAX_VISION_EVIDENCE_LEN],
        confidence=max(0.0, min(1.0, float(confidence))),
        error_code=None,
        failure_reason="",
    )


def vision_item_failed(
    image_id: int,
    code: VisionErrorCode,
    reason: str,
) -> VisionItemResult:
    try:
        _ = int(image_id)
    except (TypeError, ValueError):
        image_id = 0
    return VisionItemResult(
        image_id=int(image_id),
        status=VisionStatus.ITEM_FAILED,
        title="",
        description="",
        tags=[],
        evidence="",
        confidence=0.0,
        error_code=code,
        failure_reason=str(reason)[:MAX_VISION_REASON_LEN],
    )


def vision_systemic_failure(
    code: VisionErrorCode,
    reason: str,
    criteria: Optional[List[str]] = None,
) -> VisionTaskOutput:
    """Provider/API 级失败：整个任务失败，不代表任何单张图片。"""
    return VisionTaskOutput(
        items=[],
        total=0,
        success_count=0,
        failed_count=0,
        systemic_failed=True,
        systemic_error_code=code,
        systemic_failure_reason=str(reason)[:MAX_VISION_REASON_LEN],
        criteria=list(criteria or []),
    )


def build_vision_output(
    items: List[VisionItemResult],
    criteria: Optional[List[str]] = None,
) -> VisionTaskOutput:
    """根据逐张结果计算聚合统计（含成功/失败计数）。"""
    success = [i for i in items if i.status == VisionStatus.SUCCESS]
    failed = [i for i in items if i.status == VisionStatus.ITEM_FAILED]
    return VisionTaskOutput(
        items=list(items),
        total=len(items),
        success_count=len(success),
        failed_count=len(failed),
        systemic_failed=False,
        systemic_error_code=None,
        systemic_failure_reason="",
        criteria=list(criteria or []),
    )


def cap_evidence_total(
    items: List[VisionItemResult],
    budget: int = MAX_VISION_TOTAL_EVIDENCE_CHARS,
) -> List[VisionItemResult]:
    """聚合 evidence 总字符预算控制（默认 4000，Stage 6 Synthesizer payload 预算）。

    只截断 success item 的 evidence 文本，不改变 status / title / description /
    tags / confidence。失败 item 保持原样。
    """
    remaining = budget
    out: List[VisionItemResult] = []
    for it in items:
        if it.status != VisionStatus.SUCCESS or not it.evidence:
            out.append(it)
            continue
        ev = it.evidence
        if len(ev) > remaining:
            ev = ev[:remaining]
        remaining -= len(ev)
        out.append(it.model_copy(update={"evidence": ev}))
    return out


