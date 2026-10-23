"""Phase 1-B — 受控 Workers（规划 §6.5/§8）。

- RetrievalWorker   复用 tools.py::tool_search_images（不重实现向量搜索）
- MetadataWorker    确定性过滤 / 派生字段（不调 LLM、不写库）
- GroupImagesWorker 确定性分组（仅有限枚举字段）
- DeterministicSynthesisWorker   SYNTHESIZE 的确定性占位聚合
                                （Phase 1-C 将替换为 LLM SynthesizerAgent）

Worker 之间不自由通信；只通过 Executor 提供的 ExecutionContext 读取
前置任务的结构化结果。所有输出均为结构化 Pydantic 模型。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    AnalyzeVisualInput,
    CountNamedObjectInput,
    DEFAULT_VISION_CANDIDATES,
    DeduplicateImagesInput,
    JudgeMainSubjectInput,
    ListObjectsInput,
    ReadVisibleTextInput,
    VerifySpatialInput,
    ExpandQueryInput,
    FilterByLabelsInput,
    FilterMetadataInput,
    GroupImagesInput,
    MAX_VISUAL_CANDIDATES,
    RerankCandidatesInput,
    RetrieveImagesInput,
    RetrieveSimilarInput,
    RewriteQueryInput,
    ScopeDatasetInput,
    SelectImagesInput,
    SortImagesInput,
    SynthesizeInput,
    Task,
    TaskStatus,
    TaskType,
    VerifyVisualInput,
    VISION_ITEM_TIMEOUT_SECONDS,
    vision_available,
)
from ..tools import tool_retrieve_similar, tool_search_by_labels, tool_search_images
from ...image_analysis import (
    ImageAnalysis,
    ImageAnalysisError,
    ImageAnalysisErrorCode,
)
from .vision import (
    VisionErrorCode,
    VisionEvidence,
    VisionItemResult,
    VisionStatus,
    VisionTaskOutput,
    build_vision_output,
    cap_evidence_total,
    parse_vision_response,
    vision_item_failed,
    vision_systemic_failure,
)


# ---------------------------------------------------------------------------
# 结构化输出模型（§6.6 简化版；禁止返回 str(result)）
# ---------------------------------------------------------------------------
class ImageCandidate(BaseModel):
    """检索候选。filepath 仅内部使用，禁止进入 SSE / LLM 输出。
    vision: 成功视觉证据（Stage 5 propagation；仅 Synthesizer 消费，不进 SSE）。"""
    model_config = ConfigDict(extra="ignore")
    id: int
    score: Optional[float] = None
    title: str = ""
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    width: Optional[int] = None
    height: Optional[int] = None
    created_at: Optional[str] = None
    file_type: Optional[str] = None
    public_url: Optional[str] = None
    filepath: Optional[str] = None
    vision: Optional[VisionEvidence] = None


class RetrievalOutput(BaseModel):
    candidates: List[ImageCandidate] = Field(default_factory=list)
    query: str = ""
    total: int = 0


class MetadataOutput(BaseModel):
    candidates: List[ImageCandidate] = Field(default_factory=list)
    rejected_ids: List[int] = Field(default_factory=list)
    applied_predicates: List[Any] = Field(default_factory=list)


class ImageGroup(BaseModel):
    key: str
    image_ids: List[int] = Field(default_factory=list)
    candidates: List[ImageCandidate] = Field(default_factory=list)


class GroupOutput(BaseModel):
    group_by: str
    groups: List[ImageGroup] = Field(default_factory=list)


class CandidateListOutput(BaseModel):
    """扁平候选列表变换（rerank / dedup / sort / select）的共用底座。"""
    candidates: List[ImageCandidate] = Field(default_factory=list)
    total: int = 0


class RerankOutput(CandidateListOutput):
    query: str = ""
    provider: str = "r1"
    ordered_ids: List[int] = Field(default_factory=list)


class DedupOutput(CandidateListOutput):
    removed_ids: List[int] = Field(default_factory=list)
    removed_count: int = 0


class SortOutput(CandidateListOutput):
    sort_by: str = "score"
    order: str = "desc"


class SelectOutput(CandidateListOutput):
    selected_ids: List[int] = Field(default_factory=list)
    strategy: str = "top_n"
    limit: int = 5
    selection_reason: str = ""


class VerifyVisualItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: int
    passed: bool = False
    confidence: float = 0.0
    evidence: str = ""
    tags: List[str] = Field(default_factory=list)
    status: str = "success"
    error: str = ""


class VerifyTaskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[VerifyVisualItem] = Field(default_factory=list)
    candidates: List[ImageCandidate] = Field(default_factory=list)
    passed_ids: List[int] = Field(default_factory=list)
    condition: str = ""
    inspected_count: int = 0
    passed_count: int = 0


class VisionFacetItem(BaseModel):
    """窄视觉任务的单张结果。passed 只对会删候选的任务有筛选意义。"""
    model_config = ConfigDict(extra="forbid")
    candidate_id: int
    passed: bool = True
    text: str = ""
    objects: List[str] = Field(default_factory=list)
    subject: str = ""
    count: Optional[int] = None
    confidence: float = 0.0
    evidence: str = ""
    status: str = "success"
    error: str = ""


class VisionFacetOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    facet: str
    items: List[VisionFacetItem] = Field(default_factory=list)
    candidates: List[ImageCandidate] = Field(default_factory=list)
    inspected_count: int = 0
    passed_count: int = 0


class SynthesisOutput(BaseModel):
    image_ids: List[int] = Field(default_factory=list)
    candidates: List[ImageCandidate] = Field(default_factory=list)
    presentation: str = "flat"
    group_summary: Dict[str, int] = Field(default_factory=dict)
    # Phase 1-C：LLM SynthesizerAgent 输出
    answer: str = ""                 # 最终自然语言回答（LLM 或确定性文本）
    rationale: str = ""              # 简短推荐理由
    synthesis_mode: str = "deterministic"   # "llm" | "deterministic"
    synthesis_degraded: bool = False        # True = 曾回退到确定性合成


class TaskError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class WorkerResult(BaseModel):
    task_id: str
    task_type: TaskType
    status: TaskStatus
    output: Optional[Any] = None
    error: Optional[TaskError] = None
    duration_ms: Optional[int] = None


class ExecutionContext:
    """最小 Working Execution Context（规划 §16）。

    Worker 只读取本 Context，不接触聊天历史 / 系统提示。
    results: task_id → WorkerResult（前置任务的结构化结果）。
    """

    def __init__(self, plan: Any = None):
        self.plan = plan
        self.goal = plan.goal if plan is not None else ""
        self.results: Dict[str, WorkerResult] = {}
        self.metadata: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Worker 接口（Protocol，非继承树；显式 dispatch map，见 default_workers）
# ---------------------------------------------------------------------------
class Worker(Protocol):
    task_type: TaskType

    def run(self, task: Task, context: Any) -> WorkerResult:
        ...


# ---------------------------------------------------------------------------
# 辅助：从依赖结果收集候选 / 去重 / 派生字段
# ---------------------------------------------------------------------------
def collect_candidates(task: Task, results: Dict[str, WorkerResult]) -> List[ImageCandidate]:
    """从所有 SUCCEEDED 依赖的输出中收集候选（Stage 5 起支持 Vision 传播）。

    - Retrieval / Metadata / Group 输出 → ImageCandidate 列表
    - VisionTaskOutput（ANALYZE_VISUAL）→ 成功 item 的视觉证据按 image_id 合并进候选
    - 失败 item 不注入；vision-only id（无既有候选）不创造新候选；
      systemic 失败的 Vision 输出不注入任何证据
    """
    cands: List[ImageCandidate] = []
    vision_outputs: List[VisionTaskOutput] = []
    for dep in task.depends_on:
        res = results.get(dep)
        if res is None or res.status != TaskStatus.SUCCEEDED:
            continue
        out = res.output
        if isinstance(out, (RetrievalOutput, MetadataOutput, CandidateListOutput, VerifyTaskOutput, VisionFacetOutput)):
            cands.extend(getattr(out, "candidates", None) or [])
        elif isinstance(out, GroupOutput):
            for g in out.groups:
                cands.extend(g.candidates)
        elif isinstance(out, VisionTaskOutput):
            vision_outputs.append(out)
    return _attach_vision_evidence(cands, vision_outputs)


def _merge_labels(item: Any) -> List[str]:
    labels: List[str] = []
    for raw in list(getattr(item, "tags", None) or []) + list(getattr(item, "objects", None) or []):
        text = str(raw).strip()
        if text and text not in labels:
            labels.append(text)
    scene = str(getattr(item, "scene", "") or "").strip()
    if scene and scene not in labels:
        labels.append(scene)
    return labels[:8]


def _attach_vision_evidence(
    cands: List[ImageCandidate],
    vision_outputs: List[VisionTaskOutput],
) -> List[ImageCandidate]:
    """把成功 Vision 证据合并到已有候选（按 image_id，保持 identity / 不重复 / 不覆盖）。

    规则（Stage 5.4）：
    - image_id 是唯一 merge 键；vision 不修改候选身份、不创造新候选
    - item_failed / systemic_failed：不注入任何 title/description/tags/evidence
    - 已带 vision 的候选：不覆盖（首次出现优先）
    """
    if not vision_outputs or not cands:
        return cands
    first_by_id: Dict[int, ImageCandidate] = {}
    for c in cands:
        if c.id not in first_by_id:
            first_by_id[c.id] = c
    for vo in vision_outputs:
        if vo.systemic_failed:
            continue
        for item in vo.items:
            if item.status != VisionStatus.SUCCESS:
                continue
            cand = first_by_id.get(item.image_id)
            if cand is None or cand.vision is not None:
                continue
            cand.vision = VisionEvidence(
                image_id=item.image_id,
                title=item.title,
                description=item.description,
                tags=_merge_labels(item),
                evidence=item.evidence,
                objects=list(item.objects),
                scene=item.scene,
                colors=list(item.colors),
                ocr_text=item.ocr_text,
                confidence=item.confidence,
            )
    return cands


def dedup_candidates(cands: List[ImageCandidate]) -> List[ImageCandidate]:
    seen = set()
    out: List[ImageCandidate] = []
    for c in cands:
        if c.id in seen:
            continue
        seen.add(c.id)
        out.append(c)
    return out


def derive_orientation(cand: ImageCandidate) -> Optional[str]:
    w, h = cand.width, cand.height
    if w is None or h is None:
        return None
    if w > h:
        return "landscape"
    if w < h:
        return "portrait"
    return "square"


def _as_num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class RewriteOutput(BaseModel):
    """rewrite_query 的结果。检索任务只消费 query 和 filters，不再自己改写。"""
    query: str = ""
    filters: Dict[str, Any] = Field(default_factory=dict)
    orientation: Optional[str] = None
    needs_images: bool = True
    dataset_hint: str = ""


class ExpandQueryOutput(BaseModel):
    """expand_query 的结果：LLM 扩写后的主检索句。"""
    query: str = ""
    alternates: List[str] = Field(default_factory=list)
    source_query: str = ""


class ScopeDatasetOutput(BaseModel):
    dataset_id: int
    dataset_name: str = ""
    label_vocab: List[str] = Field(default_factory=list)


class RewriteWorker:
    """只跑意图改写规则，不访问索引。"""

    task_type = TaskType.REWRITE_QUERY

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, RewriteQueryInput):
            return _input_type_error(task, "RewriteQueryInput")
        from .intent import prepare_retrieval

        prepared = prepare_retrieval(inp.query, {})
        output = RewriteOutput(
            query=prepared["query"],
            filters=prepared["filters"],
            orientation=prepared.get("orientation"),
            needs_images=bool(prepared.get("needs_images")),
            dataset_hint=str(prepared.get("dataset_hint") or ""),
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )


class ExpandQueryWorker:
    """LLM 语义扩写；可注入 llm_fn，默认 DeepSeek chat。"""

    task_type = TaskType.EXPAND_QUERY

    def __init__(self, llm_fn: Optional[Callable[[str, Optional[List[str]]], str]] = None):
        self._llm_fn = llm_fn

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, ExpandQueryInput):
            return _input_type_error(task, "ExpandQueryInput")
        source = (inp.previous_query or inp.query or "").strip()
        if not source:
            return _failed(task, "EXPAND_EMPTY", "expand_query 需要非空 query")
        vocab: Optional[List[str]] = None
        dataset_id = inp.dataset_id
        scoped = _scope_from_deps(task, context)
        if scoped is not None:
            dataset_id = dataset_id or scoped.dataset_id
            vocab = list(scoped.label_vocab or [])
        elif dataset_id is not None:
            try:
                from ....db_func.repositories.datasets import DatasetRepository

                vocab = DatasetRepository().label_vocab(int(dataset_id))
            except Exception:
                vocab = None
        try:
            if self._llm_fn is not None:
                expanded = (self._llm_fn(source, vocab) or "").strip()
            else:
                expanded = _llm_expand_query(source, vocab)
        except Exception as e:
            return _failed(task, "EXPAND_LLM_ERROR", str(e)[:200])
        if not expanded:
            expanded = source
        alts = [expanded]
        if expanded != source:
            alts = [expanded]
        output = ExpandQueryOutput(query=expanded, alternates=alts, source_query=source)
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )


class ScopeDatasetWorker:
    """解析数据集名/id，钉死检索域。"""

    task_type = TaskType.SCOPE_DATASET

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, ScopeDatasetInput):
            return _input_type_error(task, "ScopeDatasetInput")
        from ....db_func.repositories.datasets import DatasetRepository

        repo = DatasetRepository()
        try:
            repo.ensure_schema_seed()
        except Exception:
            pass
        ds = None
        if inp.dataset_id is not None:
            ds = repo.get_by_id(int(inp.dataset_id))
        name = (inp.dataset_name or "").strip()
        if ds is None and not name:
            # 尝试从上游 rewrite 的 dataset_hint 取
            results = getattr(context, "results", None) or {}
            for dep in task.depends_on:
                out = getattr(results.get(dep), "output", None)
                if isinstance(out, RewriteOutput) and out.dataset_hint:
                    name = out.dataset_hint
                    break
        if ds is None and name:
            ds = repo.get_by_name(name)
        if ds is None:
            available = [d.get("name") for d in (repo.list_datasets() or [])]
            return _failed(
                task,
                "DATASET_NOT_FOUND",
                f"未知数据集 {name or inp.dataset_id!r}；可选: {available}",
            )
        output = ScopeDatasetOutput(
            dataset_id=int(ds["id"]),
            dataset_name=str(ds.get("name") or ""),
            label_vocab=list(ds.get("label_vocab") or []),
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )


class FilterByLabelsWorker:
    """精确类别筛选，默认 AND；可独立产生候选。"""

    task_type = TaskType.FILTER_BY_LABELS

    def __init__(self, search_tool: Optional[Callable] = None):
        self._search_tool = search_tool or tool_search_by_labels

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, FilterByLabelsInput):
            return _input_type_error(task, "FilterByLabelsInput")
        labels = [str(x).strip() for x in (inp.labels or []) if str(x).strip()]
        if not labels:
            return _failed(task, "LABELS_EMPTY", "filter_by_labels 需要至少一个类别")
        dataset_id = inp.dataset_id
        scoped = _scope_from_deps(task, context)
        if scoped is not None:
            dataset_id = dataset_id if dataset_id is not None else scoped.dataset_id
            vocab = {str(x).lower() for x in (scoped.label_vocab or [])}
            if vocab:
                unknown = [x for x in labels if x.lower() not in vocab]
                # 允许中文别名等不在英文词表的情况：仅警告式放行，不硬拒
                _ = unknown
        try:
            rows = self._search_tool(
                labels,
                dataset_id=dataset_id,
                match=inp.match,
                limit=inp.limit,
            ) or []
        except Exception as e:
            return _failed(task, "LABEL_FILTER_ERROR", str(e)[:200])
        candidates = [RetrievalWorker._to_candidate(r) for r in rows]
        output = RetrievalOutput(
            candidates=candidates,
            query=",".join(labels),
            total=len(candidates),
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )


class RetrieveSimilarWorker:
    """以图搜图，挂在检索 Agent。"""

    task_type = TaskType.RETRIEVE_SIMILAR

    def __init__(self, search_tool: Optional[Callable] = None):
        self._search_tool = search_tool or tool_retrieve_similar

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, RetrieveSimilarInput):
            return _input_type_error(task, "RetrieveSimilarInput")
        dataset_id = inp.dataset_id
        scoped = _scope_from_deps(task, context)
        if scoped is not None and dataset_id is None:
            dataset_id = scoped.dataset_id
        try:
            rows = self._search_tool(
                int(inp.image_id),
                dataset_id=dataset_id,
                limit=inp.limit,
            ) or []
        except Exception as e:
            return _failed(task, "SIMILAR_ERROR", str(e)[:200])
        candidates = [RetrievalWorker._to_candidate(r) for r in rows]
        output = RetrievalOutput(
            candidates=candidates,
            query=f"similar:{inp.image_id}",
            total=len(candidates),
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )


def _scope_from_deps(task: Task, context: Any) -> Optional[ScopeDatasetOutput]:
    results = getattr(context, "results", None) or {}
    for dep in task.depends_on:
        res = results.get(dep)
        out = getattr(res, "output", None)
        if isinstance(out, ScopeDatasetOutput):
            return out
    return None


def _rewrite_from_deps(task: Task, context: Any) -> Optional[RewriteOutput]:
    results = getattr(context, "results", None) or {}
    for dep in task.depends_on:
        res = results.get(dep)
        out = getattr(res, "output", None)
        if isinstance(out, RewriteOutput):
            return out
    return None


def _expand_from_deps(task: Task, context: Any) -> Optional[ExpandQueryOutput]:
    results = getattr(context, "results", None) or {}
    for dep in task.depends_on:
        res = results.get(dep)
        out = getattr(res, "output", None)
        if isinstance(out, ExpandQueryOutput):
            return out
    return None


def _llm_expand_query(source: str, vocab: Optional[List[str]]) -> str:
    from ....config.init_service import get_openai_client
    from ....config import settings

    client = get_openai_client()
    if client is None:
        return source
    cfg = settings.get_config()
    model = getattr(cfg, "CHAT_MODEL", None) or "deepseek-chat"
    vocab_line = ""
    if vocab:
        vocab_line = (
            "若存在数据集词表，扩写后的词尽量落在词表或其上位/近义："
            + ", ".join(list(vocab)[:40])
            + "。\n"
        )
    system = (
        "你是图像检索查询改写器。把口语检索句改成更利于 CLIP 召回的短查询。"
        "可做上位/近义扩展（如男孩→人、小孩；熊→bear）。"
        "只输出一行改写后的检索句，不要解释，不要标点装饰。"
    )
    user = f"{vocab_line}原句：{source}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=64,
    )
    text = (resp.choices[0].message.content or "").strip()
    # 只取第一行
    line = text.splitlines()[0].strip().strip("\"'")
    return line or source


# ---------------------------------------------------------------------------
# RetrievalWorker（§8.2）—— 复用旧搜索能力
# ---------------------------------------------------------------------------
class RetrievalWorker:
    task_type = TaskType.RETRIEVE_IMAGES

    def __init__(
        self,
        search_tool: Optional[Callable] = None,
        *,
        rerank_enabled: bool = True,
        reranker: Optional[Callable] = None,
    ):
        self._search_tool = search_tool or tool_search_images
        self._rerank_enabled = rerank_enabled
        self._reranker = reranker  # 可注入（测试）；None → 默认 rerank_candidates

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, RetrieveImagesInput):
            return _input_type_error(task, "RetrieveImagesInput")
        query = inp.query
        filters = dict(inp.filters or {})
        orientation = None
        rewritten = _rewrite_from_deps(task, context)
        if rewritten is not None:
            if rewritten.query:
                query = rewritten.query
            for key, value in rewritten.filters.items():
                filters.setdefault(key, value)
            orientation = rewritten.orientation
        expanded = _expand_from_deps(task, context)
        if expanded is not None and expanded.query:
            query = expanded.query
        scoped = _scope_from_deps(task, context)
        if scoped is not None:
            filters.setdefault("dataset_id", scoped.dataset_id)
        if not (query or "").strip():
            output = RetrievalOutput(candidates=[], query=query, total=0)
            return WorkerResult(
                task_id=task.task_id, task_type=task.task_type,
                status=TaskStatus.SUCCEEDED, output=output,
            )
        try:
            rows = self._search_tool(
                query=query,
                targets=list(inp.vector_targets),
                filters=filters,
                limit=inp.limit,
            ) or []
        except Exception as e:
            return _failed(task, "RETRIEVAL_ERROR", str(e)[:200])
        candidates = [self._to_candidate(r) for r in rows]
        if orientation:
            candidates = [
                c for c in candidates
                if derive_orientation(c) in (None, orientation)
            ]
        # 检索内部仍用公式重排，避免每条查询都把原图送给视觉模型。
        if self._rerank_enabled:
            try:
                if self._reranker is not None:
                    candidates = list(self._reranker(query, candidates))
                else:
                    from .model_reranker import default_model_reranker

                    candidates = list(default_model_reranker()(query, candidates))
            except Exception as e:
                print(f"[RetrievalWorker] rerank skipped (original order): {type(e).__name__}: {e}")
        output = RetrievalOutput(candidates=candidates, query=query, total=len(candidates))
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )

    @staticmethod
    def _to_candidate(row: Dict[str, Any]) -> ImageCandidate:
        try:
            img_id = int(row.get("id"))
        except (TypeError, ValueError):
            img_id = 0
        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = json.loads(tags) if tags else []
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        return ImageCandidate(
            id=img_id,
            score=row.get("score"),
            title=row.get("title") or "",
            description=row.get("description") or "",
            tags=[str(t) for t in tags] if isinstance(tags, list) else [],
            metadata=metadata if isinstance(metadata, dict) else {},
            width=row.get("width"),
            height=row.get("height"),
            created_at=row.get("created_at"),
            file_type=row.get("file_type"),
            public_url=row.get("public_url"),
            filepath=row.get("filepath"),
        )


def _query_from_deps(task: Task, results: Dict[str, WorkerResult]) -> str:
    for dep in task.depends_on:
        res = results.get(dep)
        if res is None or res.status != TaskStatus.SUCCEEDED:
            continue
        out = res.output
        q = getattr(out, "query", None)
        if q:
            return str(q)
    return ""


# ---------------------------------------------------------------------------
# RerankWorker —— 抽出 RetrievalWorker 已有 R1/R2 重排，不重新召回
# ---------------------------------------------------------------------------
class RerankWorker:
    task_type = TaskType.RERANK_CANDIDATES

    def __init__(self, reranker: Optional[Callable] = None):
        self._reranker = reranker

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, RerankCandidatesInput):
            return _input_type_error(task, "RerankCandidatesInput")
        try:
            cands = dedup_candidates(collect_candidates(task, context.results))
        except Exception as e:
            return _failed(task, "RERANK_COLLECT_ERROR", str(e)[:200])
        query = (inp.query or "").strip() or _query_from_deps(task, context.results)
        provider = inp.provider
        try:
            if self._reranker is not None:
                ordered = list(self._reranker(query, cands))
                used = "injected"
            elif provider == "r2":
                from .model_reranker import default_model_reranker

                ordered = list(default_model_reranker()(query, cands))
                used = "r2"
            else:
                from .llm_rerank import rerank_with_deepseek_vision

                ordered, used = rerank_with_deepseek_vision(query, cands)
                if ordered is cands or ordered == cands:
                    used = used
        except Exception as e:
            from .rerank import rerank_candidates

            print(f"[RerankWorker] fallback R1: {type(e).__name__}: {e}")
            ordered = list(rerank_candidates(query, cands))
            used = "r1_fallback"
        output = RerankOutput(
            candidates=ordered,
            total=len(ordered),
            query=query,
            provider=used,
            ordered_ids=[c.id for c in ordered],
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )


# ---------------------------------------------------------------------------
# DedupWorker —— 仅 image id 去重（无 pHash）
# ---------------------------------------------------------------------------
class DedupWorker:
    task_type = TaskType.DEDUPLICATE_IMAGES

    def run(self, task: Task, context: Any) -> WorkerResult:
        if not isinstance(task.input, DeduplicateImagesInput):
            return _input_type_error(task, "DeduplicateImagesInput")
        try:
            raw = collect_candidates(task, context.results)
        except Exception as e:
            return _failed(task, "DEDUP_COLLECT_ERROR", str(e)[:200])
        seen = set()
        kept: List[ImageCandidate] = []
        removed: List[int] = []
        for c in raw:
            if c.id in seen:
                removed.append(c.id)
                continue
            seen.add(c.id)
            kept.append(c)
        output = DedupOutput(
            candidates=kept, total=len(kept),
            removed_ids=removed, removed_count=len(removed),
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )


# ---------------------------------------------------------------------------
# SortWorker —— 按白名单字段排序，不检索
# ---------------------------------------------------------------------------
class SortWorker:
    task_type = TaskType.SORT_IMAGES

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, SortImagesInput):
            return _input_type_error(task, "SortImagesInput")
        try:
            cands = dedup_candidates(collect_candidates(task, context.results))
        except Exception as e:
            return _failed(task, "SORT_COLLECT_ERROR", str(e)[:200])
        reverse = inp.order == "desc"

        def _key(c: ImageCandidate):
            if inp.sort_by == "score":
                return c.score if c.score is not None else (float("-inf") if reverse else float("inf"))
            if inp.sort_by == "created_at":
                return c.created_at or ""
            if inp.sort_by == "width":
                return c.width if c.width is not None else (0 if reverse else 10**9)
            if inp.sort_by == "height":
                return c.height if c.height is not None else (0 if reverse else 10**9)
            return c.title or ""

        ordered = sorted(cands, key=_key, reverse=reverse)
        output = SortOutput(
            candidates=ordered, total=len(ordered),
            sort_by=inp.sort_by, order=inp.order,
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )


# ---------------------------------------------------------------------------
# SelectWorker —— 按数量策略取子集，不是谓词过滤
# ---------------------------------------------------------------------------
class SelectWorker:
    task_type = TaskType.SELECT_IMAGES

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, SelectImagesInput):
            return _input_type_error(task, "SelectImagesInput")
        try:
            if inp.strategy == "per_group":
                selected, reason = self._select_per_group(task, context.results, inp.limit)
            else:
                cands = dedup_candidates(collect_candidates(task, context.results))
                selected = cands[: inp.limit]
                reason = f"top_n limit={inp.limit} from {len(cands)}"
        except Exception as e:
            return _failed(task, "SELECT_COLLECT_ERROR", str(e)[:200])
        output = SelectOutput(
            candidates=selected,
            total=len(selected),
            selected_ids=[c.id for c in selected],
            strategy=inp.strategy,
            limit=inp.limit,
            selection_reason=reason,
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )

    @staticmethod
    def _select_per_group(task: Task, results: Dict[str, WorkerResult], limit: int):
        selected: List[ImageCandidate] = []
        seen = set()
        groups_used = 0
        for dep in task.depends_on:
            res = results.get(dep)
            if res is None or res.status != TaskStatus.SUCCEEDED:
                continue
            out = res.output
            if not isinstance(out, GroupOutput):
                continue
            for g in out.groups:
                groups_used += 1
                for c in (g.candidates or [])[:limit]:
                    if c.id in seen:
                        continue
                    seen.add(c.id)
                    selected.append(c)
        if groups_used == 0:
            cands = dedup_candidates(collect_candidates(task, results))
            selected = cands[:limit]
            return selected, f"per_group fallback top_n limit={limit} (no GroupOutput)"
        return selected, f"per_group limit={limit} groups={groups_used}"


# ---------------------------------------------------------------------------
# MetadataWorker（§8.3）—— 确定性过滤 / 派生字段，不调 LLM
# ---------------------------------------------------------------------------
class MetadataWorker:
    task_type = TaskType.FILTER_METADATA

    def __init__(self, image_repo: Optional[Any] = None):
        # image_repo 可选：候选 metadata 为 None 时才按 ID 补全（默认检索结果已完整）
        self._image_repo = image_repo

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, FilterMetadataInput):
            return _input_type_error(task, "FilterMetadataInput")
        try:
            cands = dedup_candidates(collect_candidates(task, context.results))
            cands = self._maybe_enrich(cands)
        except Exception as e:
            return _failed(task, "METADATA_READ_ERROR", str(e)[:200])
        kept: List[ImageCandidate] = []
        rejected: List[int] = []
        for c in cands:
            if self._matches(c, inp.predicates):
                kept.append(c)
            else:
                rejected.append(c.id)
        output = MetadataOutput(
            candidates=kept,
            rejected_ids=rejected,
            applied_predicates=list(inp.predicates),
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )

    # -- 内部 ----------------------------------------------------------------
    def _maybe_enrich(self, cands: List[ImageCandidate]) -> List[ImageCandidate]:
        missing = [c for c in cands if c.metadata is None]
        if not missing or self._image_repo is None:
            return cands
        try:
            rows = self._image_repo.get_by_ids([c.id for c in missing])
            by_id = {int(r.get("id")): r for r in rows}
        except Exception:
            return cands  # 补全失败不影响主流程
        for c in cands:
            if c.metadata is None and c.id in by_id:
                r = by_id[c.id]
                c.metadata = r.get("metadata") or {}
                c.tags = [str(t) for t in (r.get("tags") or [])]
        return cands

    @staticmethod
    def _matches(cand: ImageCandidate, predicates) -> bool:
        for p in predicates:
            val = MetadataWorker._field_value(cand, p.field)
            if not MetadataWorker._apply_op(p.operator, val, p.value):
                return False
        return True

    @staticmethod
    def _field_value(cand: ImageCandidate, field: str) -> Any:
        if field == "orientation":
            return derive_orientation(cand)
        if field == "location":
            return cand.metadata.get("location") if cand.metadata else None
        return getattr(cand, field, None)  # created_at / width / height / file_type / tags

    @staticmethod
    def _apply_op(op: str, actual: Any, expected: Any) -> bool:
        if actual is None:
            return False
        if op == "eq":
            return actual == expected
        if op == "in":
            exp = expected if isinstance(expected, list) else [expected]
            return actual in exp
        if op == "contains":
            if isinstance(actual, list):
                return any(str(e) in actual for e in (expected if isinstance(expected, list) else [expected]))
            return str(expected) in str(actual)
        if op in ("gte", "lte"):
            a, b = _as_num(actual), _as_num(expected)
            if a is None or b is None:
                return False
            return a >= b if op == "gte" else a <= b
        return False


# ---------------------------------------------------------------------------
# GroupImagesWorker（§8.4）—— 确定性分组，仅有限枚举字段
# ---------------------------------------------------------------------------
class GroupImagesWorker:
    task_type = TaskType.GROUP_IMAGES

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, GroupImagesInput):
            return _input_type_error(task, "GroupImagesInput")
        try:
            cands = dedup_candidates(collect_candidates(task, context.results))
        except Exception as e:
            return _failed(task, "GROUP_ERROR", str(e)[:200])
        unknown = (inp.unknown_group_name or "未分类").strip() or "未分类"
        buckets: Dict[str, List[ImageCandidate]] = {}
        order: List[str] = []
        for c in cands:
            key = self._group_key(c, inp.group_by)
            if not key:
                key = unknown
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(c)
        groups = [
            ImageGroup(key=k, image_ids=[c.id for c in buckets[k]], candidates=buckets[k])
            for k in order
        ]
        output = GroupOutput(group_by=inp.group_by, groups=groups)
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )

    @staticmethod
    def _group_key(cand: ImageCandidate, group_by: str) -> Optional[str]:
        if group_by == "location":
            v = cand.metadata.get("location") if cand.metadata else None
            return str(v).strip() if v else None
        if group_by == "orientation":
            return derive_orientation(cand)
        if group_by in ("year", "month"):
            return GroupImagesWorker._created_part(cand, group_by)
        if group_by == "file_type":
            return (cand.file_type or "").strip() or None
        if group_by == "tag":
            return cand.tags[0].strip() if cand.tags else None
        return None

    @staticmethod
    def _created_part(cand: ImageCandidate, part: str) -> Optional[str]:
        m = re.match(r"(\d{4})(?:-(\d{2}))?", cand.created_at or "")
        if not m:
            return None
        if part == "year":
            return m.group(1)
        return m.group(1) + "-" + (m.group(2) or "01")


# ---------------------------------------------------------------------------
# VisionWorker（Phase 2 Stage 3）—— 受控串行视觉二次分析
# Stage 4 起：default_workers() 在 vision_available() 为真时注册（capability gated）。
# ---------------------------------------------------------------------------
def build_vision_prompt(
    criteria: Optional[List[str]] = None,
    goal: str = "",
) -> str:
    """criteria-aware、bounded、image-grounded 的视觉分析提示词。

    - 只允许描述图片中实际可见内容；禁止猜测地点/人物身份/拍摄时间/EXIF/metadata
    - 输出 JSON 严格符合 Stage 1 Vision Contract 字段（title/description/tags/evidence/confidence）
    - 不包含 conversation history / tool registry / 其他图片信息
    """
    focus_items: List[str] = [str(c).strip() for c in (criteria or []) if str(c).strip()]
    if goal and str(goal).strip():
        focus_items = [str(goal).strip()] + focus_items
    focus = "；".join(focus_items) if focus_items else "（无特定关注点，仅客观描述可见内容）"
    return (
        "请对这张图片进行视觉分析。\n"
        "只能描述图片中实际可观察到的内容，禁止猜测地点、人物身份、拍摄时间、EXIF 或任何数据库/元数据；"
        "未观察到的信息一律写 unknown。\n"
        f"分析关注点：{focus}\n"
        "输出严格 JSON："
        '{"title": "10字内标题", "description": "50字内描述", "tags": ["标签"], '
        '"objects": ["画面中的物体"], "scene": "场景", "colors": ["颜色"], '
        '"ocr_text": "图中可读文字，没有则空字符串", '
        '"evidence": "图片中可见的简短证据", "confidence": 0.0到1.0}\n'
        "只返回 JSON，不要有任何其他文字。"
    )


# ImageAnalysisErrorCode → VisionErrorCode 确定性映射（provider 细节不外泄）
_IMAGE_ANALYSIS_TO_VISION_ERROR = {
    ImageAnalysisErrorCode.TIMEOUT: VisionErrorCode.TIMEOUT,
    ImageAnalysisErrorCode.PROVIDER_ERROR: VisionErrorCode.PROVIDER_ERROR,
    ImageAnalysisErrorCode.PARSE_ERROR: VisionErrorCode.PARSE_ERROR,
    ImageAnalysisErrorCode.INVALID_IMAGE: VisionErrorCode.INVALID_IMAGE_ID,
    ImageAnalysisErrorCode.NO_IMAGE_SOURCE: VisionErrorCode.INVALID_IMAGE_ID,
    ImageAnalysisErrorCode.UNSUPPORTED_MIME: VisionErrorCode.UNSUPPORTED_MIME,
    ImageAnalysisErrorCode.CLIENT_UNAVAILABLE: VisionErrorCode.CLIENT_UNAVAILABLE,
    ImageAnalysisErrorCode.UNKNOWN_ERROR: VisionErrorCode.UNKNOWN_ERROR,
}


class VisionWorker:
    """ANALYZE_VISUAL 受控 Worker（Phase 2 Stage 3）。

    流程：候选收集（复用 collect_candidates）→ dedup → 限制候选数（≤8，默认 4）
    → criteria-aware prompt → 串行逐图调用 ImageAnalysis(strict boundary)
    → 逐图失败隔离 → 聚合为 VisionTaskOutput。

    不自行检索数据库；不规划/创建任务；不改 DAG；不自动 retry / 换模型 / 换 provider。
    失败一律收敛进 VisionTaskOutput（item_failed / systemic_failed），
    Worker 本身保持 SUCCEEDED，避免阻断下游 Synthesizer（Stage 5 语义）。
    """

    task_type = TaskType.ANALYZE_VISUAL

    def __init__(
        self,
        analyzer: Optional[Any] = None,
        item_timeout: Optional[float] = None,
        model: Optional[str] = None,
    ):
        self._analyzer = analyzer  # 可注入 Fake；None 时惰性创建 ImageAnalysis
        self._item_timeout = (
            item_timeout if item_timeout is not None else VISION_ITEM_TIMEOUT_SECONDS
        )
        self._model = model  # 显式模型覆盖（运行时 override，不改 config）

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, AnalyzeVisualInput):
            return _input_type_error(task, "AnalyzeVisualInput")
        criteria = list(inp.criteria)
        goal = getattr(context, "goal", "") or ""

        # 1) 解析视觉客户端（不可用 → systemic failure，不伪装成 item 结果）
        try:
            analyzer = self._analyzer if self._analyzer is not None else ImageAnalysis()
        except Exception as e:
            return self._ok(
                task,
                vision_systemic_failure(
                    VisionErrorCode.CLIENT_UNAVAILABLE,
                    f"无法初始化视觉客户端: {str(e)[:160]}",
                    criteria=criteria,
                ),
            )
        if getattr(analyzer, "client", None) is None:
            return self._ok(
                task,
                vision_systemic_failure(
                    VisionErrorCode.CLIENT_UNAVAILABLE,
                    "视觉客户端不可用（未配置 OpenAI 兼容客户端）",
                    criteria=criteria,
                ),
            )

        # 2) 候选收集 / dedup / 非法 ID 跳过 / 上限
        try:
            cands = self._collect_valid_candidates(task, context.results)
        except Exception as e:
            return _failed(task, "VISION_COLLECT_ERROR", str(e)[:200])
        limit = self._resolve_limit(inp.max_candidates)
        selected = cands[:limit]

        # 3) 空候选：不调用 provider
        if not selected:
            return self._ok(task, build_vision_output([], criteria=criteria))

        # 4) 串行逐图调用（明确 per-item timeout + strict 边界）
        prompt = build_vision_prompt(criteria, goal)
        items: List[VisionItemResult] = []
        for cand in selected:
            items.append(self._analyze_one(analyzer, cand, inp, prompt))

        # 5) 聚合输出（evidence 总预算控制）
        output = build_vision_output(cap_evidence_total(items), criteria=criteria)
        return self._ok(task, output)

    # -- 内部 ------------------------------------------------------------
    @staticmethod
    def _collect_valid_candidates(
        task: Task, results: Dict[str, WorkerResult]
    ) -> List[ImageCandidate]:
        cands = dedup_candidates(collect_candidates(task, results))
        return [c for c in cands if isinstance(c.id, int) and c.id > 0]

    @staticmethod
    def _resolve_limit(requested: Optional[int]) -> int:
        """requested 非法/缺失 → DEFAULT_VISION_CANDIDATES(4)；否则 min(requested, 8)。"""
        if not isinstance(requested, int) or requested <= 0:
            return DEFAULT_VISION_CANDIDATES
        return min(requested, MAX_VISUAL_CANDIDATES)

    @staticmethod
    def _image_source(cand: ImageCandidate) -> Optional[Dict[str, str]]:
        """图片来源：优先合法本地 filepath；其次 public_url；都没有 → None。"""
        fp = (cand.filepath or "").strip()
        if fp and os.path.exists(fp):
            return {"local_image_path": fp}
        url = (cand.public_url or "").strip()
        if url:
            return {"image_url": url}
        return None

    def _analyze_one(
        self,
        analyzer: Any,
        cand: ImageCandidate,
        inp: AnalyzeVisualInput,
        prompt: str,
    ) -> VisionItemResult:
        source = self._image_source(cand)
        if source is None:
            return vision_item_failed(
                cand.id,
                VisionErrorCode.INVALID_IMAGE_ID,
                "无可用图片来源（filepath/public_url 均缺失）",
            )
        kwargs: Dict[str, Any] = {
            "detail": inp.detail,
            "prompt": prompt,
            "timeout": self._item_timeout,
            "raise_on_error": True,
        }
        if self._model:
            kwargs["model"] = self._model
        if "local_image_path" in source:
            kwargs["local_image_path"] = source["local_image_path"]
        else:
            kwargs["image_url"] = source["image_url"]
        try:
            result = analyzer.analyze_image(**kwargs)
        except ImageAnalysisError as e:
            code = _IMAGE_ANALYSIS_TO_VISION_ERROR.get(
                e.code, VisionErrorCode.UNKNOWN_ERROR
            )
            return vision_item_failed(cand.id, code, e.safe_message)
        except Exception as e:
            return vision_item_failed(cand.id, VisionErrorCode.UNKNOWN_ERROR, str(e)[:160])
        if not isinstance(result, dict) or "error" in result:
            return vision_item_failed(cand.id, VisionErrorCode.PARSE_ERROR, "无效的分析结果")
        # 复用 Stage 1 parser：统一字段长度 / confidence / unknown-field 校验
        return parse_vision_response(json.dumps(result, ensure_ascii=False), cand.id)

    @staticmethod
    def _ok(task: Task, output: VisionTaskOutput) -> WorkerResult:
        return WorkerResult(
            task_id=task.task_id,
            task_type=task.task_type,
            status=TaskStatus.SUCCEEDED,
            output=output,
        )


def build_verify_prompt(condition: str) -> str:
    cond = (condition or "").strip()[:80] or "（未指定条件）"
    return (
        "请判断这张图片是否满足以下视觉条件。\n"
        "只能根据图片中实际可观察内容判断，禁止猜测地点、身份、拍摄时间或元数据。\n"
        f"条件：{cond}\n"
        "输出严格 JSON："
        '{"passed": true或false, "evidence": "可见证据", "confidence": 0.0到1.0, '
        '"tags": ["标签"], "title": "10字内", "description": "50字内"}\n'
        "只返回 JSON。"
    )


def _parse_verify_payload(result: Dict[str, Any], candidate_id: int) -> VerifyVisualItem:
    passed_raw = result.get("passed")
    if isinstance(passed_raw, bool):
        passed = passed_raw
    elif isinstance(passed_raw, str):
        passed = passed_raw.strip().lower() in {"true", "yes", "1", "通过"}
    else:
        passed = False
    conf = result.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        confidence = 0.0
    tags = result.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    return VerifyVisualItem(
        candidate_id=candidate_id,
        passed=passed,
        confidence=confidence,
        evidence=str(result.get("evidence") or "")[:300],
        tags=[str(t)[:30] for t in tags[:8]],
        status="success",
    )


class VerifyVisualWorker:
    """对明确条件做 pass/fail。复用 ImageAnalysis 客户端，不改 query、不召回。"""

    task_type = TaskType.VERIFY_VISUAL_CONDITION

    def __init__(self, analyzer: Optional[Any] = None, item_timeout: Optional[float] = None):
        self._analyzer = analyzer
        self._item_timeout = (
            item_timeout if item_timeout is not None else VISION_ITEM_TIMEOUT_SECONDS
        )

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, VerifyVisualInput):
            return _input_type_error(task, "VerifyVisualInput")
        try:
            analyzer = self._analyzer if self._analyzer is not None else ImageAnalysis()
        except Exception as e:
            return _failed(task, "VERIFY_CLIENT", str(e)[:160])
        try:
            cands = dedup_candidates(collect_candidates(task, context.results))
            cands = [c for c in cands if isinstance(c.id, int) and c.id > 0]
        except Exception as e:
            return _failed(task, "VERIFY_COLLECT_ERROR", str(e)[:200])
        limit = min(int(inp.max_candidates or DEFAULT_VISION_CANDIDATES), MAX_VISUAL_CANDIDATES)
        selected = cands[:limit]
        items: List[VerifyVisualItem] = []
        prompt = build_verify_prompt(inp.condition)
        for cand in selected:
            items.append(self._verify_one(analyzer, cand, inp, prompt))
        passed_ids = [it.candidate_id for it in items if it.passed]
        by_id = {c.id: c for c in selected}
        passed_cands = [by_id[i] for i in passed_ids if i in by_id]
        output = VerifyTaskOutput(
            items=items,
            candidates=passed_cands,
            passed_ids=passed_ids,
            condition=inp.condition,
            inspected_count=len(selected),
            passed_count=len(passed_ids),
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )

    def _verify_one(self, analyzer, cand, inp, prompt) -> VerifyVisualItem:
        fp = (cand.filepath or "").strip()
        url = (cand.public_url or "").strip()
        if not (fp and os.path.exists(fp)) and not url:
            return VerifyVisualItem(
                candidate_id=cand.id, passed=False, status="failed",
                error="无可用图片来源",
            )
        if getattr(analyzer, "client", None) is None and not hasattr(analyzer, "analyze_image"):
            return VerifyVisualItem(
                candidate_id=cand.id, passed=False, status="failed",
                error="视觉客户端不可用",
            )
        kwargs: Dict[str, Any] = {
            "detail": inp.detail,
            "prompt": prompt,
            "timeout": self._item_timeout,
            "raise_on_error": False,
        }
        if fp and os.path.exists(fp):
            kwargs["local_image_path"] = fp
        else:
            kwargs["image_url"] = url
        try:
            result = analyzer.analyze_image(**kwargs)
        except Exception as e:
            return VerifyVisualItem(
                candidate_id=cand.id, passed=False, status="failed",
                error=str(e)[:160],
            )
        if not isinstance(result, dict) or result.get("error"):
            return VerifyVisualItem(
                candidate_id=cand.id, passed=False, status="failed",
                error=str((result or {}).get("error") or "invalid verify result")[:160],
            )
        return _parse_verify_payload(result, cand.id)


_FACET_PROMPTS = {
    TaskType.READ_VISIBLE_TEXT: (
        "只读取这张图片上实际可见的文字。没有文字就让 text 为空。"
        "不要描述场景。输出严格 JSON："
        '{{"text": "可读文字", "confidence": 0.0到1.0, "evidence": "可见证据"}}'
    ),
    TaskType.LIST_OBJECTS: (
        "只列出这张图片里实际可见的物体名称，不要写长描述。"
        "输出严格 JSON："
        '{{"objects": ["物体"], "confidence": 0.0到1.0, "evidence": "可见证据"}}'
    ),
    TaskType.COUNT_NAMED_OBJECT: (
        "只估计画面中「{name}」的数量。看不清就 count 为 null，不要编造精确值。"
        "这是估计，不是检测器计数。输出严格 JSON："
        '{{"count": 0或null, "confidence": 0.0到1.0, "evidence": "可见证据"}}'
    ),
    TaskType.JUDGE_MAIN_SUBJECT: (
        "判断这张图片的主体是不是「{name}」。主体指画面最主要的对象，角落里的小物体不算。"
        "输出严格 JSON："
        '{{"passed": true或false, "subject": "你看到的主体", "confidence": 0.0到1.0, "evidence": "可见证据"}}'
    ),
    TaskType.VERIFY_SPATIAL: (
        "判断这张图片是否满足空间关系：「{name}」。只能根据可见位置，禁止猜测。"
        "输出严格 JSON："
        '{{"passed": true或false, "confidence": 0.0到1.0, "evidence": "可见证据"}}'
    ),
}

_FACET_FILTERS = {TaskType.JUDGE_MAIN_SUBJECT, TaskType.VERIFY_SPATIAL}


def _facet_prompt(task_type: TaskType, focus: str) -> str:
    template = _FACET_PROMPTS[task_type]
    return template.format(name=(focus or "").strip()[:40] or "未指定") + "\n只返回 JSON。"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "通过"}
    return False


def _as_conf(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


class FocusedVisionWorker:
    """窄视觉任务：读字、列物体、计数、主体、空间关系。不召回。"""

    def __init__(self, task_type: TaskType, analyzer: Optional[Any] = None, item_timeout: Optional[float] = None):
        self.task_type = task_type
        self._analyzer = analyzer
        self._item_timeout = item_timeout if item_timeout is not None else VISION_ITEM_TIMEOUT_SECONDS

    def run(self, task: Task, context: Any) -> WorkerResult:
        if task.task_type != self.task_type:
            return _input_type_error(task, self.task_type.value)
        inp = task.input
        focus = ""
        if isinstance(inp, CountNamedObjectInput):
            focus = inp.object_name
        elif isinstance(inp, JudgeMainSubjectInput):
            focus = inp.subject
        elif isinstance(inp, VerifySpatialInput):
            focus = inp.relation
        elif not isinstance(inp, (ReadVisibleTextInput, ListObjectsInput)):
            return _input_type_error(task, self.task_type.value)
        if self.task_type in (
            TaskType.COUNT_NAMED_OBJECT, TaskType.JUDGE_MAIN_SUBJECT, TaskType.VERIFY_SPATIAL,
        ) and not (focus or "").strip():
            return _failed(task, "VISION_FOCUS_EMPTY", f"{self.task_type.value} 需要明确对象或关系")
        try:
            analyzer = self._analyzer if self._analyzer is not None else ImageAnalysis()
        except Exception as e:
            return _failed(task, "VISION_CLIENT", str(e)[:160])
        try:
            cands = [c for c in dedup_candidates(collect_candidates(task, context.results)) if c.id > 0]
        except Exception as e:
            return _failed(task, "VISION_COLLECT_ERROR", str(e)[:200])
        limit = min(int(getattr(inp, "max_candidates", None) or DEFAULT_VISION_CANDIDATES), MAX_VISUAL_CANDIDATES)
        selected = cands[:limit]
        prompt = _facet_prompt(self.task_type, focus)
        detail = getattr(inp, "detail", "low")
        items: List[VisionFacetItem] = []
        kept: List[ImageCandidate] = []
        filtering = self.task_type in _FACET_FILTERS
        for cand in selected:
            item = self._one(analyzer, cand, detail, prompt)
            items.append(item)
            if item.status != "success":
                if not filtering:
                    kept.append(cand)
                continue
            noted = self._with_evidence(cand, item)
            if filtering:
                if item.passed:
                    kept.append(noted)
            else:
                kept.append(noted)
        passed_n = sum(1 for it in items if it.status == "success" and (it.passed or not filtering))
        output = VisionFacetOutput(
            facet=self.task_type.value,
            items=items,
            candidates=kept,
            inspected_count=len(selected),
            passed_count=passed_n if filtering else len(selected),
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )

    def _one(self, analyzer: Any, cand: ImageCandidate, detail: str, prompt: str) -> VisionFacetItem:
        fp = (cand.filepath or "").strip()
        url = (cand.public_url or "").strip()
        if not ((fp and os.path.exists(fp)) or url):
            return VisionFacetItem(candidate_id=cand.id, passed=False, status="failed", error="无可用图片来源")
        kwargs: Dict[str, Any] = {
            "detail": detail,
            "prompt": prompt,
            "timeout": self._item_timeout,
            "raise_on_error": False,
        }
        if fp and os.path.exists(fp):
            kwargs["local_image_path"] = fp
        else:
            kwargs["image_url"] = url
        try:
            result = analyzer.analyze_image(**kwargs)
        except Exception as e:
            return VisionFacetItem(candidate_id=cand.id, passed=False, status="failed", error=str(e)[:160])
        if not isinstance(result, dict) or result.get("error"):
            return VisionFacetItem(
                candidate_id=cand.id, passed=False, status="failed",
                error=str((result or {}).get("error") or "invalid vision result")[:160],
            )
        objects = result.get("objects") or []
        if not isinstance(objects, list):
            objects = []
        count_raw = result.get("count")
        count = None
        if isinstance(count_raw, bool):
            count = None
        elif isinstance(count_raw, (int, float)):
            count = int(count_raw)
        passed = True
        if self.task_type in _FACET_FILTERS:
            passed = _as_bool(result.get("passed"))
        return VisionFacetItem(
            candidate_id=cand.id,
            passed=passed,
            text=str(result.get("text") or "")[:300],
            objects=[str(x)[:30] for x in objects[:8]],
            subject=str(result.get("subject") or "")[:40],
            count=count,
            confidence=_as_conf(result.get("confidence")),
            evidence=str(result.get("evidence") or "")[:300],
        )

    def _with_evidence(self, cand: ImageCandidate, item: VisionFacetItem) -> ImageCandidate:
        bits = [item.evidence]
        if item.text:
            bits.append(item.text)
        if item.subject:
            bits.append(f"主体:{item.subject}")
        if item.count is not None:
            bits.append(f"数量:{item.count}")
        if item.objects:
            bits.append("物体:" + ",".join(item.objects))
        evidence = VisionEvidence(
            image_id=cand.id,
            title=cand.title[:20],
            description="；".join(b for b in bits if b)[:80],
            tags=(item.objects or ([item.subject] if item.subject else []))[:8],
            evidence=item.evidence[:300],
            objects=item.objects[:8],
            ocr_text=item.text[:300],
            confidence=item.confidence,
        )
        return cand.model_copy(update={"vision": evidence})


# ---------------------------------------------------------------------------
# DeterministicSynthesisWorker（§21 确定性 fallback，Phase 1-C 将替换为 LLM）
# ---------------------------------------------------------------------------
class DeterministicSynthesisWorker:
    """SYNTHESIZE 的确定性占位实现。

    - 从所有 SUCCEEDED 依赖收集候选，按 score 降序去重
    - 记录分组摘要（若依赖含 GroupOutput）
    - 截断到 recommendation_limit
    不做任何 LLM 生成。
    """

    task_type = TaskType.SYNTHESIZE

    def run(self, task: Task, context: Any) -> WorkerResult:
        inp = task.input
        if not isinstance(inp, SynthesizeInput):
            return _input_type_error(task, "SynthesizeInput")
        try:
            cands = dedup_candidates(collect_candidates(task, context.results))
        except Exception as e:
            return _failed(task, "SYNTHESIS_ERROR", str(e)[:200])
        cands.sort(key=lambda c: c.score if c.score is not None else 0.0, reverse=True)
        limit = inp.recommendation_limit
        final = cands[:limit]
        group_summary: Dict[str, int] = {}
        for dep in task.depends_on:
            res = context.results.get(dep)
            if res and res.status == TaskStatus.SUCCEEDED and isinstance(res.output, GroupOutput):
                for g in res.output.groups:
                    group_summary[g.key] = len(g.image_ids)
        output = SynthesisOutput(
            image_ids=[c.id for c in final],
            candidates=final,
            presentation=inp.presentation,
            group_summary=group_summary,
        )
        return WorkerResult(
            task_id=task.task_id, task_type=task.task_type,
            status=TaskStatus.SUCCEEDED, output=output,
        )


# ---------------------------------------------------------------------------
# Worker 注册表（显式 dispatch map，非通用框架）
# ---------------------------------------------------------------------------
def default_workers() -> Dict[TaskType, Worker]:
    """capability-gated 的默认 Worker 注册表。

    ANALYZE_VISUAL 仅在 vision_available() 为真时注册（capability 在注册层成立，
    不依赖运行时失败兜底）。
    """
    workers: Dict[TaskType, Worker] = {
        TaskType.REWRITE_QUERY: RewriteWorker(),
        TaskType.EXPAND_QUERY: ExpandQueryWorker(),
        TaskType.SCOPE_DATASET: ScopeDatasetWorker(),
        TaskType.FILTER_BY_LABELS: FilterByLabelsWorker(),
        TaskType.RETRIEVE_IMAGES: RetrievalWorker(),
        TaskType.RETRIEVE_SIMILAR: RetrieveSimilarWorker(),
        TaskType.RERANK_CANDIDATES: RerankWorker(),
        TaskType.FILTER_METADATA: MetadataWorker(),
        TaskType.DEDUPLICATE_IMAGES: DedupWorker(),
        TaskType.GROUP_IMAGES: GroupImagesWorker(),
        TaskType.SORT_IMAGES: SortWorker(),
        TaskType.SELECT_IMAGES: SelectWorker(),
        TaskType.SYNTHESIZE: DeterministicSynthesisWorker(),
    }
    if vision_available():
        workers[TaskType.ANALYZE_VISUAL] = VisionWorker()
        workers[TaskType.VERIFY_VISUAL_CONDITION] = VerifyVisualWorker()
        for facet in (
            TaskType.READ_VISIBLE_TEXT,
            TaskType.LIST_OBJECTS,
            TaskType.COUNT_NAMED_OBJECT,
            TaskType.JUDGE_MAIN_SUBJECT,
            TaskType.VERIFY_SPATIAL,
        ):
            workers[facet] = FocusedVisionWorker(facet)
    return workers


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------
def _input_type_error(task: Task, expected: str) -> WorkerResult:
    return WorkerResult(
        task_id=task.task_id, task_type=task.task_type,
        status=TaskStatus.FAILED,
        error=TaskError(code="WORKER_INPUT_TYPE", message=f"{task.task_id} 需要 {expected}"),
    )


def _failed(task: Task, code: str, message: str) -> WorkerResult:
    return WorkerResult(
        task_id=task.task_id, task_type=task.task_type,
        status=TaskStatus.FAILED,
        error=TaskError(code=code, message=message),
    )


def status_summary(res: WorkerResult) -> str:
    """生成简短、无敏感数据的任务结果摘要（供事件/文本使用）。"""
    out = res.output
    if res.status != TaskStatus.SUCCEEDED or out is None:
        return res.error.code if res.error else res.status.value
    if isinstance(out, RetrievalOutput):
        return f"检索到 {out.total} 张候选"
    if isinstance(out, MetadataOutput):
        return f"筛选后保留 {len(out.candidates)} 张（剔除 {len(out.rejected_ids)}）"
    if isinstance(out, GroupOutput):
        return f"分为 {len(out.groups)} 组"
    if isinstance(out, VisionTaskOutput):
        if out.systemic_failed:
            code = out.systemic_error_code.value if out.systemic_error_code else "unknown"
            return f"视觉分析失败(systemic:{code})"
        return f"视觉分析 {out.success_count}/{out.total} 张成功"
    if isinstance(out, RerankOutput):
        return f"重排 {out.total} 张（{out.provider}）"
    if isinstance(out, DedupOutput):
        return f"去重后 {out.total} 张（移除 {out.removed_count}）"
    if isinstance(out, SortOutput):
        return f"按 {out.sort_by}/{out.order} 排序 {out.total} 张"
    if isinstance(out, SelectOutput):
        return f"选择 {len(out.selected_ids)} 张（{out.strategy}）"
    if isinstance(out, VerifyTaskOutput):
        return f"核验通过 {out.passed_count}/{out.inspected_count}"
    if isinstance(out, SynthesisOutput):
        return f"推荐 {len(out.image_ids)} 张"
    return "完成"

