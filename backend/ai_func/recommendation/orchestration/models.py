"""Typed Task Model for the Adaptive Agent Team (Phase 1).

设计依据：PHASE1_ADAPTIVE_AGENT_TEAM_IMPLEMENTATION_PLAN.md §6 / §7 / §14。
本文件包含：
  - 资源上限常量（§14）
  - TaskType / TaskStatus / RouteType 枚举（§4.2 / §6.1 / §6.2）
  - Typed Inputs（§6.3）
  - TaskDraft / TaskPlanDraft（Planner 原始输出，input 为原始 dict）
  - Task / TaskPlan（校验后不可变、input 已类型化）
  - PlanValidator（§7，确定性语义校验）
  - PlannerFailure（Planner 失败异常）
"""

from __future__ import annotations

import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# 资源上限（规划 §14）——集中定义，禁止散落 magic numbers
# ---------------------------------------------------------------------------
MAX_TASKS = 8                        # 计划内 Task 总数上限（含 SYNTHESIZE；支撑 Organizer 组合 DAG）
MAX_PLAN_DEPTH = 7                   # 最长依赖路径的节点数上限（retrieve→…→select→synthesize）
MAX_WORKERS = 20                     # 可注册 Worker 类型上限（含检索扩写/数据集/类别/相似图）
MAX_CONCURRENCY = 2                  # DAG 并发上限（Phase 1-B 使用）
MAX_RETRIEVAL_CONCURRENCY = 1        # 检索并发上限（Phase 1-B 使用）
MAX_VISION_CONCURRENCY = 1           # 视觉并发上限（Phase 1-B 使用）
MAX_RETRIEVAL_LIMIT_PER_TASK = 40    # 单个检索 Task 的 limit 上限
MAX_TOTAL_CANDIDATES = 80            # 全计划候选总量上限（Phase 1-B 使用）
MAX_VISUAL_CANDIDATES = 8            # 单次视觉分析候选上限（绝对上限）
DEFAULT_VISION_CANDIDATES = 4        # 视觉分析默认候选数（≤ MAX_VISUAL_CANDIDATES）
MAX_SYNTHESIS_CANDIDATES = 20        # 最终推荐数量上限
PLANNER_TIMEOUT_SECONDS = 20         # Planner LLM 调用超时
SYNTHESIZER_TIMEOUT_SECONDS = 45     # Synthesizer LLM 调用超时
VISION_ITEM_TIMEOUT_SECONDS = 30     # 单张图片视觉分析超时（串行逐图）

# VISION_MODEL 视为“未配置”的取值集合（配置里可能是字符串 "null"/""/"none"）
VISION_MODEL_UNAVAILABLE_VALUES = {"", "null", "none"}

_TASK_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------
class TaskType(str, Enum):
    """有限 TaskType 集合。禁止 LLM 自由创建任意类型。"""
    REWRITE_QUERY = "rewrite_query"
    EXPAND_QUERY = "expand_query"
    SCOPE_DATASET = "scope_dataset"
    FILTER_BY_LABELS = "filter_by_labels"
    RETRIEVE_IMAGES = "retrieve_images"
    RETRIEVE_SIMILAR = "retrieve_similar"
    RERANK_CANDIDATES = "rerank_candidates"
    FILTER_METADATA = "filter_metadata"
    DEDUPLICATE_IMAGES = "deduplicate_images"
    GROUP_IMAGES = "group_images"
    SORT_IMAGES = "sort_images"
    SELECT_IMAGES = "select_images"
    ANALYZE_VISUAL = "analyze_visual"
    VERIFY_VISUAL_CONDITION = "verify_visual_condition"
    READ_VISIBLE_TEXT = "read_visible_text"
    LIST_OBJECTS = "list_objects"
    COUNT_NAMED_OBJECT = "count_named_object"
    JUDGE_MAIN_SUBJECT = "judge_main_subject"
    VERIFY_SPATIAL = "verify_spatial"
    SYNTHESIZE = "synthesize"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class RouteType(str, Enum):
    LEGACY = "legacy"
    TEAM = "team"


# ---------------------------------------------------------------------------
# Complexity Gate 决策结构（§4.2）
# ---------------------------------------------------------------------------
class GateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: RouteType
    reasons: List[str] = Field(default_factory=list)
    matched_features: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Typed Inputs（§6.3）
# ---------------------------------------------------------------------------
_ALLOWED_VECTOR_TARGETS = Literal["title", "description", "image"]
_ALLOWED_METADATA_FIELDS = Literal[
    "created_at", "width", "height", "orientation", "file_type", "tags", "location"
]
_ALLOWED_GROUP_BY = Literal["location", "year", "month", "orientation", "file_type", "tag"]
_ALLOWED_OPERATORS = Literal["eq", "in", "contains", "gte", "lte"]


class RewriteQueryInput(BaseModel):
    """规则改写：客气话、分句、年份/尺寸/格式/标签/横竖；不搜图、不调 LLM。"""
    model_config = ConfigDict(extra="forbid")
    query: str


class ExpandQueryInput(BaseModel):
    """LLM 语义扩写：近义/上位（如男孩→人）；仅检索不好时使用。"""
    model_config = ConfigDict(extra="forbid")
    query: str
    dataset_id: Optional[int] = None
    previous_query: str = ""


class ScopeDatasetInput(BaseModel):
    """钉死检索域到某个数据集；不搜图。"""
    model_config = ConfigDict(extra="forbid")
    dataset_id: Optional[int] = None
    dataset_name: str = ""


class FilterByLabelsInput(BaseModel):
    """COCO 式精确类别筛选；默认多类 AND。可独立产生候选。"""
    model_config = ConfigDict(extra="forbid")
    labels: List[str] = Field(default_factory=list)
    match: Literal["all", "any"] = "all"
    dataset_id: Optional[int] = None
    limit: int = Field(default=40, ge=1, le=MAX_RETRIEVAL_LIMIT_PER_TASK)


class RetrieveImagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    vector_targets: List[_ALLOWED_VECTOR_TARGETS] = Field(
        default_factory=lambda: ["title", "description", "image"]
    )
    filters: Dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=20, ge=1, le=MAX_RETRIEVAL_LIMIT_PER_TASK)


class RetrieveSimilarInput(BaseModel):
    """以图搜图：用已有 image_id 的图像向量在同数据集内近邻。"""
    model_config = ConfigDict(extra="forbid")
    image_id: int
    dataset_id: Optional[int] = None
    limit: int = Field(default=20, ge=1, le=MAX_RETRIEVAL_LIMIT_PER_TASK)


class MetadataPredicate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: _ALLOWED_METADATA_FIELDS
    operator: _ALLOWED_OPERATORS
    value: Union[str, int, float, List[str]]


class FilterMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicates: List[MetadataPredicate] = Field(default_factory=list)
    include_fields: List[_ALLOWED_METADATA_FIELDS] = Field(default_factory=list)


class GroupImagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_by: _ALLOWED_GROUP_BY
    unknown_group_name: str = "未分类"


class AnalyzeVisualInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criteria: List[str] = Field(default_factory=list)
    max_candidates: int = Field(
        default=DEFAULT_VISION_CANDIDATES, ge=1, le=MAX_VISUAL_CANDIDATES
    )
    detail: Literal["low", "high"] = "low"


class RerankCandidatesInput(BaseModel):
    """对已有候选重排；不召回、不改 query 决策。query 可空，执行时回退上游 retrieve.query。"""
    model_config = ConfigDict(extra="forbid")
    query: str = ""
    provider: Literal["auto", "r1", "r2"] = "auto"


class DeduplicateImagesInput(BaseModel):
    """按 image id 去重。当前没有 pHash / 近重复检测。"""
    model_config = ConfigDict(extra="forbid")


class SortImagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sort_by: Literal["score", "created_at", "width", "height", "title"] = "score"
    order: Literal["asc", "desc"] = "desc"


class SelectImagesInput(BaseModel):
    """从剩余候选按数量策略取子集；不是 metadata 谓词过滤。"""
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["top_n", "per_group"] = "top_n"
    limit: int = Field(default=5, ge=1, le=MAX_SYNTHESIS_CANDIDATES)


class VerifyVisualInput(BaseModel):
    """对明确视觉条件做通过/不通过判断；不负责描述整张图。"""
    model_config = ConfigDict(extra="forbid")
    condition: str
    max_candidates: int = Field(
        default=DEFAULT_VISION_CANDIDATES, ge=1, le=MAX_VISUAL_CANDIDATES
    )
    detail: Literal["low", "high"] = "low"


class ReadVisibleTextInput(BaseModel):
    """只读画面上的字，不写场景长描述。"""
    model_config = ConfigDict(extra="forbid")
    max_candidates: int = Field(
        default=DEFAULT_VISION_CANDIDATES, ge=1, le=MAX_VISUAL_CANDIDATES
    )
    detail: Literal["low", "high"] = "low"


class ListObjectsInput(BaseModel):
    """只列可见物体，不写开放描述。"""
    model_config = ConfigDict(extra="forbid")
    max_candidates: int = Field(
        default=DEFAULT_VISION_CANDIDATES, ge=1, le=MAX_VISUAL_CANDIDATES
    )
    detail: Literal["low", "high"] = "low"


class CountNamedObjectInput(BaseModel):
    """只数查询点名的一种东西；结果是估计，不是检测器计数。"""
    model_config = ConfigDict(extra="forbid")
    object_name: str
    max_candidates: int = Field(
        default=DEFAULT_VISION_CANDIDATES, ge=1, le=MAX_VISUAL_CANDIDATES
    )
    detail: Literal["low", "high"] = "low"


class JudgeMainSubjectInput(BaseModel):
    """判断画面主体是不是指定类别；不通过的候选会被拿掉。"""
    model_config = ConfigDict(extra="forbid")
    subject: str
    max_candidates: int = Field(
        default=DEFAULT_VISION_CANDIDATES, ge=1, le=MAX_VISUAL_CANDIDATES
    )
    detail: Literal["low", "high"] = "low"


class VerifySpatialInput(BaseModel):
    """判断一句空间关系（如 A 在 B 上面）是否成立；不通过的候选会被拿掉。"""
    model_config = ConfigDict(extra="forbid")
    relation: str
    max_candidates: int = Field(
        default=DEFAULT_VISION_CANDIDATES, ge=1, le=MAX_VISUAL_CANDIDATES
    )
    detail: Literal["low", "high"] = "low"


class SynthesizeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recommendation_limit: int = Field(
        default=MAX_SYNTHESIS_CANDIDATES, ge=1, le=MAX_SYNTHESIS_CANDIDATES
    )
    presentation: Literal["flat", "grouped"] = "flat"


TaskInput = Union[
    RewriteQueryInput,
    ExpandQueryInput,
    ScopeDatasetInput,
    FilterByLabelsInput,
    RetrieveImagesInput,
    RetrieveSimilarInput,
    RerankCandidatesInput,
    FilterMetadataInput,
    DeduplicateImagesInput,
    GroupImagesInput,
    SortImagesInput,
    SelectImagesInput,
    AnalyzeVisualInput,
    VerifyVisualInput,
    ReadVisibleTextInput,
    ListObjectsInput,
    CountNamedObjectInput,
    JudgeMainSubjectInput,
    VerifySpatialInput,
    SynthesizeInput,
]

# task_type → 对应 Typed Input 模型（§6.4：input 类型必须与 task_type 一致）
INPUT_FOR_TASK_TYPE: Dict[TaskType, Type[BaseModel]] = {
    TaskType.REWRITE_QUERY: RewriteQueryInput,
    TaskType.EXPAND_QUERY: ExpandQueryInput,
    TaskType.SCOPE_DATASET: ScopeDatasetInput,
    TaskType.FILTER_BY_LABELS: FilterByLabelsInput,
    TaskType.RETRIEVE_IMAGES: RetrieveImagesInput,
    TaskType.RETRIEVE_SIMILAR: RetrieveSimilarInput,
    TaskType.RERANK_CANDIDATES: RerankCandidatesInput,
    TaskType.FILTER_METADATA: FilterMetadataInput,
    TaskType.DEDUPLICATE_IMAGES: DeduplicateImagesInput,
    TaskType.GROUP_IMAGES: GroupImagesInput,
    TaskType.SORT_IMAGES: SortImagesInput,
    TaskType.SELECT_IMAGES: SelectImagesInput,
    TaskType.ANALYZE_VISUAL: AnalyzeVisualInput,
    TaskType.VERIFY_VISUAL_CONDITION: VerifyVisualInput,
    TaskType.READ_VISIBLE_TEXT: ReadVisibleTextInput,
    TaskType.LIST_OBJECTS: ListObjectsInput,
    TaskType.COUNT_NAMED_OBJECT: CountNamedObjectInput,
    TaskType.JUDGE_MAIN_SUBJECT: JudgeMainSubjectInput,
    TaskType.VERIFY_SPATIAL: VerifySpatialInput,
    TaskType.SYNTHESIZE: SynthesizeInput,
}

# 当前已“注册”的 TaskType（对应 Phase 1-B 将实现的 Worker）。
# ANALYZE_VISUAL 依赖视觉能力；当前 VISION_MODEL 未配置，因此不注册。
REGISTERED_TASK_TYPES: set[TaskType] = {
    TaskType.REWRITE_QUERY,
    TaskType.EXPAND_QUERY,
    TaskType.SCOPE_DATASET,
    TaskType.FILTER_BY_LABELS,
    TaskType.RETRIEVE_IMAGES,
    TaskType.RETRIEVE_SIMILAR,
    TaskType.RERANK_CANDIDATES,
    TaskType.FILTER_METADATA,
    TaskType.DEDUPLICATE_IMAGES,
    TaskType.GROUP_IMAGES,
    TaskType.SORT_IMAGES,
    TaskType.SELECT_IMAGES,
    TaskType.SYNTHESIZE,
}

# 需要“能产生候选”的上游（用于 §7.2 规则 13）
CANDIDATE_PRODUCING_TYPES = {
    TaskType.RETRIEVE_IMAGES,
    TaskType.RETRIEVE_SIMILAR,
    TaskType.FILTER_BY_LABELS,
}
# 依赖候选结果的后处理类型（§7.2 规则 13）
CANDIDATE_CONSUMING_TYPES = {
    TaskType.RERANK_CANDIDATES,
    TaskType.FILTER_METADATA,
    TaskType.DEDUPLICATE_IMAGES,
    TaskType.GROUP_IMAGES,
    TaskType.SORT_IMAGES,
    TaskType.SELECT_IMAGES,
    TaskType.ANALYZE_VISUAL,
    TaskType.VERIFY_VISUAL_CONDITION,
    TaskType.READ_VISIBLE_TEXT,
    TaskType.LIST_OBJECTS,
    TaskType.COUNT_NAMED_OBJECT,
    TaskType.JUDGE_MAIN_SUBJECT,
    TaskType.VERIFY_SPATIAL,
}

# 依赖视觉模型才可执行的任务。关闭视觉时不注册。
VISION_GATED_TYPES = {
    TaskType.ANALYZE_VISUAL,
    TaskType.VERIFY_VISUAL_CONDITION,
    TaskType.READ_VISIBLE_TEXT,
    TaskType.LIST_OBJECTS,
    TaskType.COUNT_NAMED_OBJECT,
    TaskType.JUDGE_MAIN_SUBJECT,
    TaskType.VERIFY_SPATIAL,
}


# ---------------------------------------------------------------------------
# Draft 模型（Planner LLM 输出的原始结构，input 为原始 dict）
# ---------------------------------------------------------------------------
class TaskDraft(BaseModel):
    """Planner 输出的 Task 草稿。input 尚为原始 dict，等待 PlanValidator 类型化。"""
    model_config = ConfigDict(extra="forbid")
    task_id: str
    task_type: TaskType
    objective: str
    input: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)

    @field_validator("task_id")
    @classmethod
    def _check_task_id(cls, v: str) -> str:
        if not _TASK_ID_RE.match(v):
            raise ValueError(
                f"非法 task_id: {v!r}（必须匹配 ^[a-z][a-z0-9_]{{0,31}}$）"
            )
        return v


class TaskPlanDraft(BaseModel):
    """Planner 一次性输出的完整计划草稿。plan_id 由服务端生成，不信任 LLM。"""
    model_config = ConfigDict(extra="forbid")
    goal: str
    tasks: List[TaskDraft]
    synthesis_task_id: str


# ---------------------------------------------------------------------------
# 校验后不可变模型（input 已类型化）
# ---------------------------------------------------------------------------
class Task(BaseModel):
    """校验后的 Typed Task。校验通过后不可修改。"""
    model_config = ConfigDict(extra="forbid")
    task_id: str
    task_type: TaskType
    objective: str
    input: TaskInput
    depends_on: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING

    @field_validator("task_id")
    @classmethod
    def _check_task_id(cls, v: str) -> str:
        if not _TASK_ID_RE.match(v):
            raise ValueError(
                f"非法 task_id: {v!r}（必须匹配 ^[a-z][a-z0-9_]{{0,31}}$）"
            )
        return v

    @model_validator(mode="after")
    def _input_matches_task_type(self) -> "Task":
        expected = INPUT_FOR_TASK_TYPE[self.task_type]
        if not isinstance(self.input, expected):
            raise ValueError(
                f"task {self.task_id}: input 类型 {type(self.input).__name__} "
                f"与 task_type {self.task_type.value} 不匹配"
            )
        return self


class TaskPlan(BaseModel):
    """校验通过后的不可变 TaskPlan。plan_id 由服务端生成。"""
    model_config = ConfigDict(extra="forbid")
    plan_version: Literal["1.0"] = "1.0"
    plan_id: str
    goal: str
    tasks: List[Task]
    synthesis_task_id: str


class PlanValidationResult(BaseModel):
    """PlanValidator 输出。valid=True 时携带类型化后的 TaskPlan。"""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    plan: Optional[TaskPlan] = None


class PlannerFailure(Exception):
    """Planner 无法生成可解析、可通过结构校验的计划。"""


# ---------------------------------------------------------------------------
# 能力探测
# ---------------------------------------------------------------------------
def vision_available(config=None) -> bool:
    """判断当前视觉能力是否可用。

    当前 `VISION_MODEL` 为字符串 "null" 时返回 False，ANALYZE_VISUAL 不可用。
    """
    from ....config import settings
    from ....config.init_service import get_vision_client

    cfg = config if config is not None else settings.get_config()
    if cfg is None:
        return False
    vm = (getattr(cfg, "VISION_MODEL", None) or "").strip().lower()
    return bool(vm) and vm not in VISION_MODEL_UNAVAILABLE_VALUES and get_vision_client() is not None


def registered_task_types(vision_available_flag: Optional[bool] = None) -> set[TaskType]:
    """capability-aware 的「已注册 / 可执行」TaskType 集合（单一来源，Stage 4）。

    base = REGISTERED_TASK_TYPES（不含视觉任务）；
    仅当视觉能力可用（vision_available_flag=True）时加入
    ANALYZE_VISUAL 与 VERIFY_VISUAL_CONDITION。
    Planner enabled types / Validator registered types / default_workers 统一使用本函数，
    避免 capability drift。
    """
    types = set(REGISTERED_TASK_TYPES)
    flag = vision_available() if vision_available_flag is None else vision_available_flag
    if flag:
        types.update(VISION_GATED_TYPES)
    return types


# ---------------------------------------------------------------------------
# PlanValidator —— 在任何 Task 执行之前运行的确定性语义校验（§7）
# ---------------------------------------------------------------------------
class PlanValidator:
    """对 Planner 输出的 TaskPlanDraft 做结构 + 语义 + DAG 校验。

    校验失败绝不部分执行。校验通过后返回不可变 TaskPlan（input 已类型化、
    plan_id 由服务端生成）。
    """

    def __init__(
        self,
        max_tasks: int = MAX_TASKS,
        max_depth: int = MAX_PLAN_DEPTH,
        registered_types: Optional[set] = None,
    ):
        self.max_tasks = max_tasks
        self.max_depth = max_depth
        # None → validate() 时按 capability 动态解析（capability-aware 单一来源）
        self.registered_types = registered_types

    # -- 公共入口 ----------------------------------------------------------
    def validate(
        self,
        draft: TaskPlanDraft,
        vision_available_flag: Optional[bool] = None,
    ) -> PlanValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        plan: Optional[TaskPlan] = None

        # 1) 基础结构
        if draft is None or not draft.tasks:
            return PlanValidationResult(valid=False, errors=["EMPTY_PLAN: 计划不能为空"])

        # 2) Task 数量（§7.2-1）
        if len(draft.tasks) < 2:
            errors.append(f"MIN_TASKS: 至少需要 2 个 Task，当前 {len(draft.tasks)} 个")
        if len(draft.tasks) > self.max_tasks:
            errors.append(f"MAX_TASKS: Task 数量 {len(draft.tasks)} 超过上限 {self.max_tasks}")

        # 3) Task ID 唯一（格式已由 TaskDraft 校验）
        seen_ids: Dict[str, int] = {}
        for t in draft.tasks:
            seen_ids[t.task_id] = seen_ids.get(t.task_id, 0) + 1
        dup = [tid for tid, c in seen_ids.items() if c > 1]
        if dup:
            errors.append(f"DUPLICATE_TASK_ID: {dup}")

        # 5) capability：ANALYZE_VISUAL 需要视觉能力（§7.2-14）
        if vision_available_flag is None:
            vision_available_flag = vision_available()

        # 4) task_type 已注册（§7.2-3/14/15；capability-aware，与 Planner/default_workers 同源）
        if self.registered_types is None:
            registered = registered_task_types(vision_available_flag)
        else:
            registered = set(self.registered_types)
            if vision_available_flag:
                registered.update(VISION_GATED_TYPES)
        for t in draft.tasks:
            if t.task_type not in registered:
                errors.append(
                    f"WORKER_UNAVAILABLE: task {t.task_id} 的类型 {t.task_type.value} 当前未注册"
                )

        for t in draft.tasks:
            if t.task_type in VISION_GATED_TYPES and not vision_available_flag:
                errors.append(
                    f"VISION_UNAVAILABLE: task {t.task_id} 依赖视觉能力，但当前 VISION_MODEL 未启用"
                )

        # 6) 依赖引用存在、无自依赖（§7.2-5）
        id_set = set(seen_ids.keys())
        for t in draft.tasks:
            for dep in t.depends_on:
                if dep not in id_set:
                    errors.append(f"UNKNOWN_DEPENDENCY: task {t.task_id} 引用了不存在的依赖 {dep!r}")
                if dep == t.task_id:
                    errors.append(f"SELF_DEPENDENCY: task {t.task_id} 不能依赖自身")

        # 7) DAG：循环检测 + 拓扑序 + 最长路径深度（§7.2-6/7）
        order, cycle_nodes = self._kahn_order(draft.tasks)
        if cycle_nodes:
            errors.append(f"CYCLE_DETECTED: 依赖图中存在循环，涉及节点 {sorted(cycle_nodes)}")
        else:
            longest = self._longest_depth(order, draft.tasks)
            if longest > self.max_depth:
                errors.append(
                    f"MAX_PLAN_DEPTH: 最长依赖路径 {longest} 超过上限 {self.max_depth}"
                )

        # 8) 至少一个 RETRIEVE_IMAGES（§7.2-8）
        if not any(t.task_type == TaskType.RETRIEVE_IMAGES for t in draft.tasks):
            errors.append("NO_RETRIEVE: 计划必须至少包含一个 retrieve_images")

        # 9) 恰好一个 SYNTHESIZE + synthesis_task_id 指向它（§7.2-9/10）
        synth_tasks = [t for t in draft.tasks if t.task_type == TaskType.SYNTHESIZE]
        if len(synth_tasks) != 1:
            errors.append(f"SYNTHESIZE_COUNT: 必须且只能有一个 synthesize，当前 {len(synth_tasks)} 个")
        else:
            synth_id = synth_tasks[0].task_id
            if draft.synthesis_task_id != synth_id:
                errors.append(
                    f"SYNTHESIS_TARGET: synthesis_task_id={draft.synthesis_task_id!r} 与 synthesize task_id={synth_id!r} 不一致"
                )
            depended_by = [
                t.task_id
                for t in draft.tasks
                if t.task_type != TaskType.SYNTHESIZE and synth_id in t.depends_on
            ]
            if depended_by:
                errors.append(f"SYNTHESIZE_NOT_TERMINAL: synthesize 被 {depended_by} 依赖，必须是终点")

        # 10) 无孤儿：每个非 SYNTHESIZE Task 必须能到达 SYNTHESIZE（§7.2-12）
        if synth_tasks:
            synth_id = synth_tasks[0].task_id
            for t in draft.tasks:
                if t.task_type == TaskType.SYNTHESIZE:
                    continue
                if not self._can_reach(t.task_id, synth_id, draft.tasks):
                    errors.append(
                        f"ORPHAN_TASK: task {t.task_id} 无法到达 synthesize {synth_id}"
                    )

        # 11) 候选消费型 Task 必须（传递）依赖候选生产者（§7.2-13）
        for t in draft.tasks:
            if t.task_type in CANDIDATE_CONSUMING_TYPES:
                if not self._has_producer_ancestor(t, draft.tasks):
                    errors.append(
                        f"NO_PRODUCER: task {t.task_id} 必须（传递）依赖一个能产生候选的 Task"
                    )

        # 12) input 类型化：raw dict → 对应 Typed Input（§7.2-4）
        typed_tasks: List[Task] = []
        if not errors:
            for t in draft.tasks:
                typed_tasks.append(self._coerce_task(t, errors))

        if not errors:
            plan = TaskPlan(
                plan_version="1.0",
                plan_id=self._new_plan_id(),
                goal=draft.goal,
                tasks=typed_tasks,
                synthesis_task_id=draft.synthesis_task_id,
            )

        return PlanValidationResult(
            valid=not errors, errors=errors, warnings=warnings, plan=plan
        )

    # -- 内部工具 -----------------------------------------------------------
    @staticmethod
    def _new_plan_id() -> str:
        return "plan_" + uuid.uuid4().hex[:12]

    @staticmethod
    def _coerce_task(draft_task: TaskDraft, errors: List[str]) -> Task:
        """把 Draft Task 的原始 input dict 转为对应的 Typed Input 并构造 Task。"""
        model_cls = INPUT_FOR_TASK_TYPE[draft_task.task_type]
        try:
            typed_input = model_cls(**draft_task.input)
        except Exception as e:
            errors.append(
                f"INVALID_TASK_INPUT: task {draft_task.task_id} input 校验失败: {_short(e)}"
            )
            # 返回类型匹配的占位 Task（结果 valid=False，不会使用它）
            return Task(
                task_id=draft_task.task_id,
                task_type=draft_task.task_type,
                objective=draft_task.objective,
                input=_fallback_input(draft_task.task_type),
                depends_on=draft_task.depends_on,
            )
        return Task(
            task_id=draft_task.task_id,
            task_type=draft_task.task_type,
            objective=draft_task.objective,
            input=typed_input,
            depends_on=draft_task.depends_on,
        )

    @staticmethod
    def _kahn_order(tasks: List[TaskDraft]):
        """Kahn 拓扑排序。返回 (order, cycle_nodes)。存在循环时 cycle_nodes 非空。"""
        id_set = {t.task_id for t in tasks}
        deps = {t.task_id: [d for d in t.depends_on if d in id_set] for t in tasks}
        dependents: Dict[str, List[str]] = {tid: [] for tid in id_set}
        indeg = {tid: 0 for tid in id_set}
        for tid in id_set:
            for d in deps[tid]:
                indeg[tid] += 1
                dependents[d].append(tid)
        queue = [tid for tid in id_set if indeg[tid] == 0]
        order: List[str] = []
        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for j in dependents[tid]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    queue.append(j)
        if len(order) == len(id_set):
            return order, []
        return order, sorted(id_set - set(order))

    @staticmethod
    def _longest_depth(order: List[str], tasks: List[TaskDraft]) -> int:
        """基于拓扑序 DP 计算最长依赖链的节点数。"""
        dependents: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
        for t in tasks:
            for d in t.depends_on:
                if d in dependents:
                    dependents[d].append(t.task_id)
        depth = {tid: 1 for tid in order}
        for tid in order:
            for j in dependents.get(tid, []):
                depth[j] = max(depth[j], depth[tid] + 1)
        return max(depth.values()) if depth else 0

    @staticmethod
    def _can_reach(start: str, target: str, tasks: List[TaskDraft]) -> bool:
        """从 start 沿 dependents 边（task -> 依赖它的 task）BFS，能否到达 target。"""
        dependents: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
        for t in tasks:
            for d in t.depends_on:
                if d in dependents:
                    dependents[d].append(t.task_id)
        if start == target:
            return True
        visited = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in dependents.get(cur, []):
                if nxt == target:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        return False

    @staticmethod
    def _has_producer_ancestor(task: TaskDraft, tasks: List[TaskDraft]) -> bool:
        """沿 depends_on 反向 BFS，判断是否能（传递）到达候选生产者。"""
        by_id = {t.task_id: t for t in tasks}
        stack = list(task.depends_on)
        visited = set()
        while stack:
            dep = stack.pop()
            if dep in visited:
                continue
            visited.add(dep)
            t = by_id.get(dep)
            if t is None:
                continue
            if t.task_type in CANDIDATE_PRODUCING_TYPES:
                return True
            stack.extend(t.depends_on)
        return False


def _short(exc: Exception) -> str:
    msg = str(exc)
    return msg[:160] if msg else type(exc).__name__


def _fallback_input(task_type: TaskType) -> BaseModel:
    """为 input 校验失败的任务返回一个类型匹配的最小合法输入（占位用）。"""
    return {
        TaskType.RETRIEVE_IMAGES: RetrieveImagesInput(query=""),
        TaskType.RERANK_CANDIDATES: RerankCandidatesInput(),
        TaskType.FILTER_METADATA: FilterMetadataInput(),
        TaskType.DEDUPLICATE_IMAGES: DeduplicateImagesInput(),
        TaskType.GROUP_IMAGES: GroupImagesInput(group_by="location"),
        TaskType.SORT_IMAGES: SortImagesInput(),
        TaskType.SELECT_IMAGES: SelectImagesInput(),
        TaskType.REWRITE_QUERY: RewriteQueryInput(query=""),
        TaskType.EXPAND_QUERY: ExpandQueryInput(query=""),
        TaskType.SCOPE_DATASET: ScopeDatasetInput(),
        TaskType.FILTER_BY_LABELS: FilterByLabelsInput(),
        TaskType.RETRIEVE_SIMILAR: RetrieveSimilarInput(image_id=1),
        TaskType.ANALYZE_VISUAL: AnalyzeVisualInput(),
        TaskType.VERIFY_VISUAL_CONDITION: VerifyVisualInput(condition=""),
        TaskType.READ_VISIBLE_TEXT: ReadVisibleTextInput(),
        TaskType.LIST_OBJECTS: ListObjectsInput(),
        TaskType.COUNT_NAMED_OBJECT: CountNamedObjectInput(object_name=""),
        TaskType.JUDGE_MAIN_SUBJECT: JudgeMainSubjectInput(subject=""),
        TaskType.VERIFY_SPATIAL: VerifySpatialInput(relation=""),
        TaskType.SYNTHESIZE: SynthesizeInput(),
    }[task_type]
