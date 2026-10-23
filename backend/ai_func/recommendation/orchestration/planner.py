"""一次性 Planner（规划 §5）。

职责：把复杂用户请求转换为 Typed TaskPlanDraft。
- 只生成计划，绝不执行任何 Tool / 数据库 / 图像分析。
- 强制 submit_task_plan function schema；输出仍必须经过 Pydantic 解析 + PlanValidator。
- 失败一律抛 PlannerFailure，由调用层决定回退 Legacy Fast Path。
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable, Dict, List, Optional

from .models import (
    MAX_TASKS,
    MAX_PLAN_DEPTH,
    PLANNER_TIMEOUT_SECONDS,
    PlannerFailure,
    TaskPlanDraft,
    TaskType,
    registered_task_types,
    vision_available,
)
from ....config.init_service import get_openai_client
from ....config import settings

# ---------------------------------------------------------------------------
# submit_task_plan 的 JSON Schema（来自 TaskPlanDraft 的受限结构）
# ---------------------------------------------------------------------------
TASK_PLAN_DRAFT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "description": "用户目标的简写（≤120字）"},
        "synthesis_task_id": {"type": "string", "description": "必须指向计划中唯一的 synthesize task 的 task_id"},
        "tasks": {
            "type": "array",
            "minItems": 2,
            "maxItems": MAX_TASKS,
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "唯一且小写开头的 ID，如 t1"},
                    "task_type": {"type": "string", "enum": [t.value for t in TaskType]},
                    "objective": {"type": "string", "description": "该任务的目标说明"},
                    "input": {
                        "type": "object",
                        "description": "随 task_type 变化的参数，必须严格符合约束：rewrite_query: {query}; expand_query: {query, dataset_id?, previous_query?}; scope_dataset: {dataset_id?|dataset_name?}; filter_by_labels: {labels, match:all|any, dataset_id?, limit≤40}; retrieve_images: {query, vector_targets, filters, limit≤40}; retrieve_similar: {image_id, dataset_id?, limit≤40}; rerank_candidates: {query?, provider:auto|r1|r2}; filter_metadata: {predicates, include_fields}; deduplicate_images: {}; group_images: {group_by, unknown_group_name}; sort_images: {sort_by:score|created_at|width|height|title, order:asc|desc}; select_images: {strategy:top_n|per_group, limit≤20}; analyze_visual: {criteria, max_candidates≤8, detail}; verify_visual_condition: {condition, max_candidates≤8, detail}; read_visible_text: {max_candidates≤8, detail}; list_objects: {max_candidates≤8, detail}; count_named_object: {object_name, max_candidates≤8, detail}; judge_main_subject: {subject, max_candidates≤8, detail}; verify_spatial: {relation, max_candidates≤8, detail}; synthesize: {recommendation_limit≤20, presentation}",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "只能引用本计划内已存在的 task_id",
                    },
                },
                "required": ["task_id", "task_type", "objective", "input", "depends_on"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["goal", "tasks", "synthesis_task_id"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Capability Catalog（Planner 能力语义单一来源，审计 F-002b/c 最小修复）
# 每个 TaskType：purpose / evidence_source / use_when / avoid_when /
#                dependency_rule / cost / input_contract
# 同时驱动：系统提示词能力语义段 + tool schema 的 input 描述（避免两份语义漂移）。
# ---------------------------------------------------------------------------
TASK_CAPABILITY_CATALOG: Dict[str, Dict[str, Any]] = {
    TaskType.REWRITE_QUERY.value: {
        "purpose": "规则改写：去客气话、截检索句、抽出年份/宽高/格式/标签/横竖/数据集 hint。不搜图、不调 LLM。",
        "evidence_source": "确定性规则，不调用大模型。",
        "use_when": [
            "原句带客气话，或含「再/然后/按」等后处理",
            "原句里有年份、横竖、文件格式、宽度或已有标签",
        ],
        "avoid_when": [
            "纯「找几张猫/日落」且没有上述条件",
            "需要近义/上位扩写（用 expand_query，且仅检索不好时）",
            "已经有改写结果，不要再加第二次",
        ],
        "dependency_rule": "无前置依赖。retrieve_images 应依赖本任务以使用改写结果。",
        "cost": "低：本地规则，不访问模型。",
        "input_contract": "query(必填, 用户原句或当前检索句)",
    },
    TaskType.EXPAND_QUERY.value: {
        "purpose": "LLM 语义扩写：近义/上位（如男孩→人），让 CLIP 更好召回。不搜图。",
        "evidence_source": "聊天模型一次短调用；有数据集词表时尽量落在词表内。",
        "use_when": [
            "上一轮召回为空或明显不好（Supervisor RETRY_REWRITE）",
            "口语词与数据集类名不对齐，需要上位/近义",
        ],
        "avoid_when": [
            "简单「找几张猫」首次计划——不要默认加",
            "只需要去客气话/下推元数据——用 rewrite_query",
        ],
        "dependency_rule": "无强制前置；retrieve_images 应依赖本任务消费扩写句。可依赖 scope_dataset 以约束词表。",
        "cost": "中：一次 LLM 短调用。",
        "input_contract": "query(必填) · dataset_id(可选) · previous_query(可选)",
    },
    TaskType.SCOPE_DATASET.value: {
        "purpose": "把后续检索钉在某个数据集（如 voc2010），不搜图。",
        "evidence_source": "datasets 表；未知名失败并列出可选。",
        "use_when": [
            "用户点名数据集/图库域（在 VOC 里、COCO 等）",
            "类别筛选或相似图必须限定域时",
        ],
        "avoid_when": [
            "未点名数据集且全局搜即可",
        ],
        "dependency_rule": "无前置。retrieve_images / filter_by_labels / retrieve_similar 应依赖本任务。",
        "cost": "低：查库。",
        "input_contract": "dataset_id(可选) · dataset_name(可选，二者至少一个)",
    },
    TaskType.FILTER_BY_LABELS.value: {
        "purpose": "COCO 式精确类别筛选；多类默认 AND（熊+橱柜须同时有）。可独立产生候选。",
        "evidence_source": "image_labels 精确表，不是 tags LIKE。",
        "use_when": [
            "按数据集类别勾选/点名筛选",
            "纯类别浏览，可不经 CLIP",
        ],
        "avoid_when": [
            "开放语义「像厨房的」——用 retrieve_images",
            "空间关系/主成分——用视觉任务",
        ],
        "dependency_rule": "候选生产者。建议依赖 scope_dataset。",
        "cost": "低：SQL。",
        "input_contract": "labels(必填[]) · match(all|any, 默认 all) · dataset_id(可选) · limit(1..40)",
    },
    TaskType.RETRIEVE_IMAGES.value: {
        "purpose": "向量语义召回：按文本查询从图片索引召回候选（id/score/title 等）。",
        "evidence_source": "向量索引 + pgvector（CLIP 文本-图像相似度）；不读取像素细节。",
        "use_when": [
            "任何需要候选图片的任务起点",
            "按主题/风格/语义相似度找图",
        ],
        "avoid_when": [
            "不需要新候选的纯整理/汇总阶段",
            "要求对已有候选做像素级核验（应使用 analyze_visual）",
            "只要精确类别 —— 用 filter_by_labels",
            "已有图 id 找相似 —— 用 retrieve_similar",
        ],
        "dependency_rule": "候选生产者；可依赖 rewrite/expand/scope。",
        "cost": "低：本地 CLIP + pgvector；limit ≤ 40。",
        "input_contract": "query(必填, 搜索文本) · vector_targets([title|description|image]) · filters(对象, 可选) · limit(1..40)",
    },
    TaskType.RETRIEVE_SIMILAR.value: {
        "purpose": "以图搜图：用已有 image_id 的图像向量在同数据集内找近邻。",
        "evidence_source": "image_vectors 近邻；不调视觉大模型。",
        "use_when": [
            "用户给出库内图片 id，要求找相似图",
        ],
        "avoid_when": [
            "文本语义检索 —— 用 retrieve_images",
            "像素级「是否在上面」—— 用视觉核验",
        ],
        "dependency_rule": "候选生产者。建议依赖 scope_dataset。",
        "cost": "低：向量近邻。",
        "input_contract": "image_id(必填) · dataset_id(可选) · limit(1..40)",
    },
    TaskType.RERANK_CANDIDATES.value: {
        "purpose": "对已有候选重新排序，不增加、不删除、不重新召回。",
        "evidence_source": "看图重排用 DeepSeek deepseek-flash，看原图后重排，失败退回公式。GLM 不负责重排。",
        "use_when": [
            "用户明确要求重排、再排一下、或抱怨排序不准",
            "同一句里先召回再比较谁更相关",
        ],
        "avoid_when": [
            "还没有候选（必须先 retrieve_images）",
            "普通「找/推荐几张」——retrieve 内部已有排序，不要默认再加本任务",
            "需要改查询时用 rewrite_query，不要把改写塞进本任务",
        ],
        "dependency_rule": "候选消费型：必须（传递）依赖 retrieve_images。",
        "cost": "低（R1）或中（R2，失败回退 R1）。",
        "input_contract": "query(可选, 空则用上游 retrieve.query) · provider(auto|r1|r2)",
    },
    TaskType.FILTER_METADATA.value: {
        "purpose": "基于已有结构化元数据（创建时间/宽高/方向/文件类型/标签/地点）过滤候选。",
        "evidence_source": "数据库已有字段；读不到图片像素。",
        "use_when": [
            "用户按已有元数据约束筛选（年份/方向/格式/宽度/已有标签等）",
            "在视觉核验之前用便宜条件缩小候选集",
        ],
        "avoid_when": [
            "用户要求确认画面中可见的颜色/物体/场景/构图（应使用 analyze_visual）",
            "元数据字段为空或未提供时，不要把它当作事实条件",
        ],
        "dependency_rule": "候选消费型：必须（传递）依赖候选生产者。",
        "cost": "低：确定性过滤，毫秒级。",
        "input_contract": "predicates[{field: created_at|width|height|orientation|file_type|tags|location, operator: eq|in|contains|gte|lte, value}] · include_fields[]",
    },
    TaskType.DEDUPLICATE_IMAGES.value: {
        "purpose": "按 image id 去掉重复候选。当前不是近重复/感知哈希检测。",
        "evidence_source": "候选 id 集合。",
        "use_when": [
            "用户明确要求去重",
            "计划里有两路以上 retrieve（同一 id 可能重复）",
        ],
        "avoid_when": [
            "单路 retrieve 的普通检索——不要默认去重",
            "想做视觉近重复（当前没有 pHash 实现，不要规划成近重复）",
        ],
        "dependency_rule": "候选消费型：必须（传递）依赖 retrieve_images。",
        "cost": "低：确定性 id 去重。",
        "input_contract": "无额外字段（input 可为 {}）",
    },
    TaskType.GROUP_IMAGES.value: {
        "purpose": "按已有结构化字段（地点/年份/月份/方向/文件类型/标签）对候选做确定性分组。",
        "evidence_source": "数据库已有字段（metadata/created_at/width/height/file_type/tags）。",
        "use_when": [
            "用户要求按某字段分组/分类/整理",
        ],
        "avoid_when": [
            "分组字段依赖像素内容（画面场景/颜色等），应先 analyze_visual 再分组",
        ],
        "dependency_rule": "候选消费型：必须（传递）依赖候选生产者。",
        "cost": "低：确定性分组，毫秒级。",
        "input_contract": "group_by(location|year|month|orientation|file_type|tag) · unknown_group_name(可选, 默认 未分类)",
    },
    TaskType.SORT_IMAGES.value: {
        "purpose": "按明确字段对已有候选排序（score/时间/宽高/标题），不检索。",
        "evidence_source": "候选已有字段。",
        "use_when": [
            "用户要求按时间、分数或尺寸排序",
        ],
        "avoid_when": [
            "只是要相关性重排（应使用 rerank_candidates）",
        ],
        "dependency_rule": "候选消费型：必须（传递）依赖 retrieve_images。",
        "cost": "低：确定性排序。",
        "input_contract": "sort_by(score|created_at|width|height|title) · order(asc|desc)",
    },
    TaskType.SELECT_IMAGES.value: {
        "purpose": "从剩余候选按数量策略取最终子集（top_n 或每组 N 张）。不是谓词过滤。",
        "evidence_source": "上游候选或 GroupOutput。",
        "use_when": [
            "每组各选几张（必须同时有 group_images）",
            "用户明确说只要前 N 张/每组 N 张，且 N 不是 synthesize 能单独完成的默认截取",
        ],
        "avoid_when": [
            "普通「推荐几张」——用 synthesize.recommendation_limit，不要默认 select",
            "按年份/标签等条件删除候选（应使用 filter_metadata）",
        ],
        "dependency_rule": "候选消费型：必须（传递）依赖 retrieve_images；per_group 应依赖 group_images。",
        "cost": "低：确定性截取。",
        "input_contract": "strategy(top_n|per_group) · limit(1..20)",
    },
    TaskType.ANALYZE_VISUAL.value: {
        "purpose": "从候选图片的像素中产生新的视觉观察证据（title/description/tags/evidence/confidence）。",
        "evidence_source": "图片像素，经 VisionWorker → ImageAnalysis → 视觉模型。",
        "use_when": [
            "用户要求确认/核实/判断画面中是否可见某物体、颜色、场景、构图",
            "需要从像素验证或比较候选的视觉事实",
        ],
        "avoid_when": [
            "只是按主题/语义召回图片（retrieve_images 可满足）",
            "条件可通过已有元数据确定（年份/方向/格式/已有标签）",
            "尚未提供候选生产者（analyze_visual 不能凭空创建候选）",
        ],
        "dependency_rule": "候选消费型：必须（传递）依赖候选生产者（retrieve_images）；synthesize 应同时直接依赖候选生产者以取得 base candidates。",
        "cost": "昂贵（高）：默认 4 张、绝对 8 张、串行、每张约 2.5–4s；仅在必要时使用。",
        "input_contract": "criteria[](视觉核验关注点, 可空) · max_candidates(1..8, 默认 4) · detail(low|high, 默认 low)",
    },
    TaskType.VERIFY_VISUAL_CONDITION.value: {
        "purpose": "判断候选是否满足一句明确的视觉条件（通过/不通过），不做开放描述。",
        "evidence_source": "图片像素 + 条件句；输出 passed/confidence/evidence。",
        "use_when": [
            "用户要求确认画面中是否有某物体/场景",
            "需要 pass/fail 而不是标题描述",
        ],
        "avoid_when": [
            "只需开放视觉描述（应使用 analyze_visual）",
            "视觉能力关闭时不得使用",
        ],
        "dependency_rule": "候选消费型：必须（传递）依赖 retrieve_images。",
        "cost": "昂贵：串行视觉调用，默认 4 张、最多 8 张。",
        "input_contract": "condition(必填, 视觉条件) · max_candidates(1..8) · detail(low|high)",
    },
    TaskType.READ_VISIBLE_TEXT.value: {
        "purpose": "只读取候选图上可见的文字，不写场景描述，不删除候选。",
        "evidence_source": "图片像素，视觉模型。",
        "use_when": ["用户问图里写了什么字、店名、路牌、截图文字"],
        "avoid_when": ["需要整张图的开放描述（用 analyze_visual）", "视觉能力关闭时不得使用"],
        "dependency_rule": "候选消费型：必须依赖候选生产者。不缩小候选集。",
        "cost": "昂贵：串行，默认 4 张、最多 8 张。",
        "input_contract": "max_candidates(1..8) · detail(low|high)",
    },
    TaskType.LIST_OBJECTS.value: {
        "purpose": "只列出候选图里可见的物体名称，不写长描述，不删除候选。",
        "evidence_source": "图片像素，视觉模型。",
        "use_when": ["用户问画面里有什么东西"],
        "avoid_when": ["只要开放描述（用 analyze_visual）", "要按数据集类别精确筛（用 filter_by_labels）"],
        "dependency_rule": "候选消费型：必须依赖候选生产者。不缩小候选集。",
        "cost": "昂贵：串行，默认 4 张、最多 8 张。",
        "input_contract": "max_candidates(1..8) · detail(low|high)",
    },
    TaskType.COUNT_NAMED_OBJECT.value: {
        "purpose": "只估计一种点名物体的数量。结果是估计，不删除候选。",
        "evidence_source": "图片像素，视觉模型；不是检测器计数。",
        "use_when": ["用户问有几只、几个某种东西"],
        "avoid_when": ["要精确检测框计数", "开放描述用 analyze_visual"],
        "dependency_rule": "候选消费型：必须依赖候选生产者。不缩小候选集。",
        "cost": "昂贵：串行，默认 4 张、最多 8 张。",
        "input_contract": "object_name(必填) · max_candidates(1..8) · detail(low|high)",
    },
    TaskType.JUDGE_MAIN_SUBJECT.value: {
        "purpose": "判断画面主体是不是指定类别。不通过的候选会被拿掉。",
        "evidence_source": "图片像素，视觉模型。角落里的小物体不算主体。",
        "use_when": ["用户要求主体是某类，或主要成分是某物"],
        "avoid_when": ["只要标签含该类（用 filter_by_labels）", "开放描述用 analyze_visual"],
        "dependency_rule": "候选消费型：必须依赖候选生产者。会缩小候选集。",
        "cost": "昂贵：串行，默认 4 张、最多 8 张。",
        "input_contract": "subject(必填) · max_candidates(1..8) · detail(low|high)",
    },
    TaskType.VERIFY_SPATIAL.value: {
        "purpose": "判断一句空间关系是否在画面里成立，例如 A 在 B 上面。不通过的候选会被拿掉。",
        "evidence_source": "图片像素，视觉模型。CLIP 检索不能当空间硬条件。",
        "use_when": ["用户明确说在上面、在左边、在前面这类位置关系"],
        "avoid_when": ["普通找某类图（用检索或类别筛选）", "开放描述用 analyze_visual"],
        "dependency_rule": "候选消费型：必须依赖候选生产者。会缩小候选集。",
        "cost": "昂贵：串行，默认 4 张、最多 8 张。",
        "input_contract": "relation(必填) · max_candidates(1..8) · detail(low|high)",
    },
    TaskType.SYNTHESIZE.value: {
        "purpose": "汇总各任务结果，生成最终自然语言回答并挑选推荐图片。",
        "evidence_source": "其他任务的结构化输出（candidates/groups/vision evidence/failures）。",
        "use_when": [
            "每个计划必须有且只有一个 synthesize 作为终点",
        ],
        "avoid_when": [],
        "dependency_rule": "必须且只能一个；必须是终点（不被其他任务依赖）。",
        "cost": "中：LLM 合成一次调用。",
        "input_contract": "recommendation_limit(1..20) · presentation(flat|grouped, 默认 flat)",
    },
}
# 能力路由策略：按 Gate 分级注入，不整体放宽。
# 简单（Legacy）→ 最小必要 + 禁止默认套餐。
# 复杂（TEAM）→ 完整目录政策，必须覆盖用户点名的整理/核验，禁止空计划。
_CAPABILITY_ROUTING_POLICY_SIMPLE = (
    "1. 能力选择遵循“最小必要”：能用便宜的检索/元数据满足就不调用昂贵能力。\n"
    "2. filter_metadata 只能使用数据库已有结构化字段；它读不到图片像素。\n"
    "3. 颜色、物体、场景、构图等可见事实必须由 analyze_visual 从像素验证；不要映射为 filter_metadata。\n"
    "4. analyze_visual 只能消费已有候选，不能凭空创建候选；使用前必须存在 retrieve_images（或经其传递的候选）。\n"
    "5. analyze_visual 昂贵（默认 4 张、最多 8 张、串行）；仅当用户明确要求视觉核验时使用，"
    "不要因为出现“视觉/颜色/图片”一词就无条件加入。\n"
    "6. synthesize 必须能取得 base candidates：若计划含 analyze_visual，synthesize 应同时依赖 "
    "retrieve/group 等候选生产者。\n"
    "7. rerank_candidates 只重排已有候选，不增加召回；需要新候选时用 retrieve_images。\n"
    "8. filter_metadata 按谓词删除；select_images 按数量/策略取子集；二者不得互换。\n"
    "9. verify_visual_condition 输出通过/不通过；开放描述用 analyze_visual。\n"
    "10. deduplicate_images 仅按 image id 去重，不是近重复检测。\n"
    "11. 禁止默认套餐：用户没说重排/去重/只要前 N/每组 N 时，"
    "不要把 rerank_candidates、deduplicate_images、select_images 叠在 retrieve 后面。\n"
    "12. 「找几张日落/猫/夜景」的最小计划是 retrieve_images → synthesize。\n"
    "13. 原句有客气话、后处理连接词，或年份/横竖/格式/宽度/已有标签时，"
    "先 rewrite_query，再让 retrieve_images 依赖它。不要把改写写进 retrieve_images。\n"
    "14. expand_query 不要放进首次简单计划；仅当观察要求改写检索句（召回空/差）时使用。\n"
    "15. 点名数据集时先 scope_dataset；纯类别勾选用 filter_by_labels（默认 AND）。\n"
    "16. 读字用 read_visible_text，列物体用 list_objects，数一种东西用 count_named_object，"
    "主体用 judge_main_subject，位置关系用 verify_spatial。不要把这些都塞进 analyze_visual。\n"
)

_CAPABILITY_ROUTING_POLICY_COMPLEX = (
    "1. 能力选择遵循目录语义：用户点名的整理/核验步骤必须出现在计划里，"
    "不要因为便宜就省略。\n"
    "2. filter_metadata 只能使用数据库已有结构化字段；它读不到图片像素。\n"
    "3. 颜色、物体、场景、构图等可见事实必须由 analyze_visual 从像素验证；不要映射为 filter_metadata。\n"
    "4. analyze_visual 只能消费已有候选，不能凭空创建候选；使用前必须存在 retrieve_images（或经其传递的候选）。\n"
    "5. analyze_visual 昂贵（默认 4 张、最多 8 张、串行）；仅当用户明确要求视觉核验时使用，"
    "不要因为出现“视觉/颜色/图片”一词就无条件加入。\n"
    "6. synthesize 必须能取得 base candidates：若计划含 analyze_visual，synthesize 应同时依赖 "
    "retrieve/group 等候选生产者。\n"
    "7. rerank_candidates 只重排已有候选，不增加召回；需要新候选时用 retrieve_images。\n"
    "8. filter_metadata 按谓词删除；select_images 按数量/策略取子集；二者不得互换。\n"
    "9. verify_visual_condition 输出通过/不通过；开放描述用 analyze_visual。\n"
    "10. deduplicate_images 仅按 image id 去重，不是近重复检测。\n"
    "11. 复合目标必须产出可执行计划：至少 retrieve_images（或 filter_by_labels / retrieve_similar）+ synthesize，"
    "并覆盖用户明确要求的分组/过滤/排序/视觉核验；禁止空计划或只有 synthesize。\n"
    "12. 不要把「最小必要」理解成可以交白卷；用户已经点名的步骤不得省略。\n"
    "13. 点名数据集时先 scope_dataset；召回空/差时用 expand_query，不要默认塞进首次计划。\n"
)

# 兼容旧 import：默认视为简单政策。
_CAPABILITY_ROUTING_POLICY = _CAPABILITY_ROUTING_POLICY_SIMPLE

# 标准 DAG 模式（bounded）
_CAPABILITY_DAG_EXAMPLES = (
    "rewrite_query → retrieve_images → synthesize\n"
    "expand_query → retrieve_images → synthesize\n"
    "scope_dataset → filter_by_labels → synthesize\n"
    "scope_dataset → retrieve_images → synthesize\n"
    "scope_dataset → retrieve_similar → synthesize\n"
    "retrieve_images → synthesize\n"
    "retrieve_images → rerank_candidates → synthesize\n"
    "retrieve_images → filter_metadata → deduplicate_images → synthesize\n"
    "retrieve_images → group_images → select_images → synthesize\n"
    "retrieve_images → filter_metadata → group_images → select_images → synthesize\n"
    "retrieve_images → {group_images, analyze_visual} → synthesize\n"
    "retrieve_images → verify_visual_condition → synthesize\n"
    "retrieve_images → read_visible_text → synthesize\n"
    "retrieve_images → judge_main_subject → synthesize\n"
    "retrieve_images → verify_spatial → synthesize\n"
)

# 正反例（4 个紧凑示例 + 2 条一句话反例；不含 benchmark 查询的完全相同措辞）
_CAPABILITY_FEW_SHOT = (
    "例1 (仅检索，不需要 Vision)：「找几张海边日落照片」→ retrieve_images → synthesize\n"
    "例1b (先改写再检索)：「请帮我找横向 JPG 猫的照片」→ rewrite_query → retrieve_images → synthesize\n"
    "例1c (数据集+类别)：「在 VOC 里找同时有 bear 和 chair 的图」→ "
    "scope_dataset → filter_by_labels → synthesize\n"
    "例1d (召回差再扩写)：观察要求改写时 → expand_query → retrieve_images → synthesize\n"
    "例2 (仅元数据)：「找 2025 年上传的照片」→ retrieve_images → filter_metadata → synthesize\n"
    "例3 (检索 + 视觉核验)：「找派对照片，再确认画面中是否有生日蛋糕」→ "
    "retrieve_images → analyze_visual → synthesize\n"
    "例4 (元数据 + 视觉混合)：「找 2025 年照片，再确认画面中有人骑马」→ "
    "retrieve_images → filter_metadata → analyze_visual → synthesize\n"
    "例5 (分组后选择)：「按地点分组，每组选 2 张」→ "
    "retrieve_images → group_images → select_images → synthesize\n"
    "例6 (条件核验)：「找图并确认画面是否有自行车」→ "
    "retrieve_images → verify_visual_condition → synthesize\n"
    "例7 (读字)：「这些图里写了什么字」→ retrieve_images → read_visible_text → synthesize\n"
    "例8 (主体/位置)：「主体是熊，而且熊在桌子上」→ "
    "retrieve_images → judge_main_subject → verify_spatial → synthesize\n"
    "反例：「找红色花朵照片」不自动需要 Vision，语义召回可先满足；"
    "「确认候选中花朵是否确实为红色」才需要 Vision。\n"
    "反例：「找几张海边日落照片」不要加 rerank/dedup/select，retrieve → synthesize 即可。\n"
)


def _render_capability_semantics(enabled_task_types: List[str]) -> str:
    """把 catalog 渲染为 bounded 文本（仅 enabled 类型）。"""
    lines: List[str] = []
    for t in enabled_task_types:
        spec = TASK_CAPABILITY_CATALOG.get(t)
        if spec is None:
            continue
        lines.append(f"- {t}: {spec['purpose']}")
        lines.append(f"  证据来源: {spec['evidence_source']}")
        lines.append(f"  适用: {'；'.join(spec['use_when'])}")
        lines.append(f"  避免: {'；'.join(spec['avoid_when']) if spec['avoid_when'] else '无'}")
        lines.append(f"  依赖规则: {spec['dependency_rule']}")
        lines.append(f"  成本: {spec['cost']}")
    return "\n".join(lines)


def _render_input_contracts(enabled_task_types: List[str]) -> str:
    """把 catalog 的 input_contract 渲染为 bounded 文本（仅 enabled 类型）。"""
    lines: List[str] = []
    for t in enabled_task_types:
        spec = TASK_CAPABILITY_CATALOG.get(t)
        if spec and spec.get("input_contract"):
            lines.append(f"- {t}: {spec['input_contract']}")
    return "\n".join(lines)


def routing_mode_for_query(goal: str) -> str:
    """Gate TEAM → complex（完整目录政策）；其余 → simple（最小必要）。"""
    from .gate import ComplexityGate
    from .models import RouteType

    route = ComplexityGate().decide(query=goal or "").route
    if route == RouteType.TEAM:
        return "complex"
    return "simple"


def routing_policy_for_mode(mode: str) -> str:
    if (mode or "").strip().lower() == "complex":
        return _CAPABILITY_ROUTING_POLICY_COMPLEX
    return _CAPABILITY_ROUTING_POLICY_SIMPLE


def _build_task_plan_schema(enabled_task_types: List[str]) -> Dict[str, Any]:
    """基于模板深拷贝生成 per-call schema：task_type.enum 收窄 + 富化描述。

    - 不原地修改模块级 global schema；
    - Vision 关闭时 enum 不含 analyze_visual；
    - input/depends_on 描述与 catalog 同源。
    """
    schema = copy.deepcopy(TASK_PLAN_DRAFT_SCHEMA)
    items = schema["properties"]["tasks"]["items"]
    task_type_schema = items["properties"]["task_type"]
    task_type_schema["enum"] = list(enabled_task_types)
    task_type_schema["description"] = "选择完成该目标所需的最小必要能力；只能使用已启用任务类型。"
    items["properties"]["input"]["description"] = _render_input_contracts(enabled_task_types)
    items["properties"]["depends_on"]["description"] = (
        "只能引用本计划内已存在的 task_id；候选消费型任务 "
        "（rerank/filter/dedup/group/sort/select/analyze_visual/verify_visual_condition）"
        "必须（传递）依赖候选生产者（retrieve_images）；synthesize 是终点，不得被其他任务依赖。"
    )
    return schema


class PlannerAgent:
    """一次性 Planner。llm 可注入（测试用 Fake）；默认使用 OpenAI 兼容客户端。"""

    def __init__(
        self,
        llm: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        model: Optional[str] = None,
        client: Any = None,
        timeout: int = PLANNER_TIMEOUT_SECONDS,
    ):
        self._llm = llm
        self._model = model
        self._client = client
        self._timeout = timeout

    # -- 公共入口 ----------------------------------------------------------
    def create_plan(
        self,
        goal: str,
        recent_messages: Optional[List[Dict[str, str]]] = None,
        memory_hints: Optional[List[str]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        *,
        observation: Any = None,
        repair_errors: Optional[List[str]] = None,
    ) -> TaskPlanDraft:
        """生成并返回 TaskPlanDraft。任何失败都抛 PlannerFailure。

        memory_hints：bounded episodic hints（Top-N ≤ 3，由 Memory Service 渲染）。
        memory_hints=None 时行为与现有版本完全一致（capability routing 不受影响）。
        observation：仅第二轮 Replan 传入；None 时 user prompt 与第一轮完全一致。
        Planner 只根据 Observation 改策略，绝不执行检索 / 视觉 / 业务工具。
        """
        if not goal or not goal.strip():
            raise PlannerFailure("empty goal")

        capabilities = dict(capabilities or self._default_capabilities())
        if not capabilities.get("routing_mode"):
            capabilities["routing_mode"] = routing_mode_for_query(goal)
        system_prompt = self._build_system_prompt(capabilities)
        user_prompt = self._build_user_prompt(
            goal, recent_messages, memory_hints,
            observation=observation, repair_errors=repair_errors,
        )

        try:
            raw = self._call_llm(system_prompt, user_prompt, capabilities)
        except PlannerFailure:
            raise
        except Exception as e:
            raise PlannerFailure(f"LLM 调用失败: {_short(e)}")

        return self._parse_draft(raw)

    # -- LLM 调用 ----------------------------------------------------------
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self._llm is not None:
            return self._llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

        client = self._client or get_openai_client()
        if not client:
            raise PlannerFailure("OpenAI 客户端未配置，无法进行规划")

        config = settings.get_config()
        model = self._model or (getattr(config, "CHAT_MODEL", None) or "deepseek-chat")

        # per-call capability-filtered schema：task_type.enum 收窄到本次 enabled 列表
        enabled = (capabilities or {}).get("enabled_task_types") or [t.value for t in TaskType]
        schema = _build_task_plan_schema(enabled)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            stream=False,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "submit_task_plan",
                        "description": "提交最终的计划。必须且只能调用一次。",
                        "parameters": schema,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "submit_task_plan"}},
            timeout=self._timeout,
        )

        message = response.choices[0].message
        # 优先读取 submit_task_plan 的 arguments
        for tc in message.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is not None and getattr(fn, "name", None) == "submit_task_plan":
                return getattr(fn, "arguments", None) or ""
        # 兜底：部分 provider 忽略 tool_choice 直接返回 content（仍走 Pydantic 校验）
        if message.content:
            return message.content
        raise PlannerFailure("LLM 未返回 submit_task_plan 调用，也未返回 content")

    # -- 输出解析 ----------------------------------------------------------
    def _parse_draft(self, raw: str) -> TaskPlanDraft:
        text = _strip_fences(raw).strip()
        if not text:
            raise PlannerFailure("空输出")
        try:
            data = json.loads(text)
        except Exception as e:
            raise PlannerFailure(f"计划 JSON 解析失败: {_short(e)}")
        if not isinstance(data, dict):
            raise PlannerFailure("计划输出不是 JSON 对象")
        try:
            return TaskPlanDraft.model_validate(data)
        except Exception as e:
            raise PlannerFailure(f"计划结构校验失败: {_short(e)}")

    # -- Prompt 构建 -------------------------------------------------------
    @staticmethod
    def _default_capabilities() -> Dict[str, Any]:
        # capability-aware：vision 可用时 ANALYZE_VISUAL 才进入 enabled 列表
        enabled = [t for t in TaskType if t in registered_task_types()]
        enabled_values = [t.value for t in enabled]
        return {
            "enabled_task_types": enabled_values,
            "vision_available": vision_available(),
            "max_tasks": MAX_TASKS,
            "max_plan_depth": MAX_PLAN_DEPTH,
            # 仅 enabled 能力的 catalog 条目（能力语义单一来源）
            "task_capabilities": [
                {**TASK_CAPABILITY_CATALOG[t.value], "task_type": t.value}
                for t in enabled if t.value in TASK_CAPABILITY_CATALOG
            ],
        }

    @staticmethod
    def _build_system_prompt(capabilities: Dict[str, Any]) -> str:
        from ..prompts import PLANNER_SYSTEM_PROMPT_TEMPLATE

        enabled = capabilities.get("enabled_task_types") or []
        vision_ok = bool(capabilities.get("vision_available"))
        max_tasks = capabilities.get("max_tasks", MAX_TASKS)
        max_depth = capabilities.get("max_plan_depth", MAX_PLAN_DEPTH)
        vision_note = (
            "可用，可以使用 analyze_visual / verify_visual_condition"
            if vision_ok
            else "不可用，不得使用 analyze_visual / verify_visual_condition"
        )
        return PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
            enabled=enabled,
            max_tasks=max_tasks,
            max_plan_depth=max_depth,
            vision_note=vision_note,
            capability_semantics=_render_capability_semantics(enabled),
            input_contracts=_render_input_contracts(enabled),
            routing_policy=routing_policy_for_mode(
                str(capabilities.get("routing_mode") or "simple")
            ),
            dag_examples=_CAPABILITY_DAG_EXAMPLES,
            few_shot=_CAPABILITY_FEW_SHOT,
        )

    @staticmethod
    def _build_user_prompt(
        goal: str,
        recent_messages: Optional[List[Dict[str, str]]],
        memory_hints: Optional[List[str]] = None,
        observation: Any = None,
        repair_errors: Optional[List[str]] = None,
    ) -> str:
        parts = [f"用户目标：{goal}"]
        if recent_messages:
            hist = []
            for m in recent_messages[-4:]:
                role = m.get("role", "user")
                content = (m.get("content") or "")[:200]
                if content:
                    hist.append(f"{role}: {content}")
            if hist:
                parts.append("最近对话：\n" + "\n".join(hist))
        if memory_hints:
            # 独立 bounded 段落；current user query > current execution context > memory
            bounded = [h[:280] for h in memory_hints][:4]
            block = (
                "相关历史任务摘要（含可延续的筛选条件与会话画像，仅供跨轮策略参考；"
                "当前用户请求优先级最高，若与历史记忆冲突一律以当前目标为准，"
                "历史记忆不得覆盖当前请求的 TaskType 选择）：\n"
                + "\n".join(f"- {h}" for h in bounded)
            )
            parts.append(block)
        if repair_errors:
            reasons = "；".join(str(e)[:160] for e in repair_errors if e)[:400]
            parts.append(
                "上次计划不可用（空计划或无可执行 Task）。原因："
                f"{reasons or '未知'}。"
                "请重新提交可执行计划：至少包含 retrieve_images 与 synthesize，"
                "并覆盖用户已点名的步骤。"
            )
        obs_text = _render_observation(observation)
        if obs_text:
            parts.append(obs_text)
        return "\n\n".join(parts)


def _render_observation(observation: Any) -> str:
    """把结构化 Observation 渲染进第二轮 user prompt；None 时返回空串。"""
    if observation is None:
        return ""
    render = getattr(observation, "render_for_planner", None)
    if callable(render):
        return str(render() or "")
    if isinstance(observation, dict):
        try:
            from .observation import RoundObservation

            return RoundObservation.model_validate(observation).render_for_planner()
        except Exception:
            return ""
    return ""


def _strip_fences(text: str) -> str:
    """去除 ```json ... ``` 围栏。"""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    return m.group(1) if m else text


def _short(exc: Exception) -> str:
    msg = str(exc)
    return msg[:160] if msg else type(exc).__name__

