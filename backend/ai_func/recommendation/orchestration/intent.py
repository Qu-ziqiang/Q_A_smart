"""Lightweight Intent Rewrite（确定性规则层，Phase B）。

职责边界（只做规范化 + 能力信号抽取，不做任何执行决策）：
- 把用户自然语言规范化为 NormalizedIntent（raw + normalized + 能力标记）
- 复用 models.py 既有词表：TaskType / metadata field / group_by / operator
  （metadata_constraints 直接复用 MetadataPredicate 模型，字段/操作符由 Pydantic 强制校验）
- 不生成 DAG / 不做 task decomposition / 不决定 dependency / 不执行
- 不调用 LLM、不新增 Agent、不引入新能力定义

输出流向：User Query → IntentRewrite → NormalizedIntent → ComplexityGate → Planner
（normalized_query 供 Gate 判定与 Planner goal；能力标记供上层/未来 Prompt 消费）
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import MetadataPredicate, _ALLOWED_GROUP_BY


# ---------------------------------------------------------------------------
# 常量词表（均派生自既有能力语义；不复制新能力定义）
# ---------------------------------------------------------------------------
_GROUP_BY_VALUES = tuple(getattr(_ALLOWED_GROUP_BY, "__args__", ()))

_GROUP_ACTION_RE = re.compile(r"按(.{1,8}?)(?:分组|归组|分类|整理|归类)")

_YEAR_RE = re.compile(r"((?:19|20)\d{2})\s*年")
_WIDTH_RE = re.compile(r"宽度(?:至少|大于|超过|不低于|≥)?\s*(\d{3,5})")
_HEIGHT_RE = re.compile(r"高度(?:至少|大于|超过|不低于|≥)?\s*(\d{3,5})")
_ORIENTATION_WORDS = {
    "横向": "landscape", "横图": "landscape", "横版": "landscape", "横的": "landscape",
    "竖向": "portrait", "纵向": "portrait", "竖图": "portrait", "竖版": "portrait", "竖的": "portrait",
}
_FILE_TYPE_RE = re.compile(r"\b(jpg|jpeg|png|webp|gif)\b", re.IGNORECASE)
_TAGS_PATTERNS = [
    re.compile(r"标签包含[“”\"']?([^“”\"']{1,12})"),
    re.compile(r"已有标签(?:为|是|包含)[“”\"']?([^“”\"']{1,12})"),
    re.compile(r"标签(?:含|为|是)[“”\"']?([^“”\"']{1,12})"),
    re.compile(r"已有(.{1,6}?)标签"),
]
# 视觉核验信号（对应 planner.py TASK_CAPABILITY_CATALOG 中 ANALYZE_VISUAL 的 use_when 句式）
_VISION_TRIGGERS = (
    "确认", "判断", "识别", "核实", "检查", "比较", "对比", "核对",
    "能否看见", "能不能看见", "能看见", "看见", "看清", "清晰可见", "是否", "有没有",
    "画面中", "画面里", "画面有", "构图", "像素",
)
# greeting / 无关请求（用于 needs_images=False，避免简单请求误触发）
_GREETING_TOKENS = ("你好", "您好", "hi", "hello", "嗨", "哈喽", "在吗",
                    "早上好", "下午好", "晚上好", "谢谢", "感谢", "再见", "拜拜")
_IRRELEVANT_TOKENS = ("天气", "讲个笑话", "你是谁", "你会做什么", "股市", "新闻", "介绍", "自我介绍")
_SEARCH_NOUNS = ("找", "搜", "查", "推荐", "给我", "照片", "图片", "壁纸", "图库")

_CLAUSE_SPLIT_RE = re.compile(r"[，。；、,;!?！？\n]+")
_PROCESS_CONNECTORS = ("再", "然后", "接着", "并且", "并", "先", "按")
_COURTESY_LEAD_RE = re.compile(r"^(?:请|麻烦你|麻烦|帮我|帮帮我|帮我一下|我想要|我想|麻烦帮我)")
_COURTESY_TAIL_RE = re.compile(r"(?:谢谢|啦|哦|哟|哈|呢|呀|嘛)+[。.!！?？]?$")
# 数据集点名（规则层只抽 hint，真正解析在 scope_dataset）
_DATASET_HINT_PATTERNS = [
    re.compile(r"(?:在|从)\s*([A-Za-z0-9_\-]+|VOC(?:2010)?|COCO|默认)\s*(?:数据集|图库|库)?\s*(?:里|中|内)?", re.I),
    re.compile(r"([A-Za-z0-9_\-]+|VOC(?:2010)?|COCO)\s*数据集", re.I),
]


def _extract_dataset_hint(text: str) -> str:
    for pat in _DATASET_HINT_PATTERNS:
        m = pat.search(text or "")
        if m:
            return (m.group(1) or "").strip()
    return ""


def _clean_clause(text: str) -> str:
    t = _CLAUSE_SPLIT_RE.split(text)[0].strip(" ，。；、")
    return t.strip()


# ---------------------------------------------------------------------------
# NormalizedIntent —— 轻量结构化意图
# ---------------------------------------------------------------------------
class NormalizedIntent(BaseModel):
    """Intent Rewrite 的确定性输出。字段枚举/操作符复用 models.py，不新增能力定义。"""

    model_config = ConfigDict(extra="forbid")

    raw_query: str                                      # 原始用户文本（保留）
    normalized_query: str                               # 规范化目标文本（可直接作为 Planner goal）
    needs_images: bool = True                           # 是否需要图片检索
    retrieval_query: str = ""                           # 检索主条件文本（规范化后的首个检索分句）
    metadata_constraints: List[MetadataPredicate] = Field(default_factory=list)  # 复用既有 field/operator
    group_by: Optional[str] = None                      # 复用 _ALLOWED_GROUP_BY
    needs_visual_verification: bool = False             # 是否需要 analyze_visual（像素核验）
    visual_criteria: List[str] = Field(default_factory=list)  # 需核验的可见事实（短句，仅提示用）
    needs_recommendation: bool = True                   # 是否需要 synthesize 总结/推荐
    preference_hints: List[str] = Field(default_factory=list)  # 排序/偏好（预留）

    @model_validator(mode="after")
    def _check_vocab(self) -> "NormalizedIntent":
        if self.group_by is not None and self.group_by not in _GROUP_BY_VALUES:
            raise ValueError(f"group_by 超出既有词表: {self.group_by!r}")
        return self


# ---------------------------------------------------------------------------
# 确定性规则解析
# ---------------------------------------------------------------------------
def _normalize(query: str) -> str:
    """轻量规范化：去除首尾客气词/收尾语气词、规整空白。不做语义改写。"""
    t = (query or "").strip()
    t = _COURTESY_LEAD_RE.sub("", t)
    t = _COURTESY_TAIL_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[。，]+$", "", t)
    return t.strip()


def _looks_greeting_or_irrelevant(text: str) -> bool:
    lowered = text.lower()
    has_search_signal = any(k in text for k in _SEARCH_NOUNS)
    hit_greeting = any(k in lowered for k in _GREETING_TOKENS)
    hit_irrelevant = any(k in text for k in _IRRELEVANT_TOKENS)
    return (hit_greeting or hit_irrelevant) and not has_search_signal


def _detect_group(text: str) -> Optional[str]:
    """按 … 分组/归组/分类/整理/归类 → 映射既有 group_by 枚举。"""
    m = _GROUP_ACTION_RE.search(text)
    if not m:
        return None
    key = m.group(1)
    if "地点" in key:
        return "location"
    if "月份" in key or key == "月":
        return "month"
    if "年份" in key or key == "年":
        return "year"
    if "横" in key or "竖" in key or "方向" in key:
        return "orientation"
    if "文件类型" in key or "格式" in key or "类型" in key:
        return "file_type"
    if "标签" in key or "主题" in key:
        return "tag"
    return None


def _detect_metadata(text: str) -> List[MetadataPredicate]:
    """从文本提取 metadata 约束；跳过『按…分组』子句区（该区交给 group_by）。"""
    constraints: List[MetadataPredicate] = []
    excluded = [(m.start(), m.end()) for m in _GROUP_ACTION_RE.finditer(text)]

    def _in_excluded(pos: int) -> bool:
        return any(a <= pos < b for a, b in excluded)

    ym = _YEAR_RE.search(text)
    if ym:
        constraints.append(MetadataPredicate(field="created_at", operator="contains", value=ym.group(1)))
    for word, val in _ORIENTATION_WORDS.items():
        idx = text.find(word)
        if idx >= 0 and not _in_excluded(idx):
            constraints.append(MetadataPredicate(field="orientation", operator="eq", value=val))
    fmt = _FILE_TYPE_RE.search(text)
    if fmt:
        # 文件类型关键字不得出现在“按文件类型分组”区
        all_fmt = list(_FILE_TYPE_RE.finditer(text))
        if not any(_in_excluded(m2.start()) for m2 in all_fmt):
            constraints.append(MetadataPredicate(field="file_type", operator="eq", value=fmt.group(1).lower()))
    wm = _WIDTH_RE.search(text)
    if wm:
        constraints.append(MetadataPredicate(field="width", operator="gte", value=int(wm.group(1))))
    hm = _HEIGHT_RE.search(text)
    if hm:
        constraints.append(MetadataPredicate(field="height", operator="gte", value=int(hm.group(1))))
    for pat in _TAGS_PATTERNS:
        tm = pat.search(text)
        if tm:
            value = tm.group(1).strip().strip("“”\"'")
            if value:
                constraints.append(MetadataPredicate(field="tags", operator="contains", value=value))
                break
    return constraints


def _has_vision_signal(text: str) -> bool:
    return any(k in text for k in _VISION_TRIGGERS)


def _extract_visual_criteria(text: str) -> List[str]:
    """把含视觉触发词的分句截成短提示（≤28 字符），供 Planner 参考；非权威抽取。"""
    out: List[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(text):
        clause = clause.strip(" ，。；、再先然后并且并").strip()
        if clause and any(k in clause for k in _VISION_TRIGGERS):
            item = clause[:28]
            if item not in out:
                out.append(item)
        if len(out) >= 3:
            break
    return out


def _extract_retrieval_query(text: str) -> str:
    """首个检索分句：到第一个处理类连接词为止，去掉先/按等引导。

    若空前（如以『先按…筛选』开头），回退到完整规范化文本，保证始终有检索主句。"""
    idx = len(text)
    for conn in _PROCESS_CONNECTORS:
        pos = text.find(conn)
        if 0 < pos < idx:
            idx = pos
    head = text[:idx].strip(" ，。；、")
    head = re.sub(r"^先", "", head).strip()
    return head or text


# ---------------------------------------------------------------------------
# IntentRewrite —— 公开入口（确定性，幂等）
# ---------------------------------------------------------------------------
class IntentRewrite:
    """把自然语言 query 规范化为 NormalizedIntent。纯函数式规则，无状态、无 LLM。"""

    def rewrite(self, query: str) -> NormalizedIntent:
        raw = (query or "").strip()
        if not raw:
            return NormalizedIntent(raw_query=raw, normalized_query="", needs_images=False,
                                    retrieval_query="")
        normalized = _normalize(raw)
        needs_images = not _looks_greeting_or_irrelevant(raw)
        group_by = _detect_group(normalized) if needs_images else None
        metadata_constraints = _detect_metadata(normalized) if needs_images else []
        vision = _has_vision_signal(normalized) if needs_images else False
        criteria = _extract_visual_criteria(normalized) if vision else []
        retrieval_query = _extract_retrieval_query(normalized) if needs_images else ""
        return NormalizedIntent(
            raw_query=raw,
            normalized_query=normalized,
            needs_images=needs_images,
            retrieval_query=retrieval_query,
            metadata_constraints=metadata_constraints,
            group_by=group_by,
            needs_visual_verification=vision,
            visual_criteria=criteria,
            needs_recommendation=needs_images,
        )

    def rewrite_messages(self, messages: Optional[List[dict]]) -> Optional[NormalizedIntent]:
        """取最后一条 user 消息重写；无 user 消息返回 None。"""
        if not messages:
            return None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = (msg.get("content") or "").strip()
                return self.rewrite(content) if content else None
        return None


def prepare_retrieval(query: str, filters: Optional[dict] = None) -> dict:
    """检索前的规则：改写查询，并把能下推的元数据放进搜索过滤。

    规则：
    - 去掉客气话，检索用第一个处理步骤之前的那句，避免把「再分组」送进向量。
    - 年份、宽高、文件类型、标签写入搜索过滤；调用方已经写过的键不覆盖。
    - 横竖方向搜索接口没有字段，留给检索后按宽高筛。
    - 改写后没有查询词时不搜。
    """
    intent = IntentRewrite().rewrite(query or "")
    search_query = (intent.retrieval_query or intent.normalized_query or "").strip()
    merged = dict(filters or {})
    orientation = None
    if intent.needs_images:
        for pred in intent.metadata_constraints:
            if pred.field == "file_type" and not merged.get("file_type"):
                merged["file_type"] = str(pred.value).lower()
            elif pred.field == "width" and pred.operator == "gte" and "min_width" not in merged:
                merged["min_width"] = int(pred.value)
            elif pred.field == "height" and pred.operator == "gte" and "min_height" not in merged:
                merged["min_height"] = int(pred.value)
            elif pred.field == "tags" and pred.operator == "contains":
                tags = list(merged.get("tags") or [])
                value = str(pred.value).strip()
                if value and value not in tags:
                    tags.append(value)
                if tags:
                    merged["tags"] = tags
            elif pred.field == "created_at" and str(pred.value).isdigit() and len(str(pred.value)) == 4:
                year = str(pred.value)
                merged.setdefault("start_date", f"{year}-01-01")
                merged.setdefault("end_date", f"{year}-12-31")
            elif pred.field == "orientation" and pred.operator == "eq":
                orientation = str(pred.value)
    dataset_hint = _extract_dataset_hint(intent.raw_query or intent.normalized_query or "")
    return {
        "query": search_query,
        "filters": merged,
        "orientation": orientation,
        "needs_images": intent.needs_images,
        "dataset_hint": dataset_hint,
    }


