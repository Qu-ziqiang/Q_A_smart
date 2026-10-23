"""确定性 Complexity Gate（规划 §4）。

简单查询 → LEGACY（Legacy Fast Path）；高置信复合查询 → TEAM（Planner）。
- 不调用 LLM 判断复杂度。
- 默认保守：识别不出来一律 LEGACY。
- Gate 抛异常时也默认 LEGACY。
"""

from __future__ import annotations

import re
from typing import List, Optional, Union

from .models import GateDecision, RouteType

# ---------------------------------------------------------------------------
# 复杂特征（§4.4）
# ---------------------------------------------------------------------------

# 1) 分组 / 分类
_PATTERN_GROUP = [
    re.compile(r"按.{1,8}(分类|分组|归类|整理)"),
    re.compile(r"分成"),
    re.compile(r"归类"),
    re.compile(r"分别整理"),
]

# 3) 多集合任务
_PATTERN_MULTI_SET = [
    re.compile(r"(分别找|各找|每个.{0,4}各|各.{0,6}推荐|分别推荐)"),
    re.compile(r"[和与及、,，](.{1,8})各"),
]

# 4) 候选后处理
_PATTERN_POST_PROCESS = [
    re.compile(r"去重"),
    re.compile(r"按.{1,8}(时间|地点|年份|月份|类型|标签|方向|尺寸|风格).{0,8}(归组|分组|整理|归类|排序)"),
    re.compile(r"每组推荐"),
]

# 2) 多阶段连接词
_STAGE_CONNECTORS = ("先", "再", "然后", "最后", "并且", "接着")

# 检索类动词 × 处理类动词组合（“检索 + 处理”）
_SEARCH_VERBS = ("找", "搜", "查", "检索", "筛选", "选出", "找出", "推荐", "给我")
_PROCESS_VERBS = ("分类", "分组", "归类", "比较", "对比", "汇总", "总结", "整理", "排序", "归组")

# 视觉核验意图（进 TEAM，让 Planner 决定 analyze/verify；不等于无条件 Vision）
# 只抓「看像素才能回答」：确认/判断/比较 + 画面/看见/构图。
# 不抓「找红色花」「水墨风格」——那是语义召回，仍走 Legacy。
_PATTERN_VISUAL_VERIFY = [
    re.compile(r"画面(中|里|上)?(是否|有无|有没有|有)"),
    re.compile(r"(确认|判断|识别|核验|核实).{0,16}(画面|看见|可见|构图)"),
    re.compile(r"(是否能看见|能看见|清晰可见|确实同时有|确实有)"),
    re.compile(r"(再|然后)(比较|判断).{0,16}(构图|主色|对称)"),
    re.compile(r"构图是否"),
    re.compile(r"(再|然后)(确认|判断|识别|比较|核验)"),
]

_CLASSIFY_COMPLEX = "complex"
_CLASSIFY_SIMPLE = "simple"


class ComplexityGate:
    """简单 / 复杂请求路由。decide() 返回 GateDecision。"""

    def decide(
        self,
        messages: Optional[List[dict]] = None,
        query: Optional[str] = None,
        request_constraints: Optional[dict] = None,
    ) -> GateDecision:
        """根据最后一条 user 消息（或直接给定的 query）判定路由。

        request_constraints（filters/vector_targets/limit）当前不参与判定，
        仅为后续 Phase 预留；判定规则只基于文本。
        """
        try:
            text = self._extract_query(messages, query)
            if not text or not text.strip():
                return GateDecision(route=RouteType.LEGACY, reasons=["empty query"])
            return self._classify(text)
        except Exception as e:
            # Gate 自身异常绝不干扰主链路
            return GateDecision(
                route=RouteType.LEGACY, reasons=[f"gate error: {type(e).__name__}"]
            )

    # -- 内部 ---------------------------------------------------------------
    @staticmethod
    def _extract_query(messages: Optional[List[dict]], query: Optional[str]) -> str:
        if query is not None:
            return query
        if not messages:
            return ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return (msg.get("content") or "").strip()
        return ""

    def _classify(self, text: str) -> GateDecision:
        reasons: List[str] = []
        matched: List[str] = []

        # 规则 1：分组/分类强模式
        for pat in _PATTERN_GROUP:
            if pat.search(text):
                matched.append(f"group_pattern:{pat.pattern}")
        # 规则 3：多集合
        for pat in _PATTERN_MULTI_SET:
            if pat.search(text):
                matched.append(f"multi_set:{pat.pattern}")
        # 规则 4：候选后处理
        for pat in _PATTERN_POST_PROCESS:
            if pat.search(text):
                matched.append(f"post_process:{pat.pattern}")

        # 规则 2a：≥2 个阶段连接词（「然后再」算一个话语标记，不当两阶段）
        text_for_conn = text.replace("然后再", "然后")
        connector_hits = [c for c in _STAGE_CONNECTORS if c in text_for_conn]
        if len(connector_hits) >= 2:
            matched.append(f"stage_connectors:{connector_hits}")

        # 规则 2b：检索 + 处理 组合（「不要比较/判断」不计入处理动词）
        text_for_process = re.sub(
            r"(不要|别|无需|不用).{0,8}(比较|对比|判断|核验|识别|确认)",
            " ",
            text,
        )
        text_for_process = text_for_process.replace("对比度", "")
        has_search = any(v in text for v in _SEARCH_VERBS)
        has_process = any(v in text_for_process for v in _PROCESS_VERBS)
        if has_search and has_process:
            matched.append("search+process")

        # 规则 5：多分句（逗号/分号/句号切割 ≥3 个语义片段）
        clause_count = len(
            [c for c in re.split(r"[，。；、,;!?！？\n]+", text) if c.strip()]
        )
        if clause_count >= 3:
            matched.append(f"clause_count={clause_count}")

        for pat in _PATTERN_VISUAL_VERIFY:
            if pat.search(text):
                matched.append(f"visual_verify:{pat.pattern}")

        if matched:
            reasons.append("高置信复合特征命中")
            return GateDecision(
                route=RouteType.TEAM, reasons=reasons, matched_features=matched
            )

        # 未命中任何强特征 → 保守走 Legacy
        return GateDecision(route=RouteType.LEGACY, reasons=["未命中复合特征"])
