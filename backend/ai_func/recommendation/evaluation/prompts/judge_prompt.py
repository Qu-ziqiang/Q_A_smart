"""Phase 4.2-B / 4.3 — Judge Prompt Builder。

输入 JudgeInput，输出精简 Judge Context：
  {query, goal, plan, agents, tools, execution, recovery, final}，
再包装为 system/user prompt。

原则：
  - 不 dump EvaluationTrace 全量（不出现 agent_messages / tool_events 原始列表）；
    只保留判定所需 bounded 摘要；
  - 控制 token：query/goal 截断、任务/工具/agent 列表限制条数；
  - JSON 渲染确定性。
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ..judge import JudgeInput
from ..rubric import criterion_keys, rubric_for_prompt

_QUERY_CAP = 200
_GOAL_CAP = 200
_PLAN_TASK_CAP = 12
_SELECTION_CAP = 12
_TOOL_CAP = 12


class JudgePromptBuilder:
    """把 JudgeInput 转成精简上下文 + 指令 prompt。"""

    # -- 上下文 ------------------------------------------------------------
    def build_context(self, judge_input: JudgeInput) -> Dict[str, Any]:
        tool_summary = judge_input.tool_summary or {}
        tools = {
            "counts": {
                "total": tool_summary.get("total", 0),
                "success": tool_summary.get("success", 0),
                "failure": tool_summary.get("failure", 0),
            },
            "names": (tool_summary.get("names") or [])[:_TOOL_CAP],
            "total_latency_ms": tool_summary.get("total_latency_ms", None),
        }
        statuses = dict(judge_input.execution_status or {})
        plan_tasks = (judge_input.plan_tasks or [])[:_PLAN_TASK_CAP]
        plan = {
            "tasks": [
                {"task_id": t.get("task_id"), "task_type": t.get("task_type")}
                for t in plan_tasks if isinstance(t, dict)
            ],
        }
        execution = {
            "plan_ok": judge_input.plan_ok,
            "task_statuses": statuses,
            "executed": judge_input.metrics.get("executed_tasks", len(statuses)),
            "succeeded": judge_input.metrics.get("succeeded", 0),
            "failed": judge_input.metrics.get("failed", 0),
            "skipped": judge_input.metrics.get("skipped", 0),
        }
        recovery = {
            "decisions": list(judge_input.decisions),
            "had_error": judge_input.error is not None,
            "error": (judge_input.error or "")[:_QUERY_CAP],
        }
        selection = {
            "count": len(judge_input.selection_ids),
            "ids": judge_input.selection_ids[:_SELECTION_CAP],
        }
        final_state = judge_input.final_state
        context: Dict[str, Any] = {
            "query": (judge_input.query or "")[:_QUERY_CAP],
            "goal": (getattr(judge_input, "goal", None) or "")[:_GOAL_CAP],
            "plan": plan,
            "agents": list(judge_input.agent_names)[:_TOOL_CAP],
            "tools": tools,
            "execution": execution,
            "recovery": recovery,
            "final": {"final_state": final_state, "selection_count": selection["count"],
                      "selection_ids": selection["ids"]},
        }
        # Phase 4.2-B 兼容别名（既有测试/调用方依赖这些 key）
        context["plan_status"] = execution
        context["final_state"] = final_state
        context["selection"] = selection
        return context

    # -- prompts -----------------------------------------------------------
    def build_system_prompt(self) -> str:
        keys = criterion_keys()
        return (
            "你的角色：你是 Agent System Evaluation Expert。\n"
            "任务：评价一个 Multi-Agent System（SmartImager 图片检索/推荐平台）的"
            "单次执行质量。\n"
            "约束：只依据给定的 Judge Context（结构化执行摘要）判定；"
            "禁止引入外部知识或猜测未提供的信息；禁止输出 JSON 以外的文字。\n\n"
            "## Judge Context 字段\n"
            "query / goal / plan / agents / tools / execution / recovery / final。\n"
            "上下文是经裁剪的摘要（非全量 trace），缺失信息一律视为未知，不据此扣分。\n\n"
            "## 评分维度（每项 0-5，5=优秀，0=极差）\n"
            + rubric_for_prompt()
            + "\n\n## 输出要求\n"
            "只输出一个 JSON 对象：\n"
            '{"verdict": "pass|flag|fail", '
            '"criterion_scores": {'
            + ", ".join('"' + k + '": <0-5>' for k in keys)
            + '}, "reasons": ["..."]}\n'
            "判定规则：存在致命错误（plan 失败 / abort / error）→ fail；\n"
            "出现 revise/replan/工具失败但已恢复、或部分目标未达成 → flag；\n"
            "执行符合目标 → pass。"
        )

    def build_user_prompt(self, judge_input: JudgeInput) -> str:
        context = self.build_context(judge_input)
        return (
            "请评估下面这次 Agent 执行的质量（仅依据该摘要，不要编造信息）：\n"
            + json.dumps(context, ensure_ascii=False)
        )

    def build_full_prompt(self, judge_input: JudgeInput) -> str:
        return self.build_system_prompt() + "\n\n" + self.build_user_prompt(judge_input)
