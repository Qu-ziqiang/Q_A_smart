"""Observe/Replan 测试辅助：第二轮见到 Observation 后改 retrieve query。"""

from __future__ import annotations

import copy
import json


def rewrite_aware_llm(plan_json, rewrite_suffix=" 改写"):
    def llm(messages):
        data = copy.deepcopy(plan_json)
        blob = "\n".join((m.get("content") or "") for m in messages)
        if "上一轮执行观察" in blob:
            for t in data.get("tasks") or []:
                if t.get("task_type") == "retrieve_images":
                    inp = t.setdefault("input", {})
                    q = (inp.get("query") or "").strip()
                    inp["query"] = (q + rewrite_suffix).strip()
        return json.dumps(data, ensure_ascii=False)

    return llm
