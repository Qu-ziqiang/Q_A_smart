# PLANNER_BENCHMARK_V2_REPORT（Planner / Routing Benchmark v2）

> 日期：2026-09-05 ｜ Evaluation only：**未修改任何生产代码 / Planner / Validator / Executor / Intent / Rerank / v1 gold**
> v1 = 冻结的 `routing_benchmark_28.py`（28 条，历史 56 runs 结果直接复用，未重跑、未改动）
> v2 = 新增 60 条 gold（Easy 15 / Medium 20 / Hard 15 / Very Hard 10）；本次真实运行 **60×2=120 次 Planner 调用**
> 模型/参数：deepseek-chat，temperature=0，其余 Planner 参数不变；capability snapshot `vision_available=True`（runtime patch）
> 数据集/分析/结果文件：`backend/tests/agent_team/planner_benchmark_v2.py`、`run_planner_benchmark_v2.py`、`analyze_planner_benchmark_v2.py`、根目录 `planner_benchmark_v2_results.json`

---

## 1. Motivation
v1 只回答“Planner 能否正确选择 Vision 能力”。v2 扩展为回答：
**“在复杂自然语言需求下，Planner 能否正确完成能力选择、任务拆解、Vision 判断、dependency 生成与 synthesis 组织？”**
v2 不是 Retrieval benchmark，不使用任何 CLIP/VOC/Rerank 相关标签。

## 2. Dataset Size & Integrity
- v2 共 **60 条**，id 1–60 连续；integrity PASS（数量/难度/字段/route/vision/dependency 链一致性）。
- 覆盖能力组合：R / R+F / R+G / R+V / R+F+G / R+F+V / R+G+V / R+F+G+V（+S 恒在），即规格列出的 1–8 类全部出现。

## 3. Difficulty Distribution
| 难度 | 条数 | gold 平均任务数 |
|---|---|---|
| easy | 15 | 2.33 |
| medium | 20 | 3.55 |
| hard | 15 | 3.73 |
| very_hard | 10 | 4.70 |
| 合计 | 60 | 3.48 |

## 4. Gold Schema（人工/规则定义，非 Planner 输出回填）
每条 query 含：`query / difficulty / expected_route / expected_task_types(有序=理想执行序) /
vision_required / expected_dependencies({task_type:[直接依赖类型]}) / expected_synthesis_task(=synthesize)`。
Dependency gold = 按自然语言业务顺序的 canonical 线性链（例：`retrieve → filter → group → vision → synthesize`）。
**Anti-circular**：gold 先于运行定义；测评允许把“语义等价但拓扑不同”的合法计划通过「expected-edges-all-present」与「dep-exact」分开呈现。

## 5. Evaluation Method
每次 run：`ComplexityGate.decide`（无 API）→ `PlannerAgent.create_plan`（1 次真实 LLM）→
`PlanValidator.validate(vision_available_flag=True)` → 离线结构计分。
指标：TaskType Exact / Vision P·R / Dependency accuracy&exact / Plan Exact / Valid / Avg tasks / Task Count Error / Consistency / 分层错误归因。**Valid ≠ 业务正确**，分列报告。

## 6. v1 Results（复用冻结结果，28 queries × 2 runs = 56）
| 指标 | v1 |
|---|---|
| TaskType Exact Match | 1.0000 |
| Valid Plan Rate | 0.8750（49/56） |
| Vision Recall / Precision | 1.0000 / 1.0000（24/24，FP=0） |
| Dependency Exact Match（vs 链 gold） | 0.2857（16/56） |
| Dependency All-Expected-Edges-Present | 1.0000（0 missing edges；不匹配全部来自 extra 边） |
| Plan Exact Match | 0.2857 |
| Plan Consistency Rate | 0.8929 |
| Avg Tasks per Plan | 3.14（gold 3.14），Task Count Error 0.0 |
| Gate | FN=9（team 可达 11/20），FP=0 |

## 7. v2 Results（60 × 2 runs = 120）
| 指标 | v2 | v1 | 说明 |
|---|---|---|---|
| TaskType Exact Match | **0.9917（119/120）** | 1.0 | 唯一失误：q46.run2 在纯 Vision 任务上多选 filter_metadata |
| Valid Plan Rate | **0.8250（99/120）** | 0.875 | 21 个 invalid 全部可归因（见 §17） |
| Vision Recall / Precision | **1.0000 / 1.0000** | 1.0/1.0 | 58/58 需 Vision run 全选；无需 Vision 全部未选；FP=0 |
| Dependency Exact Match（严格链） | **0.2667（32/120）** | 0.2857 | 见 §12 歧义说明 |
| Dependency All-Expected-Present | **0.7667（92/120）** | 1.0 | v2 首次出现“真实缺边”30 条（v1=0） |
| Plan Exact Match | **0.2667** | 0.2857 | 主要被 dependency-exact 拉低 |
| Avg Tasks per Plan | 3.50（gold 3.48） | 3.14 | Task Count Error 0.02 |
| Plan Consistency Rate | **0.7500（45/60）** | 0.8929 | 15/60 两次结构不一致 |
| Gate | FN=8（均为 R+V 类，q26–33）、FP=0 | FN=9 | Gate 仍存在 under-routing，但 FN 未随规模爆炸 |

## 8. TaskType Exact Match —— v1 vs v2
- v1：56/56=1.0；v2：119/120=0.9917。退化来源单一且可复现：
  q46“找办公室照片，确认画面里是否有白板，只保留有白板的候选再推荐”run2 额外插入了 `filter_metadata`（把“只保留有白板的候选”误当元数据过滤）。语义上可辩（保留=过滤动作），但 TaskType 集合不再等于 gold → 如实计错。
- 结论：任务复杂度提升后 TaskType 选择**基本稳健**（唯一错误出现在“视觉结果再做排除”这类可读作过滤的 Query）。

## 9. Vision Precision / Recall
- v2：recall 1.0（58/58 Vision-positive runs 全部选中 `analyze_visual`），precision 1.0（58 次选中全部是 Vision-positive），FP=0。
- 覆盖含 metadata/group 与像素核验混合的 29 条 Vision-positive Query ×2 runs。
- v1 为 24/24。两代均无 Vision 误判 → 能力路由的“元数据 vs 像素”判别在更复杂样本上保持稳定。

## 10. Plan-level：Dependency / Plan Exact / Valid / Consistency（分难度）
| 难度 | runs | TaskType Exact | Valid | Dep-Exact | Plan-Exact |
|---|---|---|---|---|---|
| easy | 30 | 1.0 | 1.0 | 0.8667 | 0.8667 |
| medium | 40 | 1.0 | 0.85 | 0.10 | 0.10 |
| hard | 30 | 0.9667 | 0.70 | 0.0667 | 0.0667 |
| very_hard | 20 | 1.0 | 0.70 | 0.0 | 0.0 |

一致性（查询级，两次 run 结构+valid 状态）：overall 45/60=0.75；不一致 query：19,22,23,30,35,37,38,39,46,47,48,53,55,56,57。

## 11. Dependency Accuracy（边级，宽容判据）
定义：`expected 每条边是否都出现在 actual 图里`（允许 extra 边）。
- v1：all-present = 1.0（56/56；缺边 0）。
- v2：all-present = **0.7667**（92/120；共缺 **30 条 expected 边**）。缺边高度集中在一种模式：
  gold 顺序为 `retrieve→group→analyze_visual→synthesize`（G→V），而 Planner 经常把 `analyze_visual` 挂在 `retrieve`（或 filter）上、在分组前对全候选核验 → 语义合法但与 canonical gold 的 `G→V` 边不同（q35/40/43/48/51/52/54/55/56 等）。

## 12. Dependency Exact Match 低分的两重含义（重要，勿误读）
1. **v1 也有 0.2857**：v1 缺边=0 而 exact 仅 0.2857 → v1 的不匹配**全部来自 extra 边**（Planner 倾向给 synthesize/vision 加直接依赖）。这意味着“exact vs canonical chain”判据对本系统偏严。
2. **v2 额外出现真缺边（30 条）**，集中在“先分组再核验 vs 先核验再分组”的合法替代顺序。
**结论**：Dep-Exact 不是“Planner 能力全面崩塌”的证据；真正的结构性退化信号是 **all-expected-present 1.0→0.7667 + missing 30**，以及非常难 Query 中 5 任务链与 `MAX_PLAN_DEPTH=4` 的冲突（见 §17）。

## 13. Plan Exact Match
= TaskType set 正确 ∧ Vision 正确 ∧ Dep-Exact ∧ Synthesis target 正确（不含 Validator 是否放行）。
- v1 0.2857 / v2 0.2667。v2 中 dep 是唯一显著失分维度（Task/Vision/Synthesis 几乎全对，除 q46.run2 一次多选）。

## 14. Valid Plan Rate
- v2 99/120=0.825（v1 0.875）。注意 **valid ≠ 业务正确**：v2 有 dep-exact=0 却 validator-valid 的计划（拓扑合法但非 canonical gold），也有“结构完美却被 Validator 以 input 契约拒绝”的计划（见 §17）。两者必须分开读。

## 15. Avg Tasks & Task Count Error
- v2 actual 3.50 vs gold 3.48；Task Count Error 平均 0.02（v1 0.0）。
- 唯一任务数偏差来源仍是 q46.run2（多 1 个 filter_metadata）。

## 16. Consistency Rate（双 run 稳定性）
- v1 25/28=0.8929；v2 45/60=**0.75**。复杂度升高后同一 Query 两次输出结构完全一致的比例下降 14pp；不一致集中在 medium/hard/very_hard 的多消费者 Query（依赖边表示变化，TaskType 变化少）。

## 17. Error Attribution（v2，120 runs）
按主因分层（互斥主因计数；invalid 子类单独细分）：
| 层 | run 数 | 说明 |
|---|---|---|
| 无错误（plan exact） | 32 | dep-exact 通过的计划 |
| Dependency Error | 87 | Task/Vision/Synthesis 正确但依赖图 != gold（多为 extra/等价重排，30 条真缺边见 §11） |
| Task Selection Error | 1 | q46.run2 多选 filter_metadata |
| Vision Decision Error | 0 | recall/precision 均 1.0 |
| Parse/Exception | 0 | 120/120 解析成功 |

Validator 拒绝 21 个 run（input contract / 结构）细分：
- **18** × `INVALID_TASK_INPUT: FilterMetadataInput.include_fields` 枚举外字段（q19r2,23r2,32,33,37r2,38r2,42,44,46r2,47r2,51,55r2 等）→ **Planner input 契约忠实度**，与路由无关；
- **2** × `SYNTHESIS_TARGET` 自引用漂移（q48r2、q56r2）→ 结构性小缺陷；
- **2** × `MAX_PLAN_DEPTH: 最长依赖路径 5 > 4`（q58 两次）→ **canonical 5 任务链与系统深度上限的冲突**（Planner 对同类 5 任务 Query 选择了分支化合法图，如 q59 两次均 valid）。
Gate：FN=8（q26–33，全部为 “找 X 照片 + 像素核验”的 R+V Query）、FP=0 → **Gate under-routing 仍是生产路径瓶颈，Planner 自身 vision 路由无 FN/FP**。

## 18. Failure Cases（最有信息量的失败）
1. q46.run2：视觉“只保留”被解释为元数据过滤（TaskType 多选）。
2. R+G+V+S 类（q35/40/43/48/56 等）：Planner 常“先核验后分组”，与 gold“先分组后核验”不一致 → dep 缺边（语义等价的替代计划，暴露 gold 顺序假设）。
3. filter_metadata include_fields 契约类（18 runs invalid）：Planner 会用 gold 之外的字段名（如 location 类自定义名），被 Pydantic 拒绝 —— input 契约忠实度问题。
4. q58：5 任务线性链深度 5 超 `MAX_PLAN_DEPTH=4`（Validator 正确拒绝；暴露 benchmark gold 链设计需与系统深度上限对齐）。
5. 15 个不一致 Query：结构/依赖表示在两 run 间漂移，但 TaskType/Vision 保持稳定。

## 19. Conclusions
1. **能力路由（要什么任务类型、要不要 Vision）在 v2 上依旧强**：TaskType 0.9917、Vision recall/precision=1.0、FP=0、0 parse error。
2. **真正的边界在 dependency 生成与稳定性**：all-expected-present 0.7667、dep-exact 0.2667、consistency 0.75；随难度升高 dep/valid/consistency 单调下滑（easy→very_hard）。
3. **错误分层有效**：复杂任务主要卡在 Dependency（87/120 有差异）、Input Contract（18 invalid）、以及 5 任务链 vs 深度上限（2 invalid）——而不是 Vision 路由。
4. 任务复杂度上升后 Planner 确实出现可测量的性能下降（v1→v2：dep-all 1.0→0.7667、valid 0.875→0.825、consistency 0.893→0.75），这是**正常且有价值的边界证据**，不是为了让数字更好看。
5. 建议后续（不在本轮）：把 dep gold 从“单一 canonical 链”升级为“可达规范边集合 + 等价拓扑判定”，以区分“语义等价替代计划”与真正的依赖错误；并评估把 5 任务计划的 MAX_PLAN_DEPTH 语义与 planner 分解策略对齐。

## 20. Resume-ready metrics（本轮最有价值的 5 个）
| # | 指标 | 数值 | Query | 真实 run | 含义 | 为什么比 28-Query 更有价值 | 是否适合简历 |
|---|---|---|---|---|---|---|---|
| 1 | Vision Recall / Precision | 1.0 / 1.0（FP=0） | 29 条 Vision-positive（v2） | 58 | 像素 vs 元数据需求判别零误判 | 在含 metadata+group 的更难 Query 上仍保持，能抗追问 | ✅ 首选 |
| 2 | TaskType Exact Match | 0.9917（119/120） | 60（v2） | 120 | 任务集合与 gold 一致率 | 从 28→60 且含 hard/very_hard，仍近乎满分；唯一失败可解释 | ✅ |
| 3 | Dependency Edge Completeness | 0.7667（92/120 all-expected-present；缺边 30） | 60 | 120 | 期望依赖边全部命中比例 | 暴露“先分组 vs 先核验”等价顺序歧义与真缺边，是 dep 评测独有指标 | ✅（需解释判据） |
| 4 | Plan Consistency Rate | 0.75（45/60） | 60 | 120（2×） | 同 Query 两次结构完全一致率 | Agent 特有的稳定性指标；v1 0.89→v2 0.75 展示复杂度代价 | ✅ |
| 5 | Valid-Plan Rate + 分层归因 | 0.825（99/120）；invalid=18 input 契约+2 synth-target+2 depth | 60 | 120 | Validator 放行率与错误分层 | 能证明“会定位失败在哪一层”，而不是只说准确率 | ✅ |

> 诚实备注：不推荐单独引用 Dependency Exact Match=0.2667（判据过严且含等价拓扑歧义）；如需引用请用 #3 或“exact 0.27 且 all-expected 0.77，差异主要在合法顺序替代”的完整表述。

---

### Files
- dataset：`backend/tests/agent_team/planner_benchmark_v2.py`
- runner：`backend/tests/agent_team/run_planner_benchmark_v2.py`
- analyzer：`backend/tests/agent_team/analyze_planner_benchmark_v2.py`
- results：`planner_benchmark_v2_results.json`（120 runs + 60 gate rows）
- 本报告：`PLANNER_BENCHMARK_V2_REPORT.md`
- v1 frozen 保留不变：`routing_benchmark_28.py` + `planner_routing_benchmark_results.json`

