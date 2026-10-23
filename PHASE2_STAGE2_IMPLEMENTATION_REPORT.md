# PHASE2 Stage 2 — Harden ImageAnalysis Boundary 实现报告

> 阶段：Phase 2 — Vision Capability Enablement / **Stage 2 — Infrastructure Boundary Hardening**
> 结论：**PASS**（129/129 测试全绿，默认行为向后兼容，未进入 Stage 3）

---

## 1. 实际实现

### 1.1 审计结论（先审计后修改）

| 审计项 | 原状 | 证据 |
|---|---|---|
| 定义位置 | `backend/ai_func/image_analysis.py` `ImageAnalysis` 类 | 行 170-328 |
| 当前调用链 | `analyze_image_id.py:55` / `analyze_upload_image.py:32` / `routers/ai/analysis.py:15` / `routers/images.py:311`（auto_analyze） | 全部仅传 `local_image_path=` + `detail=` |
| client | 共享单例 `get_openai_client()`（openai 2.30.0，OpenAI 兼容） | `init_service.py:11-24` |
| 原 timeout | **无**：未传给 `chat.completions.create()`，仅 SDK 默认 600s | `image_analysis.py` 旧 236 行 |
| 原异常处理 | 单一 `except Exception` 吞成 error 字符串，无法区分 timeout/provider/parse | 旧 265-269 行 |
| 原 MIME | `Path(path).suffix[1:].lower()` 直接拼 `data:image/<ext>;base64`，**无白名单、无校验** | 旧 216-220 行 |
| parser | `extract_json_content()` 空文本/坏 JSON 返回 error 字典；**无大括号时隐式返回 `None`（漏洞）** | 旧 101-137 行 |

### 1.2 修改内容（最小增量）

**`backend/ai_func/image_analysis.py`（唯一修改的生产文件）**

1. **错误分类 + strict 边界**
   - 新增 `ImageAnalysisErrorCode(str, Enum)`：`client_unavailable` / `no_image_source` / `invalid_image` / `unsupported_mime` / `timeout` / `provider_error` / `parse_error` / `unknown_error`
   - 新增 `ImageAnalysisError(Exception)`：携带 `code` + `safe_message`
   - 新增 `_image_error()` 统一出口：`raise_on_error=True` 抛异常；`False`（默认）返回 error 字典
2. **Timeout**
   - `analyze_image(..., timeout: Optional[float] = None)`：非 None 时以 `create_kwargs["timeout"]` 真正传给 `chat.completions.create()`（SDK 2.x 原生支持 request-level timeout，已验证签名）
   - `None` → 不传 → 保持 SDK 默认（旧行为零变化）
   - **无自动 retry、无模型切换、无 provider 切换**
   - `openai.APITimeoutError` → `TIMEOUT`（在 `APIConnectionError` 之前捕获，因其为子类）
3. **MIME 归一化**
   - 白名单 `SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")`
   - `EXTENSION_TO_MIME`：`.jpg/.jpeg → image/jpeg`、`.png → image/png`、`.webp → image/webp`
   - `Path(path).suffix.lower()`（大小写不敏感）；白名单外 → 确定性 `UNSUPPORTED_MIME`（不发请求）
   - 文件不存在 → 确定性 `INVALID_IMAGE`（不发请求）
   - 只有白名单内才读取文件并 base64；`image_url` 直传路径不做改动
4. **Parser 兼容**
   - `extract_json_content()` **未重写**，仅补上缺失的 else 分支：无大括号时返回 `{"error": "未找到JSON内容", ...}` 而非隐式 `None`（原漏洞会导致 router 500）
   - strict 模式：parser 失败（error 字典）→ 抛 `PARSE_ERROR`
   - 默认模式：**原样返回 parser 的错误字典**（与旧行为逐字一致）
5. **新增参数签名**：`analyze_image(..., timeout=None, raise_on_error=False)`，旧调用（位置/关键字）完全不受影响

### 1.3 未修改（严格遵守范围）

`analyze_image_id.py`、`analyze_upload_image.py`、`routers/ai/analysis.py`、`routers/images.py`、`Legacy agent/tools`、`orchestration/*`、`config.yaml`、数据库 —— 全部零改动。

---

## 2. 测试证据

新增 `backend/tests/agent_team/test_image_analysis.py`（Fake/Mock OpenAI-compatible client，零真实网络调用），**27 项全部 PASS**：

| 类别 | 测试 | 结果 |
|---|---|---|
| success | local jpeg / png / webp / image_url 直传 | 4 PASS |
| timeout | 默认模式 error dict；strict 模式 `TIMEOUT`；边界真实传递（`timeout=7.5` 出现在 create kwargs） | 3 PASS |
| provider error | 默认模式 error dict；strict `PROVIDER_ERROR`（401 / 连接错误） | 3 PASS |
| malformed response | 带大括号坏 JSON → parser error；无大括号 → 非 None error 字典 | 3 PASS |
| unsupported MIME | 默认 error dict；strict `UNSUPPORTED_MIME`；**零 provider 调用** | 2 PASS |
| parser error | 默认原样返回 / strict `PARSE_ERROR` | 2 PASS |
| 大小写归一化 | `.JPG` → image/jpeg | 1 PASS |
| 无自动 retry | 失败时 `len(calls)==1` | 1 PASS |
| 兼容性 | 默认错误字典形状；`extract_json_content` 旧行为保留；无源参数 `ValueError`；client None 两种模式 | 6 PASS |
| 扩展名/无扩展名 | 白名单常量校验 | 2 PASS |

**全部回归：**

```
Phase 1-A/B/C  82/82 PASS（gate 13 / plan_validation 20 / planner 11 / workers 12 / executor 8 / orchestrator 6 / synthesizer 12）
Stage 1 Vision 20/20 PASS
Stage 2        27/27 PASS
────────────────────────
合计          129/129 PASS
```

---

## 3. Contract 说明

| Contract | 说明 |
|---|---|
| **timeout 配置/传递** | `analyze_image(timeout=<秒>)` → 直传 `chat.completions.create(timeout=...)`；`None` 保持 SDK 默认，不传参 |
| **strict error** | `raise_on_error=True` → 失败抛 `ImageAnalysisError(code, safe_message)`，8 种机器可读 code 可区分 |
| **MIME normalization** | 扩展名小写化 → 白名单映射（jpeg/png/webp）；仅本地文件路径生效 |
| **unsupported MIME 失败** | 白名单外扩展名/无扩展名 → `UNSUPPORTED_MIME`，**不读取文件、不发 provider 请求** |
| **parser error 区分** | strict → `PARSE_ERROR`；默认 → 原样返回 parser error 字典（与旧行为一致） |

## 4. Compatibility

- 旧调用 `analyze_image(local_image_path=..., detail=...)` 行为不变：成功返回 `{title, description, tags}`，失败返回 error 字典
- Legacy 路径（`analyze_image_id` / `analyze_upload_image` / 两个 router）**零改动、零语义变化**
- `images.py` auto_analyze 对 error 字典有 `analysis.get("error")` 防御，unsupported MIME 时仅跳过 DB 更新，不抛异常
- Stage 1 Vision contract（`vision.py`）未触碰
- `extract_json_content()` 各既有分支（空文本/坏 JSON/成功）逐字保留；仅补 no-braces fallback（消除 500 隐患）

## 5. Known Limitations

1. **MIME 来源 = 文件扩展名**：不做 magic bytes sniffing（避免多余文件读取）；扩展名与真实内容不符时由 provider 自行判定，本层不做二次校验。
2. **默认模式 timeout=None**：沿用 SDK 默认 600s 上限；显式 timeout 由调用方（如 VisionWorker）传入。
3. **`image_url` 直传不做 MIME 校验**：该路径不读取文件，MIME 由 provider/URL 决定（与旧行为一致）。
4. **异常分类覆盖 openai SDK 主流异常**（`APITimeoutError`/`APIConnectionError`/`APIStatusError`）；第三方 client 的异常会落到 `UNKNOWN_ERROR`。

## 6. 未实现内容（Non-goal 确认）

- [x] **VisionWorker 未实现**
- [x] **`ANALYZE_VISUAL` 未注册**
- [x] **Planner / Validator / Executor 未进入 Vision execution path**（未修改）
- [x] **未修改 DAG scheduling / Executor scheduling / Synthesizer contract / `validate_selection()`**
- [x] **未修改 frontend / router/SSE contract / database schema**
- [x] **未写入 metadata / 未 rebuild embedding**
- [x] **未修改 Legacy `agent.py` / `tools.py` / `search.py`**
- [x] **未调用真实 DeepSeek API**（Stage 0 已证明端点可用；本阶段全部 Fake/Mock）
- [x] **未打印/写入任何 API key**；测试构造 analyzer 时绕过 `get_openai_client()`，避免输出凭据片段
- [x] **未产生持久化测试图片**（tempfile + finally 清理）
- [x] **Stage 3 未开始**

## 7. Files Changed

| 文件 | 动作 | 原因 |
|---|---|---|
| `backend/ai_func/image_analysis.py` | 修改 | 增加 timeout 传递、strict error 分类、MIME 归一化、parser no-braces fallback |
| `backend/tests/agent_team/test_image_analysis.py` | 新增 | Stage 2 adapter 单测（27 项，Fake client） |

下一阶段：**Stage 3 — Implement VisionWorker**（等待指令，HARD STOP）。

