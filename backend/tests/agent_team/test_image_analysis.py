"""Phase 2 Stage 2 — ImageAnalysis Boundary Hardening 单元测试。

覆盖：success / timeout / provider error / malformed response / unsupported MIME /
parser error / default-legacy-compatible / strict mode / JPEG / PNG / WebP /
大小写归一化 / 无自动 retry / timeout 边界真实传递。

运行（repo 根目录）：
    venv\\Scripts\\python.exe backend/tests/agent_team/test_image_analysis.py
"""

import base64
import json
import os
import sys
import tempfile
import traceback

import httpx
import openai

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from backend.ai_func.image_analysis import (
    EXTENSION_TO_MIME,
    SUPPORTED_IMAGE_EXTENSIONS,
    ImageAnalysis,
    ImageAnalysisError,
    ImageAnalysisErrorCode,
    extract_json_content,
)


# ---------------------------------------------------------------------------
# Fake / Mock OpenAI-compatible client（不触发任何真实网络调用）
# ---------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def _timeout_exc():
    return openai.APITimeoutError(
        request=httpx.Request("POST", "https://example.com/chat/completions")
    )


def _auth_exc():
    req = httpx.Request("POST", "https://example.com/chat/completions")
    return openai.AuthenticationError(
        message="invalid api key",
        response=httpx.Response(401, request=req),
        body=None,
    )


def _conn_exc():
    return openai.APIConnectionError(
        message="connection refused",
        request=httpx.Request("POST", "https://example.com/chat/completions"),
    )


def _valid_content():
    return json.dumps(
        {"title": "红圆", "description": "白底上红圆与蓝三角", "tags": ["红圆", "蓝三角"]}
    )


def _make_analyzer(fake_client, prompt="PROMPT"):
    """构造 analyzer，不触发 get_openai_client()，避免打印任何凭据。"""
    inst = object.__new__(ImageAnalysis)
    inst.client = fake_client
    inst._prompt = prompt
    return inst


def _temp_image(suffix=".jpg", content=b"fake-image-bytes"):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()
    return tmp.name
def test_success_local_jpeg():
    path = _temp_image(".jpg")
    try:
        comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
        analyzer = _make_analyzer(_FakeClient(comp))
        result = analyzer.analyze_image(local_image_path=path)
        assert result.get("title") and result.get("description")
        url = comp.calls[0]["messages"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        assert "timeout" not in comp.calls[0]  # None → 不传递，保持 SDK 默认
    finally:
        os.unlink(path)


def test_success_local_png():
    path = _temp_image(".png")
    try:
        comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
        analyzer = _make_analyzer(_FakeClient(comp))
        analyzer.analyze_image(local_image_path=path)
        url = comp.calls[0]["messages"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
    finally:
        os.unlink(path)


def test_success_local_webp():
    path = _temp_image(".webp")
    try:
        comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
        analyzer = _make_analyzer(_FakeClient(comp))
        analyzer.analyze_image(local_image_path=path)
        url = comp.calls[0]["messages"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/webp;base64,")
    finally:
        os.unlink(path)


def test_uppercase_extension_normalized():
    path = _temp_image(".JPG")
    try:
        comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
        analyzer = _make_analyzer(_FakeClient(comp))
        analyzer.analyze_image(local_image_path=path)
        url = comp.calls[0]["messages"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
    finally:
        os.unlink(path)


def test_image_url_direct_passthrough():
    comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
    analyzer = _make_analyzer(_FakeClient(comp))
    analyzer.analyze_image(image_url="https://example.com/a.jpg")
    url = comp.calls[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert url == "https://example.com/a.jpg"


def test_timeout_propagated():
    path = _temp_image(".jpg")
    try:
        comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
        analyzer = _make_analyzer(_FakeClient(comp))
        analyzer.analyze_image(local_image_path=path, timeout=7.5)
        assert comp.calls[0]["timeout"] == 7.5
    finally:
        os.unlink(path)


def test_timeout_default_mode():
    comp = _FakeCompletions(exc=_timeout_exc())
    analyzer = _make_analyzer(_FakeClient(comp))
    result = analyzer.analyze_image(image_url="https://example.com/a.jpg")
    assert "error" in result and result["error"]
    assert len(comp.calls) == 1  # 无自动 retry


def test_timeout_strict_mode():
    comp = _FakeCompletions(exc=_timeout_exc())
    analyzer = _make_analyzer(_FakeClient(comp))
    try:
        analyzer.analyze_image(image_url="https://example.com/a.jpg", raise_on_error=True)
        assert False, "strict 模式应抛出 ImageAnalysisError"
    except ImageAnalysisError as e:
        assert e.code == ImageAnalysisErrorCode.TIMEOUT
        assert len(comp.calls) == 1


def test_provider_error_default_mode():
    comp = _FakeCompletions(exc=_auth_exc())
    analyzer = _make_analyzer(_FakeClient(comp))
    result = analyzer.analyze_image(image_url="https://example.com/a.jpg")
    assert "error" in result
    assert len(comp.calls) == 1


def test_provider_error_strict_mode():
    comp = _FakeCompletions(exc=_auth_exc())
    analyzer = _make_analyzer(_FakeClient(comp))
    try:
        analyzer.analyze_image(image_url="https://example.com/a.jpg", raise_on_error=True)
        assert False, "strict 模式应抛出 ImageAnalysisError"
    except ImageAnalysisError as e:
        assert e.code == ImageAnalysisErrorCode.PROVIDER_ERROR


def test_provider_connection_error_strict_mode():
    comp = _FakeCompletions(exc=_conn_exc())
    analyzer = _make_analyzer(_FakeClient(comp))
    try:
        analyzer.analyze_image(image_url="https://example.com/a.jpg", raise_on_error=True)
        assert False
    except ImageAnalysisError as e:
        assert e.code == ImageAnalysisErrorCode.PROVIDER_ERROR


def test_parse_error_default_mode():
    # 带大括号但非法 JSON → parser error 字典原样返回（旧行为兼容）
    comp = _FakeCompletions(response=_FakeResponse("prefix {broken json"))
    analyzer = _make_analyzer(_FakeClient(comp))
    result = analyzer.analyze_image(image_url="https://example.com/a.jpg")
    assert isinstance(result, dict)
    assert "error" in result


def test_parse_error_strict_mode():
    comp = _FakeCompletions(response=_FakeResponse("prefix {broken json"))
    analyzer = _make_analyzer(_FakeClient(comp))
    try:
        analyzer.analyze_image(image_url="https://example.com/a.jpg", raise_on_error=True)
        assert False, "strict 模式应抛出 parse_error"
    except ImageAnalysisError as e:
        assert e.code == ImageAnalysisErrorCode.PARSE_ERROR


def test_parse_no_braces_fallback_not_none():
    # 无大括号响应：extract_json_content 必须返回错误字典而非 None
    comp = _FakeCompletions(response=_FakeResponse("this response has no json braces"))
    analyzer = _make_analyzer(_FakeClient(comp))
    result = analyzer.analyze_image(image_url="https://example.com/a.jpg")
    assert isinstance(result, dict)
    assert "error" in result


def test_unsupported_mime_default_mode():
    path = _temp_image(".gif", content=b"GIF89a-fake")
    try:
        comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
        analyzer = _make_analyzer(_FakeClient(comp))
        result = analyzer.analyze_image(local_image_path=path)
        assert "error" in result
        assert "不支持的图片格式" in result["error"]
        assert len(comp.calls) == 0  # 未发起 provider 调用
    finally:
        os.unlink(path)


def test_unsupported_mime_strict_mode():
    path = _temp_image(".gif", content=b"GIF89a-fake")
    try:
        comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
        analyzer = _make_analyzer(_FakeClient(comp))
        try:
            analyzer.analyze_image(local_image_path=path, raise_on_error=True)
            assert False, "strict 模式应抛出 unsupported_mime"
        except ImageAnalysisError as e:
            assert e.code == ImageAnalysisErrorCode.UNSUPPORTED_MIME
        assert len(comp.calls) == 0
    finally:
        os.unlink(path)


def test_no_extension_unsupported():
    path = _temp_image("", content=b"data")
    try:
        comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
        analyzer = _make_analyzer(_FakeClient(comp))
        result = analyzer.analyze_image(local_image_path=path)
        assert "error" in result
        assert len(comp.calls) == 0
    finally:
        os.unlink(path)


def test_file_not_found_default_mode():
    comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
    analyzer = _make_analyzer(_FakeClient(comp))
    result = analyzer.analyze_image(local_image_path="Z:/no/such/file.jpg")
    assert "error" in result
    assert len(comp.calls) == 0


def test_file_not_found_strict_mode():
    comp = _FakeCompletions(response=_FakeResponse(_valid_content()))
    analyzer = _make_analyzer(_FakeClient(comp))
    try:
        analyzer.analyze_image(local_image_path="Z:/no/such/file.jpg", raise_on_error=True)
        assert False
    except ImageAnalysisError as e:
        assert e.code == ImageAnalysisErrorCode.INVALID_IMAGE


def test_no_automatic_retry_on_failure():
    # provider 失败 + strict：确保只有一次 create 调用
    comp = _FakeCompletions(exc=_auth_exc())
    analyzer = _make_analyzer(_FakeClient(comp))
    try:
        analyzer.analyze_image(image_url="https://example.com/a.jpg", raise_on_error=True)
    except ImageAnalysisError:
        pass
    assert len(comp.calls) == 1


def test_no_image_source_raises_value_error():
    analyzer = _make_analyzer(_FakeClient(_FakeCompletions()))
    try:
        analyzer.analyze_image()
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_client_unavailable_default_mode():
    analyzer = _make_analyzer(None)
    result = analyzer.analyze_image(image_url="https://example.com/a.jpg")
    assert "error" in result


def test_client_unavailable_strict_mode():
    analyzer = _make_analyzer(None)
    try:
        analyzer.analyze_image(image_url="https://example.com/a.jpg", raise_on_error=True)
        assert False
    except ImageAnalysisError as e:
        assert e.code == ImageAnalysisErrorCode.CLIENT_UNAVAILABLE


def test_extract_json_content_empty_preserved():
    result = extract_json_content("")
    assert result.get("error") == "Empty response"


def test_extract_json_content_valid_preserved():
    result = extract_json_content(_valid_content())
    assert result["title"] == "红圆"
    assert result["tags"] == ["红圆", "蓝三角"]


def test_mime_mapping_whitelist():
    assert set(SUPPORTED_IMAGE_EXTENSIONS) == {".jpg", ".jpeg", ".png", ".webp"}
    assert EXTENSION_TO_MIME[".jpg"] == "image/jpeg"
    assert EXTENSION_TO_MIME[".jpeg"] == "image/jpeg"
    assert EXTENSION_TO_MIME[".png"] == "image/png"
    assert EXTENSION_TO_MIME[".webp"] == "image/webp"


def test_default_mode_error_dict_shape():
    # 默认模式失败结果必须保持旧字典形状（error/title/description/tags）
    comp = _FakeCompletions(exc=_auth_exc())
    analyzer = _make_analyzer(_FakeClient(comp))
    result = analyzer.analyze_image(image_url="https://example.com/a.jpg")
    assert set(result.keys()) >= {"error", "title", "description", "tags"}


# ---------------------------------------------------------------------------
# 纯 python 运行器
# ---------------------------------------------------------------------------
def _main():
    failed = 0
    passed = 0
    mod_globals = globals()
    for name in sorted(mod_globals):
        if name.startswith("test_"):
            fn = mod_globals[name]
            try:
                fn()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print(f"\n== {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

