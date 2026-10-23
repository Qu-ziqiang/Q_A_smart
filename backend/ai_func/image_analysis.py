import os
import base64
import hashlib
import json
import time
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path
import logging

import openai  # 用于识别 timeout / provider 层异常

# 导入项目设置
from ..config import settings
from ..config.init_service import get_vision_client

# 配置日志记录
logger = logging.getLogger(__name__)

# 多模态提示词模板
MULTIMODAL_PROMPT = """
请分析这张图片，供图库入库和后续筛选。使用JSON格式输出。

需要识别：
1. 画面里的物体和元素
2. 场景
3. 主要颜色
4. 图片里能读到的文字；没有就空字符串
5. 10字以内标题、50字以内描述、1-3个标签

输出格式必须严格为:
{
  "title": "简洁标题(10字以内)",
  "description": "详细描述(50字以内)",
  "tags": ["标签1", "标签2", "标签3"],
  "objects": ["物体1", "物体2"],
  "scene": "场景",
  "colors": ["颜色1"],
  "ocr_text": ""
}

只返回JSON，不要有其他说明文字。
"""


# ---------------------------------------------------------------------------
# Phase 2 Stage 2 — ImageAnalysis Boundary Hardening
# 错误分类 + strict error 边界 + MIME 归一化（默认行为保持旧语义兼容）
# ---------------------------------------------------------------------------
class ImageAnalysisErrorCode(str, Enum):
    """ImageAnalysis 机器可读错误分类。"""
    CLIENT_UNAVAILABLE = "client_unavailable"   # 未配置 OpenAI 兼容客户端/API key
    NO_IMAGE_SOURCE = "no_image_source"          # 未提供 image_url 或 local_image_path
    INVALID_IMAGE = "invalid_image"              # 本地文件不存在
    UNSUPPORTED_MIME = "unsupported_mime"        # 扩展名不在受支持白名单内
    TIMEOUT = "timeout"                          # provider 请求超时（无自动 retry）
    PROVIDER_ERROR = "provider_error"            # provider/API 层错误（网络/4xx/5xx）
    PARSE_ERROR = "parse_error"                  # 响应无法解析为 title/description/tags
    UNKNOWN_ERROR = "unknown_error"              # 其他未分类异常


class ImageAnalysisError(Exception):
    """ImageAnalysis strict 模式（raise_on_error=True）下的错误边界。

    `code` 提供机器可读的错误分类；`safe_message` 为可安全展示的错误文案。
    默认模式（raise_on_error=False）不抛出，只返回 error 字典，保持旧调用行为。
    """

    def __init__(self, code: ImageAnalysisErrorCode, safe_message: str):
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


# 受支持图片扩展名白名单（大小写不敏感，统一映射到标准 MIME）
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
EXTENSION_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _image_error(
    code: ImageAnalysisErrorCode,
    safe_message: str,
    raise_on_error: bool,
) -> Dict[str, object]:
    """统一失败出口：strict 模式抛 ImageAnalysisError，默认模式返回 error 字典。"""
    if raise_on_error:
        raise ImageAnalysisError(code, safe_message)
    return {"error": safe_message, "title": "", "description": "", "tags": []}


def extract_json_content(text: str) -> Dict[str, any]:
    """
    从文本中提取JSON内容。

    参数:
        text (str): 可能包含JSON的文本

    返回:
        Dict[str, str]: 解析后的JSON字典，如果解析失败则返回包含错误信息的字典
    """
    if not text:
        return {"error": "Empty response", "title": "", "description": "", "tags": []}

    # 尝试寻找JSON的开始和结束位置
    json_start = text.find("{")
    json_end = text.rfind("}")

    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            json_text = text[json_start : json_end + 1]
            result = json.loads(json_text)
            # 确保返回的字典包含必要的键
            if "title" not in result:
                result["title"] = ""
            if "description" not in result:
                result["description"] = ""
            if "tags" not in result:
                result["tags"] = []
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.debug(f"原始文本: {text}")
            return {
                "error": f"JSON解析失败: {str(e)}",
                "title": "",
                "description": "",
                "tags": [],
            }

    # 未找到 JSON 起止大括号：确定性返回错误字典
    # （原实现此处会隐式返回 None，导致上层无法识别 parser 失败；仅补 fallback，不改变既有分支）
    return {
        "error": "未找到JSON内容",
        "title": "",
        "description": "",
        "tags": [],
    }


def image_to_base64(image_path: str) -> str:
    """
    将图像文件转换为Base64编码的字符串。

    参数:
    image_path (str): 图像文件路径。

    返回:
    str: Base64编码的字符串。
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


class ImageAnalysis:
    """
    图像文本提取器类，用于将图像内容转换为文本描述和标题。

    该类使用OpenAI的多模态模型分析图像内容，生成描述性文本和标题。
    """

    def __init__(
        self,
        prompt: str = "",
    ):
        """
        初始化 ImageAnalysis 实例。

        参数:
            prompt: 自定义提示文本，如果不提供则使用默认提示
        """
        # 使用统一的 OpenAI 客户端
        self.client = get_vision_client()
        if not self.client:
            logger.warning("OpenAI 客户端未配置，多模态功能将无法使用")
        self._prompt = prompt or MULTIMODAL_PROMPT
        logger.info("多模态图像分析器初始化完成")


    def analyze_image(
        self,
        image_url: str = None,
        local_image_path: str = None,
        model: str = None,
        detail: str = "low",
        prompt: str = None,
        temperature: float = 0.1,
        timeout: Optional[float] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, str]:
        """
        分析图像并生成JSON格式的标题和描述。

        参数:
            image_url: 图像的URL
            local_image_path: 本地图像路径（会做 MIME 归一化，仅支持 jpg/jpeg/png/webp）
            model: 使用的模型名称
            detail: 细节级别，可选值为 'low', 'high', 'auto'
            prompt: 提示文本，覆盖默认提示
            temperature: 生成文本的温度参数
            timeout: 请求超时秒数（传给 provider 调用边界；None 表示 SDK 默认，保持旧行为）
            raise_on_error: 严格错误模式；True 时失败抛出 ImageAnalysisError，False 时返回 error 字典

        返回:
            Dict[str, str]: 包含标题和描述的字典
        """
        if not self.client:
            return _image_error(
                ImageAnalysisErrorCode.CLIENT_UNAVAILABLE,
                "无法分析图像：未配置OpenAI客户端或API密钥",
                raise_on_error,
            )

        if not image_url and not local_image_path:
            raise ValueError("必须提供image_url或local_image_path中的一个")

        # 准备图像URL（仅本地文件路径做 MIME 归一化与确定性校验）
        if local_image_path:
            if not os.path.exists(local_image_path):
                return _image_error(
                    ImageAnalysisErrorCode.INVALID_IMAGE,
                    f"图片文件不存在: {local_image_path}",
                    raise_on_error,
                )
            image_extension = Path(local_image_path).suffix.lower()
            mime = EXTENSION_TO_MIME.get(image_extension)
            if mime is None:
                return _image_error(
                    ImageAnalysisErrorCode.UNSUPPORTED_MIME,
                    f"不支持的图片格式: {image_extension or '(无扩展名)'}（仅支持 .jpg/.jpeg/.png/.webp）",
                    raise_on_error,
                )
            with open(local_image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = f"data:{mime};base64,{base64_image}"

        # 获取配置对象
        config = settings.get_config()

        # 设置模型和提示词
        model = model or config.VISION_MODEL
        prompt_text = prompt or self._prompt

        # 验证细节级别参数
        if detail not in ["low", "high", "auto"]:
            detail = "low"

        start_time = time.time()
        try:
            create_kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url, "detail": detail},
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                "stream": False,
                "temperature": temperature,
            }
            # 明确 timeout：真正传递到 HTTP/provider 调用边界；不自动 retry、不切换模型/provider
            if timeout is not None:
                create_kwargs["timeout"] = timeout

            # 调用API
            response = self.client.chat.completions.create(**create_kwargs)

            # 获取结果
            result = response.choices[0].message.content

            # 解析JSON结果（复用既有 parser，不重写）
            analysis_result = extract_json_content(result)
            if analysis_result is None or "error" in analysis_result:
                if raise_on_error:
                    raise ImageAnalysisError(
                        ImageAnalysisErrorCode.PARSE_ERROR,
                        analysis_result.get("error", "未找到JSON内容")
                        if analysis_result
                        else "未找到JSON内容",
                    )
                # 默认行为保持旧语义：原样返回 parser 的错误字典
                return analysis_result

            end_time = time.time()
            logger.info(f"图像分析完成，耗时: {end_time - start_time:.2f}秒")

            return analysis_result

        except ImageAnalysisError:
            # strict 模式下已在 try 内抛出：透传，不降级为 UNKNOWN_ERROR
            raise
        except openai.APITimeoutError as e:
            end_time = time.time()
            logger.info(f"图像分析失败(超时)，耗时: {end_time - start_time:.2f}秒")
            return _image_error(
                ImageAnalysisErrorCode.TIMEOUT,
                f"图像分析超时: {str(e)}",
                raise_on_error,
            )
        except (openai.APIConnectionError, openai.APIStatusError) as e:
            end_time = time.time()
            logger.info(f"图像分析失败(服务错误)，耗时: {end_time - start_time:.2f}秒")
            return _image_error(
                ImageAnalysisErrorCode.PROVIDER_ERROR,
                f"图像分析服务错误: {str(e)}",
                raise_on_error,
            )
        except Exception as e:
            logger.error(f"图像分析出错: {str(e)}")
            end_time = time.time()
            logger.info(f"图像分析失败，耗时: {end_time - start_time:.2f}秒")
            return _image_error(
                ImageAnalysisErrorCode.UNKNOWN_ERROR,
                f"图像分析失败: {str(e)}",
                raise_on_error,
            )


if __name__ == "__main__":
    # 示例用法
    image_analyzer = ImageAnalysis()
    result = image_analyzer.analyze_image(local_image_path="example.jpg")
