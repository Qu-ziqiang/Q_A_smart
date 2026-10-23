"""RemoteCLIP（ViT-B-32）。卫星/航拍图文对齐，和通用 CLIP 不是同一个向量空间。

权重：Hugging Face chendelong/RemoteCLIP 的 RemoteCLIP-ViT-B-32.pt（OpenCLIP 格式）。
默认不下载、不编码、不写 pgvector。打开方式：RS_ENCODER_ENABLED=true，并安装 open_clip_torch。
"""

import os
from typing import List, Optional

import numpy as np
from PIL import Image

from .base import EncoderDisabled, EncoderSpec, ImageInput, TextInput


def _enabled() -> bool:
    env = os.environ.get("SM_RS_ENCODER", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    try:
        from ...config import settings
        return bool(settings.get_config().RS_ENCODER_ENABLED)
    except Exception:
        return False


def _rs_config() -> dict:
    repo = "chendelong/RemoteCLIP"
    filename = "RemoteCLIP-ViT-B-32.pt"
    arch = "ViT-B-32"
    dimension = 512
    try:
        from ...config import settings
        cfg = settings.get_config()
        repo = cfg.RS_MODEL_REPO or repo
        filename = cfg.RS_MODEL_FILE or filename
        arch = cfg.RS_ARCH or arch
        dimension = int(cfg.RS_EMBEDDING_DIMENSION or dimension)
    except Exception:
        pass
    return {
        "repo": repo,
        "filename": filename,
        "arch": arch,
        "dimension": dimension,
    }


def remote_sensing_spec() -> EncoderSpec:
    cfg = _rs_config()
    return EncoderSpec(
        name="remote_sensing",
        domain="remote_sensing",
        model_id=f"{cfg['repo']}/{cfg['filename']} ({cfg['arch']})",
        dimension=cfg["dimension"],
        shares_index_with_default=False,
        enabled=_enabled(),
        note="接口可用，默认不参与三路检索。维度同为 512，但不能写入现有 image/title/description 向量表。",
    )


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norms + 1e-12)


def _load_images(image_input: ImageInput) -> List[Image.Image]:
    raw = image_input if isinstance(image_input, list) else [image_input]
    images = []
    for item in raw:
        if isinstance(item, str):
            path = item if os.path.isabs(item) else os.path.join(os.getcwd(), item)
            images.append(Image.open(path).convert("RGB"))
        else:
            images.append(item.convert("RGB") if item.mode != "RGB" else item)
    return images


class RemoteSensingEncoder:
    """懒加载。未启用时 encode_* 直接拒绝，不会去拉 600MB 权重。"""

    def __init__(self):
        self.spec = remote_sensing_spec()
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None

    def _require_enabled(self) -> None:
        self.spec = remote_sensing_spec()
        if not self.spec.enabled:
            raise EncoderDisabled(
                "遥感编码器未启用。设置 RS_ENCODER_ENABLED=true 或环境变量 SM_RS_ENCODER=1 后才会下载并加载 RemoteCLIP。"
            )

    def _load(self):
        if self._model is not None:
            return
        self._require_enabled()
        try:
            import torch
            import open_clip
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise EncoderDisabled(
                "遥感编码依赖 open_clip_torch 与 huggingface_hub，当前环境未安装。"
            ) from exc

        cfg = _rs_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(cfg["arch"])
        ckpt_path = hf_hub_download(cfg["repo"], cfg["filename"])
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        model = model.to(device).eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(cfg["arch"])
        self._device = device

    def encode_text(self, text: TextInput) -> np.ndarray:
        import torch

        self._load()
        texts = [text] if isinstance(text, str) else list(text)
        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        vectors = _l2_normalize(features.detach().cpu().numpy().astype(np.float32))
        return vectors[0] if isinstance(text, str) else vectors

    def encode_image(self, image_input: ImageInput) -> np.ndarray:
        import torch

        self._load()
        images = _load_images(image_input)
        batch = torch.stack([self._preprocess(img) for img in images]).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
        vectors = _l2_normalize(features.detach().cpu().numpy().astype(np.float32))
        single = not isinstance(image_input, list)
        return vectors[0] if single else vectors
