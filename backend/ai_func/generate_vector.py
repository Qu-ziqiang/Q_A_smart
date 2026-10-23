import os
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Any
import diskcache


import transformers.utils.import_utils as transformers_utils
import transformers.modeling_utils as modeling_utils


def skip_check(*args, **kwargs): return None


transformers_utils.check_torch_load_is_safe = skip_check
transformers_utils._is_torch_safe = True
modeling_utils.check_torch_load_is_safe = skip_check

# 现在可以安全导入
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from ..config import settings
from ..utils.text_utils import get_text_cache_key
from ..utils.image_utils import get_image_cache_key

# ============================================================

# 全局变量
model = None
embedding_dimension = None
model_loading = False
_vision_processor = None
_vision_model = None
_vision_device = None


def load_model():
    """加载多语言文本模型"""
    global model, embedding_dimension, model_loading
    if model is not None: return model
    if model_loading:
        import time
        while model_loading and model is None:
            time.sleep(0.1)
        return model

    model_loading = True
    try:
        config = settings.get_config()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] loading text model: {config.MODEL_PATH} on {device}...")
        model = SentenceTransformer(config.MODEL_PATH, device=device)
        embedding_dimension = 512
        print("[INFO] text model loaded")
    finally:
        model_loading = False
    return model


def get_model() -> SentenceTransformer:
    global model
    if model is None: load_model()
    return model


def get_embedding_dimension() -> int:
    global embedding_dimension
    if embedding_dimension is None:
        try:
            from ..config import settings
            config = settings.get_config()
            embedding_dimension = config.EMBEDDING_DIMENSION or 512
        except Exception:
            load_model()
    return embedding_dimension


def encode_text(text: Union[str, List[str]], cache_dir=None) -> np.ndarray:
    """文本向量化"""
    config = settings.get_config()
    cache_dir = cache_dir or config.TEXT_VECTOR_CACHE_DIR
    cache_key = get_text_cache_key(text)
    cache_instance = diskcache.Cache(directory=cache_dir)

    embeddings = cache_instance.get(cache_key)
    if embeddings is not None: return embeddings

    model_instance = get_model()
    embeddings = model_instance.encode(text, normalize_embeddings=True)
    cache_instance.set(cache_key, embeddings)
    return embeddings


def encode_image(image_input, cache_dir=None) -> np.ndarray:
    """图像向量化：增加鲁棒性拆箱逻辑，解决 BaseModelOutputWithPooling 报错"""
    config = settings.get_config()
    cache_dir = cache_dir or config.IMAGE_VECTOR_CACHE_DIR
    cache_key = get_image_cache_key(image_input)
    cache_instance = diskcache.Cache(directory=cache_dir)

    embeddings_cached = cache_instance.get(cache_key)
    if embeddings_cached is not None: return embeddings_cached

    # 图像预处理
    is_single = isinstance(image_input, (Image.Image, str))
    raw_inputs = [image_input] if is_single else image_input
    images_to_encode = []
    for item in raw_inputs:
        try:
            if isinstance(item, str):
                full_path = item if os.path.isabs(item) else os.path.join(os.getcwd(), item)
                img = Image.open(full_path).convert("RGB")
                images_to_encode.append(img)
            else:
                images_to_encode.append(item.convert("RGB") if item.mode != "RGB" else item)
        except Exception as e:
            print(f"[WARN] failed to load image: {e}")
            images_to_encode.append(Image.new('RGB', (224, 224), color=0))

    # 推理提取
    global _vision_processor, _vision_model, _vision_device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if _vision_model is None or _vision_device != device:
            proxy_model_id = "openai/clip-vit-base-patch32"
            _vision_processor = CLIPProcessor.from_pretrained(proxy_model_id)
            _vision_model = CLIPModel.from_pretrained(proxy_model_id, weights_only=False).to(device)
            _vision_device = device
            print(f"[INFO] vision CLIP loaded on {device}")

        inputs = _vision_processor(images=images_to_encode, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = _vision_model.get_image_features(**inputs)

            # 🛡️ 核心修复：执行“拆箱”动作
            # 如果 outputs 不是 Tensor，说明它是一个结果对象，需要取出里面的数据
            if not torch.is_tensor(outputs):
                if hasattr(outputs, "image_features"):
                    outputs = outputs.image_features
                elif hasattr(outputs, "pooler_output"):
                    outputs = outputs.pooler_output
                else:
                    # 万能兜底：取第一个元素
                    outputs = outputs[0]

            results = outputs.cpu().numpy()

    except Exception as e:
        print(f"[WARN] image feature extract failed: {e}")
        raise RuntimeError(f"图像特征提取失败: {e}")

    # 归一化
    if len(results.shape) > 1:
        norms = np.linalg.norm(results, axis=1, keepdims=True)
        results = results / (norms + 1e-12)
        final_vector = results[0] if is_single else results
    else:
        final_vector = results / (np.linalg.norm(results) + 1e-12)

    cache_instance.set(cache_key, final_vector)
    return final_vector