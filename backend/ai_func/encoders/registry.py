"""编码器名录。检索主路不调用 get_encoder。"""

from typing import Dict, List

from .base import EncoderSpec, ImageTextEncoder
from .clip_encoder import ClipEncoder
from .remote_sensing import RemoteSensingEncoder, remote_sensing_spec


def list_encoder_specs() -> List[EncoderSpec]:
    clip = ClipEncoder.spec
    return [clip, remote_sensing_spec()]


def get_encoder(name: str) -> ImageTextEncoder:
    key = (name or "").strip().lower()
    encoders: Dict[str, ImageTextEncoder] = {
        "clip": ClipEncoder(),
        "remote_sensing": RemoteSensingEncoder(),
    }
    if key not in encoders:
        known = ", ".join(encoders)
        raise KeyError(f"未知编码器 {name!r}，可选：{known}")
    return encoders[key]
