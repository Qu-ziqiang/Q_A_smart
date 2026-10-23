"""图文编码器接口。默认检索不经过这里。"""

from dataclasses import dataclass, asdict
from typing import List, Protocol, Union

import numpy as np
from PIL import Image


ImageInput = Union[str, Image.Image, List[Union[str, Image.Image]]]
TextInput = Union[str, List[str]]


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    domain: str
    model_id: str
    dimension: int
    shares_index_with_default: bool
    enabled: bool
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


class EncoderDisabled(RuntimeError):
    """接口在，权重未启用。"""


class ImageTextEncoder(Protocol):
    spec: EncoderSpec

    def encode_text(self, text: TextInput) -> np.ndarray:
        ...

    def encode_image(self, image_input: ImageInput) -> np.ndarray:
        ...
