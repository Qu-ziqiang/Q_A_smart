"""通用 CLIP。编码实现仍在 generate_vector，这里只是同一套接口上的入口。"""

from .base import EncoderSpec, ImageInput, TextInput


class ClipEncoder:
    spec = EncoderSpec(
        name="clip",
        domain="natural_photo",
        model_id="clip-ViT-B-32-multilingual-v1 + openai/clip-vit-base-patch32",
        dimension=512,
        shares_index_with_default=True,
        enabled=True,
        note="现有入库和检索走 generate_vector，不经过编码器路由。",
    )

    def encode_text(self, text: TextInput):
        from ..generate_vector import encode_text
        return encode_text(text)

    def encode_image(self, image_input: ImageInput):
        from ..generate_vector import encode_image
        return encode_image(image_input)
