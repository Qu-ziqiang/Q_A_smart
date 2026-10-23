"""编码器查询接口。上传、三路检索、Agent 检索都不调用这些路由。"""

from fastapi import APIRouter, Body, Path

from backend.ai_func.encoders import EncoderDisabled, get_encoder, list_encoder_specs
from backend.global_schemas import ResponseModel


router = APIRouter(prefix="/api/v1/encoders", tags=["encoders"])


def _vector_payload(vector):
    values = vector.tolist()
    flat = values if isinstance(values[0], float) else values
    dim = len(values) if isinstance(values[0], float) else len(values[0])
    return {"dimension": dim, "vector": flat if isinstance(values[0], float) else values}


@router.get("")
@router.get("/")
async def list_encoders():
    specs = [spec.to_dict() for spec in list_encoder_specs()]
    return ResponseModel.success(
        data=specs,
        message="编码器列表。默认检索仍使用 CLIP，不读取本列表。",
    )


@router.get("/{name}")
async def get_encoder_spec(name: str = Path(..., description="clip 或 remote_sensing")):
    try:
        spec = get_encoder(name).spec
    except KeyError as exc:
        return ResponseModel.error(code="ENCODER_NOT_FOUND", message=str(exc), http_code=404)
    return ResponseModel.success(data=spec.to_dict())


@router.post("/{name}/encode-text")
async def encode_text_route(
    name: str = Path(..., description="clip 或 remote_sensing"),
    text: str = Body(..., embed=True),
):
    try:
        encoder = get_encoder(name)
        vector = encoder.encode_text(text)
    except KeyError as exc:
        return ResponseModel.error(code="ENCODER_NOT_FOUND", message=str(exc), http_code=404)
    except EncoderDisabled as exc:
        return ResponseModel.error(code="ENCODER_DISABLED", message=str(exc), http_code=409)
    except Exception as exc:
        return ResponseModel.error(code="ENCODE_FAILED", message=str(exc), http_code=500)
    return ResponseModel.success(
        data={"encoder": encoder.spec.to_dict(), **_vector_payload(vector)},
        message="只返回向量，不写入数据库。",
    )


@router.post("/{name}/encode-image")
async def encode_image_route(
    name: str = Path(..., description="clip 或 remote_sensing"),
    image_path: str = Body(..., embed=True),
):
    try:
        encoder = get_encoder(name)
        vector = encoder.encode_image(image_path)
    except KeyError as exc:
        return ResponseModel.error(code="ENCODER_NOT_FOUND", message=str(exc), http_code=404)
    except EncoderDisabled as exc:
        return ResponseModel.error(code="ENCODER_DISABLED", message=str(exc), http_code=409)
    except Exception as exc:
        return ResponseModel.error(code="ENCODE_FAILED", message=str(exc), http_code=500)
    return ResponseModel.success(
        data={"encoder": encoder.spec.to_dict(), **_vector_payload(vector)},
        message="只返回向量，不写入数据库。",
    )
