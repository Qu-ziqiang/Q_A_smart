"""遥感编码器只登记接口，默认不加载权重。"""

from backend.ai_func.encoders import EncoderDisabled, get_encoder, list_encoder_specs


def test_catalog_keeps_remote_sensing_off_the_default_index():
    specs = {spec.name: spec for spec in list_encoder_specs()}
    assert set(specs) == {"clip", "remote_sensing"}
    assert specs["clip"].shares_index_with_default is True
    assert specs["clip"].enabled is True
    rs = specs["remote_sensing"]
    assert rs.enabled is False
    assert rs.shares_index_with_default is False
    assert rs.dimension == 512
    assert "chendelong/RemoteCLIP" in rs.model_id
    assert "RemoteCLIP-ViT-B-32.pt" in rs.model_id


def test_remote_sensing_encode_does_not_download_when_disabled():
    encoder = get_encoder("remote_sensing")
    try:
        encoder.encode_text("机场跑道")
    except EncoderDisabled as exc:
        assert "未启用" in str(exc)
    else:
        raise AssertionError("disabled encoder should refuse before loading weights")
