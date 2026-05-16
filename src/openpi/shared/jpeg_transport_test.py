import numpy as np

from openpi.shared import jpeg_transport


def test_jpeg_encode_decode_roundtrip_shape_and_dtype():
    image = np.random.default_rng(0).integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    encoded = jpeg_transport.encode_rgb_jpeg(image)
    decoded = jpeg_transport.decode_rgb_jpeg(encoded)

    assert isinstance(encoded, bytes)
    assert decoded.shape == image.shape
    assert decoded.dtype == np.uint8


def test_jpeg_encode_is_deterministic_for_same_input():
    image = np.random.default_rng(1).integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    assert jpeg_transport.encode_rgb_jpeg(image) == jpeg_transport.encode_rgb_jpeg(image)


def test_as_rgb_uint8_accepts_chw_and_float_inputs():
    image = np.random.default_rng(2).random(size=(3, 224, 224), dtype=np.float32)

    converted = jpeg_transport.as_rgb_uint8(image)

    assert converted.shape == (224, 224, 3)
    assert converted.dtype == np.uint8


def test_jpeg_roundtrip_images_supports_batches_and_mappings():
    batch = np.random.default_rng(3).integers(0, 256, size=(2, 224, 224, 3), dtype=np.uint8)

    rounded = jpeg_transport.jpeg_roundtrip_images({"cam": batch})

    assert set(rounded) == {"cam"}
    assert rounded["cam"].shape == batch.shape
    assert rounded["cam"].dtype == np.uint8
