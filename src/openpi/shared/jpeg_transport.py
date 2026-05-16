from __future__ import annotations

from collections.abc import Mapping

import cv2
import einops
import numpy as np

IMAGE_TRANSPORT = "jpeg_q85_224_rgb_v1"
JPEG_QUALITY = 85
IMAGE_RESOLUTION = (224, 224)
MARKER_KEY = "_image_transport"


def as_rgb_uint8(image) -> np.ndarray:
    """Return an image as HWC RGB uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)

    if image.ndim != 3:
        raise ValueError(f"Expected image rank 3, got {image.shape}")
    if image.shape[0] == 3:
        return einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] == 3:
        return image
    raise ValueError(f"Expected CHW or HWC RGB image, got {image.shape}")


def encode_rgb_jpeg(image, *, quality: int = JPEG_QUALITY) -> bytes:
    rgb = as_rgb_uint8(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("OpenCV failed to encode JPEG")
    return encoded.tobytes()


def decode_rgb_jpeg(data: bytes | bytearray | memoryview) -> np.ndarray:
    encoded = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV failed to decode JPEG image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def jpeg_roundtrip_rgb(image, *, quality: int = JPEG_QUALITY) -> np.ndarray:
    return decode_rgb_jpeg(encode_rgb_jpeg(image, quality=quality))


def jpeg_roundtrip_images(images, *, quality: int = JPEG_QUALITY):
    """Apply the transport JPEG round-trip to one image, a batch, or an image mapping."""
    if isinstance(images, Mapping):
        return {name: jpeg_roundtrip_images(image, quality=quality) for name, image in images.items()}

    images = np.asarray(images)
    if images.ndim == 3:
        return jpeg_roundtrip_rgb(images, quality=quality)
    if images.ndim < 3:
        raise ValueError(f"Expected image rank at least 3, got {images.shape}")

    prefix = images.shape[:-3]
    flat = images.reshape(-1, *images.shape[-3:])
    rounded = np.stack([jpeg_roundtrip_rgb(image, quality=quality) for image in flat])
    return rounded.reshape(*prefix, *rounded.shape[-3:])
