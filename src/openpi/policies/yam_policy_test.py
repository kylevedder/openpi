import numpy as np
from openpi_client import image_tools
import pytest

from openpi import transforms
from openpi.policies import yam_policy
from openpi.shared import jpeg_transport


def test_yam_schema_matches_pi_arx_bimanual_slots():
    assert yam_policy.ARM_JOINT_ORDER == (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    )
    assert yam_policy.STATE_ORDER == (
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    )
    assert yam_policy.ACTION_SPACE == "pi0/arx_bimanual"
    assert yam_policy.GRIPPER_CONVENTION == "0.0=open, 1.0=closed"


def test_yam_i2rt_gripper_conversion_happens_at_hardware_boundary():
    closed_i2rt = np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.float32)
    open_i2rt = np.array([0, 1, 2, 3, 4, 5, 1], dtype=np.float32)

    np.testing.assert_array_equal(
        yam_policy.i2rt_arm_state_to_openpi(closed_i2rt),
        np.array([0, 1, 2, 3, 4, 5, 1], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        yam_policy.i2rt_arm_state_to_openpi(open_i2rt),
        np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        yam_policy.openpi_arm_state_to_i2rt(yam_policy.i2rt_arm_state_to_openpi(open_i2rt)),
        open_i2rt,
    )


def test_yam_inputs_accept_jpeg_transport_images():
    images = _jpeg_images()

    inputs = yam_policy.YamBimanualInputs()({"state": np.zeros((14,), dtype=np.float32), "images": images})

    assert inputs[jpeg_transport.MARKER_KEY] == jpeg_transport.IMAGE_TRANSPORT
    assert inputs["image"]["base_0_rgb"].shape == (*jpeg_transport.IMAGE_RESOLUTION, 3)
    assert inputs["image"]["base_0_rgb"].dtype == np.uint8


def test_yam_inputs_reject_mixed_jpeg_and_raw_images():
    images = _jpeg_images()
    images["cam_left_wrist"] = np.zeros((*jpeg_transport.IMAGE_RESOLUTION, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="must not mix JPEG"):
        yam_policy.YamBimanualInputs()({"state": np.zeros((14,), dtype=np.float32), "images": images})


def test_yam_training_and_inference_image_transport_match():
    rng = np.random.default_rng(0)
    raw_images = {
        name: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        for name in yam_policy.YamBimanualInputs.EXPECTED_CAMERAS
    }
    resized_images = {
        name: image_tools.convert_to_uint8(image_tools.resize_with_pad(image, *jpeg_transport.IMAGE_RESOLUTION))
        for name, image in raw_images.items()
    }

    training_inputs = yam_policy.YamBimanualInputs()(
        {"state": np.zeros((14,), dtype=np.float32), "images": raw_images, "prompt": "task"}
    )
    training_inputs = transforms.ResizeImages(*jpeg_transport.IMAGE_RESOLUTION)(training_inputs)
    training_inputs = transforms.JpegRoundTripImages()(training_inputs)

    inference_inputs = yam_policy.YamBimanualInputs()(
        {
            "state": np.zeros((14,), dtype=np.float32),
            "images": {
                name: jpeg_transport.encode_rgb_jpeg(image, quality=jpeg_transport.JPEG_QUALITY)
                for name, image in resized_images.items()
            },
            "prompt": "task",
        }
    )
    inference_inputs = transforms.ResizeImages(*jpeg_transport.IMAGE_RESOLUTION)(inference_inputs)
    inference_inputs = transforms.JpegRoundTripImages()(inference_inputs)

    for name in training_inputs["image"]:
        np.testing.assert_array_equal(training_inputs["image"][name], inference_inputs["image"][name])


def _jpeg_images() -> dict[str, bytes]:
    rng = np.random.default_rng(1)
    return {
        name: jpeg_transport.encode_rgb_jpeg(
            rng.integers(0, 256, size=(*jpeg_transport.IMAGE_RESOLUTION, 3), dtype=np.uint8)
        )
        for name in yam_policy.YamBimanualInputs.EXPECTED_CAMERAS
    }
