import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

ARM_JOINT_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
STATE_ORDER = (
    *(f"left_{joint}" for joint in ARM_JOINT_ORDER),
    "left_gripper",
    *(f"right_{joint}" for joint in ARM_JOINT_ORDER),
    "right_gripper",
)
ACTION_SPACE = "pi0/arx_bimanual"
GRIPPER_CONVENTION = "0.0=open, 1.0=closed"
I2RT_GRIPPER_CONVENTION = "0.0=closed, 1.0=open"


def make_yam_bimanual_example() -> dict:
    """Creates a random input example for the YAM bimanual policy."""
    return {
        "state": np.zeros((14,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class YamBimanualInputs(transforms.DataTransformFn):
    """Inputs for the bimanual YAM policy.

    Expected runtime input:
    - images: dict[name, img] where img is either [C, H, W] or [H, W, C].
    - state: [14], ordered [left 6 joints, left gripper, right 6 joints, right gripper].
      Grippers use the PI/ARX convention: 0.0=open, 1.0=closed.
    - actions: [action_horizon, 14], same order as state, only present during training.
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != 14:
            raise ValueError(f"Expected YAM state last dimension to be 14, got {state.shape}")

        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected YAM cameras: {tuple(sorted(unexpected))}")
        if "cam_high" not in in_images:
            raise ValueError('YAM policy requires "cam_high" image input')

        base_image = _parse_image(in_images["cam_high"])
        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        for dest, source in (
            ("left_wrist_0_rgb", "cam_left_wrist"),
            ("right_wrist_0_rgb", "cam_right_wrist"),
        ):
            if source in in_images:
                images[dest] = _parse_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 14:
                raise ValueError(f"Expected YAM actions last dimension to be 14, got {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            elif isinstance(prompt, np.ndarray) and prompt.shape == ():
                prompt = prompt.item()
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class YamBimanualOutputs(transforms.DataTransformFn):
    """Outputs for the bimanual YAM policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14], dtype=np.float32)}


def as_arm_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (7,):
        raise ValueError(f"Expected 7D arm state, got {state.shape}")
    return state


def i2rt_arm_state_to_openpi(state: np.ndarray) -> np.ndarray:
    """Convert one 7D YAM state from i2rt convention to OpenPI/ARX convention.

    i2rt linear grippers use 0=closed, 1=open. PI/ARX uses 0=open, 1=closed.
    The six arm joint slots are already base-to-wrist in both conventions.
    """
    state = as_arm_state(state)
    result = state.copy()
    result[6] = np.clip(1.0 - result[6], 0.0, 1.0)
    return result


def openpi_arm_state_to_i2rt(state: np.ndarray) -> np.ndarray:
    """Convert one 7D OpenPI/ARX state to the YAM i2rt command convention."""
    return i2rt_arm_state_to_openpi(state)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected image rank 3, got {image.shape}")
    if image.shape[0] == 3:
        return einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] == 3:
        return image.astype(np.uint8, copy=False)
    raise ValueError(f"Expected CHW or HWC RGB image, got {image.shape}")
