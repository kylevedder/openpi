from __future__ import annotations

from collections.abc import Iterator
import contextlib
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from openpi.policies import yam_policy

CAMERA_PATHS = {
    "cam_high": "/dev/yam/cam_top",
    "cam_left_wrist": "/dev/yam/cam_left_wrist",
    "cam_right_wrist": "/dev/yam/cam_right_wrist",
}

FOLLOWER_CHANNELS = {
    "left": "can_follower_l",
    "right": "can_follower_r",
}

LEADER_CHANNELS = {
    "left": "can_leader_l",
    "right": "can_leader_r",
}

CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ARM_JOINT_ORDER = yam_policy.ARM_JOINT_ORDER
STATE_ORDER = yam_policy.STATE_ORDER
ACTION_SPACE = yam_policy.ACTION_SPACE
GRIPPER_CONVENTION = yam_policy.GRIPPER_CONVENTION
I2RT_GRIPPER_CONVENTION = yam_policy.I2RT_GRIPPER_CONVENTION
i2rt_arm_state_to_openpi = yam_policy.i2rt_arm_state_to_openpi
openpi_arm_state_to_i2rt = yam_policy.openpi_arm_state_to_i2rt


def ensure_i2rt_importable() -> None:
    """Make the sibling i2rt checkout importable when running from the openpi venv."""
    if "i2rt" in sys.modules:
        return
    if os.environ.get("I2RT_ROOT"):
        candidates = [Path(os.environ["I2RT_ROOT"])]
    else:
        candidates = [Path(__file__).resolve().parents[3] / "i2rt"]
    for candidate in candidates:
        if (candidate / "i2rt").is_dir():
            sys.path.insert(0, str(candidate))
            return


class CameraSet:
    def __init__(self, paths: dict[str, str] | None = None, width: int = 640, height: int = 480) -> None:
        self.paths = paths or CAMERA_PATHS
        self._captures: dict[str, cv2.VideoCapture] = {}
        for name, path in self.paths.items():
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.close()
                raise RuntimeError(f"Failed to open camera {name}: {path}")
            self._captures[name] = cap

    def read(self) -> dict[str, np.ndarray]:
        frames = {}
        for name, cap in self._captures.items():
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame from {name}")
            frames[name] = frame
        return frames

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()

    def __enter__(self) -> CameraSet:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def make_episode_dir(root: Path, name: str | None = None) -> Path:
    episode_name = name or time.strftime("episode_%Y%m%d_%H%M%S")
    episode_dir = root / episode_name
    episode_dir.mkdir(parents=True, exist_ok=False)
    return episode_dir


def get_follower_state(robot) -> np.ndarray:
    """Return follower state in the OpenPI/ARX convention."""
    obs = robot.get_observations()
    arm = np.asarray(obs["joint_pos"], dtype=np.float32)
    gripper = np.asarray(obs.get("gripper_pos", np.array([1.0])), dtype=np.float32)
    if arm.shape != (6,) or gripper.shape != (1,):
        raise RuntimeError(f"Unexpected follower observation shapes: arm={arm.shape}, gripper={gripper.shape}")
    return yam_policy.i2rt_arm_state_to_openpi(np.concatenate([arm, gripper]).astype(np.float32))


def pack_bimanual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = yam_policy.as_arm_state(left)
    right = yam_policy.as_arm_state(right)
    return np.concatenate([left, right]).astype(np.float32)


def split_bimanual(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (14,):
        raise ValueError(f"Expected 14D bimanual action, got {action.shape}")
    return action[:7], action[7:]


def clip_bimanual_delta(
    previous: np.ndarray,
    target: np.ndarray,
    *,
    max_arm_step_rad: float,
    max_gripper_step: float,
) -> np.ndarray:
    previous = np.asarray(previous, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if previous.shape != (14,) or target.shape != (14,):
        raise ValueError(f"Expected 14D previous and target, got {previous.shape} and {target.shape}")
    max_delta = np.full((14,), max_arm_step_rad, dtype=np.float32)
    max_delta[[6, 13]] = max_gripper_step
    return previous + np.clip(target - previous, -max_delta, max_delta)


@contextlib.contextmanager
def raw_mode_stdin() -> Iterator[None]:
    if not sys.stdin.isatty():
        yield
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key_nonblocking() -> str | None:
    if not sys.stdin.isatty():
        return None
    import select

    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if readable:
        return sys.stdin.read(1)
    return None
