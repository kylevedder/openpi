from __future__ import annotations

from collections.abc import Iterator
import contextlib
import dataclasses
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2
import linuxpy.video.device as video_device
import numpy as np
import simplejpeg

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


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    frame_size: tuple[int, int] = (640, 480)
    fps: int = 30
    pixel_format: str = "MJPG"
    crop: tuple[int, int, int, int] | None = None
    resize: tuple[int, int] | None = None
    rotate_180: bool = False
    verify_mode: bool = True

    @property
    def width(self) -> int:
        return self.frame_size[0]

    @property
    def height(self) -> int:
        return self.frame_size[1]

    def as_manifest(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TimestampedFrame:
    name: str
    frame: np.ndarray
    capture_time_s: float
    monotonic_time_s: float
    capture_timestamp_ns: int
    sequence_index: int

    def metadata(self, *, now_monotonic_s: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        return {
            "capture_time_s": self.capture_time_s,
            "monotonic_time_s": self.monotonic_time_s,
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "sequence_index": self.sequence_index,
            "age_s": max(0.0, now - self.monotonic_time_s),
        }


@dataclasses.dataclass(frozen=True)
class CameraSnapshot:
    frames: dict[str, np.ndarray]
    metadata: dict[str, dict[str, Any]]


class LinuxpyV4L2Camera:
    """MonoPI-style V4L2 MJPG camera reader that returns RGB frames."""

    def __init__(self, name: str, path: str, config: CameraConfig | None = None) -> None:
        self.name = name
        self.path = path
        self.config = config or CameraConfig()
        self._device: video_device.Device | None = None
        self._stream: Any | None = None
        self._frame_count = 0
        self._last_good_frame_timestamp_ns: int | None = None
        self._last_good_frame_data: bytes | None = None

    def open(self) -> None:
        _validate_camera_config(self.config)
        device = video_device.Device(self.path)
        device.open()
        if device.info is None:
            device.close()
            raise RuntimeError(f"Could not open camera {self.name}: {self.path}")

        if self.config.verify_mode:
            issues = camera_mode_issues(camera_capabilities(device), self.config)
            if issues:
                device.close()
                detail = "; ".join(issues)
                raise RuntimeError(f"Camera {self.name} mode mismatch for {self.path}: {detail}")

        width, height = self.config.frame_size
        capture = video_device.VideoCapture(device)
        capture.set_format(width, height, pixel_format=self.config.pixel_format)
        capture.set_fps(self.config.fps)

        def _stream():
            yield from device

        self._device = device
        self._stream = _stream()
        self._frame_count = 0
        self._last_good_frame_timestamp_ns = None
        self._last_good_frame_data = None

    def read(self) -> TimestampedFrame:
        deadline = time.monotonic() + max(1.0, 2.0 / float(self.config.fps))
        while True:
            frame = self.read_next()
            if frame is not None:
                return frame
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Camera {self.name} produced only duplicate/bad frames")

    def read_next(self) -> TimestampedFrame | None:
        if self._stream is None:
            raise RuntimeError(f"Camera {self.name} is not open")
        frame = next(self._stream)
        pixel_format = getattr(frame, "pixel_format", None)
        if not _is_mjpeg_pixel_format(pixel_format):
            pixel_name = getattr(pixel_format, "name", str(pixel_format))
            raise RuntimeError(f"Unhandled camera pixel format for {self.name}: {pixel_name}")

        capture_timestamp_ns = _monotonic_timestamp_to_wall_ns(float(frame.timestamp))
        stuck_allowance_ns = 500_000_000 if self._frame_count < 10 else 100_000_000
        stuck_allowance_ns = max(stuck_allowance_ns, int(2_000_000_000 / float(self.config.fps)))
        if (
            self._last_good_frame_timestamp_ns is not None
            and capture_timestamp_ns - self._last_good_frame_timestamp_ns > stuck_allowance_ns
        ):
            raise RuntimeError(
                f"Camera {self.name} appears stuck: "
                f"{(capture_timestamp_ns - self._last_good_frame_timestamp_ns) / 1e9:.3f}s since last good frame"
            )

        frame_data = bytes(frame.data)
        try:
            simplejpeg.decode_jpeg_header(frame_data[32:])
            frame_data = frame_data[32:]
        except ValueError:
            pass

        if self._last_good_frame_data == frame_data:
            self._frame_count += 1
            return None

        try:
            image = simplejpeg.decode_jpeg(
                np.frombuffer(frame_data, dtype=np.uint8),
                colorspace="rgb",
                fastdct=True,
                fastupsample=True,
            )
        except ValueError:
            self._frame_count += 1
            return None

        image = _apply_image_transform(image, self.config)
        self._last_good_frame_data = frame_data
        self._last_good_frame_timestamp_ns = capture_timestamp_ns
        sequence_index = self._frame_count
        self._frame_count += 1
        return TimestampedFrame(
            name=self.name,
            frame=np.ascontiguousarray(image),
            capture_time_s=capture_timestamp_ns / 1_000_000_000.0,
            monotonic_time_s=time.monotonic(),
            capture_timestamp_ns=capture_timestamp_ns,
            sequence_index=sequence_index,
        )

    def capture_properties(self) -> dict[str, Any]:
        properties = {
            "path": self.path,
            "requested": self.config.as_manifest(),
            "backend": "linuxpy-v4l2",
        }
        if self._device is None:
            return properties
        info = self._device.info
        properties.update(
            {
                "bus_info": getattr(info, "bus_info", ""),
                "card": getattr(info, "card", ""),
                "driver": getattr(info, "driver", ""),
                "frame_sizes": _frame_sizes_as_dicts(info),
            }
        )
        fmt = _device_get_format(self._device)
        if fmt is not None:
            properties["actual_format"] = _object_public_attrs(fmt)
        fps = _device_get_fps(self._device)
        if fps is not None:
            properties["actual_fps"] = fps
        return properties

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.close()
        device = self._device
        self._device = None
        if device is not None:
            with contextlib.suppress(Exception):
                device.close()

    def __enter__(self) -> LinuxpyV4L2Camera:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class CameraSet:
    def __init__(
        self,
        paths: dict[str, str] | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        config: CameraConfig | None = None,
    ) -> None:
        self.paths = paths or CAMERA_PATHS
        if config is None:
            default_config = CameraConfig()
            config = dataclasses.replace(
                default_config,
                frame_size=(
                    width if width is not None else default_config.width,
                    height if height is not None else default_config.height,
                ),
            )
        elif width is not None or height is not None:
            config = dataclasses.replace(
                config,
                frame_size=(config.width if width is None else width, config.height if height is None else height),
            )
        self.config = config
        self._captures: dict[str, LinuxpyV4L2Camera] = {}
        try:
            for name, path in self.paths.items():
                camera = LinuxpyV4L2Camera(name, path, config)
                camera.open()
                self._captures[name] = camera
        except BaseException:
            self.close()
            raise

    def read(self) -> dict[str, np.ndarray]:
        frames = {}
        for name, camera in self._captures.items():
            frames[name] = camera.read().frame
        return frames

    def snapshot(self) -> CameraSnapshot:
        timestamped_frames = {name: camera.read() for name, camera in self._captures.items()}
        now = time.monotonic()
        return CameraSnapshot(
            frames={name: frame.frame for name, frame in timestamped_frames.items()},
            metadata={name: frame.metadata(now_monotonic_s=now) for name, frame in timestamped_frames.items()},
        )

    def close(self) -> None:
        for camera in self._captures.values():
            camera.close()
        self._captures.clear()

    def __enter__(self) -> CameraSet:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def capture_properties(self) -> dict[str, dict[str, Any]]:
        return {name: camera.capture_properties() for name, camera in self._captures.items()}


class AsyncCameraSet:
    """Continuously reads cameras so the control/record row loop can use latest frames."""

    def __init__(
        self,
        paths: dict[str, str] | None = None,
        *,
        config: CameraConfig | None = None,
        startup_timeout_s: float = 5.0,
    ) -> None:
        self.paths = paths or CAMERA_PATHS
        self.config = config or CameraConfig()
        self._captures: dict[str, LinuxpyV4L2Camera] = {}
        self._latest: dict[str, TimestampedFrame] = {}
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        try:
            for name, path in self.paths.items():
                camera = LinuxpyV4L2Camera(name, path, self.config)
                camera.open()
                self._captures[name] = camera
        except BaseException:
            self.close()
            raise
        for name, camera in self._captures.items():
            thread = threading.Thread(
                target=self._reader_loop, args=(name, camera), name=f"yam-camera-{name}", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        self.wait_for_frames(timeout_s=startup_timeout_s)

    def wait_for_frames(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._raise_if_error()
            with self._lock:
                if all(name in self._latest for name in self.paths):
                    return
            time.sleep(0.005)
        missing = sorted(set(self.paths) - set(self._latest))
        raise RuntimeError(f"Timed out waiting for initial camera frames: {missing}")

    def snapshot(self, *, max_age_s: float | None = None) -> CameraSnapshot:
        self._raise_if_error()
        now = time.monotonic()
        with self._lock:
            missing = sorted(set(self.paths) - set(self._latest))
            if missing:
                raise RuntimeError(f"Missing camera frames: {missing}")
            latest = dict(self._latest)
        stale = {
            name: now - frame.monotonic_time_s
            for name, frame in latest.items()
            if max_age_s is not None and now - frame.monotonic_time_s > max_age_s
        }
        if stale:
            details = ", ".join(f"{name}={age:.3f}s" for name, age in sorted(stale.items()))
            raise RuntimeError(f"Stale camera frame(s): {details}; max_age_s={max_age_s:.3f}")
        return CameraSnapshot(
            frames={name: latest[name].frame for name in self.paths},
            metadata={name: latest[name].metadata(now_monotonic_s=now) for name in self.paths},
        )

    def capture_properties(self) -> dict[str, dict[str, Any]]:
        return {name: camera.capture_properties() for name, camera in self._captures.items()}

    def close(self) -> None:
        self._stop.set()
        for camera in self._captures.values():
            camera.close()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()
        self._captures.clear()

    def __enter__(self) -> AsyncCameraSet:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _reader_loop(self, name: str, camera: LinuxpyV4L2Camera) -> None:
        try:
            while not self._stop.is_set():
                captured = camera.read()
                with self._lock:
                    self._latest[name] = captured
        except BaseException as exc:
            if self._stop.is_set():
                return
            with self._lock:
                if self._error is None:
                    self._error = exc
            self._stop.set()

    def _raise_if_error(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Camera reader failed: {error}") from error


def camera_capabilities(device: video_device.Device) -> dict[str, Any]:
    info = device.info
    return {
        "frame_sizes": _frame_sizes_as_dicts(info),
        "bus_info": getattr(info, "bus_info", ""),
        "card": getattr(info, "card", ""),
        "driver": getattr(info, "driver", ""),
    }


def camera_mode_issues(properties: dict[str, Any], config: CameraConfig) -> list[str]:
    issues = []
    frame_sizes = properties.get("frame_sizes", ())
    supported = [
        frame_size
        for frame_size in frame_sizes
        if frame_size.get("width") == config.width
        and frame_size.get("height") == config.height
        and _pixel_formats_match(frame_size.get("pixel_format", ""), config.pixel_format)
        and float(frame_size.get("min_fps", 0.0)) <= float(config.fps)
        and float(config.fps) <= float(frame_size.get("max_fps", float("inf")))
    ]
    if not supported:
        issues.append(f"mode {config.width}x{config.height}@{config.fps}fps not reported by V4L2 frame sizes")
    return issues


def fourcc_to_str(value: int) -> str:
    chars = "".join(chr((value >> (8 * idx)) & 0xFF) for idx in range(4))
    return chars if chars.strip("\x00") else str(value)


def _validate_camera_config(config: CameraConfig) -> None:
    if config.width <= 0 or config.height <= 0:
        raise ValueError(f"Camera width/height must be positive, got {config.width}x{config.height}")
    if config.fps <= 0:
        raise ValueError(f"Camera fps must be positive, got {config.fps}")
    if not config.pixel_format:
        raise ValueError("Camera pixel_format must be non-empty")


def _monotonic_timestamp_to_wall_ns(monotonic_timestamp_s: float) -> int:
    wall_now_ns = time.time_ns()
    monotonic_now_ns = time.monotonic_ns()
    return wall_now_ns + int(monotonic_timestamp_s * 1_000_000_000) - monotonic_now_ns


def _is_mjpeg_pixel_format(pixel_format: Any) -> bool:
    if pixel_format == video_device.PixelFormat.MJPEG:
        return True
    name = getattr(pixel_format, "name", "")
    value = getattr(pixel_format, "value", pixel_format)
    return str(name).upper() in {"MJPEG", "MJPG"} or str(value).upper() in {"MJPEG", "MJPG"}


def _apply_image_transform(image: np.ndarray, config: CameraConfig) -> np.ndarray:
    if config.crop:
        left, top, right, bottom = config.crop
        image = image[top:bottom, left:right, :]
    target_size = config.resize
    if target_size is not None and (image.shape[1], image.shape[0]) != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    if config.rotate_180:
        image = image[::-1, ::-1, :]
    return image


def _frame_sizes_as_dicts(info: Any) -> list[dict[str, Any]]:
    frame_sizes = getattr(info, "frame_sizes", ())
    if callable(frame_sizes):
        frame_sizes = frame_sizes()

    result = []
    for frame_size in frame_sizes or ():
        width = _frame_size_attr(frame_size, "width")
        height = _frame_size_attr(frame_size, "height")
        pixel_format = getattr(frame_size, "pixel_format", "")
        intervals = _fps_intervals(info, pixel_format, width, height)
        if intervals:
            result.extend(
                [
                    {
                        "width": int(getattr(interval, "width", width)),
                        "height": int(getattr(interval, "height", height)),
                        "min_fps": float(getattr(interval, "min_fps", 0.0)),
                        "max_fps": float(getattr(interval, "max_fps", 0.0)),
                        "pixel_format": _pixel_format_label(getattr(interval, "pixel_format", pixel_format)),
                    }
                    for interval in intervals
                ]
            )
        else:
            result.append(
                {
                    "width": int(width),
                    "height": int(height),
                    "min_fps": float(getattr(frame_size, "min_fps", 0.0)),
                    "max_fps": float(getattr(frame_size, "max_fps", 0.0)),
                    "pixel_format": _pixel_format_label(pixel_format),
                }
            )
    return result


def _frame_size_attr(frame_size: Any, name: str) -> int:
    value = getattr(frame_size, name, None)
    if value is None and hasattr(frame_size, "info"):
        value = getattr(frame_size.info, name, None)
    return int(value or 0)


def _fps_intervals(info: Any, pixel_format: Any, width: int, height: int) -> list[Any]:
    fps_intervals = getattr(info, "fps_intervals", None)
    if fps_intervals is None or not pixel_format or not width or not height:
        return []
    with contextlib.suppress(Exception):
        return list(fps_intervals(pixel_format, width, height))
    return []


def _pixel_format_label(pixel_format: Any) -> str:
    human_str = getattr(pixel_format, "human_str", None)
    if callable(human_str):
        with contextlib.suppress(Exception):
            return str(human_str())
    name = getattr(pixel_format, "name", "")
    if name:
        return str(name)
    return str(pixel_format)


def _pixel_formats_match(actual: Any, requested: str) -> bool:
    actual_normalized = str(actual).upper()
    if not actual_normalized:
        return True
    requested_normalized = requested.upper()
    aliases = {"MJPG": {"MJPG", "MJPEG"}, "MJPEG": {"MJPG", "MJPEG"}}
    return actual_normalized in aliases.get(requested_normalized, {requested_normalized})


def _device_get_format(device: Any) -> Any | None:
    get_format = getattr(device, "get_format", None)
    if get_format is None:
        return None
    buffer_type = getattr(getattr(video_device, "BufferType", None), "VIDEO_CAPTURE", None)
    arg_options = ((buffer_type,), ()) if buffer_type is not None else ((),)
    for args in arg_options:
        with contextlib.suppress(Exception):
            return get_format(*args)
    return None


def _device_get_fps(device: Any) -> float | None:
    get_fps = getattr(device, "get_fps", None)
    if get_fps is None:
        return None
    buffer_type = getattr(getattr(video_device, "BufferType", None), "VIDEO_CAPTURE", None)
    arg_options = ((buffer_type,), ()) if buffer_type is not None else ((),)
    for args in arg_options:
        with contextlib.suppress(Exception):
            return float(get_fps(*args))
    return None


def _object_public_attrs(value: Any) -> dict[str, Any]:
    attrs = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:
            continue
        encoded = _jsonable_camera_property(attr)
        if encoded is not None:
            attrs[name] = encoded
    return attrs


def _jsonable_camera_property(value: Any) -> Any | None:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, tuple | list):
        return [_jsonable_camera_property(item) for item in value]
    name = getattr(value, "name", None)
    if name:
        return str(name)
    human_str = getattr(value, "human_str", None)
    if callable(human_str):
        with contextlib.suppress(Exception):
            return str(human_str())
    return None


def make_episode_dir(root: Path, name: str | None = None) -> Path:
    episode_name = name or time.strftime("episode_%Y%m%d_%H%M%S")
    episode_dir = root / episode_name
    episode_dir.mkdir(parents=True, exist_ok=False)
    return episode_dir


def save_frames(episode_dir: Path, frame_index: int, frames: dict[str, np.ndarray]) -> dict[str, str]:
    rel_paths = {}
    for camera_name, frame in frames.items():
        rel_path = Path("images") / camera_name / f"{frame_index:06d}.jpg"
        out_path = episode_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(out_path), bgr):
            raise RuntimeError(f"Failed to write image {out_path}")
        rel_paths[camera_name] = str(rel_path)
    return rel_paths


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
