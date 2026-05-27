from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from examples.yam_real import common
from examples.yam_real import record_episode


def test_record_episode_defaults_match_standard_mcap_record_command() -> None:
    args = record_episode.Args()

    assert args.output_dir == Path("yam_data/raw")
    assert args.fps == 50.0
    assert args.camera_fps == 30.0
    assert args.camera_width == 640
    assert args.camera_height == 480
    assert args.camera_pixel_format == "MJPG"
    assert args.use_gravity_comp is True


def test_linuxpy_camera_rejects_unsupported_50hz_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = FakeCameraEnv(max_fps=30.0)
    fake_env.install(monkeypatch)

    config = common.CameraConfig(frame_size=(640, 480), fps=50, pixel_format="MJPG", verify_mode=True)

    with pytest.raises(RuntimeError, match="640x480@50fps"), common.LinuxpyV4L2Camera(
        "cam_high", "/dev/fake", config
    ):
        pass

    assert fake_env.devices[0].closed


def test_linuxpy_camera_sets_mjpg_mode_and_returns_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = FakeCameraEnv(max_fps=30.0)
    fake_env.install(monkeypatch)

    config = common.CameraConfig(frame_size=(640, 480), fps=30, pixel_format="MJPG", verify_mode=True)
    with common.LinuxpyV4L2Camera("cam_high", "/dev/fake", config) as camera:
        frame = camera.read()

    assert fake_env.captures[0].format_call == (640, 480, "MJPG")
    assert fake_env.captures[0].fps_call == 30
    assert frame.frame.shape == (480, 640, 3)
    assert frame.frame.dtype == np.uint8
    assert frame.sequence_index == 0
    np.testing.assert_allclose(frame.frame[0, 0], np.asarray([0, 20, 40], dtype=np.uint8), atol=1)


def test_linuxpy_camera_accepts_method_frame_sizes_with_fps_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = FakeCameraEnv(max_fps=30.0, linuxpy_024_info=True)
    fake_env.install(monkeypatch)

    config = common.CameraConfig(frame_size=(640, 480), fps=30, pixel_format="MJPG", verify_mode=True)
    with common.LinuxpyV4L2Camera("cam_high", "/dev/fake", config) as camera:
        frame = camera.read()

    assert frame.frame.shape == (480, 640, 3)
    assert fake_env.captures[0].format_call == (640, 480, "MJPG")


def test_linuxpy_camera_capture_properties_include_actual_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = FakeCameraEnv(max_fps=30.0)
    fake_env.install(monkeypatch)

    config = common.CameraConfig(frame_size=(640, 480), fps=30, pixel_format="MJPG", verify_mode=True)
    with common.LinuxpyV4L2Camera("cam_high", "/dev/fake", config) as camera:
        properties = camera.capture_properties()

    assert properties["bus_info"] == "usb-fake"
    assert properties["driver"] == "fake"
    assert properties["frame_sizes"][0]["width"] == 640
    assert properties["actual_fps"] == 30.0
    assert properties["actual_format"]["width"] == 640
    assert properties["actual_format"]["height"] == 480
    assert properties["actual_format"]["pixel_format"] == "MJPG"


def test_async_camera_set_returns_latest_frame_without_waiting_for_next_read(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = FakeCameraEnv(max_fps=30.0, read_delay_s=0.03)
    fake_env.install(monkeypatch)

    config = common.CameraConfig(fps=30, verify_mode=False)
    with common.AsyncCameraSet(paths={"cam_high": "/dev/fake"}, config=config, startup_timeout_s=1.0) as cameras:
        first = cameras.snapshot(max_age_s=1.0)
        second = cameras.snapshot(max_age_s=1.0)

    assert first.metadata["cam_high"]["sequence_index"] == second.metadata["cam_high"]["sequence_index"]


def test_teleop_pair_unsynced_action_equals_follower_state() -> None:
    follower = FakeFollower()
    leader = FakeLeader()
    pair = record_episode.TeleopPair("left", leader, follower, bilateral_kp=0.2)

    state, action, buttons = pair.step()

    np.testing.assert_allclose(action, state)
    assert buttons.tolist() == [0.0, 0.0]
    assert follower.commands == []


class FakeCameraEnv:
    def __init__(self, *, max_fps: float, read_delay_s: float = 0.0, linuxpy_024_info: bool = False) -> None:
        self.max_fps = max_fps
        self.read_delay_s = read_delay_s
        self.linuxpy_024_info = linuxpy_024_info
        self.devices: list[FakeDevice] = []
        self.captures: list[FakeVideoCapture] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = self

        class DeviceFactory(FakeDevice):
            def __init__(self, path: str) -> None:
                super().__init__(
                    path,
                    max_fps=env.max_fps,
                    read_delay_s=env.read_delay_s,
                    linuxpy_024_info=env.linuxpy_024_info,
                )
                env.devices.append(self)

        class VideoCaptureFactory(FakeVideoCapture):
            def __init__(self, device: FakeDevice) -> None:
                super().__init__(device)
                env.captures.append(self)

        monkeypatch.setattr(common.video_device, "Device", DeviceFactory)
        monkeypatch.setattr(common.video_device, "VideoCapture", VideoCaptureFactory)


class FakeFrameSize:
    width = 640
    height = 480
    min_fps = 1.0

    def __init__(self, max_fps: float) -> None:
        self.max_fps = max_fps


class FakeInfo:
    bus_info = "usb-fake"
    card = "FakeCam"
    driver = "fake"

    def __init__(self, max_fps: float) -> None:
        self.frame_sizes = [FakeFrameSize(max_fps)]


class FakeSize:
    width = 640
    height = 480


class FakeLinuxpy024FrameSize:
    pixel_format = common.video_device.PixelFormat.MJPEG
    info = FakeSize()


class FakeFrameInterval:
    pixel_format = common.video_device.PixelFormat.MJPEG
    width = 640
    height = 480
    min_fps = 1.0

    def __init__(self, max_fps: float) -> None:
        self.max_fps = max_fps


class FakeActualFormat:
    def __init__(self, width: int, height: int, pixel_format: str) -> None:
        self.width = width
        self.height = height
        self.pixel_format = pixel_format


class FakeLinuxpy024Info:
    bus_info = "usb-fake"
    card = "FakeCam"
    driver = "fake"

    def __init__(self, max_fps: float) -> None:
        self.max_fps = max_fps

    def frame_sizes(self) -> list[FakeLinuxpy024FrameSize]:
        return [FakeLinuxpy024FrameSize()]

    def fps_intervals(self, pixel_format, width: int, height: int) -> list[FakeFrameInterval]:
        assert pixel_format == common.video_device.PixelFormat.MJPEG
        assert (width, height) == (640, 480)
        return [FakeFrameInterval(self.max_fps)]


class FakeFrame:
    pixel_format = common.video_device.PixelFormat.MJPEG

    def __init__(self, index: int) -> None:
        self.timestamp = time.monotonic()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 0] = index
        image[:, :, 1] = 20
        image[:, :, 2] = 40
        self.data = common.simplejpeg.encode_jpeg(image, quality=95, colorspace="rgb")


class FakeDevice:
    def __init__(self, path: str, *, max_fps: float, read_delay_s: float, linuxpy_024_info: bool) -> None:
        self.path = path
        self.info = FakeLinuxpy024Info(max_fps) if linuxpy_024_info else FakeInfo(max_fps)
        self.read_delay_s = read_delay_s
        self.opened = False
        self.closed = False
        self.frame_index = 0
        self.actual_format: FakeActualFormat | None = None
        self.actual_fps: float | None = None

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def get_format(self):
        return self.actual_format

    def get_fps(self):
        return self.actual_fps

    def __iter__(self):
        while not self.closed:
            if self.read_delay_s:
                time.sleep(self.read_delay_s)
            frame = FakeFrame(self.frame_index)
            self.frame_index += 1
            yield frame


class FakeVideoCapture:
    def __init__(self, device: FakeDevice) -> None:
        self.device = device
        self.format_call: tuple[int, int, str] | None = None
        self.fps_call: float | None = None

    def set_format(self, width: int, height: int, pixel_format: str = "MJPG") -> None:
        self.format_call = (width, height, pixel_format)
        self.device.actual_format = FakeActualFormat(width, height, pixel_format)

    def set_fps(self, fps: float) -> None:
        self.fps_call = fps
        self.device.actual_fps = fps


class FakeFollower:
    def __init__(self) -> None:
        self.commands: list[np.ndarray] = []

    def get_observations(self) -> dict[str, np.ndarray]:
        return {
            "joint_pos": np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32),
            "gripper_pos": np.asarray([0.25], dtype=np.float32),
        }

    def command_joint_pos(self, command: np.ndarray) -> None:
        self.commands.append(np.asarray(command, dtype=np.float32))


class FakeLeader:
    def get_info(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((7,), dtype=np.float32), np.zeros((2,), dtype=np.float32)

    def command_arm_pos(self, qpos_6d: np.ndarray) -> None:
        del qpos_6d

    def set_bilateral(self, *, enabled: bool, gain: float) -> None:
        del enabled, gain

    def pulse_haptic(self, *, pattern_s: tuple[float, ...], gain: float) -> None:
        del pattern_s, gain
