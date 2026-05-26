from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from examples.yam_real import common
from examples.yam_real import data_regression
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


def test_async_camera_set_returns_latest_frame_without_waiting_for_next_read(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = FakeCameraEnv(max_fps=30.0, read_delay_s=0.03)
    fake_env.install(monkeypatch)

    config = common.CameraConfig(fps=30, verify_mode=False)
    with common.AsyncCameraSet(paths={"cam_high": "/dev/fake"}, config=config, startup_timeout_s=1.0) as cameras:
        first = cameras.snapshot(max_age_s=1.0)
        second = cameras.snapshot(max_age_s=1.0)

    assert first.metadata["cam_high"]["sequence_index"] == second.metadata["cam_high"]["sequence_index"]


def test_camera_metadata_counts_unique_frames_and_reused_rows() -> None:
    metadata = np.asarray(
        [
            {"cam_high": {"sequence_index": 1, "capture_time_s": 10.0}},
            {"cam_high": {"sequence_index": 1, "capture_time_s": 10.0}},
            {"cam_high": {"sequence_index": 2, "capture_time_s": 10.05}},
        ],
        dtype=object,
    )

    timestamps, reuse = data_regression._camera_timestamps_from_npz_metadata(metadata, num_rows=3)  # noqa: SLF001

    np.testing.assert_allclose(timestamps["cam_high"], np.asarray([10.0, 10.05]))
    assert reuse["cam_high"] == {"rows": 3, "unique_frames": 2, "reused_rows": 1, "missing_rows": 0}


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

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

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

    def set_fps(self, fps: float) -> None:
        self.fps_call = fps
