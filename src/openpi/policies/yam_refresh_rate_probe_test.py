from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from examples.yam_real import refresh_rate_probe


class FakeClock:
    def __init__(self) -> None:
        self.time_s = 100.0

    def now(self) -> float:
        return self.time_s

    def advance(self, seconds: float) -> str:
        self.time_s += seconds
        return "ok"


class FakeDelayedDevice:
    def __init__(self, clock: FakeClock, delay_s: float) -> None:
        self.clock = clock
        self.delay_s = delay_s

    def read(self) -> str:
        return self.clock.advance(self.delay_s)


def test_measure_stage_sequence_supports_fake_delayed_camera_and_robot() -> None:
    clock = FakeClock()
    robot = FakeDelayedDevice(clock, 0.007)
    camera = FakeDelayedDevice(clock, 0.018)
    image_writer = FakeDelayedDevice(clock, 0.011)

    durations_ms, outputs = refresh_rate_probe.measure_stage_sequence(
        [
            ("fake_robot", robot.read),
            ("fake_camera", camera.read),
            ("fake_save", image_writer.read),
        ],
        clock=clock.now,
    )

    assert outputs == {"fake_robot": "ok", "fake_camera": "ok", "fake_save": "ok"}
    assert durations_ms["fake_robot"] == pytest.approx(7.0)
    assert durations_ms["fake_camera"] == pytest.approx(18.0)
    assert durations_ms["fake_save"] == pytest.approx(11.0)
    assert durations_ms["total"] == pytest.approx(36.0)


def test_analyzer_flags_slow_single_camera_as_dominant_bottleneck() -> None:
    samples = [
        _sample("camera-single", "cam_high", 60.0),
        _sample("camera-single", "cam_high", 60.0),
        _sample("camera-single", "cam_left_wrist", 10.0),
        _sample("camera-single", "cam_right_wrist", 10.0),
        _sample("image-save", "all", 8.0),
    ]

    summary = refresh_rate_probe.analyze_samples(samples, fps=50.0)

    assert summary["dominant_bottleneck"] == "single_camera_rate"
    assert any(finding["kind"] == "single_camera_rate" for finding in summary["findings"])
    assert not summary["groups"]["camera-single/cam_high"]["meets_20hz_p95"]


def test_analyzer_flags_image_save_when_camera_reads_are_fast() -> None:
    samples = [
        _sample("camera-single", "cam_high", 10.0),
        _sample("camera-single", "cam_left_wrist", 10.0),
        _sample("camera-single", "cam_right_wrist", 10.0),
        _sample("camera-all-sequential", "all", 18.0),
        _sample("image-save", "all", 31.0),
        _sample("image-save", "all", 32.0),
    ]

    summary = refresh_rate_probe.analyze_samples(samples, fps=50.0)

    assert summary["dominant_bottleneck"] == "image_save_budget"
    assert any(finding["kind"] == "image_save_budget" for finding in summary["findings"])
    assert not summary["groups"]["image-save/all"]["meets_50hz_p95"]


def test_analyzer_flags_record_loop_save_delta() -> None:
    samples = [
        _sample("record-loop-dry", "robots_cameras_no_save", 14.0),
        _sample("record-loop-dry", "robots_cameras_no_save", 15.0),
        _sample("record-loop-dry", "robots_cameras_scratch_save", 34.0),
        _sample("record-loop-dry", "robots_cameras_scratch_save", 36.0),
    ]

    summary = refresh_rate_probe.analyze_samples(samples, fps=50.0)

    assert summary["dominant_bottleneck"] == "record_loop_save_delta"
    assert any(finding["kind"] == "record_loop_save_delta" for finding in summary["findings"])


def _sample(mode: str, target: str, total_ms: float) -> dict:
    return {
        "type": "sample",
        "mode": mode,
        "target": target,
        "sample_index": 0,
        "durations_ms": {"total": total_ms},
    }
