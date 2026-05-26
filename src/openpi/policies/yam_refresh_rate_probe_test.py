import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from examples.yam_real import process_runtime
from examples.yam_real import record_episode
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
        _sample("record-loop-dry", "robots_cameras", 14.0),
        _sample("record-loop-dry", "robots_cameras", 15.0),
        _sample("record-loop-dry", "robots_cameras_mcap_write", 34.0),
        _sample("record-loop-dry", "robots_cameras_mcap_write", 36.0),
    ]

    summary = refresh_rate_probe.analyze_samples(samples, fps=50.0)

    assert summary["dominant_bottleneck"] == "record_loop_mcap_write_delta"
    assert any(finding["kind"] == "record_loop_mcap_write_delta" for finding in summary["findings"])


def test_analyzer_flags_process_record_loop_budget() -> None:
    samples = [
        _sample("process-record-loop-dry", "full", 24.0),
        _sample("process-record-loop-dry", "full", 28.0),
    ]

    summary = refresh_rate_probe.analyze_samples(samples, fps=50.0)

    assert summary["dominant_bottleneck"] == "process_record_loop_budget"
    assert any(finding["kind"] == "process_record_loop_budget" for finding in summary["findings"])


def test_recorder_diagnostics_classifies_synthetic_loop_bottleneck() -> None:
    records = [
        {
            "type": "loop",
            "recorded": True,
            "timestamp_ns": 1_000_000_000,
            "durations_ms": {"left_teleop_step": 4.0, "mcap_write_step": 6.0, "total_loop": 18.0},
        },
        {
            "type": "loop",
            "recorded": True,
            "timestamp_ns": 1_020_000_000,
            "durations_ms": {"left_teleop_step": 5.0, "mcap_write_step": 32.0, "total_loop": 42.0},
        },
    ]

    summary = record_episode.RecorderDiagnostics.summarize_records(records, fps=50.0)

    assert summary["recorded_rows"] == 2
    assert summary["row_fps"] == 50.0
    assert summary["bottleneck"].startswith("mcap_write_step")


def test_mcap_write_probe_writes_only_to_scratch_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "scratch" / "mcap-write"
    args = refresh_rate_probe.Args(
        duration_s=0.01,
        max_samples_per_mode=2,
        camera_width=64,
        camera_height=48,
        fps=50.0,
        camera_fps=30.0,
    )

    samples = refresh_rate_probe._probe_mcap_write(args, output_dir)  # noqa: SLF001

    assert any(sample["mode"] == "mcap-write" and sample["target"] == "cached_rgb" for sample in samples)
    assert output_dir.is_dir()
    assert all(path.is_relative_to(output_dir) for path in output_dir.rglob("*"))
    assert any(path.name.startswith("episode_part") for path in output_dir.iterdir())


def test_shared_camera_ring_detects_overwritten_slot() -> None:
    ctx = mp.get_context("spawn")
    ring = process_runtime.SharedCameraRing.create(
        camera_name="cam_high",
        ring_size=2,
        frame_shape=(4, 4, 3),
        dtype="uint8",
        lock=ctx.Lock(),
    )
    try:
        first_ref = process_runtime.CameraFrameRef(
            camera_name="cam_high",
            slot_index=0,
            sequence_index=0,
            capture_timestamp_ns=1,
            capture_time_s=0.0,
            monotonic_time_s=0.0,
        )
        second_ref = process_runtime.CameraFrameRef(
            camera_name="cam_high",
            slot_index=0,
            sequence_index=2,
            capture_timestamp_ns=2,
            capture_time_s=0.0,
            monotonic_time_s=0.0,
        )
        ring.write_frame(first_ref, np.ones((4, 4, 3), dtype=np.uint8))
        ring.write_frame(second_ref, np.zeros((4, 4, 3), dtype=np.uint8))

        with pytest.raises(RuntimeError, match="slot overwritten"):
            ring.read_frame(first_ref)
    finally:
        ring.close()
        ring.unlink()


def test_process_writer_probe_writes_only_to_scratch_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "scratch" / "process-writer"
    args = refresh_rate_probe.Args(
        duration_s=0.01,
        max_samples_per_mode=2,
        camera_width=64,
        camera_height=48,
        fps=50.0,
        camera_fps=30.0,
    )

    samples = refresh_rate_probe._probe_process_writer(args, output_dir)  # noqa: SLF001

    assert any(sample["mode"] == "process-writer" and sample["target"] == "cached_rgb" for sample in samples)
    assert output_dir.is_dir()
    assert all(path.is_relative_to(output_dir) for path in output_dir.rglob("*"))
    assert any(path.name.startswith("episode_part") for path in output_dir.iterdir())


def _sample(mode: str, target: str, total_ms: float) -> dict:
    return {
        "type": "sample",
        "mode": mode,
        "target": target,
        "sample_index": 0,
        "durations_ms": {"total": total_ms},
    }
