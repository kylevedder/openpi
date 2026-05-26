from __future__ import annotations

import contextlib
import dataclasses
from itertools import pairwise
import json
from pathlib import Path
import shutil
import time
from typing import Any, Literal

import numpy as np
import tyro

from examples.yam_real import common
from examples.yam_real import process_runtime
from examples.yam_real import teleop

SYNC_BUTTON_INDEX = teleop.SYNC_BUTTON_INDEX
RECORD_BUTTON_INDEX = teleop.RECORD_BUTTON_INDEX
YamLeader = teleop.YamLeader
TeleopPair = teleop.TeleopPair


@dataclasses.dataclass
class Args:
    output_dir: Path = Path("yam_data/raw")
    episode_name: str | None = None
    task: str = "perform the demonstrated bimanual task"
    fps: float = 50.0
    camera_fps: float = 30.0
    camera_width: int = 640
    camera_height: int = 480
    camera_pixel_format: str = "MJPG"
    camera_startup_timeout_s: float = 5.0
    max_camera_age_s: float = 0.25
    skip_camera_mode_verify: bool = False
    max_duration_s: float | None = None
    gripper: Literal["crank_4310", "linear_3507", "linear_4310"] = "linear_4310"
    bilateral_kp: float = 0.2
    ee_mass: float | None = None
    use_gravity_comp: bool = True
    status_haptic_cue: bool = True
    status_cue_gain: float = 0.08
    record_button_debounce_s: float = 1.0
    record_only_when_synced: bool = True
    startup_check_only: bool = False
    diagnostics_jsonl: Path | None = None
    auto_start: bool = False
    exit_after_max_duration: bool = False
    record_loop_sleep_s: float = 0.005
    camera_ipc_ring_size: int = 16
    writer_queue_size: int = 16
    ipc_startup_timeout_s: float = 10.0
    ipc_step_timeout_s: float = 2.0
    writer_drain_timeout_s: float = 10.0


def _make_next_episode_dir(output_dir: Path, requested_name: str | None, episode_index: int) -> Path:
    if requested_name is None:
        base_name = time.strftime("episode_%Y%m%d_%H%M%S")
    elif episode_index == 1:
        base_name = requested_name
    else:
        base_name = f"{requested_name}_{episode_index:03d}"

    for suffix in range(1000):
        episode_name = base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        try:
            return common.make_episode_dir(output_dir, episode_name)
        except FileExistsError:
            continue

    raise FileExistsError(f"Could not create a unique episode directory under {output_dir}")


def _print_recording_banner(title: str, details: list[str] | None = None) -> None:
    width = 72
    inner_width = width - 4
    print()
    print("#" * width)
    print(f"# {title.center(inner_width)} #")
    if details:
        print(f"# {' '.center(inner_width)} #")
        for detail in details:
            line = f"{detail[: inner_width - 3]}..." if len(detail) > inner_width else detail
            print(f"# {line.center(inner_width)} #")
    print("#" * width)
    print(flush=True)


class RecorderDiagnostics:
    def __init__(self, path: Path | None, *, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._records: list[dict[str, Any]] = []
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w")

    def write_context(self, payload: dict[str, Any]) -> None:
        self._write({"type": "context", **payload})

    def record_loop(self, record: dict[str, Any]) -> None:
        record = {"type": "loop", **record}
        self._records.append(record)
        self._write(record)

    def write_summary(self, *, episode_dir: Path | None = None, reason: str = "summary") -> dict[str, Any]:
        summary = self.summarize_records(
            self._records,
            fps=self.fps,
            episode_dir=str(episode_dir) if episode_dir is not None else None,
        )
        self._write(
            {
                "type": "summary",
                "reason": reason,
                "episode_dir": str(episode_dir) if episode_dir is not None else None,
                "summary": summary,
            }
        )
        return summary

    def close(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None

    @staticmethod
    def summarize_records(records: list[dict[str, Any]], *, fps: float, episode_dir: str | None = None) -> dict[str, Any]:
        selected = [
            record
            for record in records
            if record.get("type") == "loop" and (episode_dir is None or record.get("episode_dir") == episode_dir)
        ]
        recorded = [record for record in selected if record.get("recorded")]
        row_timestamps_ns = [int(record["timestamp_ns"]) for record in recorded if record.get("timestamp_ns") is not None]
        row_dt_ms = [(b - a) / 1_000_000.0 for a, b in pairwise(row_timestamps_ns)]
        stage_stats = _stage_stats(selected)
        camera_summary = _camera_summary(recorded)
        row_stats = _duration_stats(row_dt_ms)
        row_fps = 0.0 if row_stats["median_ms"] <= 0 else 1000.0 / float(row_stats["median_ms"])
        return {
            "loop_count": len(selected),
            "recorded_rows": len(recorded),
            "requested_fps": fps,
            "row_dt_ms": row_stats,
            "row_fps": row_fps,
            "camera": camera_summary,
            "stages_ms": stage_stats,
            "bottleneck": _classify_bottleneck(stage_stats, fps=fps),
        }

    def _write(self, payload: dict[str, Any]) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
        self._file.flush()


def _stage_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = {}
    for record in records:
        for stage, value in record.get("durations_ms", {}).items():
            values.setdefault(stage, []).append(float(value))
    return {stage: _duration_stats(stage_values) for stage, stage_values in sorted(values.items())}


def _camera_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for camera_name in common.CAMERA_NAMES:
        rows = 0
        sequence_ids = []
        timestamps_ns = []
        ages_ms = []
        for record in records:
            camera_sequence_ids = record.get("camera_sequence_ids", {})
            camera_timestamps_ns = record.get("camera_capture_timestamp_ns", {})
            camera_age_ms = record.get("camera_age_ms", {})
            if camera_name not in camera_sequence_ids:
                continue
            rows += 1
            sequence_ids.append(int(camera_sequence_ids[camera_name]))
            if camera_name in camera_timestamps_ns:
                timestamps_ns.append(int(camera_timestamps_ns[camera_name]))
            if camera_name in camera_age_ms:
                ages_ms.append(float(camera_age_ms[camera_name]))
        unique_sequences = _unique_preserve_order(sequence_ids)
        unique_timestamps_ns = _unique_preserve_order(timestamps_ns)
        dt_ms = [(b - a) / 1_000_000.0 for a, b in pairwise(unique_timestamps_ns)]
        dt_stats = _duration_stats(dt_ms)
        summary[camera_name] = {
            "rows": rows,
            "unique_frames": len(unique_sequences),
            "reused_rows": max(0, rows - len(unique_sequences)),
            "reuse_ratio": 0.0 if rows == 0 else max(0, rows - len(unique_sequences)) / rows,
            "dt_ms": dt_stats,
            "fps": 0.0 if dt_stats["median_ms"] <= 0 else 1000.0 / float(dt_stats["median_ms"]),
            "age_ms": _duration_stats(ages_ms),
        }
    return summary


def _classify_bottleneck(stage_stats: dict[str, dict[str, float | int]], *, fps: float) -> str:
    if not stage_stats:
        return "no_samples"
    budget_ms = 1000.0 / fps if fps > 0 else float("inf")
    total_p95_ms = float(stage_stats.get("total_loop", stage_stats.get("total", {})).get("p95_ms", 0.0))
    if total_p95_ms <= budget_ms:
        return "none_detected"

    candidates = {
        stage: stats
        for stage, stats in stage_stats.items()
        if stage not in {"total", "total_loop", "sleep"} and stats.get("count", 0)
    }
    if not candidates:
        return "loop_over_budget"
    stage, stats = max(candidates.items(), key=lambda item: float(item[1].get("p95_ms", 0.0)))
    return f"{stage}_p95_{float(stats.get('p95_ms', 0.0)):.1f}ms"


def _duration_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.quantile(arr, 0.95)),
        "max_ms": float(np.max(arr)),
        "min_ms": float(np.min(arr)),
    }


def _unique_preserve_order(values: list[int]) -> list[int]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _elapsed_ms(start: float) -> float:
    return 1000.0 * (time.perf_counter() - start)


def main(args: Args) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.record_button_debounce_s < 0:
        raise ValueError("--record-button-debounce-s must be non-negative")
    if args.record_loop_sleep_s < 0:
        raise ValueError("--record-loop-sleep-s must be non-negative")
    if args.camera_ipc_ring_size <= 1:
        raise ValueError("--camera-ipc-ring-size must be greater than 1")
    if args.writer_queue_size <= 0:
        raise ValueError("--writer-queue-size must be positive")
    if args.ipc_startup_timeout_s <= 0:
        raise ValueError("--ipc-startup-timeout-s must be positive")
    if args.ipc_step_timeout_s <= 0:
        raise ValueError("--ipc-step-timeout-s must be positive")
    if args.writer_drain_timeout_s <= 0:
        raise ValueError("--writer-drain-timeout-s must be positive")

    print(f"Output root: {args.output_dir}")
    print("Controls: leader top buttons toggle sync per side; leader bottom button or 'r' starts/stops and saves.")
    print("'q' saves any active recording and exits.")
    print("Recording haptic cue: one leader pulse=start, two leader pulses=stop.")
    print("Hardware backend: multiprocessing workers (cameras, teleop, writer).")

    episode_dir: Path | None = None
    recording = False
    start_time = None
    frame_idx = 0
    next_record_t = time.monotonic()
    last_record_timestamp_ns: int | None = None
    last_record_button = 0.0
    last_record_button_toggle_t = -float("inf")
    episode_index = 0
    camera_actual_modes: dict[str, dict] = {}
    diagnostics = RecorderDiagnostics(args.diagnostics_jsonl, fps=args.fps)
    camera_config = common.CameraConfig(
        frame_size=(args.camera_width, args.camera_height),
        fps=int(args.camera_fps),
        pixel_format=args.camera_pixel_format,
        verify_mode=not args.skip_camera_mode_verify,
    )
    runtime = process_runtime.YamMultiprocessRuntime(
        camera_config=camera_config,
        gripper=args.gripper,
        bilateral_kp=args.bilateral_kp,
        ee_mass=args.ee_mass,
        use_gravity_comp=args.use_gravity_comp,
        camera_ring_size=args.camera_ipc_ring_size,
        writer_queue_size=args.writer_queue_size,
        startup_timeout_s=max(args.ipc_startup_timeout_s, args.camera_startup_timeout_s),
        step_timeout_s=args.ipc_step_timeout_s,
        writer_drain_timeout_s=args.writer_drain_timeout_s,
    )
    record_period_s = 1.0 / args.fps

    def reset_episode_buffers() -> None:
        nonlocal frame_idx, start_time, next_record_t, last_record_timestamp_ns
        frame_idx = 0
        start_time = None
        next_record_t = time.monotonic()
        last_record_timestamp_ns = None

    def save_episode() -> bool:
        nonlocal frame_idx
        if episode_dir is None or frame_idx == 0:
            return False

        writer_summary = runtime.stop_episode()
        num_steps = int(writer_summary.get("written_rows", frame_idx))
        print(f"Recorded {num_steps} rows to MCAP episode {episode_dir}")
        diagnostics.write_summary(episode_dir=episode_dir, reason="episode_saved")
        return True

    def start_episode() -> None:
        nonlocal episode_dir, episode_index, recording, start_time, next_record_t
        if recording:
            return

        episode_index += 1
        episode_dir = _make_next_episode_dir(args.output_dir, args.episode_name, episode_index)
        runtime.start_episode(
            episode_dir=episode_dir,
            task=args.task,
            fps=args.fps,
            camera_fps=args.camera_fps,
            image_width=args.camera_width,
            image_height=args.camera_height,
            camera_config={
                "requested": camera_config.as_manifest(),
                "actual_modes": camera_actual_modes,
                "camera_paths": common.CAMERA_PATHS,
            },
        )
        reset_episode_buffers()
        recording = True
        start_time = time.monotonic()
        next_record_t = start_time
        print(f"recording={recording}")
        print(f"Recording directory: {episode_dir}")
        _print_recording_banner(
            "RECORDING STARTED",
            [
                f"Episode: {episode_dir}",
                f"FPS: {args.fps:g}",
            ],
        )
        if args.status_haptic_cue:
            runtime.emit_recording_status(recording=True, gain=args.status_cue_gain)

    def stop_episode() -> None:
        nonlocal episode_dir, recording
        was_recording = recording
        recording = False
        if was_recording:
            print(f"recording={recording}")
            _print_recording_banner(
                "RECORDING STOPPED",
                [
                    f"Rows captured: {frame_idx}",
                    "Saving episode now",
                ],
            )
            if args.status_haptic_cue:
                runtime.emit_recording_status(recording=False, gain=args.status_cue_gain)

        saved = save_episode()
        if saved:
            print("Ready for next episode.")
        elif episode_dir is not None:
            print("No frames recorded; discarding empty episode directory.")
            with contextlib.suppress(Exception):
                runtime.stop_episode()
            shutil.rmtree(episode_dir, ignore_errors=True)

        episode_dir = None
        reset_episode_buffers()

    def toggle_recording() -> None:
        if recording:
            stop_episode()
        else:
            start_episode()

    try:
        with runtime, common.raw_mode_stdin():
            camera_actual_modes = runtime.camera_actual_modes
            print("Camera capture mode: multiprocessing_shared_memory")
            print(f"Camera requested mode: {camera_config.as_manifest()}")
            print(f"Camera actual modes: {camera_actual_modes}")
            print(f"Worker PIDs: {runtime.process_ids}")
            diagnostics.write_context(
                {
                    "args": dataclasses.asdict(args),
                    "camera_requested": camera_config.as_manifest(),
                    "camera_actual_modes": camera_actual_modes,
                    "worker_pids": runtime.process_ids,
                    "hardware_backend": "multiprocessing",
                }
            )

            if args.startup_check_only:
                teleop_step = runtime.step_teleop()
                camera_snapshot = runtime.camera_snapshot(max_age_s=args.max_camera_age_s)
                state_l = teleop_step["left"].state
                state_r = teleop_step["right"].state
                leader_l_buttons = teleop_step["left"].buttons
                leader_r_buttons = teleop_step["right"].buttons
                camera_shapes = {
                    name: list(spec.frame_shape) for name, spec in runtime.cameras.ring_specs.items()
                }
                camera_properties = camera_actual_modes
                print("Startup check OK.")
                print(f"left follower state shape={state_l.shape}, right follower state shape={state_r.shape}")
                print(f"left leader buttons={leader_l_buttons.tolist()}")
                print(f"right leader buttons={leader_r_buttons.tolist()}")
                print(f"camera shapes={camera_shapes}")
                print(f"camera refs={ {name: ref.as_payload() for name, ref in camera_snapshot.refs.items()} }")
                print(f"camera properties={camera_properties}")
                return

            if args.auto_start:
                start_episode()
            while True:
                loop_start = time.perf_counter()
                durations_ms: dict[str, float] = {}
                recorded = False
                row_dt_s = None
                timestamp_ns = None
                camera_sequence_ids: dict[str, int] = {}
                camera_capture_timestamp_ns: dict[str, int] = {}
                camera_age_ms: dict[str, float] = {}
                writer_queue_depth = None
                scheduler_lag_ms = 0.0
                skipped_periods = 0
                stop_due_to_max_duration = False
                current_episode_dir = str(episode_dir) if episode_dir is not None else None
                runtime.poll()
                key = common.read_key_nonblocking()
                if key == "q":
                    if recording or frame_idx:
                        stop_episode()
                    break
                if key == "r":
                    toggle_recording()

                step_start = time.perf_counter()
                teleop_step = runtime.step_teleop()
                durations_ms["teleop_ipc_step"] = _elapsed_ms(step_start)
                left_step = teleop_step["left"]
                right_step = teleop_step["right"]
                durations_ms["left_teleop_ipc_latency"] = left_step.ipc_latency_ms
                durations_ms["right_teleop_ipc_latency"] = right_step.ipc_latency_ms
                durations_ms.update({f"left_{name}": value for name, value in left_step.timings_ms.items()})
                durations_ms.update({f"right_{name}": value for name, value in right_step.timings_ms.items()})
                state_l = left_step.state
                action_l = left_step.action
                buttons_l = left_step.buttons
                state_r = right_step.state
                action_r = right_step.action
                buttons_r = right_step.buttons
                now = time.monotonic()
                record_button = max(
                    float(buttons_l[RECORD_BUTTON_INDEX]) if buttons_l.size > RECORD_BUTTON_INDEX else 0.0,
                    float(buttons_r[RECORD_BUTTON_INDEX]) if buttons_r.size > RECORD_BUTTON_INDEX else 0.0,
                )
                if record_button > 0.5 and last_record_button <= 0.5:
                    if now - last_record_button_toggle_t >= args.record_button_debounce_s:
                        toggle_recording()
                        last_record_button_toggle_t = now
                    else:
                        remaining_s = args.record_button_debounce_s - (now - last_record_button_toggle_t)
                        print(f"Ignoring record button bounce ({remaining_s:.2f}s debounce remaining).")
                last_record_button = record_button
                if recording and now >= next_record_t + record_period_s:
                    skipped_periods = int((now - next_record_t) // record_period_s)
                    next_record_t += skipped_periods * record_period_s
                should_record = recording and now >= next_record_t
                if args.record_only_when_synced and not (left_step.synchronized and right_step.synchronized):
                    should_record = False

                if should_record:
                    if episode_dir is None:
                        raise RuntimeError("Recording is active without an episode directory.")
                    scheduler_lag_ms = 1000.0 * max(0.0, now - next_record_t)
                    snapshot_start = time.perf_counter()
                    snapshot = runtime.camera_snapshot(max_age_s=args.max_camera_age_s)
                    durations_ms["camera_snapshot"] = _elapsed_ms(snapshot_start)
                    for camera_name, metadata in snapshot.metadata.items():
                        if "sequence_index" in metadata:
                            camera_sequence_ids[camera_name] = int(metadata["sequence_index"])
                        if "capture_timestamp_ns" in metadata:
                            camera_capture_timestamp_ns[camera_name] = int(metadata["capture_timestamp_ns"])
                        if "age_s" in metadata:
                            camera_age_ms[camera_name] = float(metadata["age_s"]) * 1000.0
                    write_start = time.perf_counter()
                    timestamp_ns = time.time_ns()
                    writer_queue_depth = runtime.write_row(
                        frame_index=frame_idx,
                        timestamp_ns=timestamp_ns,
                        state=common.pack_bimanual(state_l, state_r),
                        action=common.pack_bimanual(action_l, action_r),
                        camera_snapshot=snapshot,
                    )
                    durations_ms["writer_enqueue"] = _elapsed_ms(write_start)
                    if last_record_timestamp_ns is not None:
                        row_dt_s = (timestamp_ns - last_record_timestamp_ns) / 1_000_000_000.0
                    last_record_timestamp_ns = timestamp_ns
                    frame_idx += 1
                    next_record_t += record_period_s
                    recorded = True

                if (
                    args.max_duration_s is not None
                    and start_time is not None
                    and time.monotonic() - start_time >= args.max_duration_s
                    and recording
                ):
                    stop_due_to_max_duration = True

                sleep_start = time.perf_counter()
                sleep_s = args.record_loop_sleep_s
                if recording:
                    sleep_s = min(sleep_s, max(0.0, next_record_t - time.monotonic()))
                if sleep_s > 0:
                    time.sleep(sleep_s)
                durations_ms["sleep"] = _elapsed_ms(sleep_start)
                durations_ms["total_loop"] = _elapsed_ms(loop_start)
                current_episode_dir = str(episode_dir) if episode_dir is not None else current_episode_dir
                diagnostics.record_loop(
                    {
                        "episode_dir": current_episode_dir,
                        "frame_idx": frame_idx - 1 if recorded else None,
                        "recorded": recorded,
                        "recording": recording,
                        "synchronized": dict(runtime.teleop.synchronized),
                        "scheduler_lag_ms": scheduler_lag_ms,
                        "skipped_periods": skipped_periods,
                        "durations_ms": durations_ms,
                        "row_dt_s": row_dt_s,
                        "timestamp_ns": timestamp_ns,
                        "camera_sequence_ids": camera_sequence_ids,
                        "camera_capture_timestamp_ns": camera_capture_timestamp_ns,
                        "camera_age_ms": camera_age_ms,
                        "writer_queue_depth": writer_queue_depth,
                        "worker_pids": runtime.process_ids,
                    }
                )
                if stop_due_to_max_duration:
                    stop_episode()
                    if args.exit_after_max_duration:
                        break
    except process_runtime.WorkerRuntimeError as exc:
        print(f"Worker failure; fail-closing recorder: {exc}")
        if recording or frame_idx:
            with contextlib.suppress(Exception):
                stop_episode()
        raise
    except KeyboardInterrupt:
        print("Interrupted; saving recorded frames before exit.")
        if recording or frame_idx:
            stop_episode()
    finally:
        runtime.close()
        diagnostics.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
