from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tyro

from examples.yam_real import common
from examples.yam_real import mcap_episode


@dataclasses.dataclass(frozen=True)
class Args:
    episode_dir: Path
    expected_row_fps: float = 50.0
    expected_camera_fps: float = 30.0
    strict: bool = False
    output_dir: Path = Path("yam_data/logs/live_refresh")


def main(args: Args) -> None:
    summary = validate_episode(args)
    output_dir = args.output_dir / args.episode_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation_summary.json"
    md_path = output_dir / "validation_summary.md"
    contact_sheet_path = output_dir / "contact_sheet.jpg"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    md_path.write_text(_markdown_summary(summary, json_path=json_path, contact_sheet_path=contact_sheet_path))
    if summary.get("contact_sheet_written"):
        print(f"Contact sheet: {contact_sheet_path}")
    print(f"Validation summary: {json_path}")
    print(f"Strict pass: {summary['strict_pass']}")
    if args.strict and not summary["strict_pass"]:
        raise RuntimeError(f"Strict validation failed for {args.episode_dir}: {summary['strict_failures']}")


def validate_episode(args: Args) -> dict[str, Any]:
    episode_metadata = mcap_episode.read_episode_metadata(args.episode_dir)
    recording_context = mcap_episode.read_recording_context(args.episode_dir)
    episode = mcap_episode.read_episode(args.episode_dir, decode_images=True)
    row_timing = _timestamp_stats_ns(episode.timestamps_ns, fps=args.expected_row_fps)
    camera_timing = {
        camera_name: _timestamp_stats_ns(timestamps_ns, fps=args.expected_camera_fps)
        for camera_name, timestamps_ns in sorted((episode.camera_timestamps_ns or {}).items())
    }
    row_to_camera_age = _row_to_camera_age(episode)
    image_summary = _image_summary(args.episode_dir, episode.images)
    contact_sheet_written = False
    if episode.images is not None:
        contact_sheet_written = _write_contact_sheet(args.output_dir / args.episode_dir.name / "contact_sheet.jpg", episode.images)

    state = np.asarray(episode.state)
    action = np.asarray(episode.action)
    action_state_delta = action - state if action.shape == state.shape else np.asarray([])
    finite = {
        "state": bool(np.all(np.isfinite(state))),
        "action": bool(np.all(np.isfinite(action))),
    }
    gripper = {
        "state": _gripper_ranges(state),
        "action": _gripper_ranges(action),
    }
    dynamic_demo = _is_dynamic_gripper_demo(args.episode_dir)
    strict_failures = _strict_failures(
        recording_context=recording_context,
        row_timing=row_timing,
        camera_timing=camera_timing,
        image_summary=image_summary,
        finite=finite,
        gripper=gripper,
        dynamic_demo=dynamic_demo,
        expected_row_fps=args.expected_row_fps,
        expected_camera_fps=args.expected_camera_fps,
    )
    return {
        "episode_dir": str(args.episode_dir),
        "task": episode.task,
        "rows": int(state.shape[0]),
        "expected_row_fps": args.expected_row_fps,
        "expected_camera_fps": args.expected_camera_fps,
        "episode_metadata": episode_metadata,
        "recording_context": recording_context,
        "recording_context_present": bool(recording_context),
        "camera_requested": recording_context.get("camera_requested", {}),
        "camera_actual_modes": recording_context.get("camera_actual_modes", {}),
        "camera_paths": recording_context.get("camera_paths", {}),
        "row_timing": row_timing,
        "camera_timing": camera_timing,
        "camera_frame_reuse": episode.camera_frame_reuse or {},
        "row_to_camera_age": row_to_camera_age,
        "state": {"shape": list(state.shape), "finite": finite["state"]},
        "action": {"shape": list(action.shape), "finite": finite["action"]},
        "gripper": gripper,
        "action_state_delta": _matrix_abs_stats(action_state_delta),
        "images": image_summary,
        "dynamic_gripper_demo": dynamic_demo,
        "contact_sheet_written": contact_sheet_written,
        "strict_failures": strict_failures,
        "strict_pass": not strict_failures,
    }


def _timestamp_stats_ns(timestamps_ns: np.ndarray, *, fps: float) -> dict[str, float | int]:
    timestamps = np.asarray(timestamps_ns, dtype=np.float64) / 1_000_000_000.0
    expected_dt_s = 0.0 if fps <= 0 else 1.0 / fps
    if timestamps.size < 2:
        return {
            "num_intervals": 0,
            "expected_dt_s": expected_dt_s,
            "median_dt_s": 0.0,
            "mean_dt_s": 0.0,
            "p95_dt_s": 0.0,
            "max_dt_s": 0.0,
            "min_dt_s": 0.0,
            "median_fps": 0.0,
        }
    dt = np.diff(timestamps)
    median_dt_s = float(np.median(dt))
    return {
        "num_intervals": int(dt.size),
        "expected_dt_s": expected_dt_s,
        "median_dt_s": median_dt_s,
        "mean_dt_s": float(np.mean(dt)),
        "p95_dt_s": float(np.quantile(dt, 0.95)),
        "max_dt_s": float(np.max(dt)),
        "min_dt_s": float(np.min(dt)),
        "median_fps": 0.0 if median_dt_s <= 0 else float(1.0 / median_dt_s),
    }


def _row_to_camera_age(episode: mcap_episode.YamMcapEpisode) -> dict[str, dict[str, float | int]]:
    if not episode.camera_row_timestamps_ns:
        return {}
    row_timestamps_ns = np.asarray(episode.timestamps_ns, dtype=np.int64)
    result = {}
    for camera_name, raw_camera_timestamps_ns in sorted(episode.camera_row_timestamps_ns.items()):
        camera_timestamps_ns = np.asarray(raw_camera_timestamps_ns, dtype=np.int64)
        if camera_timestamps_ns.shape != row_timestamps_ns.shape:
            continue
        age_ms = (row_timestamps_ns - camera_timestamps_ns).astype(np.float64) / 1_000_000.0
        result[camera_name] = _duration_stats(age_ms)
    return result


def _image_summary(episode_dir: Path, images: dict[str, np.ndarray] | None) -> dict[str, dict[str, Any]]:
    if images is None:
        return {camera_name: {"present": False} for camera_name in common.CAMERA_NAMES}
    result = {}
    for camera_name in common.CAMERA_NAMES:
        camera_images = images.get(camera_name)
        if camera_images is None:
            result[camera_name] = {"present": False}
            continue
        array = np.asarray(camera_images)
        if array.ndim != 4 or array.shape[-1] != 3:
            raise RuntimeError(f"Bad image array for {camera_name} in {episode_dir}: {array.shape}")
        per_frame_std = np.std(array.astype(np.float32), axis=(1, 2, 3))
        result[camera_name] = {
            "present": True,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "blank_frames": int(np.count_nonzero(per_frame_std < 1.0)),
            "mean_rgb": np.mean(array, axis=(0, 1, 2)).astype(float).tolist() if array.size else [0.0, 0.0, 0.0],
            "std_rgb": np.std(array, axis=(0, 1, 2)).astype(float).tolist() if array.size else [0.0, 0.0, 0.0],
        }
    return result


def _write_contact_sheet(path: Path, images: dict[str, np.ndarray]) -> bool:
    tiles = []
    for camera_name in common.CAMERA_NAMES:
        camera_images = images.get(camera_name)
        if camera_images is None or len(camera_images) == 0:
            continue
        indices = [0, len(camera_images) // 2, len(camera_images) - 1]
        row_tiles = []
        for label, idx in zip(("first", "middle", "last"), indices, strict=True):
            tile = np.asarray(camera_images[idx]).copy()
            tile = cv2.resize(tile, (320, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(
                tile,
                f"{camera_name} {label}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"{camera_name} {label}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            row_tiles.append(tile)
        tiles.append(np.concatenate(row_tiles, axis=1))
    if not tiles:
        return False
    sheet = np.concatenate(tiles, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)))


def _gripper_ranges(matrix: np.ndarray) -> dict[str, dict[str, float | int]]:
    if matrix.ndim != 2 or matrix.shape[1] < 14 or matrix.shape[0] == 0:
        return {
            "left": {"min": 0.0, "max": 0.0, "range": 0.0, "close_count": 0},
            "right": {"min": 0.0, "max": 0.0, "range": 0.0, "close_count": 0},
        }
    return {
        "left": _one_gripper_range(matrix[:, 6]),
        "right": _one_gripper_range(matrix[:, 13]),
    }


def _one_gripper_range(values: np.ndarray) -> dict[str, float | int]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "close_count": int(np.count_nonzero(values >= 0.75)),
    }


def _matrix_abs_stats(matrix: np.ndarray) -> dict[str, float]:
    if matrix.size == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "p95_abs": 0.0}
    abs_value = np.abs(np.asarray(matrix, dtype=np.float64))
    return {
        "max_abs": float(np.max(abs_value)),
        "mean_abs": float(np.mean(abs_value)),
        "p95_abs": float(np.quantile(abs_value, 0.95)),
    }


def _strict_failures(
    *,
    recording_context: dict[str, Any],
    row_timing: dict[str, float | int],
    camera_timing: dict[str, dict[str, float | int]],
    image_summary: dict[str, dict[str, Any]],
    finite: dict[str, bool],
    gripper: dict[str, dict[str, dict[str, float | int]]],
    dynamic_demo: bool,
    expected_row_fps: float,
    expected_camera_fps: float,
) -> list[str]:
    failures = []
    if not recording_context:
        failures.append("missing recording_context.json sidecar")
    expected_row_dt_s = 1.0 / expected_row_fps
    row_median_dt_s = float(row_timing["median_dt_s"])
    if _relative_error(row_median_dt_s, expected_row_dt_s) > 0.25:
        failures.append(f"row median dt {row_median_dt_s:.4f}s is outside 25% of {expected_row_dt_s:.4f}s")
    if float(row_timing["p95_dt_s"]) >= 0.050:
        failures.append(f"row p95 dt {float(row_timing['p95_dt_s']):.4f}s is not under 0.0500s")

    expected_camera_dt_s = 1.0 / expected_camera_fps
    for camera_name, timing in camera_timing.items():
        if int(timing.get("num_intervals", 0)) == 0:
            failures.append(f"{camera_name} has no camera timing intervals")
            continue
        median_dt_s = float(timing["median_dt_s"])
        if _relative_error(median_dt_s, expected_camera_dt_s) > 0.25:
            failures.append(
                f"{camera_name} median dt {median_dt_s:.4f}s is outside 25% of {expected_camera_dt_s:.4f}s"
            )

    for camera_name, summary in image_summary.items():
        if not summary.get("present"):
            failures.append(f"{camera_name} images are missing")
        elif int(summary.get("blank_frames", 0)) > 0:
            failures.append(f"{camera_name} has {summary['blank_frames']} blank frame(s)")

    for name, is_finite in finite.items():
        if not is_finite:
            failures.append(f"{name} contains non-finite values")

    if dynamic_demo:
        failures.extend(
            f"{side} action gripper does not close to >= 0.75 during dynamic demo"
            for side in ("left", "right")
            if int(gripper["action"][side]["close_count"]) == 0
        )
    return failures


def _relative_error(actual: float, expected: float) -> float:
    return float("inf") if expected <= 0 else abs(actual - expected) / expected


def _is_dynamic_gripper_demo(episode_dir: Path) -> bool:
    name = episode_dir.name.lower()
    return "motion" in name or "dynamic" in name or "gripper" in name


def _duration_stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(np.max(values)),
        "min_ms": float(np.min(values)),
    }


def _markdown_summary(summary: dict[str, Any], *, json_path: Path, contact_sheet_path: Path) -> str:
    lines = [
        "# YAM Recording Validation",
        "",
        f"- Episode: `{summary['episode_dir']}`",
        f"- JSON: `{json_path}`",
        f"- Contact sheet: `{contact_sheet_path}`",
        f"- Strict pass: `{summary['strict_pass']}`",
        f"- Rows: {summary['rows']}",
        f"- Row median FPS: {summary['row_timing']['median_fps']:.2f}",
        "",
        "## Cameras",
        "",
        "| Camera | median FPS | reused rows | blank frames |",
        "| --- | ---: | ---: | ---: |",
    ]
    for camera_name in common.CAMERA_NAMES:
        timing = summary["camera_timing"].get(camera_name, {})
        reuse = summary["camera_frame_reuse"].get(camera_name, {})
        image = summary["images"].get(camera_name, {})
        lines.append(
            f"| `{camera_name}` | {timing.get('median_fps', 0.0):.2f} | "
            f"{reuse.get('reused_rows', 0)} | {image.get('blank_frames', 0)} |"
        )
    failures = summary.get("strict_failures", [])
    lines.extend(["", "## Strict Failures"])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main(tyro.cli(Args))
