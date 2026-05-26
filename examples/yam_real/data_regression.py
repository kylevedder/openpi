from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import time
from typing import Any, Literal

import cv2
import numpy as np
import tyro

from examples.yam_real import common

GRIPPER_DIMS = (6, 13)
ARM_DIMS = tuple(dim for dim in range(14) if dim not in GRIPPER_DIMS)
CAMERA_NAMES = common.CAMERA_NAMES
DIM_NAMES = common.STATE_ORDER
DEFAULT_OLD_GOOD_DIR = Path("yam_data/archive/raw_before_moved_robot_20260525_145623")
DEFAULT_CURRENT_RAW_DIR = Path("yam_data/raw")
DEFAULT_LOG_DIR = Path("yam_data/logs/data_regression")


@dataclasses.dataclass(frozen=True)
class CanonicalEpisode:
    source_path: Path
    source_format: str
    task: str
    fps: float
    state: np.ndarray
    action: np.ndarray
    timestamps_s: np.ndarray
    image_counts: dict[str, int]
    row_fps: float | None = None
    camera_fps: float | None = None
    camera_timestamps_s: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    camera_frame_reuse: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.state.shape[0])


@dataclasses.dataclass(frozen=True)
class Args:
    inputs: tuple[Path, ...] = ()
    old_good_dir: Path | None = DEFAULT_OLD_GOOD_DIR
    current_raw_dir: Path | None = DEFAULT_CURRENT_RAW_DIR
    lerobot_repo_id: str | None = None
    lerobot_root: Path | None = None
    lerobot_episode_index: int = 0
    source_format: Literal["auto", "npz", "mcap", "lerobot"] = "auto"
    output_dir: Path = DEFAULT_LOG_DIR
    max_episodes_per_source: int | None = None
    strict: bool = False
    max_arm_abs_rad: float = 3.2
    max_arm_step_rad: float = 1.25
    max_median_dt_error_ratio: float = 0.25
    max_p95_dt_error_ratio: float = 2.5
    min_gripper_range: float = 0.05
    open_threshold: float = 0.25
    close_threshold: float = 0.75


def main(args: Args) -> None:
    episodes = []
    for path in _default_and_user_paths(args):
        episodes.extend(discover_episodes(path, source_format=args.source_format))
    if args.max_episodes_per_source is not None:
        episodes = episodes[: args.max_episodes_per_source]
    if args.lerobot_repo_id is not None or args.source_format == "lerobot":
        episodes.append(args.lerobot_root or Path(args.lerobot_repo_id or ""))

    if not episodes:
        raise FileNotFoundError("No YAM episode inputs found.")

    records = []
    for episode_path in episodes:
        episode = load_canonical_episode(
            episode_path,
            source_format=args.source_format,
            lerobot_repo_id=args.lerobot_repo_id,
            lerobot_root=args.lerobot_root,
            lerobot_episode_index=args.lerobot_episode_index,
        )
        record = summarize_episode(
            episode,
            max_arm_abs_rad=args.max_arm_abs_rad,
            max_arm_step_rad=args.max_arm_step_rad,
            max_median_dt_error_ratio=args.max_median_dt_error_ratio,
            max_p95_dt_error_ratio=args.max_p95_dt_error_ratio,
            min_gripper_range=args.min_gripper_range,
            open_threshold=args.open_threshold,
            close_threshold=args.close_threshold,
        )
        records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"yam_gripper_regression_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    with output_path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"Wrote {len(records)} episode summaries to {output_path}")
    print_summary_table(records)

    warnings = [warning for record in records for warning in record["warnings"]]
    if args.strict and warnings:
        raise RuntimeError(f"YAM gripper regression found {len(warnings)} warning(s); see {output_path}")


def load_canonical_episode(
    path: Path,
    *,
    source_format: Literal["auto", "npz", "mcap", "lerobot"] = "auto",
    lerobot_repo_id: str | None = None,
    lerobot_root: Path | None = None,
    lerobot_episode_index: int = 0,
) -> CanonicalEpisode:
    path = Path(path)
    resolved_format = resolve_source_format(path, source_format, lerobot_repo_id=lerobot_repo_id)
    if resolved_format == "npz":
        return load_npz_episode(path)
    if resolved_format == "mcap":
        return load_mcap_episode(path)
    if resolved_format == "lerobot":
        repo_id = lerobot_repo_id or _infer_lerobot_repo_id(path)
        root = lerobot_root or path
        return load_lerobot_episode(repo_id, root=root, episode_index=lerobot_episode_index)
    raise ValueError(f"Unsupported source format: {resolved_format}")


def resolve_source_format(
    path: Path,
    requested: Literal["auto", "npz", "mcap", "lerobot"],
    *,
    lerobot_repo_id: str | None = None,
) -> Literal["npz", "mcap", "lerobot"]:
    if requested != "auto":
        return requested
    if lerobot_repo_id is not None:
        return "lerobot"
    if (path / "episode.npz").exists():
        return "npz"
    if any(path.glob("episode_part*.mcap")):
        return "mcap"
    if (path / "meta").exists() and (path / "data").exists():
        return "lerobot"
    raise ValueError(f"Could not infer YAM episode format for {path}")


def discover_episodes(path: Path, *, source_format: Literal["auto", "npz", "mcap", "lerobot"] = "auto") -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    if source_format in ("auto", "npz") and (path / "episode.npz").exists():
        return [path]
    if source_format in ("auto", "mcap") and any(path.glob("episode_part*.mcap")):
        return [path]
    if source_format in ("auto", "lerobot") and (path / "meta").exists() and (path / "data").exists():
        return [path]

    episodes = []
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        if (source_format in ("auto", "npz") and (child / "episode.npz").exists()) or (
            source_format in ("auto", "mcap") and any(child.glob("episode_part*.mcap"))
        ):
            episodes.append(child)
    return episodes


def load_npz_episode(episode_dir: Path) -> CanonicalEpisode:
    manifest, arrays = common.load_episode(episode_dir)
    state = _as_float_matrix(arrays["state"], "state", episode_dir)
    action = _as_float_matrix(arrays["action"], "action", episode_dir)
    _validate_state_action_shapes(state, action, episode_dir)

    timestamps_s = np.asarray(arrays.get("timestamp", np.arange(len(state))), dtype=np.float64)
    image_paths = arrays.get("image_paths")
    image_counts = _image_counts_from_npz(image_paths)
    camera_timestamps_s, camera_frame_reuse = _camera_timestamps_from_npz_metadata(
        arrays.get("camera_frame_metadata"),
        num_rows=len(state),
    )
    fps = float(manifest.get("row_fps", manifest.get("fps", 0.0)))
    return CanonicalEpisode(
        source_path=episode_dir,
        source_format="npz",
        task=str(manifest.get("task", "")),
        fps=fps,
        state=state,
        action=action,
        timestamps_s=timestamps_s,
        image_counts=image_counts,
        row_fps=fps,
        camera_fps=float(manifest["camera_fps"]) if manifest.get("camera_fps") is not None else None,
        camera_timestamps_s=camera_timestamps_s,
        camera_frame_reuse=camera_frame_reuse,
    )


def load_mcap_episode(episode_dir: Path) -> CanonicalEpisode:
    from examples.yam_real import mcap_episode

    episode = mcap_episode.read_episode(episode_dir, decode_images=False)
    state = _as_float_matrix(episode.state, "state", episode_dir)
    action = _as_float_matrix(episode.action, "action", episode_dir)
    _validate_state_action_shapes(state, action, episode_dir)
    timestamps_s = np.asarray(episode.timestamps_ns, dtype=np.float64) / 1_000_000_000.0
    camera_timestamps_s = {
        camera_name: np.asarray(timestamps_ns, dtype=np.float64) / 1_000_000_000.0
        for camera_name, timestamps_ns in (episode.camera_timestamps_ns or {}).items()
    }
    return CanonicalEpisode(
        source_path=episode_dir,
        source_format="mcap",
        task=episode.task,
        fps=float(episode.fps),
        state=state,
        action=action,
        timestamps_s=timestamps_s,
        image_counts={camera_name: len(state) for camera_name in CAMERA_NAMES},
        row_fps=float(episode.fps),
        camera_fps=_infer_camera_fps(camera_timestamps_s),
        camera_timestamps_s=camera_timestamps_s,
        camera_frame_reuse=episode.camera_frame_reuse or {},
    )


def load_lerobot_episode(repo_id: str, *, root: Path | None, episode_index: int) -> CanonicalEpisode:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root, episodes=[episode_index])
    states = []
    actions = []
    timestamps = []
    task = ""
    for idx in range(len(dataset)):
        item = dataset[idx]
        states.append(_to_numpy(item["observation.state"]))
        actions.append(_to_numpy(item["action"]))
        timestamps.append(float(_to_numpy(item["timestamp"])))
        if not task:
            task = str(item.get("task", ""))

    state = _as_float_matrix(np.asarray(states), "observation.state", root or Path(repo_id))
    action = _as_float_matrix(np.asarray(actions), "action", root or Path(repo_id))
    _validate_state_action_shapes(state, action, root or Path(repo_id))
    image_keys = tuple(getattr(dataset.meta, "image_keys", ()))
    return CanonicalEpisode(
        source_path=root or Path(repo_id),
        source_format="lerobot",
        task=task,
        fps=float(dataset.meta.fps),
        state=state,
        action=action,
        timestamps_s=np.asarray(timestamps, dtype=np.float64),
        image_counts={_camera_name_from_lerobot_key(key): len(state) for key in image_keys},
    )


def summarize_episode(
    episode: CanonicalEpisode,
    *,
    max_arm_abs_rad: float = 3.2,
    max_arm_step_rad: float = 1.25,
    max_median_dt_error_ratio: float = 0.25,
    max_p95_dt_error_ratio: float = 2.5,
    min_gripper_range: float = 0.05,
    open_threshold: float = 0.25,
    close_threshold: float = 0.75,
) -> dict:
    warnings = []
    record = {
        "source_path": str(episode.source_path),
        "source_format": episode.source_format,
        "task": episode.task,
        "fps": episode.fps,
        "row_fps": episode.row_fps,
        "camera_fps": episode.camera_fps,
        "num_frames": episode.num_frames,
        "duration_s": _duration_s(episode.timestamps_s),
        "timing": _timestamp_stats(episode.timestamps_s, fps=episode.fps),
        "camera_timing": _camera_timing_stats(episode),
        "camera_frame_reuse": episode.camera_frame_reuse,
        "image_counts": episode.image_counts,
        "state_stats": _matrix_stats(episode.state),
        "action_stats": _matrix_stats(episode.action),
        "state_stats_by_name": _named_matrix_stats(episode.state),
        "action_stats_by_name": _named_matrix_stats(episode.action),
        "arm_joints": {
            "state": _arm_joint_stats(episode.state),
            "action": _arm_joint_stats(episode.action),
        },
        "gripper": {
            "state": _gripper_stats(episode.state, open_threshold=open_threshold, close_threshold=close_threshold),
            "action": _gripper_stats(episode.action, open_threshold=open_threshold, close_threshold=close_threshold),
        },
        "warnings": warnings,
    }

    if episode.num_frames == 0:
        warnings.append("episode has no frames")
    if episode.num_frames > 1 and episode.fps > 0:
        expected_dt_s = 1.0 / episode.fps
        timing = record["timing"]
        median_dt_s = timing["median_dt_s"]
        p95_dt_s = timing["p95_dt_s"]
        median_error_ratio = abs(median_dt_s - expected_dt_s) / expected_dt_s
        if median_error_ratio > max_median_dt_error_ratio:
            warnings.append(
                "timestamp median dt "
                f"{median_dt_s:.4f}s does not match fps={episode.fps:g} "
                f"(expected {expected_dt_s:.4f}s, error_ratio={median_error_ratio:.2f})"
            )
        if p95_dt_s > expected_dt_s * max_p95_dt_error_ratio:
            warnings.append(
                "timestamp p95 dt "
                f"{p95_dt_s:.4f}s exceeds {max_p95_dt_error_ratio:.2f}x expected dt "
                f"{expected_dt_s:.4f}s for fps={episode.fps:g}"
            )
    for name, matrix in (("state", episode.state), ("action", episode.action)):
        if not np.all(np.isfinite(matrix)):
            warnings.append(f"{name} contains non-finite values")
        grippers = matrix[:, GRIPPER_DIMS]
        out_of_range = np.logical_or(grippers < -1e-4, grippers > 1.0001)
        if np.any(out_of_range):
            warnings.append(f"{name} gripper values outside [0, 1]: {int(np.count_nonzero(out_of_range))}")

        arm_values = matrix[:, ARM_DIMS]
        arm_abs_max = float(np.max(np.abs(arm_values))) if arm_values.size else 0.0
        if arm_abs_max > max_arm_abs_rad:
            warnings.append(f"{name} arm joint abs max {arm_abs_max:.4f} exceeds {max_arm_abs_rad:.4f} rad")

        arm_step_max = _max_abs_step(arm_values)
        if arm_step_max > max_arm_step_rad:
            warnings.append(f"{name} arm joint step max {arm_step_max:.4f} exceeds {max_arm_step_rad:.4f} rad")

    for key, side in (("left", 0), ("right", 1)):
        values = episode.action[:, GRIPPER_DIMS[side]]
        action_range = float(np.max(values) - np.min(values)) if len(values) else 0.0
        if action_range < min_gripper_range:
            warnings.append(f"{key} action gripper range is flat: {action_range:.4f}")
        if np.count_nonzero(values >= close_threshold) == 0:
            warnings.append(f"{key} action gripper has no close samples >= {close_threshold}")

    warnings.extend(
        f"{camera_name} image count mismatch: {episode.image_counts.get(camera_name, 0)} vs {episode.num_frames}"
        for camera_name in CAMERA_NAMES
        if episode.image_counts.get(camera_name, 0) != episode.num_frames
    )
    if episode.camera_fps:
        expected_dt_s = 1.0 / episode.camera_fps
        for camera_name, timing in record["camera_timing"].items():
            if timing["num_intervals"] == 0:
                continue
            median_dt_s = float(timing["median_dt_s"])
            median_error_ratio = abs(median_dt_s - expected_dt_s) / expected_dt_s
            if median_error_ratio > max_median_dt_error_ratio:
                warnings.append(
                    f"{camera_name} camera median dt {median_dt_s:.4f}s does not match camera_fps="
                    f"{episode.camera_fps:g} (expected {expected_dt_s:.4f}s, error_ratio={median_error_ratio:.2f})"
                )
    return record


def compare_canonical_episodes(
    expected: CanonicalEpisode,
    actual: CanonicalEpisode,
    *,
    atol: float = 1e-6,
) -> dict[str, float]:
    _validate_state_action_shapes(expected.state, expected.action, expected.source_path)
    _validate_state_action_shapes(actual.state, actual.action, actual.source_path)
    if expected.state.shape != actual.state.shape:
        raise AssertionError(f"state shape mismatch: {expected.state.shape} != {actual.state.shape}")
    if expected.action.shape != actual.action.shape:
        raise AssertionError(f"action shape mismatch: {expected.action.shape} != {actual.action.shape}")
    np.testing.assert_allclose(actual.state, expected.state, atol=atol, rtol=0)
    np.testing.assert_allclose(actual.action, expected.action, atol=atol, rtol=0)
    return {
        "max_state_abs_error": float(np.max(np.abs(actual.state - expected.state))),
        "max_action_abs_error": float(np.max(np.abs(actual.action - expected.action))),
    }


def make_synthetic_episode(num_frames: int = 6) -> CanonicalEpisode:
    if num_frames < 4:
        raise ValueError("Synthetic YAM regression episode needs at least four frames.")
    state = np.zeros((num_frames, 14), dtype=np.float32)
    action = np.zeros((num_frames, 14), dtype=np.float32)
    left_arm = np.linspace(-0.5, 0.5, 6, dtype=np.float32)
    right_arm = np.linspace(0.4, -0.4, 6, dtype=np.float32)
    for idx in range(num_frames):
        base = np.float32(idx / 100.0)
        state[idx, :6] = base + left_arm
        state[idx, 7:13] = -base + right_arm
        action[idx, :6] = state[idx, :6] + 0.01
        action[idx, 7:13] = state[idx, 7:13] - 0.01

    pattern = np.asarray([0.0, 0.2, 0.5, 0.85, 1.0, 0.0], dtype=np.float32)
    gripper = np.resize(pattern, num_frames)
    state[:, 6] = gripper
    state[:, 13] = np.roll(gripper, 1)
    action[:, 6] = gripper
    action[:, 13] = np.roll(gripper, 2)
    timestamps_s = np.arange(num_frames, dtype=np.float64) / 50.0
    return CanonicalEpisode(
        source_path=Path("<synthetic>"),
        source_format="synthetic",
        task="synthetic gripper regression",
        fps=50.0,
        state=state,
        action=action,
        timestamps_s=timestamps_s,
        image_counts=dict.fromkeys(CAMERA_NAMES, num_frames),
    )


def write_npz_fixture(episode: CanonicalEpisode, episode_dir: Path) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=False)
    image_paths = []
    for camera_name in CAMERA_NAMES:
        (episode_dir / "images" / camera_name).mkdir(parents=True, exist_ok=True)
    for frame_idx in range(episode.num_frames):
        frame_paths = {}
        for camera_idx, camera_name in enumerate(CAMERA_NAMES):
            rel_path = Path("images") / camera_name / f"{frame_idx:06d}.jpg"
            image = np.full((480, 640, 3), frame_idx * 10 + camera_idx, dtype=np.uint8)
            if not cv2.imwrite(str(episode_dir / rel_path), image):
                raise RuntimeError(f"Failed to write fixture image {episode_dir / rel_path}")
            frame_paths[camera_name] = str(rel_path)
        image_paths.append(frame_paths)

    manifest = common.EpisodeManifest(
        task=episode.task,
        fps=episode.fps,
        created_at=0.0,
        state_order=tuple(common.STATE_ORDER),
        action_space=common.ACTION_SPACE,
        gripper_convention=common.GRIPPER_CONVENTION,
        camera_paths={camera_name: f"/dev/null/{camera_name}" for camera_name in CAMERA_NAMES},
        leader_channels=dict(common.LEADER_CHANNELS),
        follower_channels=dict(common.FOLLOWER_CHANNELS),
        num_frames=episode.num_frames,
    )
    manifest.write(episode_dir / "manifest.json")
    np.savez_compressed(
        episode_dir / "episode.npz",
        state=episode.state.astype(np.float32),
        action=episode.action.astype(np.float32),
        timestamp=episode.timestamps_s.astype(np.float64),
        image_paths=np.asarray(image_paths, dtype=object),
    )
    return episode_dir


def write_mcap_fixture(episode: CanonicalEpisode, episode_dir: Path) -> Path:
    from examples.yam_real import mcap_episode

    writer = mcap_episode.YamMcapEpisodeWriter(episode_dir, task=episode.task, fps=episode.fps)
    try:
        for frame_idx in range(episode.num_frames):
            frames = {
                camera_name: np.full((480, 640, 3), frame_idx * 10 + camera_idx, dtype=np.uint8)
                for camera_idx, camera_name in enumerate(CAMERA_NAMES)
            }
            timestamp_ns = int(episode.timestamps_s[frame_idx] * 1_000_000_000)
            writer.write_step(
                frames_rgb=frames,
                camera_metadata={
                    camera_name: {
                        "sequence_index": frame_idx,
                        "capture_timestamp_ns": timestamp_ns,
                        "capture_time_s": timestamp_ns / 1_000_000_000.0,
                    }
                    for camera_name in CAMERA_NAMES
                },
                state=episode.state[frame_idx],
                action=episode.action[frame_idx],
                timestamp_ns=timestamp_ns,
            )
    finally:
        writer.close()
    return episode_dir


def print_summary_table(records: list[dict]) -> None:
    print("format frames left_action_range right_action_range warnings path")
    for record in records:
        left = record["gripper"]["action"]["left_gripper"]["range"]
        right = record["gripper"]["action"]["right_gripper"]["range"]
        print(
            f"{record['source_format']:>7} {record['num_frames']:>6} "
            f"{left:>17.4f} {right:>18.4f} {len(record['warnings']):>8} {record['source_path']}"
        )


def _default_and_user_paths(args: Args) -> list[Path]:
    paths = list(args.inputs)
    if args.old_good_dir is not None:
        paths.append(args.old_good_dir)
    if args.current_raw_dir is not None:
        paths.append(args.current_raw_dir)
    return paths


def _image_counts_from_npz(image_paths: np.ndarray | None) -> dict[str, int]:
    counts = dict.fromkeys(CAMERA_NAMES, 0)
    if image_paths is None:
        return counts
    for entry in image_paths:
        paths = entry.item() if hasattr(entry, "item") else entry
        for camera_name in CAMERA_NAMES:
            if camera_name in paths:
                counts[camera_name] += 1
    return counts


def _camera_timestamps_from_npz_metadata(
    metadata_array: np.ndarray | None,
    *,
    num_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    if metadata_array is None:
        return {}, {}

    timestamps: dict[str, list[float]] = {camera_name: [] for camera_name in CAMERA_NAMES}
    row_counts: dict[str, int] = dict.fromkeys(CAMERA_NAMES, 0)
    last_sequence: dict[str, int] = {}
    for entry in metadata_array:
        metadata = entry.item() if hasattr(entry, "item") else entry
        if not isinstance(metadata, dict):
            continue
        for camera_name in CAMERA_NAMES:
            camera_metadata = metadata.get(camera_name)
            if not isinstance(camera_metadata, dict):
                continue
            row_counts[camera_name] += 1
            sequence_index = _metadata_int(camera_metadata.get("sequence_index"))
            capture_time_s = _metadata_float(camera_metadata.get("capture_time_s"))
            if capture_time_s is None:
                continue
            if sequence_index is None or last_sequence.get(camera_name) != sequence_index:
                timestamps[camera_name].append(capture_time_s)
                if sequence_index is not None:
                    last_sequence[camera_name] = sequence_index

    arrays = {camera_name: np.asarray(values, dtype=np.float64) for camera_name, values in timestamps.items() if values}
    reuse = {}
    for camera_name in CAMERA_NAMES:
        unique_frames = len(arrays.get(camera_name, ()))
        rows_with_camera = int(row_counts.get(camera_name, 0))
        if rows_with_camera:
            reuse[camera_name] = {
                "rows": rows_with_camera,
                "unique_frames": unique_frames,
                "reused_rows": max(0, rows_with_camera - unique_frames),
                "missing_rows": max(0, num_rows - rows_with_camera),
            }
    return arrays, reuse


def _metadata_float(value: Any) -> float | None:
    if value is None:
        return None
    with np.errstate(all="ignore"):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
    return result if np.isfinite(result) else None


def _metadata_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _camera_timing_stats(episode: CanonicalEpisode) -> dict[str, dict[str, float | int]]:
    fps = float(episode.camera_fps or 0.0)
    return {
        camera_name: _timestamp_stats(timestamps_s, fps=fps)
        for camera_name, timestamps_s in sorted(episode.camera_timestamps_s.items())
    }


def _infer_camera_fps(camera_timestamps_s: dict[str, np.ndarray]) -> float | None:
    medians = []
    for timestamps in camera_timestamps_s.values():
        if len(timestamps) < 2:
            continue
        diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
        if diffs.size and np.median(diffs) > 0:
            medians.append(float(1.0 / np.median(diffs)))
    if not medians:
        return None
    return float(np.median(medians))


def _gripper_stats(
    matrix: np.ndarray,
    *,
    open_threshold: float,
    close_threshold: float,
) -> dict[str, dict[str, float | int]]:
    return {
        "left_gripper": _one_gripper_stats(matrix[:, 6], open_threshold, close_threshold),
        "right_gripper": _one_gripper_stats(matrix[:, 13], open_threshold, close_threshold),
    }


def _one_gripper_stats(values: np.ndarray, open_threshold: float, close_threshold: float) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "open_count": 0,
            "close_count": 0,
            "close_crossings": 0,
            "open_crossings": 0,
            "out_of_range_count": 0,
        }
    closed = values >= close_threshold
    opened = values <= open_threshold
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "open_count": int(np.count_nonzero(opened)),
        "close_count": int(np.count_nonzero(closed)),
        "close_crossings": int(np.count_nonzero(np.diff(closed.astype(np.int8)) == 1)),
        "open_crossings": int(np.count_nonzero(np.diff(opened.astype(np.int8)) == 1)),
        "out_of_range_count": int(np.count_nonzero(np.logical_or(values < -1e-4, values > 1.0001))),
    }


def _matrix_stats(matrix: np.ndarray) -> dict[str, list[float]]:
    if matrix.size == 0:
        return {"min": [], "max": [], "mean": [], "std": []}
    return {
        "min": np.min(matrix, axis=0).astype(float).tolist(),
        "max": np.max(matrix, axis=0).astype(float).tolist(),
        "mean": np.mean(matrix, axis=0).astype(float).tolist(),
        "std": np.std(matrix, axis=0).astype(float).tolist(),
    }


def _named_matrix_stats(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    stats = _matrix_stats(matrix)
    if not stats["min"]:
        return {}
    return {
        name: {
            "min": stats["min"][dim],
            "max": stats["max"][dim],
            "mean": stats["mean"][dim],
            "std": stats["std"][dim],
        }
        for dim, name in enumerate(DIM_NAMES)
    }


def _arm_joint_stats(matrix: np.ndarray) -> dict[str, float]:
    if matrix.size == 0:
        return {"min": 0.0, "max": 0.0, "abs_max": 0.0, "step_abs_max": 0.0, "step_abs_p99": 0.0}
    arm_values = matrix[:, ARM_DIMS]
    diffs = np.abs(np.diff(arm_values, axis=0)) if len(arm_values) > 1 else np.zeros((0, len(ARM_DIMS)))
    return {
        "min": float(np.min(arm_values)),
        "max": float(np.max(arm_values)),
        "abs_max": float(np.max(np.abs(arm_values))),
        "step_abs_max": float(np.max(diffs)) if diffs.size else 0.0,
        "step_abs_p99": float(np.quantile(diffs, 0.99)) if diffs.size else 0.0,
    }


def _max_abs_step(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(values, axis=0))))


def _duration_s(timestamps_s: np.ndarray) -> float:
    if len(timestamps_s) < 2:
        return 0.0
    return float(timestamps_s[-1] - timestamps_s[0])


def _timestamp_stats(timestamps_s: np.ndarray, *, fps: float) -> dict[str, float | int]:
    if len(timestamps_s) < 2:
        expected_dt_s = 0.0 if fps <= 0 else 1.0 / fps
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

    dt = np.diff(np.asarray(timestamps_s, dtype=np.float64))
    median_dt_s = float(np.median(dt))
    return {
        "num_intervals": int(dt.size),
        "expected_dt_s": 0.0 if fps <= 0 else float(1.0 / fps),
        "median_dt_s": median_dt_s,
        "mean_dt_s": float(np.mean(dt)),
        "p95_dt_s": float(np.quantile(dt, 0.95)),
        "max_dt_s": float(np.max(dt)),
        "min_dt_s": float(np.min(dt)),
        "median_fps": 0.0 if median_dt_s <= 0 else float(1.0 / median_dt_s),
    }


def _as_float_matrix(value: np.ndarray, name: str, source_path: Path) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim == 3 and matrix.shape[1] == 1:
        matrix = matrix[:, 0, :]
    if matrix.ndim != 2:
        raise RuntimeError(f"Expected {name} to be rank-2 in {source_path}, got {matrix.shape}")
    return matrix


def _validate_state_action_shapes(state: np.ndarray, action: np.ndarray, source_path: Path) -> None:
    if state.shape != action.shape or state.shape[-1] != 14:
        raise RuntimeError(f"Bad state/action shapes in {source_path}: {state.shape}, {action.shape}")


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _infer_lerobot_repo_id(path: Path) -> str:
    if path.parent.name:
        return f"{path.parent.name}/{path.name}"
    return path.name


def _camera_name_from_lerobot_key(key: str) -> str:
    return key.removeprefix("observation.images.")


if __name__ == "__main__":
    main(tyro.cli(Args))
