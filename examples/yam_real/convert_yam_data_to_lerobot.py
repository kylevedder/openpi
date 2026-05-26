from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import gc
from pathlib import Path
import shutil
from typing import Literal

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

from examples.yam_real import common
from examples.yam_real import data_regression


@dataclasses.dataclass
class Args:
    raw_dir: Path = Path("yam_data/raw")
    episode_manifest: Path | None = None
    repo_id: str = "local/yam_bimanual"
    raw_format: Literal["auto", "npz", "mcap"] = "auto"
    mode: Literal["image", "video"] = "image"
    resume: bool = False
    image_writer_processes: int = 0
    image_writer_threads: int = 0
    image_writer_flush_interval_frames: int = 0
    push_to_hub: bool = False
    overwrite: bool = True
    allow_timing_mismatch: bool = False
    max_median_dt_error_ratio: float = 0.25
    max_p95_dt_error_ratio: float = 2.5


def main(args: Args) -> None:
    output_path = HF_LEROBOT_HOME / args.repo_id
    resume_from = 0
    if output_path.exists() and args.resume:
        shutil.rmtree(output_path / "images", ignore_errors=True)
        dataset = LeRobotDataset(args.repo_id, root=output_path)
        resume_from = int(dataset.meta.total_episodes)
        print(f"Resuming existing LeRobot dataset at {output_path}: {resume_from} episodes already saved")
    elif output_path.exists():
        if not args.overwrite:
            raise FileExistsError(output_path)
        shutil.rmtree(output_path)
        dataset = None
    else:
        dataset = None

    episode_dirs = _episode_dirs(args.raw_dir, args.episode_manifest, args.raw_format)
    if not episode_dirs:
        raise FileNotFoundError(f"No recorded YAM episodes found under {args.raw_dir}")
    if resume_from > len(episode_dirs):
        raise ValueError(f"Cannot resume from {resume_from} episodes; manifest only has {len(episode_dirs)}")
    episode_dirs = episode_dirs[resume_from:]
    if not episode_dirs:
        print(f"No remaining episodes to convert; dataset is already complete at {output_path}")
        return

    first_episode = data_regression.load_canonical_episode(episode_dirs[0], source_format=args.raw_format)
    fps = round(float(first_episode.fps))
    frame_counts = [_validate_episode(episode_dir, args) for episode_dir in episode_dirs]
    print(f"Selected {len(episode_dirs)} YAM episodes")
    print(f"Total frames: {sum(frame_counts)}")

    if dataset is None:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=output_path,
            robot_type="yam_bimanual",
            fps=fps,
            features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["action"],
                },
                "observation.images.cam_high": _image_feature(args.mode),
                "observation.images.cam_left_wrist": _image_feature(args.mode),
                "observation.images.cam_right_wrist": _image_feature(args.mode),
            },
            use_videos=args.mode == "video",
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )

    for episode_dir in tqdm.tqdm(episode_dirs, desc="Converting YAM episodes"):
        task, states, actions, images_or_paths = _load_episode_for_conversion(episode_dir, args.raw_format)
        if states.shape != actions.shape or states.shape[-1] != 14:
            raise RuntimeError(f"Bad state/action shapes in {episode_dir}: {states.shape}, {actions.shape}")
        if isinstance(images_or_paths, dict):
            for camera_name, images in images_or_paths.items():
                if len(images) != len(states):
                    raise RuntimeError(
                        f"Image count mismatch for {camera_name} in {episode_dir}: {len(images)} vs {len(states)}"
                    )
        elif len(images_or_paths) != len(states):
            raise RuntimeError(f"Image count mismatch in {episode_dir}: {len(images_or_paths)} vs {len(states)}")

        for idx in range(len(states)):
            frame = {
                "observation.state": states[idx],
                "action": actions[idx],
                "task": task,
            }
            frame.update(_load_images_for_frame(episode_dir, images_or_paths, idx))
            dataset.add_frame(frame)
            if args.image_writer_flush_interval_frames and (idx + 1) % args.image_writer_flush_interval_frames == 0:
                dataset._wait_image_writer()  # noqa: SLF001
        dataset.save_episode()
        del states, actions, images_or_paths
        gc.collect()
        _malloc_trim()

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    if args.push_to_hub:
        dataset.push_to_hub()
    print(f"Wrote LeRobot dataset to {output_path}")


def _malloc_trim() -> None:
    if not hasattr(ctypes, "CDLL"):
        return
    with contextlib.suppress(Exception):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def _episode_dirs(raw_dir: Path, episode_manifest: Path | None, raw_format: str) -> list[Path]:
    if episode_manifest is None:
        return data_regression.discover_episodes(raw_dir, source_format=raw_format)

    if not episode_manifest.exists():
        raise FileNotFoundError(episode_manifest)

    episode_dirs = []
    seen = set()
    for line_number, line in enumerate(episode_manifest.read_text().splitlines(), start=1):
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        rel_path = Path(entry)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"Invalid episode path on line {line_number}: {entry}")
        if entry in seen:
            raise ValueError(f"Duplicate episode on line {line_number}: {entry}")
        seen.add(entry)
        episode_dir = raw_dir / rel_path
        if not data_regression.discover_episodes(episode_dir, source_format=raw_format):
            raise FileNotFoundError(f"Manifest-listed episode is not a {raw_format} YAM episode: {episode_dir}")
        episode_dirs.append(episode_dir)
    return episode_dirs


def _validate_episode(episode_dir: Path, args: Args) -> int:
    episode = data_regression.load_canonical_episode(episode_dir, source_format=args.raw_format)
    states = episode.state
    actions = episode.action
    if states.shape != actions.shape or states.shape[-1] != 14:
        raise RuntimeError(f"Bad state/action shapes in {episode_dir}: {states.shape}, {actions.shape}")
    for camera_name in common.CAMERA_NAMES:
        if episode.image_counts.get(camera_name, 0) != len(states):
            raise RuntimeError(
                f"Image count mismatch for {camera_name} in {episode_dir}: "
                f"{episode.image_counts.get(camera_name, 0)} vs {len(states)}"
            )
    summary = data_regression.summarize_episode(
        episode,
        max_median_dt_error_ratio=args.max_median_dt_error_ratio,
        max_p95_dt_error_ratio=args.max_p95_dt_error_ratio,
    )
    timing_warnings = [warning for warning in summary["warnings"] if warning.startswith("timestamp ")]
    if timing_warnings and not args.allow_timing_mismatch:
        detail = "; ".join(timing_warnings)
        raise RuntimeError(
            f"Timing mismatch in {episode_dir}: {detail}. "
            "Fix collection FPS or pass --allow-timing-mismatch for an explicit one-off conversion."
        )
    return int(states.shape[0])


def _load_episode_for_conversion(episode_dir: Path, raw_format: str) -> tuple[str, np.ndarray, np.ndarray, object]:
    resolved_format = data_regression.resolve_source_format(episode_dir, raw_format, lerobot_repo_id=None)
    if resolved_format == "npz":
        manifest, arrays = common.load_episode(episode_dir)
        return (
            str(manifest["task"]),
            np.asarray(arrays["state"], dtype=np.float32),
            np.asarray(arrays["action"], dtype=np.float32),
            arrays["image_paths"],
        )
    if resolved_format == "mcap":
        from examples.yam_real import mcap_episode

        episode = mcap_episode.read_episode(episode_dir, decode_images=True)
        if episode.images is None:
            raise RuntimeError(f"MCAP episode did not decode images: {episode_dir}")
        return (
            episode.task,
            np.asarray(episode.state, dtype=np.float32),
            np.asarray(episode.action, dtype=np.float32),
            episode.images,
        )
    raise ValueError(f"Unsupported YAM raw format for conversion: {resolved_format}")


def _load_images_for_frame(episode_dir: Path, images_or_paths: object, idx: int) -> dict[str, np.ndarray]:
    if isinstance(images_or_paths, dict):
        return {
            "observation.images.cam_high": _hwc_rgb_to_chw(images_or_paths["cam_high"][idx]),
            "observation.images.cam_left_wrist": _hwc_rgb_to_chw(images_or_paths["cam_left_wrist"][idx]),
            "observation.images.cam_right_wrist": _hwc_rgb_to_chw(images_or_paths["cam_right_wrist"][idx]),
        }

    paths = images_or_paths[idx].item() if hasattr(images_or_paths[idx], "item") else images_or_paths[idx]
    return {
        "observation.images.cam_high": _load_chw_rgb(episode_dir / paths["cam_high"]),
        "observation.images.cam_left_wrist": _load_chw_rgb(episode_dir / paths["cam_left_wrist"]),
        "observation.images.cam_right_wrist": _load_chw_rgb(episode_dir / paths["cam_right_wrist"]),
    }


def _image_feature(mode: str) -> dict:
    return {
        "dtype": mode,
        "shape": (3, 480, 640),
        "names": ["channel", "height", "width"],
    }


def _load_chw_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return _hwc_rgb_to_chw(image)


def _hwc_rgb_to_chw(image: np.ndarray) -> np.ndarray:
    return np.transpose(image, (2, 0, 1))


if __name__ == "__main__":
    main(tyro.cli(Args))
