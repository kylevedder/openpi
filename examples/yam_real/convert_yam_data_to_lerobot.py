from __future__ import annotations

import dataclasses
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


@dataclasses.dataclass
class Args:
    raw_dir: Path = Path("yam_data/raw")
    episode_manifest: Path | None = None
    repo_id: str = "local/yam_bimanual"
    mode: Literal["image", "video"] = "image"
    push_to_hub: bool = False
    overwrite: bool = True


def main(args: Args) -> None:
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(output_path)
        shutil.rmtree(output_path)

    episode_dirs = _episode_dirs(args.raw_dir, args.episode_manifest)
    if not episode_dirs:
        raise FileNotFoundError(f"No recorded YAM episodes found under {args.raw_dir}")

    first_manifest, _ = common.load_episode(episode_dirs[0])
    fps = round(float(first_manifest["fps"]))
    frame_counts = [_validate_episode(episode_dir) for episode_dir in episode_dirs]
    print(f"Selected {len(episode_dirs)} YAM episodes")
    print(f"Total frames: {sum(frame_counts)}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
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
        image_writer_processes=5,
        image_writer_threads=10,
    )

    for episode_dir in tqdm.tqdm(episode_dirs, desc="Converting YAM episodes"):
        manifest, arrays = common.load_episode(episode_dir)
        states = np.asarray(arrays["state"], dtype=np.float32)
        actions = np.asarray(arrays["action"], dtype=np.float32)
        image_paths = arrays["image_paths"]
        if states.shape != actions.shape or states.shape[-1] != 14:
            raise RuntimeError(f"Bad state/action shapes in {episode_dir}: {states.shape}, {actions.shape}")
        if len(image_paths) != len(states):
            raise RuntimeError(f"Image path count mismatch in {episode_dir}: {len(image_paths)} vs {len(states)}")

        for idx in range(len(states)):
            frame = {
                "observation.state": states[idx],
                "action": actions[idx],
                "task": manifest["task"],
            }
            paths = image_paths[idx].item() if hasattr(image_paths[idx], "item") else image_paths[idx]
            frame["observation.images.cam_high"] = _load_chw_rgb(episode_dir / paths["cam_high"])
            frame["observation.images.cam_left_wrist"] = _load_chw_rgb(episode_dir / paths["cam_left_wrist"])
            frame["observation.images.cam_right_wrist"] = _load_chw_rgb(episode_dir / paths["cam_right_wrist"])
            dataset.add_frame(frame)
        dataset.save_episode()

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    if args.push_to_hub:
        dataset.push_to_hub()
    print(f"Wrote LeRobot dataset to {output_path}")


def _episode_dirs(raw_dir: Path, episode_manifest: Path | None) -> list[Path]:
    if episode_manifest is None:
        return sorted(path for path in raw_dir.iterdir() if (path / "manifest.json").exists())

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
        if not (episode_dir / "manifest.json").exists():
            raise FileNotFoundError(f"Manifest-listed episode is missing manifest.json: {episode_dir}")
        episode_dirs.append(episode_dir)
    return episode_dirs


def _validate_episode(episode_dir: Path) -> int:
    _, arrays = common.load_episode(episode_dir)
    states = np.asarray(arrays["state"], dtype=np.float32)
    actions = np.asarray(arrays["action"], dtype=np.float32)
    image_paths = arrays["image_paths"]
    if states.shape != actions.shape or states.shape[-1] != 14:
        raise RuntimeError(f"Bad state/action shapes in {episode_dir}: {states.shape}, {actions.shape}")
    if len(image_paths) != len(states):
        raise RuntimeError(f"Image path count mismatch in {episode_dir}: {len(image_paths)} vs {len(states)}")
    return int(states.shape[0])


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
    return np.transpose(image, (2, 0, 1))


if __name__ == "__main__":
    main(tyro.cli(Args))
