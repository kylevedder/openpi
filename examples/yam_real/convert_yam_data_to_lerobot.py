from __future__ import annotations

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

from examples.yam_real import mcap_episode


@dataclasses.dataclass
class Args:
    raw_dir: Path = Path("yam_data/raw")
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

    episode_dirs = mcap_episode.iter_episode_dirs(args.raw_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No recorded YAM episodes found under {args.raw_dir}")

    fps = round(_episode_fps(episode_dirs[0]))

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
        _add_mcap_episode(dataset, episode_dir)
        dataset.save_episode()

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    if args.push_to_hub:
        dataset.push_to_hub()
    print(f"Wrote LeRobot dataset to {output_path}")


def _image_feature(mode: str) -> dict:
    return {
        "dtype": mode,
        "shape": (3, 480, 640),
        "names": ["channel", "height", "width"],
    }


def _episode_fps(episode_dir: Path) -> float:
    return mcap_episode.read_episode(episode_dir, decode_images=False).fps


def _add_mcap_episode(dataset: LeRobotDataset, episode_dir: Path) -> None:
    episode = mcap_episode.read_episode(episode_dir, decode_images=True)
    if episode.images is None:
        raise RuntimeError(f"MCAP episode did not decode images: {episode_dir}")
    _validate_state_action(episode_dir, episode.state, episode.action)
    for idx in range(len(episode.state)):
        dataset.add_frame(
            {
                "observation.state": episode.state[idx],
                "action": episode.action[idx],
                "task": episode.task,
                "observation.images.cam_high": _to_chw(episode.images["cam_high"][idx]),
                "observation.images.cam_left_wrist": _to_chw(episode.images["cam_left_wrist"][idx]),
                "observation.images.cam_right_wrist": _to_chw(episode.images["cam_right_wrist"][idx]),
            }
        )


def _validate_state_action(episode_dir: Path, states: np.ndarray, actions: np.ndarray) -> None:
    if states.shape != actions.shape or states.shape[-1] != 14:
        raise RuntimeError(f"Bad state/action shapes in {episode_dir}: {states.shape}, {actions.shape}")


def _to_chw(image_rgb: np.ndarray) -> np.ndarray:
    return np.transpose(image_rgb, (2, 0, 1))


if __name__ == "__main__":
    main(tyro.cli(Args))
