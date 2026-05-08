from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
import time
from typing import Literal

import numpy as np
import tyro

from examples.yam_real import common


@dataclasses.dataclass
class Args:
    episode_dir: Path
    execute: bool = False
    fps: float | None = None
    gripper: Literal["crank_4310", "linear_3507", "linear_4310"] = "linear_4310"
    max_steps: int | None = None
    max_arm_step_rad: float = 0.03
    max_gripper_step: float = 0.03
    move_to_first_s: float = 2.0
    use_gravity_comp: bool = False


def main(args: Args) -> None:
    manifest, arrays = common.load_episode(args.episode_dir)
    actions = np.asarray(arrays["action"], dtype=np.float32)
    if args.max_steps is not None:
        actions = actions[: args.max_steps]
    fps = args.fps or float(manifest["fps"])

    print(f"Episode: {args.episode_dir}")
    print(f"Frames: {len(actions)}, fps={fps}, execute={args.execute}")
    print(f"Task: {manifest.get('task', '')}")
    if not args.execute:
        print("Dry run only. Re-run with --execute to command the followers.")
        print(f"First action: {actions[0].round(4).tolist()}")
        print(f"Last action: {actions[-1].round(4).tolist()}")
        return

    common.ensure_i2rt_importable()
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    gripper_type = GripperType.from_string_name(args.gripper)
    robots = []
    try:
        follower_l = get_yam_robot(
            channel=common.FOLLOWER_CHANNELS["left"],
            gripper_type=gripper_type,
            use_gravity_comp=args.use_gravity_comp,
            zero_gravity_mode=False,
        )
        robots.append(follower_l)
        follower_r = get_yam_robot(
            channel=common.FOLLOWER_CHANNELS["right"],
            gripper_type=gripper_type,
            use_gravity_comp=args.use_gravity_comp,
            zero_gravity_mode=False,
        )
        robots.append(follower_r)

        first_l, first_r = common.split_bimanual(actions[0])
        follower_l.move_joints(common.openpi_arm_state_to_i2rt(first_l), time_interval_s=args.move_to_first_s)
        follower_r.move_joints(common.openpi_arm_state_to_i2rt(first_r), time_interval_s=args.move_to_first_s)

        command = common.pack_bimanual(common.get_follower_state(follower_l), common.get_follower_state(follower_r))
        dt = 1.0 / fps
        next_t = time.monotonic()
        for target in actions:
            command = common.clip_bimanual_delta(
                command,
                target,
                max_arm_step_rad=args.max_arm_step_rad,
                max_gripper_step=args.max_gripper_step,
            )
            left, right = common.split_bimanual(command)
            follower_l.command_joint_pos(common.openpi_arm_state_to_i2rt(left))
            follower_r.command_joint_pos(common.openpi_arm_state_to_i2rt(right))
            next_t += dt
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        for robot in robots:
            with contextlib.suppress(Exception):
                robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
