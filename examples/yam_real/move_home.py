from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import Literal

import numpy as np
import tyro

from examples.yam_real import common


@dataclasses.dataclass
class Args:
    side: Literal["both", "left", "right"] = "both"
    gripper: Literal["crank_4310", "linear_3507", "linear_4310"] = "linear_4310"
    duration_s: float = 5.0
    hz: float = 50.0
    hold_s: float = 1.0
    hold_forever: bool = False
    gripper_target: Literal["current", "open", "closed"] = "current"
    use_gravity_comp: bool = True
    execute: bool = False


def main(args: Args) -> None:
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    if args.hz <= 0:
        raise ValueError("--hz must be positive")
    if args.hold_s < 0:
        raise ValueError("--hold-s must be non-negative")

    common.ensure_i2rt_importable()
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    gripper_type = GripperType.from_string_name(args.gripper)
    sides = _selected_sides(args.side)
    robots = []

    try:
        for side in sides:
            robot = get_yam_robot(
                channel=common.FOLLOWER_CHANNELS[side],
                gripper_type=gripper_type,
                use_gravity_comp=args.use_gravity_comp,
                zero_gravity_mode=False,
            )
            robots.append((side, robot))

        starts: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        for side, robot in robots:
            start = _read_i2rt_state(robot)
            target = _home_target(start, args.gripper_target)
            starts[side] = start
            targets[side] = target
            print(
                f"{side}: start={_fmt(start)} target={_fmt(target)} "
                f"gripper_target={args.gripper_target}",
                flush=True,
            )

        if not args.execute:
            print("Dry run only. Re-run with --execute to command the follower arms.", flush=True)
            return

        steps = max(1, round(args.duration_s * args.hz))
        dt = 1.0 / args.hz
        for i in range(1, steps + 1):
            alpha = i / steps
            for side, robot in robots:
                command = (1.0 - alpha) * starts[side] + alpha * targets[side]
                robot.command_joint_pos(command.astype(np.float32))
            time.sleep(dt)

        for side, robot in robots:
            robot.command_joint_pos(targets[side])

        if args.hold_forever:
            print("Home reached. Holding command until interrupted.", flush=True)
            while True:
                time.sleep(1.0)
        elif args.hold_s > 0:
            time.sleep(args.hold_s)

        for side, robot in robots:
            print(f"{side}: final={_fmt(_read_i2rt_state(robot))}", flush=True)
    finally:
        for _, robot in robots:
            with contextlib.suppress(Exception):
                robot.close()


def _selected_sides(side: str) -> tuple[str, ...]:
    if side == "both":
        return ("left", "right")
    return (side,)


def _read_i2rt_state(robot) -> np.ndarray:
    obs = robot.get_observations()
    arm = np.asarray(obs["joint_pos"], dtype=np.float32)
    gripper = np.asarray(obs.get("gripper_pos", np.array([1.0], dtype=np.float32)), dtype=np.float32)
    if arm.shape != (6,) or gripper.shape != (1,):
        raise RuntimeError(f"Unexpected observation shapes: arm={arm.shape}, gripper={gripper.shape}")
    return np.concatenate([arm, gripper]).astype(np.float32)


def _home_target(start: np.ndarray, gripper_target: str) -> np.ndarray:
    target = np.asarray(start, dtype=np.float32).copy()
    target[:6] = 0.0
    if gripper_target == "open":
        target[6] = 1.0
    elif gripper_target == "closed":
        target[6] = 0.0
    elif gripper_target != "current":
        raise ValueError(f"Unknown gripper target: {gripper_target}")
    return target


def _fmt(values: np.ndarray) -> str:
    return np.array2string(np.asarray(values), precision=4, suppress_small=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
