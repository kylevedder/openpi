from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import Literal

import cv2
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro

from examples.yam_real import common


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    prompt: str = "perform the demonstrated bimanual task"
    execute: bool = False
    fps: float = 20.0
    max_steps: int = 200
    action_horizon: int = 8
    gripper: Literal["crank_4310", "linear_3507", "linear_4310"] = "linear_4310"
    max_arm_step_rad: float = 0.02
    max_gripper_step: float = 0.02
    use_gravity_comp: bool = False


def main(args: Args) -> None:
    common.ensure_i2rt_importable()
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    policy_port = None if args.host.startswith("ws") else args.port
    policy = action_chunk_broker.ActionChunkBroker(
        websocket_client_policy.WebsocketClientPolicy(host=args.host, port=policy_port),
        action_horizon=args.action_horizon,
    )
    gripper_type = GripperType.from_string_name(args.gripper)
    robots = []
    policy_endpoint = args.host if policy_port is None else f"{args.host}:{policy_port}"
    print(f"execute={args.execute}; host={policy_endpoint}; max_steps={args.max_steps}")
    if not args.execute:
        print("Dry run only. Observations will be sent to the server, but followers will not be commanded.")

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
        command = common.pack_bimanual(common.get_follower_state(follower_l), common.get_follower_state(follower_r))

        with common.CameraSet() as cameras:
            dt = 1.0 / args.fps
            next_t = time.monotonic()
            for step in range(args.max_steps):
                state = common.pack_bimanual(
                    common.get_follower_state(follower_l), common.get_follower_state(follower_r)
                )
                observation = {
                    "state": state,
                    "images": _policy_images(cameras.read()),
                    "prompt": args.prompt,
                }
                result = policy.infer(observation)
                action = np.asarray(result["actions"], dtype=np.float32)
                if action.shape != (14,):
                    raise RuntimeError(f"Expected one 14D action from broker, got {action.shape}")
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Policy returned non-finite action")

                command = common.clip_bimanual_delta(
                    command,
                    action,
                    max_arm_step_rad=args.max_arm_step_rad,
                    max_gripper_step=args.max_gripper_step,
                )
                if args.execute:
                    left, right = common.split_bimanual(command)
                    follower_l.command_joint_pos(common.openpi_arm_state_to_i2rt(left))
                    follower_r.command_joint_pos(common.openpi_arm_state_to_i2rt(right))
                print(f"step={step} action_norm={float(np.linalg.norm(action)):.4f}")

                next_t += dt
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        for robot in robots:
            with contextlib.suppress(Exception):
                robot.close()


def _policy_images(frames_bgr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    images = {}
    for name, frame_bgr in frames_bgr.items():
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, 224, 224))
        images[name] = np.transpose(rgb, (2, 0, 1))
    return images


if __name__ == "__main__":
    main(tyro.cli(Args))
