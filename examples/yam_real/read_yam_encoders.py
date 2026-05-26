from __future__ import annotations

import argparse
import dataclasses
import math
import time
from typing import Literal

import numpy as np

from examples.yam_real import common

JOINT_NAMES = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
)


@dataclasses.dataclass(frozen=True)
class MotorRead:
    motor_id: int
    joint_name: str
    motor_type: str
    direction: int
    raw_rad: float
    software_rad: float
    velocity: float
    torque: float
    error_code: str
    temp_mos: float
    temp_rotor: float


def _channel_for(role: str, side: str, override: str | None) -> str:
    if override:
        return override
    if role == "follower":
        return common.FOLLOWER_CHANNELS[side]
    return common.LEADER_CHANNELS[side]


def build_motor_config(arm: str, gripper: str, role: str) -> tuple[list[tuple[int, str]], np.ndarray]:
    common.ensure_i2rt_importable()
    from i2rt.robots.utils import ArmType
    from i2rt.robots.utils import GripperType
    from i2rt.robots.utils import _load_arm_config

    arm_type = ArmType.from_string_name(arm)
    gripper_type = GripperType.from_string_name(gripper)

    hw = _load_arm_config(arm_type)
    motor_list = [(int(can_id), str(motor_type)) for can_id, motor_type in hw.motor_list]
    directions = list(hw.directions)

    # Leaders use the passive teaching handle, so they only expose the 6 arm motors here.
    with_gripper = role == "follower" and gripper_type not in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER)
    if with_gripper:
        motor_list.append((0x07, gripper_type.get_motor_type(arm_type)))
        directions.append(gripper_type.get_motor_direction(arm_type))

    return motor_list, np.asarray(directions, dtype=np.int8)


def read_motors(
    *,
    channel: str,
    arm: str,
    gripper: str,
    role: str,
    motor_off_after_read: bool,
) -> list[MotorRead]:
    common.ensure_i2rt_importable()
    from i2rt.motor_drivers.dm_driver import ControlMode
    from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface

    motor_list, directions = build_motor_config(arm=arm, gripper=gripper, role=role)
    iface = DMSingleMotorCanInterface(
        channel=channel,
        bustype="socketcan",
        control_mode=ControlMode.MIT,
        name=f"yam_encoder_read_{channel}",
    )
    reads: list[MotorRead] = []
    try:
        for _ in range(7):
            iface.try_receive_message(timeout=0.001)

        for index, ((motor_id, motor_type), direction) in enumerate(zip(motor_list, directions, strict=True)):
            feedback = iface.motor_on(motor_id, motor_type)
            raw_rad = float(feedback.position)
            reads.append(
                MotorRead(
                    motor_id=motor_id,
                    joint_name=JOINT_NAMES[index],
                    motor_type=motor_type,
                    direction=int(direction),
                    raw_rad=raw_rad,
                    software_rad=raw_rad * float(direction),
                    velocity=float(feedback.velocity) * float(direction),
                    torque=float(feedback.torque) * float(direction),
                    error_code=str(feedback.error_code),
                    temp_mos=float(feedback.temperature_mos),
                    temp_rotor=float(feedback.temperature_rotor),
                )
            )
            if motor_off_after_read:
                iface.motor_off(motor_id)
            time.sleep(0.003)
    finally:
        iface.close()

    return reads


def print_reads(label: str, reads: list[MotorRead], shoulder_threshold_rad: float) -> None:
    print(f"\n{label}")
    print("joint          id  type    raw_rad   raw_deg  software_rad  software_deg  vel_rad_s  torque  error  temp")
    for read in reads:
        print(
            f"{read.joint_name:<13} {read.motor_id:>2}  {read.motor_type:<6} "
            f"{read.raw_rad:>8.4f} {math.degrees(read.raw_rad):>8.2f} "
            f"{read.software_rad:>12.4f} {math.degrees(read.software_rad):>12.2f} "
            f"{read.velocity:>9.4f} {read.torque:>7.4f} {read.error_code:>6} "
            f"{read.temp_mos:>4.0f}/{read.temp_rotor:<4.0f}"
        )

    shoulder = next((read for read in reads if read.joint_name == "shoulder"), None)
    if shoulder is None:
        return

    shoulder_abs = abs(shoulder.software_rad)
    if shoulder_abs <= shoulder_threshold_rad:
        print(
            f"shoulder check: encoder reports NEAR ZERO "
            f"({shoulder.software_rad:.4f} rad / {math.degrees(shoulder.software_rad):.2f} deg)."
        )
        print("If the arm is physically pitched forward, this points to a shoulder zero/encoder-offset problem.")
    else:
        print(
            f"shoulder check: encoder reports TILTED "
            f"({shoulder.software_rad:.4f} rad / {math.degrees(shoulder.software_rad):.2f} deg)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Read YAM motor encoder positions without robot calibration/control.")
    parser.add_argument("--role", choices=("follower", "leader"), default="follower")
    parser.add_argument("--side", choices=("left", "right", "both"), default="left")
    parser.add_argument("--channel", default=None, help="Override CAN channel; only valid with --side left/right.")
    parser.add_argument("--arm", default="yam")
    parser.add_argument("--gripper", default="linear_4310")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=0.25)
    parser.add_argument("--shoulder-threshold-rad", type=float, default=0.2)
    parser.add_argument(
        "--motor-off-after-read",
        action="store_true",
        help="Send motor-off after each read. Default leaves the current motor enable state alone.",
    )
    args = parser.parse_args()

    if args.channel and args.side == "both":
        raise ValueError("--channel can only be used with --side left or --side right")

    sides: tuple[Literal["left", "right"], ...]
    sides = ("left", "right") if args.side == "both" else (args.side,)

    for sample_index in range(args.samples):
        if args.samples > 1:
            print(f"\n=== sample {sample_index + 1}/{args.samples} ===")
        for side in sides:
            channel = _channel_for(args.role, side, args.channel)
            reads = read_motors(
                channel=channel,
                arm=args.arm,
                gripper=args.gripper,
                role=args.role,
                motor_off_after_read=args.motor_off_after_read,
            )
            print_reads(f"{args.role} {side} ({channel})", reads, args.shoulder_threshold_rad)
        if sample_index + 1 < args.samples:
            time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
