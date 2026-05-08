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
    output_dir: Path = Path("yam_data/raw")
    episode_name: str | None = None
    task: str = "perform the demonstrated bimanual task"
    fps: float = 20.0
    max_duration_s: float | None = None
    gripper: Literal["crank_4310", "linear_3507", "linear_4310"] = "linear_4310"
    bilateral_kp: float = 0.2
    ee_mass: float | None = None
    use_gravity_comp: bool = False
    record_only_when_synced: bool = True
    startup_check_only: bool = False


class YamLeader:
    def __init__(self, robot):
        self.robot = robot
        self._motor_chain = robot.motor_chain
        self._nominal_kp = np.asarray(robot.get_robot_info()["kp"], dtype=np.float32).copy()

    def get_info(self) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(self.robot.get_observations()["joint_pos"], dtype=np.float32)
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.005)
        gripper_cmd = np.float32(1.0 - encoder_obs[0].position)
        buttons = np.asarray(encoder_obs[0].io_inputs, dtype=np.float32)
        return np.concatenate([qpos, [gripper_cmd]]).astype(np.float32), buttons

    def command_arm_pos(self, qpos_6d: np.ndarray) -> None:
        self.robot.command_joint_pos(np.asarray(qpos_6d, dtype=np.float32))

    def set_bilateral(self, *, enabled: bool, gain: float) -> None:
        if enabled:
            self.robot.update_kp_kd(kp=self._nominal_kp * gain, kd=np.zeros(6))
        else:
            self.robot.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))


class TeleopPair:
    def __init__(self, side: str, leader: YamLeader, follower, bilateral_kp: float) -> None:
        self.side = side
        self.leader = leader
        self.follower = follower
        self.bilateral_kp = bilateral_kp
        self.synchronized = False
        self._last_button = 0.0
        self._last_command = common.get_follower_state(follower)

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        leader_i2rt, buttons = self.leader.get_info()
        follower_openpi = common.get_follower_state(self.follower)
        button = float(buttons[0]) if buttons.size else 0.0

        if button > 0.5 and self._last_button <= 0.5:
            self._toggle_sync(leader_i2rt, follower_openpi)
        self._last_button = button

        if self.synchronized:
            self.follower.command_joint_pos(leader_i2rt)
            self.leader.command_arm_pos(follower_openpi[:6])
            self._last_command = common.i2rt_arm_state_to_openpi(leader_i2rt)
        else:
            self._last_command = follower_openpi

        return follower_openpi, self._last_command

    def close(self) -> None:
        self.leader.set_bilateral(enabled=False, gain=self.bilateral_kp)

    def _toggle_sync(self, leader_i2rt: np.ndarray, follower_openpi: np.ndarray) -> None:
        self.synchronized = not self.synchronized
        if self.synchronized:
            print(f"[{self.side}] sync enabled")
            self.leader.set_bilateral(enabled=True, gain=self.bilateral_kp)
            self.leader.command_arm_pos(leader_i2rt[:6])
            self._slow_move_follower(common.openpi_arm_state_to_i2rt(follower_openpi), leader_i2rt)
        else:
            print(f"[{self.side}] sync disabled")
            self.leader.set_bilateral(enabled=False, gain=self.bilateral_kp)

    def _slow_move_follower(self, start: np.ndarray, target: np.ndarray, duration_s: float = 1.0) -> None:
        steps = 100
        for i in range(1, steps + 1):
            alpha = i / steps
            command = (1.0 - alpha) * start + alpha * target
            self.follower.command_joint_pos(command)
            time.sleep(duration_s / steps)


def main(args: Args) -> None:
    common.ensure_i2rt_importable()
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    gripper_type = GripperType.from_string_name(args.gripper)
    episode_dir = common.make_episode_dir(args.output_dir, args.episode_name)
    print(f"Recording directory: {episode_dir}")
    print("Controls: press 'r' to start/stop recording, 'q' to stop. Leader top buttons toggle sync per side.")

    robots = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    timestamps: list[float] = []
    image_paths: list[dict[str, str]] = []
    recording = False
    start_time = None
    frame_idx = 0
    next_record_t = time.monotonic()

    try:
        follower_l = get_yam_robot(
            channel=common.FOLLOWER_CHANNELS["left"],
            gripper_type=gripper_type,
            use_gravity_comp=args.use_gravity_comp,
            zero_gravity_mode=False,
            ee_mass=args.ee_mass,
        )
        robots.append(follower_l)
        follower_r = get_yam_robot(
            channel=common.FOLLOWER_CHANNELS["right"],
            gripper_type=gripper_type,
            use_gravity_comp=args.use_gravity_comp,
            zero_gravity_mode=False,
            ee_mass=args.ee_mass,
        )
        robots.append(follower_r)
        leader_l = YamLeader(
            get_yam_robot(
                channel=common.LEADER_CHANNELS["left"],
                gripper_type=GripperType.YAM_TEACHING_HANDLE,
                use_gravity_comp=args.use_gravity_comp,
                zero_gravity_mode=True,
                ee_mass=args.ee_mass,
            )
        )
        robots.append(leader_l.robot)
        leader_r = YamLeader(
            get_yam_robot(
                channel=common.LEADER_CHANNELS["right"],
                gripper_type=GripperType.YAM_TEACHING_HANDLE,
                use_gravity_comp=args.use_gravity_comp,
                zero_gravity_mode=True,
                ee_mass=args.ee_mass,
            )
        )
        robots.append(leader_r.robot)
        pair_l = TeleopPair("left", leader_l, follower_l, args.bilateral_kp)
        pair_r = TeleopPair("right", leader_r, follower_r, args.bilateral_kp)

        if args.startup_check_only:
            with common.CameraSet() as cameras:
                frames = cameras.read()
            state_l = common.get_follower_state(follower_l)
            state_r = common.get_follower_state(follower_r)
            leader_l_state, leader_l_buttons = leader_l.get_info()
            leader_r_state, leader_r_buttons = leader_r.get_info()
            print("Startup check OK.")
            print(f"left follower state shape={state_l.shape}, right follower state shape={state_r.shape}")
            print(f"left leader state shape={leader_l_state.shape}, buttons={leader_l_buttons.tolist()}")
            print(f"right leader state shape={leader_r_state.shape}, buttons={leader_r_buttons.tolist()}")
            print(f"camera shapes={ {name: frame.shape for name, frame in frames.items()} }")
            return

        with common.CameraSet() as cameras, common.raw_mode_stdin():
            while True:
                key = common.read_key_nonblocking()
                if key == "q":
                    break
                if key == "r":
                    recording = not recording
                    print(f"recording={recording}")
                    if recording and start_time is None:
                        start_time = time.monotonic()
                    next_record_t = time.monotonic()

                state_l, action_l = pair_l.step()
                state_r, action_r = pair_r.step()
                now = time.monotonic()
                should_record = recording and now >= next_record_t
                if args.record_only_when_synced and not (pair_l.synchronized and pair_r.synchronized):
                    should_record = False

                if should_record:
                    frames = cameras.read()
                    image_paths.append(common.save_frames(episode_dir, frame_idx, frames))
                    states.append(common.pack_bimanual(state_l, state_r))
                    actions.append(common.pack_bimanual(action_l, action_r))
                    timestamps.append(time.time())
                    frame_idx += 1
                    next_record_t += 1.0 / args.fps

                if (
                    args.max_duration_s is not None
                    and start_time is not None
                    and time.monotonic() - start_time >= args.max_duration_s
                ):
                    break

                time.sleep(0.005)
    finally:
        for robot in robots:
            with contextlib.suppress(Exception):
                robot.close()

    if not states:
        raise RuntimeError("No frames recorded. Enable sync and press 'r' before stopping.")

    np.savez_compressed(
        episode_dir / "episode.npz",
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        timestamp=np.asarray(timestamps, dtype=np.float64),
        image_paths=np.asarray(image_paths, dtype=object),
    )
    manifest = common.EpisodeManifest(
        task=args.task,
        fps=args.fps,
        created_at=time.time(),
        state_order=common.STATE_ORDER,
        action_space=common.ACTION_SPACE,
        gripper_convention=common.GRIPPER_CONVENTION,
        camera_paths=common.CAMERA_PATHS,
        leader_channels=common.LEADER_CHANNELS,
        follower_channels=common.FOLLOWER_CHANNELS,
        num_frames=len(states),
    )
    manifest.write(episode_dir / "manifest.json")
    print(f"Recorded {len(states)} frames to {episode_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
