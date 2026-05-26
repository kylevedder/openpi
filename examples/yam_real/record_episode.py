from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
import shutil
import time
from typing import Literal

import numpy as np
import tyro

from examples.yam_real import common
from examples.yam_real import mcap_episode

SYNC_BUTTON_INDEX = 0
RECORD_BUTTON_INDEX = 1
STATUS_CUE_START_PATTERN = (0.12,)
STATUS_CUE_STOP_PATTERN = (0.06, 0.06)
STATUS_CUE_GAP_S = 0.06


@dataclasses.dataclass
class Args:
    output_dir: Path = Path("yam_data/raw")
    episode_name: str | None = None
    task: str = "perform the demonstrated bimanual task"
    fps: float = 50.0
    camera_fps: float = 30.0
    camera_width: int = 640
    camera_height: int = 480
    camera_pixel_format: str = "MJPG"
    camera_startup_timeout_s: float = 5.0
    max_camera_age_s: float = 0.25
    skip_camera_mode_verify: bool = False
    max_duration_s: float | None = None
    gripper: Literal["crank_4310", "linear_3507", "linear_4310"] = "linear_4310"
    bilateral_kp: float = 0.2
    ee_mass: float | None = None
    use_gravity_comp: bool = True
    status_haptic_cue: bool = True
    status_cue_gain: float = 0.08
    record_button_debounce_s: float = 1.0
    record_only_when_synced: bool = True
    startup_check_only: bool = False


class YamLeader:
    def __init__(self, robot):
        self.robot = robot
        self._motor_chain = robot.motor_chain
        self._nominal_kp = np.asarray(robot.get_robot_info()["kp"], dtype=np.float32).copy()
        self._nominal_kd = np.asarray(robot.get_robot_info()["kd"], dtype=np.float32).copy()

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
        self.command_arm_pos(self.current_arm_pos())

    def current_arm_pos(self) -> np.ndarray:
        return np.asarray(self.robot.get_observations()["joint_pos"], dtype=np.float32)

    def pulse_haptic(self, *, pattern_s: tuple[float, ...], gain: float) -> None:
        qpos = self.current_arm_pos()
        kp = self._nominal_kp * gain
        kd = self._nominal_kd * gain
        zero_k = np.zeros_like(kp)
        for duration_s in pattern_s:
            self.robot.update_kp_kd(kp=kp, kd=kd)
            self.command_arm_pos(qpos)
            time.sleep(duration_s)
            self.robot.update_kp_kd(kp=zero_k, kd=zero_k)
            self.command_arm_pos(qpos)
            time.sleep(STATUS_CUE_GAP_S)


class TeleopPair:
    def __init__(self, side: str, leader: YamLeader, follower, bilateral_kp: float) -> None:
        self.side = side
        self.leader = leader
        self.follower = follower
        self.bilateral_kp = bilateral_kp
        self.synchronized = False
        self._last_button = 0.0
        self._last_command = common.get_follower_state(follower)

    def step(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        leader_i2rt, buttons = self.leader.get_info()
        follower_openpi = common.get_follower_state(self.follower)
        button = float(buttons[SYNC_BUTTON_INDEX]) if buttons.size > SYNC_BUTTON_INDEX else 0.0

        if button > 0.5 and self._last_button <= 0.5:
            self._toggle_sync(leader_i2rt, follower_openpi)
        self._last_button = button

        if self.synchronized:
            self.follower.command_joint_pos(leader_i2rt)
            self.leader.command_arm_pos(follower_openpi[:6])
            self._last_command = common.i2rt_arm_state_to_openpi(leader_i2rt)
        else:
            self._last_command = follower_openpi

        return follower_openpi, self._last_command, buttons

    def close(self) -> None:
        self.leader.set_bilateral(enabled=False, gain=self.bilateral_kp)

    def emit_recording_status(self, *, recording: bool, gain: float) -> None:
        pattern = STATUS_CUE_START_PATTERN if recording else STATUS_CUE_STOP_PATTERN
        self.leader.pulse_haptic(pattern_s=pattern, gain=gain)
        self.leader.set_bilateral(enabled=self.synchronized, gain=self.bilateral_kp)

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


def _make_next_episode_dir(output_dir: Path, requested_name: str | None, episode_index: int) -> Path:
    if requested_name is None:
        base_name = time.strftime("episode_%Y%m%d_%H%M%S")
    elif episode_index == 1:
        base_name = requested_name
    else:
        base_name = f"{requested_name}_{episode_index:03d}"

    for suffix in range(1000):
        episode_name = base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        try:
            return common.make_episode_dir(output_dir, episode_name)
        except FileExistsError:
            continue

    raise FileExistsError(f"Could not create a unique episode directory under {output_dir}")


def _print_recording_banner(title: str, details: list[str] | None = None) -> None:
    width = 72
    inner_width = width - 4
    print()
    print("#" * width)
    print(f"# {title.center(inner_width)} #")
    if details:
        print(f"# {' '.center(inner_width)} #")
        for detail in details:
            line = f"{detail[: inner_width - 3]}..." if len(detail) > inner_width else detail
            print(f"# {line.center(inner_width)} #")
    print("#" * width)
    print(flush=True)


def main(args: Args) -> None:
    if args.record_button_debounce_s < 0:
        raise ValueError("--record-button-debounce-s must be non-negative")

    common.ensure_i2rt_importable()
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    gripper_type = GripperType.from_string_name(args.gripper)
    print(f"Output root: {args.output_dir}")
    print("Controls: leader top buttons toggle sync per side; leader bottom button or 'r' starts/stops and saves.")
    print("'q' saves any active recording and exits.")
    print("Recording haptic cue: one leader pulse=start, two leader pulses=stop.")

    robots = []
    episode_dir: Path | None = None
    episode_writer: mcap_episode.YamMcapEpisodeWriter | None = None
    recording = False
    start_time = None
    frame_idx = 0
    next_record_t = time.monotonic()
    last_record_button = 0.0
    last_record_button_toggle_t = -float("inf")
    episode_index = 0
    camera_actual_modes: dict[str, dict] = {}
    camera_config = common.CameraConfig(
        frame_size=(args.camera_width, args.camera_height),
        fps=int(args.camera_fps),
        pixel_format=args.camera_pixel_format,
        verify_mode=not args.skip_camera_mode_verify,
    )

    def reset_episode_buffers() -> None:
        nonlocal frame_idx, start_time, next_record_t
        frame_idx = 0
        start_time = None
        next_record_t = time.monotonic()

    def save_episode() -> bool:
        nonlocal episode_writer
        if episode_dir is None or episode_writer is None or episode_writer.num_steps == 0:
            return False

        num_steps = episode_writer.num_steps
        episode_writer.close()
        episode_writer = None
        print(f"Recorded {num_steps} rows to MCAP episode {episode_dir}")
        return True

    def start_episode() -> None:
        nonlocal episode_dir, episode_index, episode_writer, recording, start_time, next_record_t
        if recording:
            return
        if episode_writer is not None and episode_writer.num_steps:
            save_episode()
            reset_episode_buffers()

        episode_index += 1
        episode_dir = _make_next_episode_dir(args.output_dir, args.episode_name, episode_index)
        episode_writer = mcap_episode.YamMcapEpisodeWriter(
            episode_dir,
            task=args.task,
            fps=args.fps,
            camera_fps=args.camera_fps,
            image_width=args.camera_width,
            image_height=args.camera_height,
            camera_config={
                "requested": camera_config.as_manifest(),
                "actual_modes": camera_actual_modes,
                "camera_paths": common.CAMERA_PATHS,
            },
        )
        reset_episode_buffers()
        recording = True
        start_time = time.monotonic()
        next_record_t = start_time
        print(f"recording={recording}")
        print(f"Recording directory: {episode_dir}")
        _print_recording_banner(
            "RECORDING STARTED",
            [
                f"Episode: {episode_dir}",
                f"FPS: {args.fps:g}",
            ],
        )
        if args.status_haptic_cue:
            pair_l.emit_recording_status(recording=True, gain=args.status_cue_gain)
            pair_r.emit_recording_status(recording=True, gain=args.status_cue_gain)

    def stop_episode() -> None:
        nonlocal episode_dir, episode_writer, recording
        was_recording = recording
        recording = False
        if was_recording:
            print(f"recording={recording}")
            _print_recording_banner(
                "RECORDING STOPPED",
                [
                    f"Rows captured: {episode_writer.num_steps if episode_writer is not None else 0}",
                    "Saving episode now",
                ],
            )
            if args.status_haptic_cue:
                pair_l.emit_recording_status(recording=False, gain=args.status_cue_gain)
                pair_r.emit_recording_status(recording=False, gain=args.status_cue_gain)

        saved = save_episode()
        if saved:
            print("Ready for next episode.")
        elif episode_dir is not None:
            print("No frames recorded; discarding empty episode directory.")
            if episode_writer is not None:
                with contextlib.suppress(Exception):
                    episode_writer.close()
                episode_writer = None
            shutil.rmtree(episode_dir, ignore_errors=True)

        episode_dir = None
        reset_episode_buffers()

    def toggle_recording() -> None:
        if recording:
            stop_episode()
        else:
            start_episode()

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
            with common.CameraSet(config=camera_config) as cameras:
                frames = cameras.read()
                camera_properties = cameras.capture_properties()
            state_l = common.get_follower_state(follower_l)
            state_r = common.get_follower_state(follower_r)
            leader_l_state, leader_l_buttons = leader_l.get_info()
            leader_r_state, leader_r_buttons = leader_r.get_info()
            print("Startup check OK.")
            print(f"left follower state shape={state_l.shape}, right follower state shape={state_r.shape}")
            print(f"left leader state shape={leader_l_state.shape}, buttons={leader_l_buttons.tolist()}")
            print(f"right leader state shape={leader_r_state.shape}, buttons={leader_r_buttons.tolist()}")
            print(f"camera shapes={ {name: frame.shape for name, frame in frames.items()} }")
            print(f"camera properties={camera_properties}")
            return

        with common.AsyncCameraSet(config=camera_config, startup_timeout_s=args.camera_startup_timeout_s) as cameras, common.raw_mode_stdin():
            camera_actual_modes = cameras.capture_properties()
            print("Camera capture mode: async_latest_linuxpy_v4l2")
            print(f"Camera requested mode: {camera_config.as_manifest()}")
            print(f"Camera actual modes: {camera_actual_modes}")
            while True:
                key = common.read_key_nonblocking()
                if key == "q":
                    if recording or (episode_writer is not None and episode_writer.num_steps):
                        stop_episode()
                    break
                if key == "r":
                    toggle_recording()

                state_l, action_l, buttons_l = pair_l.step()
                state_r, action_r, buttons_r = pair_r.step()
                now = time.monotonic()
                record_button = max(
                    float(buttons_l[RECORD_BUTTON_INDEX]) if buttons_l.size > RECORD_BUTTON_INDEX else 0.0,
                    float(buttons_r[RECORD_BUTTON_INDEX]) if buttons_r.size > RECORD_BUTTON_INDEX else 0.0,
                )
                if record_button > 0.5 and last_record_button <= 0.5:
                    if now - last_record_button_toggle_t >= args.record_button_debounce_s:
                        toggle_recording()
                        last_record_button_toggle_t = now
                    else:
                        remaining_s = args.record_button_debounce_s - (now - last_record_button_toggle_t)
                        print(f"Ignoring record button bounce ({remaining_s:.2f}s debounce remaining).")
                last_record_button = record_button
                should_record = recording and now >= next_record_t
                if args.record_only_when_synced and not (pair_l.synchronized and pair_r.synchronized):
                    should_record = False

                if should_record:
                    if episode_dir is None or episode_writer is None:
                        raise RuntimeError("Recording is active without an episode directory.")
                    snapshot = cameras.snapshot(max_age_s=args.max_camera_age_s)
                    episode_writer.write_step(
                        camera_snapshot=snapshot,
                        state=common.pack_bimanual(state_l, state_r),
                        action=common.pack_bimanual(action_l, action_r),
                        timestamp_ns=time.time_ns(),
                    )
                    frame_idx += 1
                    next_record_t += 1.0 / args.fps

                if (
                    args.max_duration_s is not None
                    and start_time is not None
                    and time.monotonic() - start_time >= args.max_duration_s
                    and recording
                ):
                    stop_episode()

                time.sleep(0.005)
    except KeyboardInterrupt:
        print("Interrupted; saving recorded frames before exit.")
        if recording or (episode_writer is not None and episode_writer.num_steps):
            stop_episode()
    finally:
        if episode_writer is not None:
            with contextlib.suppress(Exception):
                episode_writer.close()
        for robot in robots:
            with contextlib.suppress(Exception):
                robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
