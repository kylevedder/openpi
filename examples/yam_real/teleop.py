from __future__ import annotations

import time
from typing import Any

import numpy as np

from examples.yam_real import common

SYNC_BUTTON_INDEX = 0
RECORD_BUTTON_INDEX = 1
STATUS_CUE_START_PATTERN = (0.12,)
STATUS_CUE_STOP_PATTERN = (0.06, 0.06)
STATUS_CUE_GAP_S = 0.06


class YamLeader:
    def __init__(self, robot: Any) -> None:
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
    def __init__(self, side: str, leader: YamLeader, follower: Any, bilateral_kp: float) -> None:
        self.side = side
        self.leader = leader
        self.follower = follower
        self.bilateral_kp = bilateral_kp
        self.synchronized = False
        self._last_button = 0.0
        self._last_command = common.get_follower_state(follower)

    def step(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, action, buttons, _timings_ms = self.step_with_timings()
        return state, action, buttons

    def step_with_timings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        timings_ms: dict[str, float] = {}
        total_start = time.perf_counter()
        leader_start = time.perf_counter()
        leader_i2rt, buttons = self.leader.get_info()
        timings_ms["leader_get_info"] = _elapsed_ms(leader_start)
        follower_start = time.perf_counter()
        follower_openpi = common.get_follower_state(self.follower)
        timings_ms["follower_get_state"] = _elapsed_ms(follower_start)
        button = float(buttons[SYNC_BUTTON_INDEX]) if buttons.size > SYNC_BUTTON_INDEX else 0.0

        if button > 0.5 and self._last_button <= 0.5:
            sync_start = time.perf_counter()
            self._toggle_sync(leader_i2rt, follower_openpi)
            timings_ms["sync_toggle"] = _elapsed_ms(sync_start)
        self._last_button = button

        if self.synchronized:
            follower_command_start = time.perf_counter()
            self.follower.command_joint_pos(leader_i2rt)
            timings_ms["follower_command_joint_pos"] = _elapsed_ms(follower_command_start)
            leader_command_start = time.perf_counter()
            self.leader.command_arm_pos(follower_openpi[:6])
            timings_ms["leader_command_arm_pos"] = _elapsed_ms(leader_command_start)
            self._last_command = common.i2rt_arm_state_to_openpi(leader_i2rt)
        else:
            self._last_command = follower_openpi

        timings_ms["total"] = _elapsed_ms(total_start)
        return follower_openpi, self._last_command, buttons, timings_ms

    def close(self) -> None:
        self.leader.set_bilateral(enabled=False, gain=self.bilateral_kp)

    def emit_recording_status(self, *, recording: bool, gain: float) -> None:
        pattern = STATUS_CUE_START_PATTERN if recording else STATUS_CUE_STOP_PATTERN
        self.leader.pulse_haptic(pattern_s=pattern, gain=gain)
        self.leader.set_bilateral(enabled=self.synchronized, gain=self.bilateral_kp)

    def _toggle_sync(self, leader_i2rt: np.ndarray, follower_openpi: np.ndarray) -> None:
        self.synchronized = not self.synchronized
        if self.synchronized:
            print(f"[{self.side}] sync enabled", flush=True)
            self.leader.set_bilateral(enabled=True, gain=self.bilateral_kp)
            self.leader.command_arm_pos(leader_i2rt[:6])
            self._slow_move_follower(common.openpi_arm_state_to_i2rt(follower_openpi), leader_i2rt)
        else:
            print(f"[{self.side}] sync disabled", flush=True)
            self.leader.set_bilateral(enabled=False, gain=self.bilateral_kp)

    def _slow_move_follower(self, start: np.ndarray, target: np.ndarray, duration_s: float = 1.0) -> None:
        steps = 100
        for i in range(1, steps + 1):
            alpha = i / steps
            command = (1.0 - alpha) * start + alpha * target
            self.follower.command_joint_pos(command)
            time.sleep(duration_s / steps)


def _elapsed_ms(start: float) -> float:
    return 1000.0 * (time.perf_counter() - start)
