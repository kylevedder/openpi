from __future__ import annotations

from collections.abc import Callable
import contextlib
import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Literal

import cv2
import numpy as np
import tyro

from examples.yam_real import common
from examples.yam_real import read_yam_encoders

ProbeMode = Literal[
    "camera-controls",
    "camera-single",
    "camera-all-sequential",
    "camera-grab-retrieve",
    "image-save",
    "camera-exposure-sweep",
    "can-raw",
    "robot-state",
    "teleop-step-nosync",
    "record-loop-dry",
]

DEFAULT_MODES: tuple[ProbeMode, ...] = (
    "camera-controls",
    "camera-single",
    "camera-all-sequential",
    "camera-grab-retrieve",
    "image-save",
    "can-raw",
    "robot-state",
    "teleop-step-nosync",
    "record-loop-dry",
)

FIFTY_HZ_BUDGET_MS = 20.0
TWENTY_HZ_BUDGET_MS = 50.0
LEADER_GET_INFO_SLEEP_MS = 5.0

Sample = dict[str, Any]
StageFn = Callable[[], Any]
ClockFn = Callable[[], float]


@dataclasses.dataclass(frozen=True)
class Args:
    duration_s: float = 12.0
    fps: float = 50.0
    modes: tuple[ProbeMode, ...] = DEFAULT_MODES
    output_dir: Path = Path("yam_data/logs/refresh_rate")
    scratch_dir: Path = Path("/tmp/yam_refresh_probe")
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 30.0
    camera_pixel_format: str = "MJPG"
    skip_camera_mode_verify: bool = False
    record_camera_capture_mode: Literal["async_latest", "blocking"] = "async_latest"
    camera_startup_timeout_s: float = 5.0
    max_camera_age_s: float = 0.25
    warmup_frames: int = 3
    max_samples_per_mode: int = 800
    image_save_max_samples: int = 300
    can_samples: int = 10
    allow_camera_control_writes: bool = False
    exposure_sweep: tuple[float, ...] = (40.0, 80.0, 120.0, 156.0)
    arm: str = "yam"
    gripper: Literal["crank_4310", "linear_3507", "linear_4310"] = "linear_4310"
    ee_mass: float | None = None
    use_gravity_comp: bool = False


def main(args: Args) -> None:
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    started_at = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"yam_refresh_rate_{started_at}"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_run_dir = args.scratch_dir / run_id
    scratch_run_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    context = _system_context()
    records.append({"type": "context", "created_at": started_at, "args": dataclasses.asdict(args), "context": context})

    samples: list[Sample] = []
    for mode in args.modes:
        mode_start = time.perf_counter()
        print(f"\n=== {mode} ===", flush=True)
        try:
            mode_samples = _run_mode(mode, args, scratch_run_dir)
        except Exception as exc:
            mode_samples = [_error_sample(mode, str(exc))]
            print(f"[error] {mode}: {exc}", flush=True)
        elapsed_ms = _elapsed_ms(mode_start)
        samples.extend(mode_samples)
        records.extend(mode_samples)
        records.append({"type": "mode_complete", "mode": mode, "elapsed_ms": elapsed_ms, "samples": len(mode_samples)})
        print(f"{mode}: {len(mode_samples)} sample record(s), elapsed={elapsed_ms:.1f} ms", flush=True)

    summary = analyze_samples(samples, fps=args.fps)
    records.append({"type": "summary", "summary": summary})
    _write_outputs(records, summary, output_dir, run_id)

    print("\nYAM refresh-rate probe complete")
    print(f"Dominant bottleneck: {summary.get('dominant_bottleneck', 'unknown')}")
    print(f"Report: {output_dir / 'latest.md'}")
    print(f"JSONL: {output_dir / 'latest.jsonl'}")


def _run_mode(mode: ProbeMode, args: Args, scratch_run_dir: Path) -> list[Sample]:
    if mode == "camera-controls":
        return _probe_camera_controls(args)
    if mode == "camera-single":
        return _probe_camera_single(args)
    if mode == "camera-all-sequential":
        return _probe_camera_all_sequential(args)
    if mode == "camera-grab-retrieve":
        return _probe_camera_grab_retrieve(args)
    if mode == "image-save":
        return _probe_image_save(args, scratch_run_dir / "image-save")
    if mode == "camera-exposure-sweep":
        return _probe_camera_exposure_sweep(args)
    if mode == "can-raw":
        return _probe_can_raw(args)
    if mode == "robot-state":
        return _probe_robot_state(args)
    if mode == "teleop-step-nosync":
        return _probe_teleop_step_no_sync(args)
    if mode == "record-loop-dry":
        return _probe_record_loop_dry(args, scratch_run_dir / "record-loop-dry")
    raise ValueError(f"Unsupported probe mode: {mode}")


def measure_stage_sequence(
    stages: list[tuple[str, StageFn]],
    *,
    clock: ClockFn = time.perf_counter,
) -> tuple[dict[str, float], dict[str, Any]]:
    durations_ms = {}
    outputs = {}
    total_start = clock()
    for name, stage in stages:
        start = clock()
        outputs[name] = stage()
        durations_ms[name] = 1000.0 * (clock() - start)
    durations_ms["total"] = 1000.0 * (clock() - total_start)
    return durations_ms, outputs


def analyze_samples(samples: list[Sample], *, fps: float) -> dict[str, Any]:
    sample_records = [sample for sample in samples if sample.get("type") == "sample"]
    error_records = [sample for sample in samples if sample.get("type") == "error"]
    groups = _summarize_sample_groups(sample_records)
    findings = _build_findings(groups, error_records)
    dominant = _dominant_bottleneck(groups, findings, fps=fps)
    return {
        "fps": fps,
        "budgets": {
            "50hz_ms": FIFTY_HZ_BUDGET_MS,
            "20hz_ms": TWENTY_HZ_BUDGET_MS,
            "requested_fps_ms": 1000.0 / fps,
        },
        "sample_count": len(sample_records),
        "error_count": len(error_records),
        "groups": groups,
        "findings": findings,
        "dominant_bottleneck": dominant,
    }


def _probe_camera_controls(args: Args) -> list[Sample]:
    samples: list[Sample] = []
    config = dataclasses.replace(_camera_config(args), verify_mode=False)
    for camera_name, camera_path in common.CAMERA_PATHS.items():
        camera = _open_camera(camera_name, camera_path, config=config)
        try:
            samples.append(
                _sample(
                    "camera-controls",
                    target=camera_name,
                    sample_index=0,
                    durations_ms={"total": 0.0},
                    extra={
                        "camera_path": camera_path,
                        "requested_config": config.as_manifest(),
                        "capture_properties": camera.capture_properties(),
                        "capture_controls": {},
                        "backend": "linuxpy-v4l2",
                    },
                )
            )
        finally:
            camera.close()
    return samples


def _probe_camera_single(args: Args) -> list[Sample]:
    samples: list[Sample] = []
    for camera_name, camera_path in common.CAMERA_PATHS.items():
        camera = _open_camera(camera_name, camera_path, config=_camera_config(args))
        try:
            _warmup_capture(camera, args.warmup_frames)
            camera_mode = _capture_properties(camera)
            deadline = time.perf_counter() + args.duration_s
            sample_index = 0
            while time.perf_counter() < deadline and sample_index < args.max_samples_per_mode:
                durations_ms, outputs = measure_stage_sequence([("read", lambda camera=camera: _read_capture(camera))])
                frame = outputs["read"]
                samples.append(
                    _sample(
                        "camera-single",
                        target=camera_name,
                        sample_index=sample_index,
                        durations_ms=durations_ms,
                        extra={
                            "camera_path": camera_path,
                            "capture_properties": camera_mode,
                            "frame": _frame_stats(frame),
                        },
                    )
                )
                sample_index += 1
        finally:
            camera.close()
    return samples


def _probe_camera_all_sequential(args: Args) -> list[Sample]:
    samples: list[Sample] = []
    with common.CameraSet(config=_camera_config(args)) as cameras:
        _warmup_camera_set(cameras, args.warmup_frames)
        properties = cameras.capture_properties()
        deadline = time.perf_counter() + args.duration_s
        sample_index = 0
        while time.perf_counter() < deadline and sample_index < args.max_samples_per_mode:
            durations_ms: dict[str, float] = {}
            frames = {}
            total_start = time.perf_counter()
            for camera_name, camera in cameras._captures.items():  # noqa: SLF001
                start = time.perf_counter()
                frames[camera_name] = _read_capture(camera)
                durations_ms[f"read_{camera_name}"] = _elapsed_ms(start)
            durations_ms["total"] = _elapsed_ms(total_start)
            samples.append(
                _sample(
                    "camera-all-sequential",
                    target="all",
                    sample_index=sample_index,
                    durations_ms=durations_ms,
                    extra={
                        "capture_properties": properties,
                        "frames": {name: _frame_stats(frame) for name, frame in frames.items()},
                    },
                )
            )
            sample_index += 1
    return samples


def _probe_camera_grab_retrieve(args: Args) -> list[Sample]:
    samples: list[Sample] = []
    with common.AsyncCameraSet(
        config=_camera_config(args),
        startup_timeout_s=args.camera_startup_timeout_s,
    ) as cameras:
        deadline = time.perf_counter() + args.duration_s
        sample_index = 0
        while time.perf_counter() < deadline and sample_index < args.max_samples_per_mode:
            total_start = time.perf_counter()
            snapshot = cameras.snapshot(max_age_s=args.max_camera_age_s)
            durations_ms = {"snapshot_latest": _elapsed_ms(total_start), "total": _elapsed_ms(total_start)}

            samples.append(
                _sample(
                    "camera-grab-retrieve",
                    target="all",
                    sample_index=sample_index,
                    durations_ms=durations_ms,
                    extra={
                        "backend": "linuxpy-async-latest",
                        "note": "OpenCV grab/retrieve is not used by the MonoPI-style camera backend.",
                        "frames": {name: _frame_stats(frame) for name, frame in snapshot.frames.items()},
                        "camera_frame_metadata": snapshot.metadata,
                    },
                )
            )
            sample_index += 1
    return samples


def _probe_image_save(args: Args, output_dir: Path) -> list[Sample]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with common.CameraSet(config=_camera_config(args)) as cameras:
        _warmup_camera_set(cameras, args.warmup_frames)
        frames = cameras.read()

    for camera_name in common.CAMERA_NAMES:
        (output_dir / "images" / camera_name).mkdir(parents=True, exist_ok=True)

    samples = []
    deadline = time.perf_counter() + args.duration_s
    sample_index = 0
    max_samples = min(args.max_samples_per_mode, args.image_save_max_samples)
    while time.perf_counter() < deadline and sample_index < max_samples:
        durations_ms: dict[str, float] = {}
        total_start = time.perf_counter()
        for camera_name, frame in frames.items():
            path = output_dir / "images" / camera_name / f"{sample_index:06d}.jpg"
            start = time.perf_counter()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError(f"Failed to write image {path}")
            durations_ms[f"save_{camera_name}"] = _elapsed_ms(start)
        durations_ms["total"] = _elapsed_ms(total_start)
        samples.append(
            _sample(
                "image-save",
                target="all",
                sample_index=sample_index,
                durations_ms=durations_ms,
                extra={"scratch_dir": str(output_dir), "camera_count": len(frames)},
            )
        )
        sample_index += 1
    return samples


def _probe_camera_exposure_sweep(args: Args) -> list[Sample]:
    del args
    return [
        _error_sample(
            "camera-exposure-sweep",
            "Exposure writes are not implemented in the linuxpy MonoPI-style backend; use v4l2-ctl manually.",
        )
    ]


def _probe_can_raw(args: Args) -> list[Sample]:
    samples: list[Sample] = []
    for role, side, channel in _can_targets():
        try:
            samples.extend(_probe_can_channel(args, role=role, side=side, channel=channel))
        except Exception as exc:
            samples.append(_error_sample("can-raw", f"{role}/{side}/{channel}: {exc}", target=f"{role}_{side}"))
    return samples


def _probe_can_channel(args: Args, *, role: str, side: str, channel: str) -> list[Sample]:
    common.ensure_i2rt_importable()
    from i2rt.motor_drivers.dm_driver import ControlMode
    from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface

    motor_list, _directions = read_yam_encoders.build_motor_config(arm=args.arm, gripper=args.gripper, role=role)
    iface = DMSingleMotorCanInterface(
        channel=channel,
        bustype="socketcan",
        control_mode=ControlMode.MIT,
        name=f"yam_refresh_probe_{channel}",
    )
    samples: list[Sample] = []
    try:
        for _ in range(len(motor_list)):
            iface.try_receive_message(timeout=0.001)

        for sample_index in range(args.can_samples):
            durations_ms: dict[str, float] = {}
            total_start = time.perf_counter()
            for motor_id, motor_type in motor_list:
                start = time.perf_counter()
                iface.motor_on(motor_id, motor_type)
                durations_ms[f"motor_{motor_id:02d}"] = _elapsed_ms(start)
            durations_ms["total"] = _elapsed_ms(total_start)
            samples.append(
                _sample(
                    "can-raw",
                    target=f"{role}_{side}",
                    sample_index=sample_index,
                    durations_ms=durations_ms,
                    extra={"channel": channel, "motor_count": len(motor_list)},
                )
            )
    finally:
        iface.close()
    return samples


def _probe_robot_state(args: Args) -> list[Sample]:
    robots = _open_yam_robots(args)
    try:
        return _sample_robot_state(args, robots, mode="robot-state")
    finally:
        _close_robots(robots)


def _probe_teleop_step_no_sync(args: Args) -> list[Sample]:
    robots = _open_yam_robots(args)
    try:
        samples: list[Sample] = []
        deadline = time.perf_counter() + args.duration_s
        sample_index = 0
        while time.perf_counter() < deadline and sample_index < args.max_samples_per_mode:
            for side in ("left", "right"):
                leader = robots[f"leader_{side}"]
                follower = robots[f"follower_{side}"]

                def read_leader(leader=leader):
                    return leader.get_info()

                def read_follower(follower=follower):
                    return common.get_follower_state(follower)

                durations_ms, outputs = measure_stage_sequence(
                    [
                        ("leader_get_info", read_leader),
                        ("follower_get_state", read_follower),
                    ]
                )
                leader_state, buttons = outputs["leader_get_info"]
                samples.append(
                    _sample(
                        "teleop-step-nosync",
                        target=side,
                        sample_index=sample_index,
                        durations_ms=durations_ms,
                        extra={
                            "leader_state_shape": list(np.asarray(leader_state).shape),
                            "buttons": np.asarray(buttons, dtype=float).tolist(),
                            "sync_transitions_ignored": True,
                            "no_command_forwarding": True,
                        },
                    )
                )
            sample_index += 1
        return samples
    finally:
        _close_robots(robots)


def _probe_record_loop_dry(args: Args, scratch_dir: Path) -> list[Sample]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    robots = _open_yam_robots(args)
    try:
        camera_context = _record_loop_camera_context(args)
        with camera_context as cameras:
            if isinstance(cameras, common.CameraSet):
                _warmup_camera_set(cameras, args.warmup_frames)
            variants = (
                ("robots_only", True, False, False),
                ("cameras_only", False, True, False),
                ("robots_cameras_no_save", True, True, False),
                ("robots_cameras_scratch_save", True, True, True),
            )
            for variant_name, include_robots, include_cameras, include_save in variants:
                variant_dir = scratch_dir / variant_name
                if include_save:
                    _prepare_image_output_dir(variant_dir)
                samples.extend(
                    _run_record_loop_variant(
                        args,
                        robots=robots,
                        cameras=cameras,
                        variant_name=variant_name,
                        output_dir=variant_dir,
                        include_robots=include_robots,
                        include_cameras=include_cameras,
                        include_save=include_save,
                    )
                )
    finally:
        _close_robots(robots)
    return samples


def _sample_robot_state(args: Args, robots: dict[str, Any], *, mode: str) -> list[Sample]:
    samples: list[Sample] = []
    targets: tuple[tuple[str, Callable[[], Any], str], ...] = (
        ("follower_left", lambda: common.get_follower_state(robots["follower_left"]), "follower get_follower_state"),
        ("follower_right", lambda: common.get_follower_state(robots["follower_right"]), "follower get_follower_state"),
        ("leader_left", lambda: robots["leader_left"].get_info(), "leader YamLeader.get_info"),
        ("leader_right", lambda: robots["leader_right"].get_info(), "leader YamLeader.get_info"),
    )
    deadline = time.perf_counter() + args.duration_s
    sample_index = 0
    while time.perf_counter() < deadline and sample_index < args.max_samples_per_mode:
        for target, stage, label in targets:
            durations_ms, outputs = measure_stage_sequence([("read", stage)])
            output = outputs["read"]
            extra = {
                "read_label": label,
                "leader_get_info_sleep_ms": LEADER_GET_INFO_SLEEP_MS if target.startswith("leader_") else 0.0,
            }
            if target.startswith("leader_"):
                leader_state, buttons = output
                extra["state_shape"] = list(np.asarray(leader_state).shape)
                extra["buttons"] = np.asarray(buttons, dtype=float).tolist()
            else:
                extra["state_shape"] = list(np.asarray(output).shape)
            samples.append(
                _sample(
                    mode,
                    target=target,
                    sample_index=sample_index,
                    durations_ms=durations_ms,
                    extra=extra,
                )
            )
        sample_index += 1
    return samples


def _run_record_loop_variant(
    args: Args,
    *,
    robots: dict[str, Any],
    cameras: common.CameraSet | common.AsyncCameraSet,
    variant_name: str,
    output_dir: Path,
    include_robots: bool,
    include_cameras: bool,
    include_save: bool,
) -> list[Sample]:
    samples: list[Sample] = []
    next_record_t = time.perf_counter()
    deadline = time.perf_counter() + args.duration_s
    frame_index = 0
    while time.perf_counter() < deadline and frame_index < args.max_samples_per_mode:
        loop_start = time.perf_counter()
        durations_ms: dict[str, float] = {}
        frames = None

        if include_robots:
            robot_start = time.perf_counter()
            _read_all_robot_states(robots)
            durations_ms["robot_loop"] = _elapsed_ms(robot_start)

        now = time.perf_counter()
        if now >= next_record_t:
            scheduler_lag_ms = 1000.0 * max(0.0, now - next_record_t)
            camera_metadata = {}
            if include_cameras:
                camera_start = time.perf_counter()
                snapshot = _record_loop_camera_snapshot(cameras, frame_index, max_age_s=args.max_camera_age_s)
                frames = snapshot.frames
                camera_metadata = snapshot.metadata
                durations_ms["camera_read"] = _elapsed_ms(camera_start)
            if include_save:
                if frames is None:
                    raise RuntimeError("record-loop save variant requires camera frames")
                save_start = time.perf_counter()
                _save_frames_for_probe(output_dir, frame_index, frames)
                durations_ms["image_save"] = _elapsed_ms(save_start)
            durations_ms["total"] = _elapsed_ms(loop_start)
            samples.append(
                _sample(
                    "record-loop-dry",
                    target=variant_name,
                    sample_index=frame_index,
                    durations_ms=durations_ms,
                    extra={
                        "scheduler_lag_ms": scheduler_lag_ms,
                        "include_robots": include_robots,
                        "include_cameras": include_cameras,
                        "include_save": include_save,
                        "camera_capture_mode": args.record_camera_capture_mode,
                        "camera_frame_metadata": camera_metadata,
                    },
                )
            )
            frame_index += 1
            next_record_t += 1.0 / args.fps

        time.sleep(0.005)
    return samples


def _record_loop_camera_context(args: Args) -> common.CameraSet | common.AsyncCameraSet:
    config = _camera_config(args)
    if args.record_camera_capture_mode == "async_latest":
        return common.AsyncCameraSet(config=config, startup_timeout_s=args.camera_startup_timeout_s)
    return common.CameraSet(config=config)


def _record_loop_camera_snapshot(
    cameras: common.CameraSet | common.AsyncCameraSet,
    frame_index: int,
    *,
    max_age_s: float,
) -> common.CameraSnapshot:
    if isinstance(cameras, common.AsyncCameraSet):
        return cameras.snapshot(max_age_s=max_age_s)
    del frame_index
    return cameras.snapshot()


def _read_all_robot_states(robots: dict[str, Any]) -> None:
    common.get_follower_state(robots["follower_left"])
    common.get_follower_state(robots["follower_right"])
    robots["leader_left"].get_info()
    robots["leader_right"].get_info()


def _open_yam_robots(args: Args) -> dict[str, Any]:
    common.ensure_i2rt_importable()
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    from examples.yam_real import record_episode

    gripper_type = GripperType.from_string_name(args.gripper)
    robots: dict[str, Any] = {}
    try:
        robots["follower_left"] = get_yam_robot(
            channel=common.FOLLOWER_CHANNELS["left"],
            gripper_type=gripper_type,
            use_gravity_comp=args.use_gravity_comp,
            zero_gravity_mode=False,
            ee_mass=args.ee_mass,
        )
        robots["follower_right"] = get_yam_robot(
            channel=common.FOLLOWER_CHANNELS["right"],
            gripper_type=gripper_type,
            use_gravity_comp=args.use_gravity_comp,
            zero_gravity_mode=False,
            ee_mass=args.ee_mass,
        )
        robots["leader_left"] = record_episode.YamLeader(
            get_yam_robot(
                channel=common.LEADER_CHANNELS["left"],
                gripper_type=GripperType.YAM_TEACHING_HANDLE,
                use_gravity_comp=args.use_gravity_comp,
                zero_gravity_mode=True,
                ee_mass=args.ee_mass,
            )
        )
        robots["leader_right"] = record_episode.YamLeader(
            get_yam_robot(
                channel=common.LEADER_CHANNELS["right"],
                gripper_type=GripperType.YAM_TEACHING_HANDLE,
                use_gravity_comp=args.use_gravity_comp,
                zero_gravity_mode=True,
                ee_mass=args.ee_mass,
            )
        )
    except BaseException:
        _close_robots(robots)
        raise
    return robots


def _close_robots(robots: dict[str, Any]) -> None:
    for robot in robots.values():
        underlying = getattr(robot, "robot", robot)
        with contextlib.suppress(Exception):
            underlying.close()


def _camera_config(args: Args) -> common.CameraConfig:
    return common.CameraConfig(
        frame_size=(args.camera_width, args.camera_height),
        fps=int(args.camera_fps),
        pixel_format=args.camera_pixel_format,
        verify_mode=not args.skip_camera_mode_verify,
    )


def _open_camera(name: str, path: str, *, config: common.CameraConfig) -> common.LinuxpyV4L2Camera:
    camera = common.LinuxpyV4L2Camera(name, path, config)
    camera.open()
    return camera


def _read_capture(camera: common.LinuxpyV4L2Camera) -> np.ndarray:
    return camera.read().frame


def _warmup_capture(camera: common.LinuxpyV4L2Camera, warmup_frames: int) -> None:
    for _ in range(max(0, warmup_frames)):
        _read_capture(camera)


def _warmup_camera_set(cameras: common.CameraSet, warmup_frames: int) -> None:
    for _ in range(max(0, warmup_frames)):
        cameras.read()


def _capture_properties(camera: common.LinuxpyV4L2Camera) -> dict[str, Any]:
    properties = camera.capture_properties()
    properties["controls"] = {}
    return properties


def _fourcc_to_str(value: int) -> str:
    return common.fourcc_to_str(value)


def _frame_stats(frame: np.ndarray) -> dict[str, Any]:
    array = np.asarray(frame)
    std = float(np.std(array))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(np.mean(array)),
        "std": std,
        "nonblank": std >= 1.0,
    }


def _prepare_image_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for camera_name in common.CAMERA_NAMES:
        (output_dir / "images" / camera_name).mkdir(parents=True, exist_ok=True)


def _save_frames_for_probe(output_dir: Path, frame_index: int, frames: dict[str, np.ndarray]) -> None:
    for camera_name, frame in frames.items():
        path = output_dir / "images" / camera_name / f"{frame_index:06d}.jpg"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"Failed to write image {path}")


def _can_targets() -> tuple[tuple[str, str, str], ...]:
    return (
        ("follower", "left", common.FOLLOWER_CHANNELS["left"]),
        ("follower", "right", common.FOLLOWER_CHANNELS["right"]),
        ("leader", "left", common.LEADER_CHANNELS["left"]),
        ("leader", "right", common.LEADER_CHANNELS["right"]),
    )


def _sample(
    mode: str,
    *,
    target: str,
    sample_index: int,
    durations_ms: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> Sample:
    sample = {
        "type": "sample",
        "mode": mode,
        "target": target,
        "sample_index": sample_index,
        "durations_ms": durations_ms,
    }
    if extra:
        sample.update(extra)
    return sample


def _error_sample(mode: str, error: str, *, target: str = "all") -> Sample:
    return {"type": "error", "mode": mode, "target": target, "error": error}


def _summarize_sample_groups(samples: list[Sample]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for sample in samples:
        key = (str(sample["mode"]), str(sample.get("target", "all")))
        stage_values = grouped.setdefault(key, {})
        for stage, value in sample.get("durations_ms", {}).items():
            stage_values.setdefault(stage, []).append(float(value))

    result = {}
    for (mode, target), stage_values in sorted(grouped.items()):
        stages = {stage: duration_stats(values) for stage, values in sorted(stage_values.items())}
        total = stages.get("total", {})
        group_key = f"{mode}/{target}"
        result[group_key] = {
            "mode": mode,
            "target": target,
            "stages": stages,
            "total": total,
            "meets_50hz_p95": bool(total.get("p95_ms", float("inf")) <= FIFTY_HZ_BUDGET_MS),
            "meets_20hz_p95": bool(total.get("p95_ms", float("inf")) <= TWENTY_HZ_BUDGET_MS),
        }
    return result


def duration_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "min_ms": 0.0,
            "median_fps": 0.0,
            "p95_fps": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    median_ms = float(np.median(arr))
    p95_ms = float(np.quantile(arr, 0.95))
    return {
        "count": int(arr.size),
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "max_ms": float(np.max(arr)),
        "min_ms": float(np.min(arr)),
        "median_fps": _fps_from_ms(median_ms),
        "p95_fps": _fps_from_ms(p95_ms),
    }


def _build_findings(groups: dict[str, Any], errors: list[Sample]) -> list[dict[str, Any]]:
    findings = [
        {
            "kind": "probe_error",
            "severity": "error",
            "message": f"{error.get('mode')}/{error.get('target')}: {error.get('error')}",
        }
        for error in errors
    ]

    slow_single_cameras = []
    for group in groups.values():
        if group["mode"] != "camera-single":
            continue
        median_fps = float(group["total"].get("median_fps", 0.0))
        if median_fps and median_fps <= 25.0:
            slow_single_cameras.append((group["target"], median_fps))
    if slow_single_cameras:
        findings.append(
            {
                "kind": "single_camera_rate",
                "severity": "blocker",
                "message": "At least one camera is slow by itself; suspect exposure, driver mode, USB bandwidth, or camera hardware.",
                "evidence": [{"camera": name, "median_fps": fps} for name, fps in slow_single_cameras],
            }
        )

    all_seq = _group_total(groups, "camera-all-sequential", "all")
    if all_seq and all_seq.get("p95_ms", 0.0) > TWENTY_HZ_BUDGET_MS and not slow_single_cameras:
        findings.append(
            {
                "kind": "multi_camera_sequential",
                "severity": "blocker",
                "message": "Single cameras are not slow, but all-camera sequential reads miss 20 Hz; suspect sequential blocking or USB scheduling.",
                "evidence": all_seq,
            }
        )

    image_save = _group_total(groups, "image-save", "all")
    if image_save and image_save.get("p95_ms", 0.0) > FIFTY_HZ_BUDGET_MS:
        findings.append(
            {
                "kind": "image_save_budget",
                "severity": "blocker",
                "message": "JPEG/disk image saving alone exceeds the 50 Hz row budget.",
                "evidence": image_save,
            }
        )
    elif image_save and image_save.get("p95_ms", 0.0) > FIFTY_HZ_BUDGET_MS * 0.5:
        findings.append(
            {
                "kind": "image_save_significant",
                "severity": "warning",
                "message": "JPEG/disk image saving consumes more than half of the 50 Hz row budget.",
                "evidence": image_save,
            }
        )

    for group in groups.values():
        if group["mode"] not in {"robot-state", "teleop-step-nosync", "can-raw"}:
            continue
        total = group["total"]
        if total.get("p95_ms", 0.0) > FIFTY_HZ_BUDGET_MS * 0.5:
            findings.append(
                {
                    "kind": f"{group['mode']}_budget",
                    "severity": "warning",
                    "message": f"{group['mode']} target {group['target']} consumes significant 50 Hz budget.",
                    "evidence": {"target": group["target"], **total},
                }
            )

    no_save = _group_total(groups, "record-loop-dry", "robots_cameras_no_save")
    with_save = _group_total(groups, "record-loop-dry", "robots_cameras_scratch_save")
    if no_save and with_save:
        if no_save.get("p95_ms", 0.0) <= FIFTY_HZ_BUDGET_MS and with_save.get("p95_ms", 0.0) > FIFTY_HZ_BUDGET_MS:
            findings.append(
                {
                    "kind": "record_loop_save_delta",
                    "severity": "blocker",
                    "message": "Record loop only misses 50 Hz when scratch image saving is enabled; current JPEG recording is likely the bottleneck.",
                    "evidence": {"without_save": no_save, "with_save": with_save},
                }
            )
        elif no_save.get("p95_ms", 0.0) > FIFTY_HZ_BUDGET_MS:
            findings.append(
                {
                    "kind": "record_loop_combined_no_save",
                    "severity": "blocker",
                    "message": "Combined robot+camera record loop misses 50 Hz even without saving; suspect capture contention or scheduling.",
                    "evidence": no_save,
                }
            )
    return findings


def _dominant_bottleneck(groups: dict[str, Any], findings: list[dict[str, Any]], *, fps: float) -> str:
    blockers = [finding for finding in findings if finding.get("severity") == "blocker"]
    if blockers:
        return str(blockers[0]["kind"])

    requested_budget_ms = 1000.0 / fps
    worst_key = None
    worst_ratio = 0.0
    for key, group in groups.items():
        p95_ms = float(group["total"].get("p95_ms", 0.0))
        ratio = p95_ms / requested_budget_ms if requested_budget_ms > 0 else 0.0
        if ratio > worst_ratio:
            worst_key = key
            worst_ratio = ratio
    if worst_key is None:
        return "none_detected" if groups else "no_samples"
    if worst_ratio <= 1.0:
        return "none_detected"
    return f"highest_p95:{worst_key}"


def _group_total(groups: dict[str, Any], mode: str, target: str) -> dict[str, Any] | None:
    group = groups.get(f"{mode}/{target}")
    if not group:
        return None
    return group.get("total")


def _fps_from_ms(duration_ms: float) -> float:
    return 0.0 if duration_ms <= 0 else float(1000.0 / duration_ms)


def _system_context() -> dict[str, Any]:
    return {
        "camera_paths": common.CAMERA_PATHS,
        "camera_symlinks": _camera_symlinks(),
        "can_interfaces": _run_command(["ip", "-details", "-br", "link", "show", "type", "can"])["stdout"],
        "lsusb_tree": _run_command(["lsusb", "-t"])["stdout"],
        "gst_device_monitor": _run_command(["gst-device-monitor-1.0", "Video/Source"]),
        "loadavg": Path("/proc/loadavg").read_text().strip() if Path("/proc/loadavg").exists() else "",
        "disk_usage": _disk_usage(Path.cwd()),
        "v4l2": {
            name: _run_command(["v4l2-ctl", "--device", path, "--all"])
            for name, path in common.CAMERA_PATHS.items()
            if Path(path).exists()
        },
    }


def _camera_symlinks() -> dict[str, str]:
    result = {}
    for name, path in common.CAMERA_PATHS.items():
        link_path = Path(path)
        result[name] = str(link_path.resolve()) if link_path.exists() else "missing"
    return result


def _disk_usage(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "used": usage.used, "free": usage.free}


def _run_command(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": 124, "stdout": exc.stdout or "", "stderr": exc.stderr or "timeout"}
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _write_outputs(records: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path, run_id: str) -> None:
    jsonl_path = output_dir / f"{run_id}.jsonl"
    with jsonl_path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, default=str, sort_keys=True) + "\n")
    latest_jsonl = output_dir / "latest.jsonl"
    shutil.copyfile(jsonl_path, latest_jsonl)

    markdown_path = output_dir / f"{run_id}.md"
    markdown_path.write_text(_markdown_summary(summary, jsonl_path))
    latest_md = output_dir / "latest.md"
    shutil.copyfile(markdown_path, latest_md)


def _markdown_summary(summary: dict[str, Any], jsonl_path: Path) -> str:
    lines = [
        "# YAM Refresh-Rate Probe",
        "",
        f"- JSONL: `{jsonl_path}`",
        f"- Requested FPS: {summary['fps']:g}",
        f"- Dominant bottleneck: `{summary.get('dominant_bottleneck', 'unknown')}`",
        f"- Samples: {summary.get('sample_count', 0)}",
        f"- Errors: {summary.get('error_count', 0)}",
        "",
        "## Findings",
    ]
    findings = summary.get("findings", [])
    if findings:
        lines.extend(f"- [{finding['severity']}] {finding['kind']}: {finding['message']}" for finding in findings)
    else:
        lines.append("- No blocking finding detected from collected samples.")

    lines.extend(
        [
            "",
            "## Timing Groups",
            "",
            "| Group | p50 ms | p95 ms | p95 fps | 50 Hz p95 | 20 Hz p95 |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for key, group in summary.get("groups", {}).items():
        total = group.get("total", {})
        lines.append(
            f"| `{key}` | {total.get('median_ms', 0.0):.2f} | {total.get('p95_ms', 0.0):.2f} | "
            f"{total.get('p95_fps', 0.0):.2f} | {_yes_no(group.get('meets_50hz_p95', False))} | "
            f"{_yes_no(group.get('meets_20hz_p95', False))} |"
        )
    return "\n".join(lines) + "\n"


def _yes_no(value) -> str:
    return "yes" if value else "no"


def _elapsed_ms(start: float) -> float:
    return 1000.0 * (time.perf_counter() - start)


if __name__ == "__main__":
    main(tyro.cli(Args))
