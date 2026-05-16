from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Literal

import cv2
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro

from examples.yam_real import common
from openpi.shared import jpeg_transport


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    prompt: str = "perform the demonstrated bimanual task"
    execute: bool = False
    fps: float = 50.0
    max_steps: int = 200
    action_horizon: int = 50
    gripper: Literal["crank_4310", "linear_3507", "linear_4310"] = "linear_4310"
    max_arm_step_rad: float = 0.02
    max_gripper_step: float = 0.02
    use_gravity_comp: bool = False
    connect_timeout_s: float | None = 20.0
    connection_retry_interval_s: float = 5.0
    response_status_interval_s: float | None = 5.0
    per_step_log_interval: int = 1
    log_dir: Path = Path("yam_data/logs/run_policy")
    image_transport: Literal["auto", "raw", "jpeg_q85_224_rgb_v1"] = "auto"


def main(args: Args) -> None:
    log_path = _make_run_log_path(args.log_dir)
    with _tee_output(log_path):
        _log(f"Run log: {log_path}")
        try:
            _run_policy(args)
        except KeyboardInterrupt:
            print("Interrupted by user.", file=sys.stderr, flush=True)
            raise SystemExit(130) from None
        except Exception:
            traceback.print_exc()
            raise SystemExit(1) from None


def _run_policy(args: Args) -> None:
    common.ensure_i2rt_importable()
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    policy_host = _normalize_policy_host(args.host)
    policy_port = None if policy_host.startswith("ws") else args.port
    policy_endpoint = policy_host if policy_port is None else f"{policy_host}:{policy_port}"
    _log(
        f"execute={args.execute}; host={policy_endpoint}; max_steps={args.max_steps}; "
        f"fps={args.fps:g}; action_horizon={args.action_horizon}"
    )
    if not args.execute:
        _log("Dry run only. Observations will be sent to the server, but followers will not be commanded.")
    ws_policy = websocket_client_policy.WebsocketClientPolicy(
        host=policy_host,
        port=policy_port,
        connect_timeout_s=args.connect_timeout_s,
        retry_interval_s=args.connection_retry_interval_s,
        response_status_interval_s=args.response_status_interval_s,
        status_callback=_log,
    )
    server_metadata = ws_policy.get_server_metadata()
    _validate_server_metadata(server_metadata, expected_fps=args.fps, expected_action_horizon=args.action_horizon)
    image_transport = _resolve_image_transport(server_metadata, args.image_transport)
    _log(f"Using image_transport={image_transport}")
    gripper_type = GripperType.from_string_name(args.gripper)
    robots = []

    try:
        _log(f"Connecting left follower on {common.FOLLOWER_CHANNELS['left']}")
        follower_l = get_yam_robot(
            channel=common.FOLLOWER_CHANNELS["left"],
            gripper_type=gripper_type,
            use_gravity_comp=args.use_gravity_comp,
            zero_gravity_mode=False,
        )
        robots.append(follower_l)
        _log(f"Connecting right follower on {common.FOLLOWER_CHANNELS['right']}")
        follower_r = get_yam_robot(
            channel=common.FOLLOWER_CHANNELS["right"],
            gripper_type=gripper_type,
            use_gravity_comp=args.use_gravity_comp,
            zero_gravity_mode=False,
        )
        robots.append(follower_r)
        command = common.pack_bimanual(common.get_follower_state(follower_l), common.get_follower_state(follower_r))

        _log("Opening cameras")
        with common.CameraSet() as cameras:
            _log(f"Cameras ready: {', '.join(common.CAMERA_NAMES)}")
            dt = 1.0 / args.fps
            next_command_t = time.monotonic()
            last_command_t: float | None = None
            chunk_actions: np.ndarray | None = None
            chunk_step = args.action_horizon
            chunk_index = -1
            _log("Starting policy loop")
            for step in range(args.max_steps):
                loop_start = time.monotonic()
                state_ms = camera_ms = image_ms = server_ms = obs_ms = None
                server_model_ms = server_prev_total_ms = None
                client_pack_ms = client_send_ms = client_recv_wait_ms = client_unpack_ms = client_total_ms = None
                request_bytes = response_bytes = None
                request_new_chunk = chunk_actions is None or chunk_step >= len(chunk_actions)
                if request_new_chunk:
                    obs_start = time.monotonic()
                    state_start = time.monotonic()
                    state = common.pack_bimanual(
                        common.get_follower_state(follower_l), common.get_follower_state(follower_r)
                    )
                    state_ms = _elapsed_ms(state_start)

                    camera_start = time.monotonic()
                    frames = cameras.read()
                    camera_ms = _elapsed_ms(camera_start)

                    image_start = time.monotonic()
                    images = _policy_images(frames, image_transport=image_transport)
                    image_ms = _elapsed_ms(image_start)
                    obs_ms = _elapsed_ms(obs_start)
                    observation = {
                        "state": state,
                        "images": images,
                        "prompt": args.prompt,
                    }
                    _log(
                        f"step={step}: requesting action chunk "
                        f"(horizon={args.action_horizon}, obs_ms={obs_ms:.1f}, state_ms={state_ms:.1f}, "
                        f"camera_ms={camera_ms:.1f}, image_ms={image_ms:.1f}, "
                        f"image_transport={image_transport})"
                    )
                    infer_start = time.monotonic()
                    result = ws_policy.infer(observation)
                    server_ms = _elapsed_ms(infer_start)
                    server_timing = result.get("server_timing", {})
                    server_model_ms = _optional_float(server_timing.get("infer_ms"))
                    server_prev_total_ms = _optional_float(server_timing.get("prev_total_ms"))
                    client_timing = result.get("client_timing", {})
                    client_pack_ms = _optional_float(client_timing.get("pack_ms"))
                    client_send_ms = _optional_float(client_timing.get("send_ms"))
                    client_recv_wait_ms = _optional_float(client_timing.get("recv_wait_ms"))
                    client_unpack_ms = _optional_float(client_timing.get("unpack_ms"))
                    client_total_ms = _optional_float(client_timing.get("total_ms"))
                    request_bytes = _optional_int(client_timing.get("request_bytes"))
                    response_bytes = _optional_int(client_timing.get("response_bytes"))
                    chunk_actions = np.asarray(result["actions"], dtype=np.float32)
                    _validate_action_chunk(chunk_actions, expected_horizon=args.action_horizon)
                    chunk_step = 0
                    chunk_index += 1
                    _log(
                        f"step={step}: received chunk={chunk_index} shape={chunk_actions.shape} "
                        f"roundtrip_ms={server_ms:.1f} server_model_ms={_format_optional_ms(server_model_ms)} "
                        f"server_prev_total_ms={_format_optional_ms(server_prev_total_ms)} "
                        f"client_pack_ms={_format_optional_ms(client_pack_ms)} "
                        f"client_send_ms={_format_optional_ms(client_send_ms)} "
                        f"client_recv_wait_ms={_format_optional_ms(client_recv_wait_ms)} "
                        f"client_unpack_ms={_format_optional_ms(client_unpack_ms)} "
                        f"request_bytes={_format_optional_int(request_bytes)} "
                        f"response_bytes={_format_optional_int(response_bytes)}"
                    )

                action = chunk_actions[chunk_step]
                action_step = chunk_step
                chunk_step += 1

                previous_command = command
                command = common.clip_bimanual_delta(
                    command,
                    action,
                    max_arm_step_rad=args.max_arm_step_rad,
                    max_gripper_step=args.max_gripper_step,
                )

                pre_command_wait_s = next_command_t - time.monotonic()
                if pre_command_wait_s > 0:
                    time.sleep(pre_command_wait_s)
                command_t = time.monotonic()
                command_period_ms = None if last_command_t is None else 1000.0 * (command_t - last_command_t)
                command_late_ms = max(0.0, 1000.0 * (command_t - next_command_t))
                last_command_t = command_t
                next_command_t = command_t + dt

                if args.execute:
                    left, right = common.split_bimanual(command)
                    follower_l.command_joint_pos(common.openpi_arm_state_to_i2rt(left))
                    follower_r.command_joint_pos(common.openpi_arm_state_to_i2rt(right))

                loop_ms = 1000.0 * (time.monotonic() - loop_start)
                sleep_ms = max(0.0, pre_command_wait_s) * 1000.0
                max_delta = float(np.max(np.abs(command - previous_command)))
                if step % max(1, args.per_step_log_interval) == 0:
                    print(
                        f"step={step:05d} chunk={chunk_index}:{action_step + 1}/{len(chunk_actions)} "
                        f"source={'server' if request_new_chunk else 'cache'} "
                        f"obs_ms={_format_optional_ms(obs_ms)} state_ms={_format_optional_ms(state_ms)} "
                        f"camera_ms={_format_optional_ms(camera_ms)} image_ms={_format_optional_ms(image_ms)} "
                        f"roundtrip_ms={_format_optional_ms(server_ms)} "
                        f"server_model_ms={_format_optional_ms(server_model_ms)} "
                        f"server_prev_total_ms={_format_optional_ms(server_prev_total_ms)} "
                        f"client_total_ms={_format_optional_ms(client_total_ms)} "
                        f"client_pack_ms={_format_optional_ms(client_pack_ms)} "
                        f"client_send_ms={_format_optional_ms(client_send_ms)} "
                        f"client_recv_wait_ms={_format_optional_ms(client_recv_wait_ms)} "
                        f"client_unpack_ms={_format_optional_ms(client_unpack_ms)} "
                        f"request_bytes={_format_optional_int(request_bytes)} "
                        f"response_bytes={_format_optional_int(response_bytes)} loop_ms={loop_ms:.1f} "
                        f"sleep_ms={sleep_ms:.1f} command_period_ms={_format_optional_ms(command_period_ms)} "
                        f"command_late_ms={command_late_ms:.1f} "
                        f"action_norm={float(np.linalg.norm(action)):.4f} "
                        f"delta_norm={float(np.linalg.norm(command - previous_command)):.4f} "
                        f"max_delta={max_delta:.4f}",
                        flush=True,
                    )
    finally:
        _log("Closing robot connections")
        for robot in robots:
            with contextlib.suppress(Exception):
                robot.close()
        _log("Policy runner stopped")


def _policy_images(frames_bgr: dict[str, np.ndarray], *, image_transport: str) -> dict[str, np.ndarray | bytes]:
    images = {}
    for name, frame_bgr in frames_bgr.items():
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, *jpeg_transport.IMAGE_RESOLUTION))
        if image_transport == jpeg_transport.IMAGE_TRANSPORT:
            images[name] = jpeg_transport.encode_rgb_jpeg(rgb)
        elif image_transport == "raw":
            images[name] = np.transpose(rgb, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported image_transport={image_transport!r}")
    return images


def _make_run_log_path(log_dir: Path) -> Path:
    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_policy_{timestamp}.log"
    suffix = 1
    while log_path.exists():
        log_path = log_dir / f"run_policy_{timestamp}_{suffix}.log"
        suffix += 1

    latest_path = log_dir / "latest.log"
    with contextlib.suppress(FileNotFoundError):
        latest_path.unlink()
    try:
        latest_path.symlink_to(log_path.name)
    except OSError:
        latest_path.write_text(f"{log_path}\n")
    return log_path


@contextlib.contextmanager
def _tee_output(log_path: Path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", buffering=1) as log_file:
        sys.stdout = _TeeStream(original_stdout, log_file)
        sys.stderr = _TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


class _TeeStream:
    def __init__(self, primary, log_file) -> None:
        self._primary = primary
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._log_file.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._log_file.flush()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


def _validate_action_chunk(actions: np.ndarray, *, expected_horizon: int) -> None:
    if actions.shape != (expected_horizon, 14):
        raise RuntimeError(f"Expected action chunk shape {(expected_horizon, 14)}, got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Policy returned non-finite action chunk")


def _elapsed_ms(start: float) -> float:
    return 1000.0 * (time.monotonic() - start)


def _format_optional_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _normalize_policy_host(host: str) -> str:
    if host.startswith("https://"):
        return f"wss://{host.removeprefix('https://')}"
    if host.startswith("http://"):
        return f"ws://{host.removeprefix('http://')}"
    return host


def _resolve_image_transport(
    metadata: dict[str, Any],
    requested_image_transport: Literal["auto", "raw", "jpeg_q85_224_rgb_v1"],
) -> str:
    server_image_transport = metadata.get("image_transport", "raw")
    if requested_image_transport == "auto":
        return server_image_transport
    if requested_image_transport != server_image_transport:
        raise RuntimeError(
            "Policy image_transport mismatch: "
            f"server advertises {server_image_transport!r}, requested {requested_image_transport!r}"
        )
    return requested_image_transport


def _validate_server_metadata(metadata: dict[str, Any], *, expected_fps: float, expected_action_horizon: int) -> None:
    action_space = metadata.get("action_space")
    if action_space != common.ACTION_SPACE:
        raise RuntimeError(f"Policy action_space mismatch: expected {common.ACTION_SPACE!r}, got {action_space!r}")

    state_order = tuple(metadata.get("state_order", ()))
    if state_order != common.STATE_ORDER:
        raise RuntimeError(f"Policy state_order mismatch: expected {common.STATE_ORDER!r}, got {state_order!r}")

    gripper_convention = metadata.get("gripper_convention")
    if gripper_convention != common.GRIPPER_CONVENTION:
        raise RuntimeError(
            f"Policy gripper convention mismatch: expected {common.GRIPPER_CONVENTION!r}, got {gripper_convention!r}"
        )

    action_horizon = metadata.get("action_horizon")
    if action_horizon != expected_action_horizon:
        raise RuntimeError(
            f"Policy action_horizon mismatch: expected {expected_action_horizon!r}, got {action_horizon!r}"
        )

    fps = metadata.get("fps")
    if fps is None or abs(float(fps) - float(expected_fps)) > 1e-6:
        raise RuntimeError(f"Policy fps mismatch: expected {expected_fps!r}, got {fps!r}")

    image_transport = metadata.get("image_transport", "raw")
    if image_transport not in ("raw", jpeg_transport.IMAGE_TRANSPORT):
        raise RuntimeError(f"Unsupported policy image_transport: {image_transport!r}")
    if image_transport == jpeg_transport.IMAGE_TRANSPORT:
        jpeg_quality = metadata.get("jpeg_quality")
        if jpeg_quality != jpeg_transport.JPEG_QUALITY:
            raise RuntimeError(
                f"Policy jpeg_quality mismatch: expected {jpeg_transport.JPEG_QUALITY!r}, got {jpeg_quality!r}"
            )

        image_resolution = metadata.get("image_resolution")
        if image_resolution is None or tuple(image_resolution) != jpeg_transport.IMAGE_RESOLUTION:
            raise RuntimeError(
                "Policy image_resolution mismatch: "
                f"expected {jpeg_transport.IMAGE_RESOLUTION!r}, got {image_resolution!r}"
            )

    print(
        "server metadata OK: "
        f"action_space={action_space}, gripper_convention={gripper_convention}, state_dim={len(state_order)}, "
        f"action_horizon={action_horizon}, fps={fps}, image_transport={image_transport}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
