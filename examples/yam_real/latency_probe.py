from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path
import statistics
import time
import traceback
from typing import Any, Literal

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy
import tyro
import websockets.exceptions
import websockets.sync.client

from openpi.serving import modal_quic
from openpi.shared import jpeg_transport


@dataclasses.dataclass
class Args:
    protocol: Literal["echo", "policy"] = "echo"
    transport: Literal["websocket", "modal-quic"] = "websocket"
    host: str = "localhost"
    port: int | None = 8765
    modal_app_name: str = "yam-openpi-latency"
    modal_class_name: str = "QuicLatencyServer"
    modal_method_name: str = "serve"
    modal_region: str | None = None
    modal_gpu: str | None = None
    modal_server_start_timeout_s: float = 60.0
    modal_punch_timeout_s: int = 10
    iterations: int = 50
    warmup: int = 5
    payload_bytes: int = 115_875
    response_bytes: int = 3_000
    sleep_ms: float = 0.0
    prompt: str = "perform the demonstrated bimanual task"
    connect_timeout_s: float | None = 20.0
    retry_interval_s: float = 5.0
    response_status_interval_s: float | None = 10.0
    log_dir: Path = Path("yam_data/logs/latency")
    label: str | None = None


def main(args: Args) -> None:
    log_path = _make_log_path(args.log_dir, args.label or f"{args.transport}_{args.protocol}")
    print(f"Latency log: {log_path}", flush=True)
    try:
        if args.protocol == "echo":
            rows = _run_echo_probe(args, log_path)
        elif args.protocol == "policy":
            rows = _run_policy_probe(args, log_path)
        else:
            raise ValueError(f"Unsupported protocol: {args.protocol}")
    except Exception:
        with log_path.open("a", buffering=1) as log_file:
            _write_jsonl(
                log_file,
                {
                    "type": "error",
                    "transport": args.transport,
                    "protocol": args.protocol,
                    "traceback": traceback.format_exc(),
                    "args": dataclasses.asdict(args),
                },
            )
        traceback.print_exc()
        raise SystemExit(1) from None

    _print_summary(rows)


def _run_echo_probe(args: Args, log_path: Path) -> list[dict[str, Any]]:
    if args.transport == "modal-quic":
        return _run_quic_echo_probe(args, log_path)

    uri = _normalize_ws_uri(args.host, args.port)
    packer = msgpack_numpy.Packer()
    conn, metadata = _connect_echo(
        uri,
        connect_timeout_s=args.connect_timeout_s,
        retry_interval_s=args.retry_interval_s,
        response_status_interval_s=args.response_status_interval_s,
    )
    rows = []
    with log_path.open("a", buffering=1) as log_file:
        _write_jsonl(
            log_file,
            {
                "type": "metadata",
                "protocol": args.protocol,
                "transport": args.transport,
                "uri": uri,
                "server_metadata": metadata,
                "args": dataclasses.asdict(args),
            },
        )
        for index in range(args.warmup + args.iterations):
            warmup = index < args.warmup
            request = {
                "sequence": index,
                "payload": bytes(args.payload_bytes),
                "response_bytes": args.response_bytes,
                "sleep_ms": args.sleep_ms,
            }
            request_start = time.monotonic()
            pack_start = time.monotonic()
            packed = packer.pack(request)
            pack_ms = _elapsed_ms(pack_start)
            send_start = time.monotonic()
            conn.send(packed)
            send_ms = _elapsed_ms(send_start)
            recv_start = time.monotonic()
            response = _recv_with_status(conn, args.response_status_interval_s, "latency echo response")
            recv_wait_ms = _elapsed_ms(recv_start)
            if isinstance(response, str):
                raise RuntimeError(f"Latency echo server returned an error:\n{response}")
            unpack_start = time.monotonic()
            result = msgpack_numpy.unpackb(response)
            unpack_ms = _elapsed_ms(unpack_start)
            total_ms = _elapsed_ms(request_start)
            row = {
                "type": "sample",
                "protocol": args.protocol,
                "transport": args.transport,
                "iteration": index,
                "warmup": warmup,
                "total_ms": total_ms,
                "pack_ms": pack_ms,
                "send_ms": send_ms,
                "recv_wait_ms": recv_wait_ms,
                "unpack_ms": unpack_ms,
                "request_bytes": len(packed),
                "response_bytes": len(response),
                "server_timing": result.get("server_timing", {}),
            }
            rows.append(row)
            _write_jsonl(log_file, row)
            _print_row(row)
    return rows


def _run_quic_echo_probe(args: Args, log_path: Path) -> list[dict[str, Any]]:
    client = modal_quic.ModalQuicClient(
        _modal_quic_options(args, mode="echo"),
        status_callback=lambda msg: print(f"[status] {msg}", flush=True),
    )
    metadata = client.get_server_metadata()
    rows = []
    try:
        with log_path.open("a", buffering=1) as log_file:
            _write_jsonl(
                log_file,
                {
                    "type": "metadata",
                    "protocol": args.protocol,
                    "transport": args.transport,
                    "modal_app_name": args.modal_app_name,
                    "modal_class_name": args.modal_class_name,
                    "server_metadata": metadata,
                    "args": dataclasses.asdict(args),
                },
            )
            for index in range(args.warmup + args.iterations):
                warmup = index < args.warmup
                request = {
                    "sequence": index,
                    "payload": bytes(args.payload_bytes),
                    "response_bytes": args.response_bytes,
                    "sleep_ms": args.sleep_ms,
                }
                result = client.request(request, label="latency echo response")
                client_timing = result.get("client_timing", {})
                row = {
                    "type": "sample",
                    "protocol": args.protocol,
                    "transport": args.transport,
                    "iteration": index,
                    "warmup": warmup,
                    "total_ms": _optional_float(client_timing.get("total_ms")),
                    "pack_ms": _optional_float(client_timing.get("pack_ms")),
                    "send_ms": _optional_float(client_timing.get("send_ms")),
                    "recv_wait_ms": _optional_float(client_timing.get("recv_wait_ms")),
                    "unpack_ms": _optional_float(client_timing.get("unpack_ms")),
                    "request_bytes": _optional_int(client_timing.get("request_bytes")),
                    "response_bytes": _optional_int(client_timing.get("response_bytes")),
                    "server_timing": result.get("server_timing", {}),
                }
                rows.append(row)
                _write_jsonl(log_file, row)
                _print_row(row)
    finally:
        client.close()
    return rows


def _run_policy_probe(args: Args, log_path: Path) -> list[dict[str, Any]]:
    if args.transport == "modal-quic":
        policy_host = f"modal://{args.modal_app_name}/{args.modal_class_name}.{args.modal_method_name}"
        policy_port = None
        policy = modal_quic.ModalQuicClientPolicy(
            _modal_quic_options(args, mode="noop-policy"),
            status_callback=lambda msg: print(f"[status] {msg}", flush=True),
        )
    else:
        policy_host = _normalize_policy_host(args.host)
        policy_port = None if policy_host.startswith("ws") else args.port
        policy = websocket_client_policy.WebsocketClientPolicy(
            host=policy_host,
            port=policy_port,
            connect_timeout_s=args.connect_timeout_s,
            retry_interval_s=args.retry_interval_s,
            response_status_interval_s=args.response_status_interval_s,
            status_callback=lambda msg: print(f"[status] {msg}", flush=True),
        )
    metadata = policy.get_server_metadata()
    observation = _make_policy_observation(args, metadata)
    rows = []
    try:
        with log_path.open("a", buffering=1) as log_file:
            _write_jsonl(
                log_file,
                {
                    "type": "metadata",
                    "protocol": args.protocol,
                    "transport": args.transport,
                    "host": policy_host,
                    "port": policy_port,
                    "server_metadata": metadata,
                    "args": dataclasses.asdict(args),
                },
            )
            for index in range(args.warmup + args.iterations):
                warmup = index < args.warmup
                if args.sleep_ms > 0:
                    observation["diagnostic_sleep_ms"] = args.sleep_ms
                else:
                    observation.pop("diagnostic_sleep_ms", None)
                result = policy.infer(observation)
                actions = np.asarray(result["actions"])
                client_timing = result.get("client_timing", {})
                row = {
                    "type": "sample",
                    "protocol": args.protocol,
                    "transport": args.transport,
                    "iteration": index,
                    "warmup": warmup,
                    "total_ms": _optional_float(client_timing.get("total_ms")),
                    "pack_ms": _optional_float(client_timing.get("pack_ms")),
                    "send_ms": _optional_float(client_timing.get("send_ms")),
                    "recv_wait_ms": _optional_float(client_timing.get("recv_wait_ms")),
                    "unpack_ms": _optional_float(client_timing.get("unpack_ms")),
                    "request_bytes": _optional_int(client_timing.get("request_bytes")),
                    "response_bytes": _optional_int(client_timing.get("response_bytes")),
                    "actions_shape": tuple(int(dim) for dim in actions.shape),
                    "server_timing": result.get("server_timing", {}),
                }
                rows.append(row)
                _write_jsonl(log_file, row)
                _print_row(row)
    finally:
        close = getattr(policy, "close", None)
        if close is not None:
            close()
    return rows


def _modal_quic_options(args: Args, *, mode: str) -> modal_quic.ModalQuicOptions:
    return modal_quic.ModalQuicOptions(
        app_name=args.modal_app_name,
        class_name=args.modal_class_name,
        method_name=args.modal_method_name,
        modal_region=args.modal_region,
        modal_gpu=args.modal_gpu,
        server_start_timeout_s=args.modal_server_start_timeout_s,
        punch_timeout_s=args.modal_punch_timeout_s,
        response_status_interval_s=args.response_status_interval_s,
        connect_retry_interval_s=args.retry_interval_s,
        serve_kwargs={
            "mode": mode,
            "fixed_sleep_ms": args.sleep_ms,
            "response_bytes": args.response_bytes,
        },
    )


def _make_policy_observation(args: Args, metadata: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    image_transport = metadata.get("image_transport", jpeg_transport.IMAGE_TRANSPORT)
    images = {}
    for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        image = rng.integers(0, 256, size=(*jpeg_transport.IMAGE_RESOLUTION, 3), dtype=np.uint8)
        if image_transport == jpeg_transport.IMAGE_TRANSPORT:
            images[name] = jpeg_transport.encode_rgb_jpeg(image)
        elif image_transport == "raw":
            images[name] = np.transpose(image, (2, 0, 1))
        else:
            raise RuntimeError(f"Unsupported policy image_transport={image_transport!r}")
    return {
        "state": np.zeros((14,), dtype=np.float32),
        "images": images,
        "prompt": args.prompt,
    }


def _connect_echo(
    uri: str,
    *,
    connect_timeout_s: float | None,
    retry_interval_s: float,
    response_status_interval_s: float | None,
):
    attempt = 0
    wait_start = time.monotonic()
    while True:
        attempt += 1
        attempt_start = time.monotonic()
        try:
            print(f"[status] opening websocket attempt {attempt} uri={uri}", flush=True)
            conn = websockets.sync.client.connect(
                uri,
                compression=None,
                max_size=None,
                open_timeout=connect_timeout_s,
            )
            print(f"[status] connected in {time.monotonic() - attempt_start:.1f}s; waiting for metadata", flush=True)
            metadata = msgpack_numpy.unpackb(_recv_with_status(conn, response_status_interval_s, "latency metadata"))
            print(f"[status] server ready after {time.monotonic() - wait_start:.1f}s", flush=True)
            return conn, metadata
        except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
            elapsed_s = time.monotonic() - wait_start
            retry_s = max(0.0, retry_interval_s)
            print(
                f"[status] server not ready after {elapsed_s:.1f}s "
                f"(attempt {attempt}: {type(exc).__name__}: {exc}); retrying in {retry_s:g}s",
                flush=True,
            )
            time.sleep(retry_s)


def _recv_with_status(conn, response_status_interval_s: float | None, label: str):
    if response_status_interval_s is None:
        return conn.recv()

    wait_start = time.monotonic()
    while True:
        try:
            return conn.recv(timeout=response_status_interval_s)
        except TimeoutError:
            print(f"[status] waiting for {label} for {time.monotonic() - wait_start:.1f}s", flush=True)


def _print_row(row: dict[str, Any]) -> None:
    server_timing = row.get("server_timing", {})
    print(
        f"iter={row['iteration']:03d} warmup={row['warmup']} "
        f"total_ms={_format_optional_ms(row.get('total_ms'))} "
        f"send_ms={_format_optional_ms(row.get('send_ms'))} "
        f"recv_wait_ms={_format_optional_ms(row.get('recv_wait_ms'))} "
        f"request_bytes={row.get('request_bytes')} response_bytes={row.get('response_bytes')} "
        f"server_infer_ms={_format_optional_ms(server_timing.get('infer_ms'))} "
        f"server_active_prev_ms={_format_optional_ms(server_timing.get('prev_handler_active_ms'))}",
        flush=True,
    )


def _print_summary(rows: list[dict[str, Any]]) -> None:
    warm_rows = [row for row in rows if not row.get("warmup")]
    print("\nWarm summary", flush=True)
    for key in ("total_ms", "recv_wait_ms", "send_ms", "pack_ms", "unpack_ms", "request_bytes", "response_bytes"):
        _print_metric_summary(warm_rows, key)
    for key in ("infer_ms", "policy_infer_ms", "prev_handler_active_ms", "prev_response_send_ms"):
        _print_server_metric_summary(warm_rows, key)


def _print_metric_summary(rows: list[dict[str, Any]], key: str) -> None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    _print_values(key, values)


def _print_server_metric_summary(rows: list[dict[str, Any]], key: str) -> None:
    values = [float(row["server_timing"][key]) for row in rows if row.get("server_timing", {}).get(key) is not None]
    _print_values(f"server_{key}", values)


def _print_values(name: str, values: list[float]) -> None:
    if not values:
        return
    p50, p90, p99 = np.percentile(values, [50, 90, 99])
    print(
        f"{name}: mean={statistics.mean(values):.1f} p50={p50:.1f} "
        f"p90={p90:.1f} p99={p99:.1f} min={min(values):.1f} max={max(values):.1f}",
        flush=True,
    )


def _make_log_path(log_dir: Path, label: str) -> Path:
    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{safe_label}.jsonl"
    latest_path = log_dir / "latest.jsonl"
    with contextlib.suppress(FileNotFoundError):
        latest_path.unlink()
    try:
        latest_path.symlink_to(log_path.name)
    except OSError:
        latest_path.write_text(f"{log_path}\n")
    return log_path


def _write_jsonl(log_file, item: dict[str, Any]) -> None:
    log_file.write(json.dumps(_json_safe(item), sort_keys=True) + "\n")


def _json_safe(value):
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _elapsed_ms(start: float) -> float:
    return 1000.0 * (time.monotonic() - start)


def _format_optional_ms(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_ws_uri(host: str, port: int | None) -> str:
    if host.startswith("ws"):
        return host
    if host.startswith("https://"):
        return f"wss://{host.removeprefix('https://')}"
    if host.startswith("http://"):
        return f"ws://{host.removeprefix('http://')}"
    uri = f"ws://{host}"
    if port is not None:
        uri += f":{port}"
    return uri


def _normalize_policy_host(host: str) -> str:
    if host.startswith("https://"):
        return f"wss://{host.removeprefix('https://')}"
    if host.startswith("http://"):
        return f"ws://{host.removeprefix('http://')}"
    return host


if __name__ == "__main__":
    main(tyro.cli(Args))
