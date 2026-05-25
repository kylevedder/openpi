from __future__ import annotations

from collections.abc import Callable
import contextlib
import dataclasses
import logging
import random
import time
import traceback
from typing import Any

import msgpack
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)

STUN_SERVERS: list[tuple[str, int]] = [
    ("stun.l.google.com", 19302),
    ("stun1.l.google.com", 19302),
]


@dataclasses.dataclass(frozen=True)
class ModalQuicOptions:
    app_name: str
    class_name: str
    method_name: str = "serve"
    modal_region: str | None = None
    modal_gpu: str | None = None
    serve_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    connect_attempts: int = 3
    connect_retry_interval_s: float = 1.0
    server_start_timeout_s: float = 60.0
    punch_timeout_s: int = 10
    response_status_interval_s: float | None = None


class ModalQuicClient:
    """Persistent Modal QUIC client using a Modal Dict rendezvous."""

    def __init__(
        self,
        options: ModalQuicOptions,
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._options = options
        self._status_callback = status_callback
        self._packer = msgpack_numpy.Packer()
        self._portal = None
        self._handle = None
        self._server_metadata = self._connect_and_read_metadata()

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._portal is not None:
                self._portal.close()
        self._portal = None
        with contextlib.suppress(Exception):
            if self._handle is not None:
                self._handle.cancel()
        self._handle = None

    def request(self, payload: dict[str, Any], *, label: str = "Modal QUIC response") -> dict[str, Any]:
        if self._portal is None:
            raise RuntimeError("Modal QUIC client is not connected")

        request_start = time.monotonic()
        pack_start = time.monotonic()
        request = self._packer.pack(payload)
        pack_ms = _elapsed_ms(pack_start)

        send_start = time.monotonic()
        self._portal.send(request)
        send_ms = _elapsed_ms(send_start)

        recv_start = time.monotonic()
        response = self._recv_with_status(label)
        recv_wait_ms = _elapsed_ms(recv_start)
        if response is None:
            raise RuntimeError(f"Modal QUIC connection closed while waiting for {label}")

        unpack_start = time.monotonic()
        result = msgpack_numpy.unpackb(response)
        unpack_ms = _elapsed_ms(unpack_start)
        if not isinstance(result, dict):
            raise RuntimeError(f"Modal QUIC server returned {type(result).__name__}, expected dict")
        if "error" in result:
            raise RuntimeError(f"Error from Modal QUIC server:\n{result['error']}")

        result["client_timing"] = {
            "pack_ms": pack_ms,
            "send_ms": send_ms,
            "recv_wait_ms": recv_wait_ms,
            "unpack_ms": unpack_ms,
            "total_ms": _elapsed_ms(request_start),
            "request_bytes": len(request),
            "response_bytes": len(response),
            "transport": "modal-quic",
        }
        return result

    def _connect_and_read_metadata(self) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self._options.connect_attempts + 1):
            attempt_start = time.monotonic()
            try:
                self._emit_status(
                    "Opening Modal QUIC connection "
                    f"attempt {attempt}/{self._options.connect_attempts} "
                    f"app={self._options.app_name} class={self._options.class_name}"
                )
                self._connect_once()
                assert self._portal is not None
                server_ip = getattr(self._portal, "server_ip", None)
                self._emit_status(
                    "Modal QUIC connected "
                    f"in {time.monotonic() - attempt_start:.1f}s"
                    + (f"; server_ip={server_ip}" if server_ip else "")
                )
                metadata_response = self._recv_with_status("Modal QUIC metadata")
                if metadata_response is None:
                    raise RuntimeError("Modal QUIC server closed before sending metadata")
                metadata = msgpack_numpy.unpackb(metadata_response)
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"Modal QUIC metadata was {type(metadata).__name__}, expected dict")
                self._emit_status(f"Modal QUIC policy server ready after {time.monotonic() - attempt_start:.1f}s")
                return metadata
            except Exception as exc:
                last_exc = exc
                self.close()
                if attempt >= self._options.connect_attempts:
                    break
                self._emit_status(
                    "Modal QUIC connection failed "
                    f"(attempt {attempt}: {type(exc).__name__}: {exc}); "
                    f"retrying in {self._options.connect_retry_interval_s:g}s"
                )
                time.sleep(max(0.0, self._options.connect_retry_interval_s))

        raise RuntimeError(
            "Failed to establish Modal QUIC connection after "
            f"{self._options.connect_attempts} attempts using STUN servers {STUN_SERVERS}"
        ) from last_exc

    def _connect_once(self) -> None:
        import modal
        import quic_portal

        server_cls = modal.Cls.from_name(self._options.app_name, self._options.class_name)
        modal_options = {}
        if self._options.modal_region:
            modal_options["region"] = self._options.modal_region
        if self._options.modal_gpu:
            modal_options["gpu"] = self._options.modal_gpu
        if modal_options:
            self._emit_status(f"Applying Modal runtime options: {modal_options}")
            server_cls = server_cls.with_options(**modal_options)

        server = server_cls()
        server_method = getattr(server, self._options.method_name)
        with modal.Dict.ephemeral() as rendezvous:
            self._emit_status("Spawning Modal QUIC server method")
            spawn_start = time.monotonic()
            self._handle = server_method.spawn(rendezvous, **self._options.serve_kwargs)
            self._emit_status(f"Modal QUIC server method spawned in {time.monotonic() - spawn_start:.1f}s")
            self._wait_for_server_endpoint(rendezvous)
            local_port = random.randint(5555, 65535)
            self._emit_status(
                "Creating QUIC portal "
                f"(local_port={local_port}, punch_timeout={self._options.punch_timeout_s}s, "
                f"stun_servers={STUN_SERVERS})"
            )
            portal_start = time.monotonic()
            self._portal = quic_portal.Portal.create_client(
                rendezvous,
                local_port=local_port,
                stun_servers=STUN_SERVERS,
                punch_timeout=self._options.punch_timeout_s,
                transport_options=quic_portal.QuicTransportOptions(
                    keep_alive_interval_secs=1,
                    max_idle_timeout_secs=20,
                ),
            )
            self._emit_status(f"QUIC portal created in {time.monotonic() - portal_start:.1f}s")
            self._portal.send(b"hello")

    def _wait_for_server_endpoint(self, rendezvous) -> None:
        wait_start = time.monotonic()
        last_status_time = wait_start
        status_interval_s = self._options.response_status_interval_s or 5.0
        while "server" not in rendezvous:
            if "server_create_error" in rendezvous:
                raise RuntimeError(f"Modal QUIC server failed before rendezvous: {rendezvous['server_create_error']}")
            elapsed_s = time.monotonic() - wait_start
            if elapsed_s > self._options.server_start_timeout_s:
                raise TimeoutError(
                    "Timed out waiting for Modal QUIC server endpoint after "
                    f"{elapsed_s:.1f}s; STUN servers={STUN_SERVERS}"
                )
            if time.monotonic() - last_status_time >= status_interval_s:
                started = rendezvous.get("server_started", None)
                self._emit_status(
                    "Waiting for Modal QUIC server endpoint "
                    f"for {elapsed_s:.1f}s"
                    + (f"; server_started={started}" if started else "")
                )
                last_status_time = time.monotonic()
            time.sleep(0.2)
        endpoint = rendezvous["server"]
        self._emit_status(f"Modal QUIC server endpoint registered: {endpoint}")

    def _recv_with_status(self, label: str):
        assert self._portal is not None
        interval_s = self._options.response_status_interval_s
        if interval_s is None:
            return self._portal.recv()

        wait_start = time.monotonic()
        timeout_ms = max(1, int(interval_s * 1000))
        while True:
            response = self._portal.recv(timeout_ms=timeout_ms)
            if response is not None:
                return response
            self._emit_status(f"Waiting for {label} for {time.monotonic() - wait_start:.1f}s")

    def _emit_status(self, message: str) -> None:
        logging.info(message)
        if self._status_callback is not None:
            self._status_callback(message)


class ModalQuicClientPolicy(_base_policy.BasePolicy):
    """Policy-shaped wrapper around ModalQuicClient."""

    def __init__(
        self,
        options: ModalQuicOptions,
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._client = ModalQuicClient(options, status_callback=status_callback)

    def get_server_metadata(self) -> dict[str, Any]:
        return self._client.get_server_metadata()

    def infer(self, obs: dict) -> dict:
        return self._client.request(obs, label="Modal QUIC policy inference response")

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._client.close()


def serve_policy(
    *,
    rendezvous,
    policy: _base_policy.BasePolicy,
    metadata: dict[str, Any],
    receive_timeout_s: float = 10 * 60,
) -> None:
    def handle(payload: dict[str, Any]) -> dict[str, Any]:
        return policy.infer(payload)

    _serve_loop(
        rendezvous=rendezvous,
        metadata=metadata,
        handler=handle,
        receive_timeout_s=receive_timeout_s,
    )


def serve_echo(
    *,
    rendezvous,
    metadata: dict[str, Any],
    fixed_sleep_ms: float = 0.0,
    default_response_bytes: int = 3_000,
    receive_timeout_s: float = 10 * 60,
) -> None:
    def handle(payload: dict[str, Any]) -> dict[str, Any]:
        sleep_ms = float(payload.get("sleep_ms", fixed_sleep_ms))
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
        response_bytes = int(payload.get("response_bytes", default_response_bytes))
        return {
            "sequence": int(payload.get("sequence", -1)),
            "payload": bytes(response_bytes),
            "server_timing": {
                "noop_sleep_ms": sleep_ms,
            },
        }

    _serve_loop(
        rendezvous=rendezvous,
        metadata=metadata,
        handler=handle,
        receive_timeout_s=receive_timeout_s,
    )


def _serve_loop(
    *,
    rendezvous,
    metadata: dict[str, Any],
    handler: Callable[[dict[str, Any]], dict[str, Any]],
    receive_timeout_s: float,
) -> None:
    import quic_portal

    packer = msgpack_numpy.Packer()
    local_port = random.randint(5555, 65535)
    rendezvous["server_started"] = {
        "local_port": local_port,
        "stun_servers": STUN_SERVERS,
        "timestamp": time.time(),
    }
    logger.info("Creating Modal QUIC server portal on local_port=%s", local_port)
    try:
        portal = quic_portal.Portal.create_server(
            rendezvous,
            local_port=local_port,
            stun_servers=STUN_SERVERS,
            punch_timeout=10,
            transport_options=quic_portal.QuicTransportOptions(
                keep_alive_interval_secs=1,
                max_idle_timeout_secs=20,
            ),
        )
    except Exception as exc:
        rendezvous["server_create_error"] = {
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "local_port": local_port,
            "stun_servers": STUN_SERVERS,
        }
        logger.exception("Failed to create Modal QUIC server portal")
        raise
    try:
        hello = portal.recv(timeout_ms=10_000)
        if hello != b"hello":
            raise RuntimeError(f"Expected Modal QUIC hello, got {hello!r}")
        portal.send(packer.pack(metadata))
        logger.info("Modal QUIC server connected on local_port=%s", local_port)
        _serve_connected_portal(
            portal=portal,
            packer=packer,
            handler=handler,
            receive_timeout_s=receive_timeout_s,
        )
    finally:
        with contextlib.suppress(Exception):
            portal.close()


def _serve_connected_portal(
    *,
    portal,
    packer: msgpack_numpy.Packer,
    handler: Callable[[dict[str, Any]], dict[str, Any]],
    receive_timeout_s: float,
) -> None:
    prev_total_time = None
    prev_handler_active_time = None
    prev_response_send_time = None
    prev_response_bytes = None
    timeout_ms = int(receive_timeout_s * 1000)

    while True:
        try:
            wait_start_time = time.monotonic()
            try:
                request = portal.recv(timeout_ms=timeout_ms)
            except Exception as exc:
                if _is_connection_closed_error(exc):
                    logger.info("Modal QUIC client closed connection")
                    break
                raise
            request_received_time = time.monotonic()
            if request is None:
                logger.info("No Modal QUIC request received for %.1fs; closing", receive_timeout_s)
                break
            request_wait_time = request_received_time - wait_start_time

            unpack_start = time.monotonic()
            payload = _unpack_request_payload(request)
            unpack_time = time.monotonic() - unpack_start
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected packed dict payload, got {type(payload).__name__}")

            infer_start = time.monotonic()
            response = handler(payload)
            infer_time = time.monotonic() - infer_start
            if not isinstance(response, dict):
                raise RuntimeError(f"Expected handler dict response, got {type(response).__name__}")

            server_timing = response.setdefault("server_timing", {})
            server_timing.update(
                {
                    "server_waiting_for_request_ms": request_wait_time * 1000,
                    "request_bytes": len(request),
                    "request_unpack_ms": unpack_time * 1000,
                    "infer_ms": infer_time * 1000,
                    "policy_infer_ms": infer_time * 1000,
                    "transport": "modal-quic",
                }
            )
            if prev_total_time is not None:
                server_timing["prev_total_ms"] = prev_total_time * 1000
                server_timing["prev_handler_active_ms"] = prev_handler_active_time * 1000
                server_timing["prev_response_send_ms"] = prev_response_send_time * 1000
                server_timing["prev_response_bytes"] = prev_response_bytes

            pack_start = time.monotonic()
            packed_response = packer.pack(response)
            pack_time = time.monotonic() - pack_start
            server_timing["response_pack_ms_approx"] = pack_time * 1000
            server_timing["response_bytes_approx"] = len(packed_response)
            packed_response = packer.pack(response)

            send_start = time.monotonic()
            portal.send(packed_response)
            prev_response_send_time = time.monotonic() - send_start
            prev_response_bytes = len(packed_response)
            prev_handler_active_time = time.monotonic() - request_received_time
            prev_total_time = time.monotonic() - wait_start_time

        except Exception:
            logger.exception("Modal QUIC server request failed")
            with contextlib.suppress(Exception):
                portal.send(packer.pack({"error": traceback.format_exc()}))
            break


def _is_connection_closed_error(exc: Exception) -> bool:
    message = str(exc)
    return "ConnectionLost" in message or "ApplicationClosed" in message or "closed" in message.lower()


def _unpack_request_payload(request: bytes) -> Any:
    try:
        return msgpack_numpy.unpackb(request)
    except msgpack.ExtraData:
        unpacker = msgpack_numpy.Unpacker()
        unpacker.feed(request)
        objects = list(unpacker)
        logger.warning(
            "Received %s coalesced Modal QUIC msgpack objects in one frame; "
            "first_types=%s request_bytes=%s request_prefix_hex=%s",
            len(objects),
            [type(obj).__name__ for obj in objects[:4]],
            len(request),
            request[:32].hex(),
        )
        objects = [obj for obj in objects if obj not in (b"hello", "hello")]
        if len(objects) == 1:
            return objects[0]
        raise


def _elapsed_ms(start: float) -> float:
    return 1000.0 * (time.monotonic() - start)
