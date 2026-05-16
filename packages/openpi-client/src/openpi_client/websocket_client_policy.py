import logging
import time
from collections.abc import Callable
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.exceptions
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        connect_timeout_s: float | None = 60,
        retry_interval_s: float = 5,
        response_status_interval_s: float | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._connect_timeout_s = connect_timeout_s
        self._retry_interval_s = retry_interval_s
        self._response_status_interval_s = response_status_interval_s
        self._status_callback = status_callback
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        self._emit_status(f"Waiting for policy server at {self._uri}")
        attempt = 0
        wait_start = time.monotonic()
        while True:
            attempt += 1
            attempt_start = time.monotonic()
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                self._emit_status(
                    f"Opening websocket attempt {attempt} "
                    f"(timeout={self._connect_timeout_s if self._connect_timeout_s is not None else 'none'}s)"
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=self._connect_timeout_s,
                )
                self._emit_status(
                    f"Websocket connected in {time.monotonic() - attempt_start:.1f}s; waiting for metadata"
                )
                metadata = msgpack_numpy.unpackb(self._recv_with_status(conn, "policy metadata"))
                self._emit_status(f"Policy server ready after {time.monotonic() - wait_start:.1f}s")
                return conn, metadata
            except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
                elapsed_s = time.monotonic() - wait_start
                retry_s = max(0.0, self._retry_interval_s)
                self._emit_status(
                    f"Policy server not ready after {elapsed_s:.1f}s "
                    f"(attempt {attempt}: {type(exc).__name__}: {exc}); retrying in {retry_s:g}s"
                )
                logging.debug("Websocket connection attempt failed: %s", exc)
                time.sleep(retry_s)

    def _emit_status(self, message: str) -> None:
        logging.info(message)
        if self._status_callback is not None:
            self._status_callback(message)

    def _recv_with_status(self, conn: websockets.sync.client.ClientConnection, label: str):
        if self._response_status_interval_s is None:
            return conn.recv()

        wait_start = time.monotonic()
        while True:
            try:
                return conn.recv(timeout=self._response_status_interval_s)
            except TimeoutError:
                self._emit_status(f"Waiting for {label} for {time.monotonic() - wait_start:.1f}s")

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        request_start = time.monotonic()
        pack_start = time.monotonic()
        data = self._packer.pack(obs)
        pack_ms = 1000.0 * (time.monotonic() - pack_start)
        send_start = time.monotonic()
        self._ws.send(data)
        send_ms = 1000.0 * (time.monotonic() - send_start)
        recv_start = time.monotonic()
        response = self._recv_with_status(self._ws, "policy inference response")
        recv_wait_ms = 1000.0 * (time.monotonic() - recv_start)
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        unpack_start = time.monotonic()
        result = msgpack_numpy.unpackb(response)
        unpack_ms = 1000.0 * (time.monotonic() - unpack_start)
        result["client_timing"] = {
            "pack_ms": pack_ms,
            "send_ms": send_ms,
            "recv_wait_ms": recv_wait_ms,
            "unpack_ms": unpack_ms,
            "total_ms": 1000.0 * (time.monotonic() - request_start),
            "request_bytes": len(data),
            "response_bytes": len(response),
        }
        return result

    @override
    def reset(self) -> None:
        pass
