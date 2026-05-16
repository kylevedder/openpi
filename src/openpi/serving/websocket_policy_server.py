import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        prev_response_send_time = None
        prev_response_bytes = None
        while True:
            try:
                start_time = time.monotonic()
                request = await websocket.recv()
                request_recv_wait_time = time.monotonic() - start_time
                request_unpack_start = time.monotonic()
                obs = msgpack_numpy.unpackb(request)
                request_unpack_time = time.monotonic() - request_unpack_start

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "request_recv_wait_ms": request_recv_wait_time * 1000,
                    "request_bytes": len(request),
                    "request_unpack_ms": request_unpack_time * 1000,
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                    action["server_timing"]["prev_response_send_ms"] = prev_response_send_time * 1000
                    action["server_timing"]["prev_response_bytes"] = prev_response_bytes

                response_pack_start = time.monotonic()
                response = packer.pack(action)
                response_pack_time = time.monotonic() - response_pack_start
                action["server_timing"]["response_pack_ms_approx"] = response_pack_time * 1000
                action["server_timing"]["response_bytes_approx"] = len(response)
                response = packer.pack(action)

                response_send_start = time.monotonic()
                await websocket.send(response)
                prev_response_send_time = time.monotonic() - response_send_start
                prev_response_bytes = len(response)
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
