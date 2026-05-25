from __future__ import annotations

import asyncio
import dataclasses
import http
import logging
import time
import traceback
from typing import Literal

import numpy as np
from openpi_client import base_policy
from openpi_client import msgpack_numpy
import tyro
import websockets.asyncio.server as _server
import websockets.frames

from openpi.policies import yam_policy
from openpi.serving import websocket_policy_server
from openpi.shared import jpeg_transport


@dataclasses.dataclass
class Args:
    mode: Literal["echo", "noop-policy"] = "echo"
    host: str = "0.0.0.0"
    port: int = 8765
    fixed_sleep_ms: float = 0.0
    response_bytes: int = 3_000
    action_horizon: int = 50
    action_dim: int = 14


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    if args.mode == "echo":
        EchoServer(args).serve_forever()
    elif args.mode == "noop-policy":
        policy = NoopPolicy(
            fixed_sleep_ms=args.fixed_sleep_ms,
            action_horizon=args.action_horizon,
            action_dim=args.action_dim,
        )
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=policy.metadata,
        )
        server.serve_forever()
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


class EchoServer:
    def __init__(self, args: Args) -> None:
        self._args = args
        self._metadata = {
            "protocol": "yam_latency_echo_v1",
            "fixed_sleep_ms": args.fixed_sleep_ms,
            "default_response_bytes": args.response_bytes,
        }

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._args.host,
            self._args.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logging.info("Latency echo connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        prev_handler_active_time = None
        prev_response_send_time = None
        prev_response_bytes = None
        while True:
            try:
                wait_start_time = time.monotonic()
                request = await websocket.recv()
                request_received_time = time.monotonic()
                request_wait_time = request_received_time - wait_start_time

                unpack_start = time.monotonic()
                obs = msgpack_numpy.unpackb(request)
                unpack_time = time.monotonic() - unpack_start

                sleep_ms = float(obs.get("sleep_ms", self._args.fixed_sleep_ms))
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)

                response_bytes = int(obs.get("response_bytes", self._args.response_bytes))
                response = {
                    "sequence": int(obs.get("sequence", -1)),
                    "payload": bytes(response_bytes),
                    "server_timing": {
                        "server_waiting_for_request_ms": request_wait_time * 1000,
                        "request_bytes": len(request),
                        "request_unpack_ms": unpack_time * 1000,
                        "noop_sleep_ms": sleep_ms,
                    },
                }
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                    response["server_timing"]["prev_handler_active_ms"] = prev_handler_active_time * 1000
                    response["server_timing"]["prev_response_send_ms"] = prev_response_send_time * 1000
                    response["server_timing"]["prev_response_bytes"] = prev_response_bytes

                pack_start = time.monotonic()
                packed_response = packer.pack(response)
                pack_time = time.monotonic() - pack_start
                response["server_timing"]["response_pack_ms_approx"] = pack_time * 1000
                response["server_timing"]["response_bytes_approx"] = len(packed_response)
                packed_response = packer.pack(response)

                send_start = time.monotonic()
                await websocket.send(packed_response)
                prev_response_send_time = time.monotonic() - send_start
                prev_response_bytes = len(packed_response)
                prev_handler_active_time = time.monotonic() - request_received_time
                prev_total_time = time.monotonic() - wait_start_time

            except websockets.ConnectionClosed:
                logging.info("Latency echo connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


class NoopPolicy(base_policy.BasePolicy):
    def __init__(self, *, fixed_sleep_ms: float, action_horizon: int, action_dim: int) -> None:
        self._fixed_sleep_ms = fixed_sleep_ms
        self._actions = np.zeros((action_horizon, action_dim), dtype=np.float32)
        self.metadata = {
            "protocol": "yam_latency_noop_policy_v1",
            "fixed_sleep_ms": fixed_sleep_ms,
            "reset_pose": [0.0] * action_dim,
            "action_space": yam_policy.ACTION_SPACE,
            "gripper_convention": yam_policy.GRIPPER_CONVENTION,
            "state_order": list(yam_policy.STATE_ORDER),
            "action_horizon": action_horizon,
            "fps": 50.0,
            "image_transport": jpeg_transport.IMAGE_TRANSPORT,
            "jpeg_quality": jpeg_transport.JPEG_QUALITY,
            "image_resolution": list(jpeg_transport.IMAGE_RESOLUTION),
        }

    def infer(self, obs: dict) -> dict:
        sleep_ms = float(obs.get("diagnostic_sleep_ms", self._fixed_sleep_ms))
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
        return {"actions": self._actions.copy()}


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


if __name__ == "__main__":
    main(tyro.cli(Args))
