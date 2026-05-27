from __future__ import annotations

import contextlib
import dataclasses
import multiprocessing as mp
from multiprocessing import connection
from multiprocessing import shared_memory
from pathlib import Path
import queue
import time
import traceback
from typing import Any, Literal

import numpy as np

from examples.yam_real import common
from examples.yam_real import mcap_episode
from examples.yam_real import teleop

WorkerMessage = dict[str, Any]


class WorkerRuntimeError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class SharedCameraRingSpec:
    camera_name: str
    frame_shm_name: str
    sequence_shm_name: str
    ring_size: int
    frame_shape: tuple[int, int, int]
    dtype: str
    lock: Any


@dataclasses.dataclass(frozen=True)
class CameraFrameRef:
    camera_name: str
    slot_index: int
    sequence_index: int
    capture_timestamp_ns: int
    capture_time_s: float
    monotonic_time_s: float

    def metadata(self, *, now_monotonic_s: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        return {
            "capture_time_s": self.capture_time_s,
            "monotonic_time_s": self.monotonic_time_s,
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "sequence_index": self.sequence_index,
            "slot_index": self.slot_index,
            "age_s": max(0.0, now - self.monotonic_time_s),
            "ipc_backend": "multiprocessing_shared_memory",
        }

    def as_payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ProcessCameraSnapshot:
    refs: dict[str, CameraFrameRef]
    metadata: dict[str, dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class TeleopStepResult:
    side: str
    state: np.ndarray
    action: np.ndarray
    buttons: np.ndarray
    synchronized: bool
    timings_ms: dict[str, float]
    ipc_latency_ms: float


class SharedCameraRing:
    def __init__(self, spec: SharedCameraRingSpec, *, create: bool) -> None:
        self.spec = spec
        frame_size = int(np.prod((spec.ring_size, *spec.frame_shape)) * np.dtype(spec.dtype).itemsize)
        seq_size = int(spec.ring_size * np.dtype(np.int64).itemsize)
        if create:
            self.frame_shm = shared_memory.SharedMemory(create=True, size=frame_size)
            self.sequence_shm = shared_memory.SharedMemory(create=True, size=seq_size)
            self.spec = dataclasses.replace(
                spec,
                frame_shm_name=self.frame_shm.name,
                sequence_shm_name=self.sequence_shm.name,
            )
        else:
            self.frame_shm = shared_memory.SharedMemory(name=spec.frame_shm_name)
            self.sequence_shm = shared_memory.SharedMemory(name=spec.sequence_shm_name)
        self.frames = np.ndarray(
            (spec.ring_size, *spec.frame_shape),
            dtype=np.dtype(spec.dtype),
            buffer=self.frame_shm.buf,
        )
        self.sequences = np.ndarray((spec.ring_size,), dtype=np.int64, buffer=self.sequence_shm.buf)
        if create:
            self.frames.fill(0)
            self.sequences.fill(-1)

    @classmethod
    def create(
        cls,
        *,
        camera_name: str,
        ring_size: int,
        frame_shape: tuple[int, int, int],
        dtype: str,
        lock: Any,
    ) -> SharedCameraRing:
        placeholder = SharedCameraRingSpec(
            camera_name=camera_name,
            frame_shm_name="",
            sequence_shm_name="",
            ring_size=ring_size,
            frame_shape=frame_shape,
            dtype=dtype,
            lock=lock,
        )
        return cls(placeholder, create=True)

    @classmethod
    def open(cls, spec: SharedCameraRingSpec) -> SharedCameraRing:
        return cls(spec, create=False)

    def write_frame(self, frame_ref: CameraFrameRef, frame: np.ndarray) -> None:
        expected_shape = self.spec.frame_shape
        if tuple(frame.shape) != expected_shape:
            raise ValueError(f"{self.spec.camera_name} frame shape {frame.shape} != {expected_shape}")
        with self.spec.lock:
            self.frames[frame_ref.slot_index, ...] = frame
            self.sequences[frame_ref.slot_index] = frame_ref.sequence_index

    def read_frame(self, frame_ref: CameraFrameRef) -> np.ndarray:
        with self.spec.lock:
            current_sequence = int(self.sequences[frame_ref.slot_index])
            if current_sequence != frame_ref.sequence_index:
                raise RuntimeError(
                    f"{frame_ref.camera_name} shared-memory slot overwritten: "
                    f"slot={frame_ref.slot_index}, expected_seq={frame_ref.sequence_index}, current_seq={current_sequence}"
                )
            return np.array(self.frames[frame_ref.slot_index, ...], copy=True)

    def close(self) -> None:
        self.frame_shm.close()
        self.sequence_shm.close()

    def unlink(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.frame_shm.unlink()
        with contextlib.suppress(FileNotFoundError):
            self.sequence_shm.unlink()


class CameraProcessGroup:
    def __init__(
        self,
        *,
        ctx: mp.context.BaseContext,
        paths: dict[str, str],
        config: common.CameraConfig,
        ring_size: int,
        startup_timeout_s: float,
        frame_queue_size: int = 256,
    ) -> None:
        self._ctx = ctx
        self._paths = paths
        self._config = config
        self._ring_size = ring_size
        self._startup_timeout_s = startup_timeout_s
        self._frame_queue = ctx.Queue(maxsize=frame_queue_size)
        self._stop_event = ctx.Event()
        self._rings: dict[str, SharedCameraRing] = {}
        self._processes: dict[str, mp.Process] = {}
        self._latest_refs: dict[str, CameraFrameRef] = {}
        self.actual_modes: dict[str, dict[str, Any]] = {}

    @property
    def ring_specs(self) -> dict[str, SharedCameraRingSpec]:
        return {name: ring.spec for name, ring in self._rings.items()}

    @property
    def process_ids(self) -> dict[str, int | None]:
        return {name: process.pid for name, process in self._processes.items()}

    def start(self) -> None:
        if self._processes:
            return
        frame_shape = (self._config.height, self._config.width, 3)
        for camera_name, camera_path in self._paths.items():
            lock = self._ctx.Lock()
            ring = SharedCameraRing.create(
                camera_name=camera_name,
                ring_size=self._ring_size,
                frame_shape=frame_shape,
                dtype="uint8",
                lock=lock,
            )
            self._rings[camera_name] = ring
            process = self._ctx.Process(
                target=_camera_worker_main,
                name=f"yam-camera-{camera_name}",
                args=(camera_name, camera_path, self._config, ring.spec, self._frame_queue, self._stop_event),
            )
            process.start()
            self._processes[camera_name] = process
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout_s
        ready = set()
        while time.monotonic() < deadline and ready != set(self._paths):
            self._raise_for_dead_workers()
            try:
                message = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self._handle_message(message)
            if message.get("type") == "camera_ready":
                ready.add(str(message["camera_name"]))
        missing = sorted(set(self._paths) - ready)
        if missing:
            raise WorkerRuntimeError(f"Timed out waiting for camera worker(s): {missing}")

    def poll(self) -> None:
        while True:
            try:
                message = self._frame_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_message(message)
        self._raise_for_dead_workers()

    def snapshot(self, *, max_age_s: float | None) -> ProcessCameraSnapshot:
        self.poll()
        missing = sorted(set(self._paths) - set(self._latest_refs))
        if missing:
            raise WorkerRuntimeError(f"Missing camera frame refs: {missing}")
        now = time.monotonic()
        refs = dict(self._latest_refs)
        metadata = {name: ref.metadata(now_monotonic_s=now) for name, ref in refs.items()}
        stale = {
            name: float(meta["age_s"])
            for name, meta in metadata.items()
            if max_age_s is not None and float(meta["age_s"]) > max_age_s
        }
        if stale:
            details = ", ".join(f"{name}={age:.3f}s" for name, age in sorted(stale.items()))
            raise WorkerRuntimeError(f"Stale camera frame(s): {details}; max_age_s={max_age_s:.3f}")
        return ProcessCameraSnapshot(refs=refs, metadata=metadata)

    def close(self) -> None:
        self._stop_event.set()
        for process in self._processes.values():
            process.join(timeout=1.0)
        for process in self._processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        for ring in self._rings.values():
            with contextlib.suppress(Exception):
                ring.close()
            with contextlib.suppress(Exception):
                ring.unlink()
        self._processes.clear()
        self._rings.clear()

    def _handle_message(self, message: WorkerMessage) -> None:
        message_type = message.get("type")
        if message_type == "camera_ready":
            self.actual_modes[str(message["camera_name"])] = dict(message.get("capture_properties", {}))
            return
        if message_type == "camera_frame":
            frame_ref = CameraFrameRef(**message["frame_ref"])
            self._latest_refs[frame_ref.camera_name] = frame_ref
            return
        if message_type == "error":
            raise WorkerRuntimeError(_format_worker_error(message))
        raise WorkerRuntimeError(f"Unexpected camera worker message: {message}")

    def _raise_for_dead_workers(self) -> None:
        for camera_name, process in self._processes.items():
            if process.exitcode not in (None, 0):
                raise WorkerRuntimeError(f"Camera worker {camera_name} exited with code {process.exitcode}")


class TeleopProcessGroup:
    def __init__(
        self,
        *,
        ctx: mp.context.BaseContext,
        gripper: str,
        bilateral_kp: float,
        ee_mass: float | None,
        use_gravity_comp: bool,
        startup_timeout_s: float,
    ) -> None:
        self._ctx = ctx
        self._gripper = gripper
        self._bilateral_kp = bilateral_kp
        self._ee_mass = ee_mass
        self._use_gravity_comp = use_gravity_comp
        self._startup_timeout_s = startup_timeout_s
        self._connections: dict[str, connection.Connection] = {}
        self._processes: dict[str, mp.Process] = {}
        self._request_index = 0
        self.synchronized: dict[str, bool] = {"left": False, "right": False}

    @property
    def process_ids(self) -> dict[str, int | None]:
        return {name: process.pid for name, process in self._processes.items()}

    def start(self) -> None:
        if self._processes:
            return
        for side in ("left", "right"):
            parent_conn, child_conn = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_teleop_worker_main,
                name=f"yam-teleop-{side}",
                args=(
                    side,
                    child_conn,
                    self._gripper,
                    self._bilateral_kp,
                    self._ee_mass,
                ),
                kwargs={"use_gravity_comp": self._use_gravity_comp},
            )
            process.start()
            child_conn.close()
            self._connections[side] = parent_conn
            self._processes[side] = process
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout_s
        pending = set(self._connections)
        while pending and time.monotonic() < deadline:
            self._raise_for_dead_workers()
            ready_conns = connection.wait([self._connections[side] for side in pending], timeout=0.05)
            for conn in ready_conns:
                message = conn.recv()
                if message.get("type") != "ready":
                    if message.get("type") == "error":
                        raise WorkerRuntimeError(_format_worker_error(message))
                    raise WorkerRuntimeError(f"Unexpected teleop startup message: {message}")
                pending.remove(str(message["side"]))
        if pending:
            raise WorkerRuntimeError(f"Timed out waiting for teleop worker(s): {sorted(pending)}")

    def step(self, *, timeout_s: float) -> dict[str, TeleopStepResult]:
        self._request_index += 1
        request_id = self._request_index
        send_time_s = time.perf_counter()
        for side, conn in self._connections.items():
            conn.send({"type": "step", "request_id": request_id, "send_time_s": send_time_s, "side": side})
        responses: dict[str, TeleopStepResult] = {}
        deadline = time.monotonic() + timeout_s
        pending = set(self._connections)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerRuntimeError(f"Timed out waiting for teleop step response(s): {sorted(pending)}")
            self._raise_for_dead_workers()
            ready_conns = connection.wait([self._connections[side] for side in pending], timeout=remaining)
            if not ready_conns:
                continue
            for conn in ready_conns:
                message = conn.recv()
                if message.get("type") == "error":
                    raise WorkerRuntimeError(_format_worker_error(message))
                if message.get("type") != "step":
                    raise WorkerRuntimeError(f"Unexpected teleop step message: {message}")
                if int(message.get("request_id", -1)) != request_id:
                    raise WorkerRuntimeError(f"Unexpected teleop request id: {message}")
                side = str(message["side"])
                result = TeleopStepResult(
                    side=side,
                    state=np.asarray(message["state"], dtype=np.float32),
                    action=np.asarray(message["action"], dtype=np.float32),
                    buttons=np.asarray(message["buttons"], dtype=np.float32),
                    synchronized=bool(message["synchronized"]),
                    timings_ms={str(k): float(v) for k, v in message.get("timings_ms", {}).items()},
                    ipc_latency_ms=1000.0 * (time.perf_counter() - send_time_s),
                )
                self.synchronized[side] = result.synchronized
                responses[side] = result
                pending.remove(side)
        return responses

    def close(self) -> None:
        request_id = f"shutdown-{time.monotonic_ns()}"
        for conn in self._connections.values():
            with contextlib.suppress(Exception):
                conn.send({"type": "shutdown", "request_id": request_id})
        with contextlib.suppress(Exception):
            self._wait_for_acks("shutdown_ack", request_id=request_id, timeout_s=2.0)
        for conn in self._connections.values():
            with contextlib.suppress(Exception):
                conn.close()
        for process in self._processes.values():
            process.join(timeout=1.0)
        for process in self._processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self._connections.clear()
        self._processes.clear()

    def _wait_for_acks(self, ack_type: str, *, request_id: Any, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        pending = set(self._connections)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerRuntimeError(f"Timed out waiting for teleop {ack_type}: {sorted(pending)}")
            self._raise_for_dead_workers()
            ready_conns = connection.wait([self._connections[side] for side in pending], timeout=remaining)
            for conn in ready_conns:
                message = conn.recv()
                if message.get("type") == "error":
                    raise WorkerRuntimeError(_format_worker_error(message))
                if message.get("type") != ack_type or message.get("request_id") != request_id:
                    raise WorkerRuntimeError(f"Unexpected teleop ack message: {message}")
                pending.remove(str(message["side"]))

    def _raise_for_dead_workers(self) -> None:
        for side, process in self._processes.items():
            if process.exitcode not in (None, 0):
                raise WorkerRuntimeError(f"Teleop worker {side} exited with code {process.exitcode}")


class WriterProcessClient:
    def __init__(
        self,
        *,
        ctx: mp.context.BaseContext,
        camera_specs: dict[str, SharedCameraRingSpec],
        queue_size: int,
        startup_timeout_s: float,
    ) -> None:
        self._ctx = ctx
        self._camera_specs = camera_specs
        self._command_queue = ctx.Queue(maxsize=queue_size)
        self._status_queue = ctx.Queue()
        self._stop_event = ctx.Event()
        self._startup_timeout_s = startup_timeout_s
        self._process: mp.Process | None = None
        self.accepted_rows = 0
        self.written_rows = 0
        self.active_episode_dir: Path | None = None

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = self._ctx.Process(
            target=_writer_worker_main,
            name="yam-mcap-writer",
            args=(self._camera_specs, self._command_queue, self._status_queue, self._stop_event),
        )
        self._process.start()
        deadline = time.monotonic() + self._startup_timeout_s
        while time.monotonic() < deadline:
            try:
                message = self._status_queue.get(timeout=0.05)
            except queue.Empty:
                self._raise_for_dead_worker()
                continue
            if message.get("type") == "writer_ready":
                return
            if message.get("type") == "error":
                raise WorkerRuntimeError(_format_worker_error(message))
            raise WorkerRuntimeError(f"Unexpected writer startup message: {message}")
        raise WorkerRuntimeError("Timed out waiting for writer worker")

    def start_episode(
        self,
        *,
        episode_dir: Path,
        task: str,
        fps: float,
        camera_fps: float,
        image_width: int,
        image_height: int,
        timeout_s: float,
    ) -> None:
        self.poll()
        self._put_command(
            {
                "type": "start_episode",
                "episode_dir": str(episode_dir),
                "task": task,
                "fps": fps,
                "camera_fps": camera_fps,
                "image_width": image_width,
                "image_height": image_height,
            },
            timeout_s=timeout_s,
        )
        message = self._wait_for_message("episode_started", timeout_s=timeout_s)
        self.active_episode_dir = Path(message["episode_dir"])
        self.accepted_rows = 0
        self.written_rows = 0

    def write_row(
        self,
        *,
        frame_index: int,
        timestamp_ns: int,
        state: np.ndarray,
        action: np.ndarray,
        camera_refs: dict[str, CameraFrameRef],
        camera_metadata: dict[str, dict[str, Any]],
        timeout_s: float,
    ) -> int:
        self.poll()
        self._put_command(
            {
                "type": "write_row",
                "frame_index": frame_index,
                "timestamp_ns": timestamp_ns,
                "state": np.asarray(state, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "camera_refs": {name: ref.as_payload() for name, ref in camera_refs.items()},
                "camera_metadata": camera_metadata,
            },
            timeout_s=timeout_s,
        )
        self.accepted_rows += 1
        return self.queue_depth()

    def stop_episode(self, *, timeout_s: float) -> dict[str, Any]:
        self.poll()
        self._put_command({"type": "stop_episode"}, timeout_s=timeout_s)
        message = self._wait_for_message("episode_stopped", timeout_s=timeout_s)
        self.active_episode_dir = None
        self.written_rows = int(message.get("written_rows", self.written_rows))
        return message

    def poll(self) -> None:
        while True:
            try:
                message = self._status_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_status(message)
        self._raise_for_dead_worker()

    def close(self) -> None:
        if self._process is None:
            return
        with contextlib.suppress(Exception):
            self._command_queue.put({"type": "shutdown"}, timeout=0.2)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._stop_event.set()
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._process = None

    def queue_depth(self) -> int:
        with contextlib.suppress(NotImplementedError, OSError):
            return int(self._command_queue.qsize())
        return -1

    def _put_command(self, command: dict[str, Any], *, timeout_s: float) -> None:
        self._raise_for_dead_worker()
        try:
            self._command_queue.put(command, timeout=timeout_s)
        except queue.Full as exc:
            raise WorkerRuntimeError(f"Writer queue full while sending {command.get('type')}") from exc

    def _wait_for_message(self, message_type: str, *, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._raise_for_dead_worker()
            try:
                message = self._status_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if message.get("type") == message_type:
                return message
            self._handle_status(message)
        raise WorkerRuntimeError(f"Timed out waiting for writer message {message_type}")

    def _handle_status(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "row_written":
            self.written_rows = max(self.written_rows, int(message.get("written_rows", 0)))
            return
        if message_type == "error":
            raise WorkerRuntimeError(_format_worker_error(message))
        if message_type in {"writer_ready", "episode_started", "episode_stopped", "shutdown_ack"}:
            return
        raise WorkerRuntimeError(f"Unexpected writer status message: {message}")

    def _raise_for_dead_worker(self) -> None:
        if self._process is not None and self._process.exitcode not in (None, 0):
            raise WorkerRuntimeError(f"Writer worker exited with code {self._process.exitcode}")


class YamMultiprocessRuntime:
    def __init__(
        self,
        *,
        camera_config: common.CameraConfig,
        gripper: str,
        bilateral_kp: float,
        ee_mass: float | None,
        use_gravity_comp: bool,
        camera_ring_size: int,
        writer_queue_size: int,
        startup_timeout_s: float,
        step_timeout_s: float,
        writer_drain_timeout_s: float,
    ) -> None:
        self._ctx = mp.get_context("spawn")
        self._camera_config = camera_config
        self._startup_timeout_s = startup_timeout_s
        self._step_timeout_s = step_timeout_s
        self._writer_drain_timeout_s = writer_drain_timeout_s
        self.cameras = CameraProcessGroup(
            ctx=self._ctx,
            paths=common.CAMERA_PATHS,
            config=camera_config,
            ring_size=camera_ring_size,
            startup_timeout_s=startup_timeout_s,
        )
        self.teleop = TeleopProcessGroup(
            ctx=self._ctx,
            gripper=gripper,
            bilateral_kp=bilateral_kp,
            ee_mass=ee_mass,
            use_gravity_comp=use_gravity_comp,
            startup_timeout_s=startup_timeout_s,
        )
        self.writer: WriterProcessClient | None = None
        self._writer_queue_size = writer_queue_size

    @property
    def camera_actual_modes(self) -> dict[str, dict[str, Any]]:
        return self.cameras.actual_modes

    @property
    def process_ids(self) -> dict[str, Any]:
        return {
            "cameras": self.cameras.process_ids,
            "teleop": self.teleop.process_ids,
            "writer": None if self.writer is None else self.writer.pid,
        }

    def start(self) -> None:
        common.ensure_i2rt_importable()
        self.cameras.start()
        self.teleop.start()
        self.writer = WriterProcessClient(
            ctx=self._ctx,
            camera_specs=self.cameras.ring_specs,
            queue_size=self._writer_queue_size,
            startup_timeout_s=self._startup_timeout_s,
        )
        self.writer.start()

    def poll(self) -> None:
        self.cameras.poll()
        self.teleop._raise_for_dead_workers()  # noqa: SLF001
        if self.writer is not None:
            self.writer.poll()

    def step_teleop(self) -> dict[str, TeleopStepResult]:
        return self.teleop.step(timeout_s=self._step_timeout_s)

    def camera_snapshot(self, *, max_age_s: float | None) -> ProcessCameraSnapshot:
        return self.cameras.snapshot(max_age_s=max_age_s)

    def start_episode(
        self,
        *,
        episode_dir: Path,
        task: str,
        fps: float,
        camera_fps: float,
        image_width: int,
        image_height: int,
    ) -> None:
        if self.writer is None:
            raise WorkerRuntimeError("Writer process is not started")
        self.writer.start_episode(
            episode_dir=episode_dir,
            task=task,
            fps=fps,
            camera_fps=camera_fps,
            image_width=image_width,
            image_height=image_height,
            timeout_s=self._writer_drain_timeout_s,
        )

    def write_row(
        self,
        *,
        frame_index: int,
        timestamp_ns: int,
        state: np.ndarray,
        action: np.ndarray,
        camera_snapshot: ProcessCameraSnapshot,
    ) -> int:
        if self.writer is None:
            raise WorkerRuntimeError("Writer process is not started")
        return self.writer.write_row(
            frame_index=frame_index,
            timestamp_ns=timestamp_ns,
            state=state,
            action=action,
            camera_refs=camera_snapshot.refs,
            camera_metadata=camera_snapshot.metadata,
            timeout_s=min(0.01, self._step_timeout_s),
        )

    def stop_episode(self) -> dict[str, Any]:
        if self.writer is None:
            return {"type": "episode_stopped", "written_rows": 0}
        return self.writer.stop_episode(timeout_s=self._writer_drain_timeout_s)

    def close(self) -> None:
        if self.writer is not None:
            with contextlib.suppress(Exception):
                self.writer.close()
        with contextlib.suppress(Exception):
            self.teleop.close()
        with contextlib.suppress(Exception):
            self.cameras.close()

    def __enter__(self) -> YamMultiprocessRuntime:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _camera_worker_main(
    camera_name: str,
    camera_path: str,
    config: common.CameraConfig,
    ring_spec: SharedCameraRingSpec,
    frame_queue: Any,
    stop_event: Any,
) -> None:
    ring = None
    camera = None
    try:
        ring = SharedCameraRing.open(ring_spec)
        camera = common.LinuxpyV4L2Camera(camera_name, camera_path, config)
        camera.open()
        _put_drop_old(
            frame_queue,
            {
                "type": "camera_ready",
                "camera_name": camera_name,
                "capture_properties": camera.capture_properties(),
                "ring_spec": dataclasses.asdict(dataclasses.replace(ring.spec, lock=None)),
            },
        )
        while not stop_event.is_set():
            captured = camera.read()
            frame_ref = CameraFrameRef(
                camera_name=camera_name,
                slot_index=int(captured.sequence_index % ring.spec.ring_size),
                sequence_index=int(captured.sequence_index),
                capture_timestamp_ns=int(captured.capture_timestamp_ns),
                capture_time_s=float(captured.capture_time_s),
                monotonic_time_s=float(captured.monotonic_time_s),
            )
            ring.write_frame(frame_ref, captured.frame)
            _put_drop_old(
                frame_queue,
                {
                    "type": "camera_frame",
                    "camera_name": camera_name,
                    "frame_ref": frame_ref.as_payload(),
                },
            )
    except BaseException as exc:
        _put_drop_old(frame_queue, _error_message("camera", camera_name, exc))
    finally:
        if camera is not None:
            with contextlib.suppress(Exception):
                camera.close()
        if ring is not None:
            with contextlib.suppress(Exception):
                ring.close()


def _teleop_worker_main(
    side: Literal["left", "right"],
    conn: connection.Connection,
    gripper: str,
    bilateral_kp: float,
    ee_mass: float | None,
    *,
    use_gravity_comp: bool,
) -> None:
    robots = []
    pair = None
    try:
        common.ensure_i2rt_importable()
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        gripper_type = GripperType.from_string_name(gripper)
        follower = get_yam_robot(
            channel=common.FOLLOWER_CHANNELS[side],
            gripper_type=gripper_type,
            use_gravity_comp=use_gravity_comp,
            zero_gravity_mode=False,
            ee_mass=ee_mass,
        )
        robots.append(follower)
        leader = teleop.YamLeader(
            get_yam_robot(
                channel=common.LEADER_CHANNELS[side],
                gripper_type=GripperType.YAM_TEACHING_HANDLE,
                use_gravity_comp=use_gravity_comp,
                zero_gravity_mode=True,
                ee_mass=ee_mass,
            )
        )
        robots.append(leader.robot)
        pair = teleop.TeleopPair(side, leader, follower, bilateral_kp)
        conn.send({"type": "ready", "side": side, "synchronized": pair.synchronized})
        while True:
            message = conn.recv()
            message_type = message.get("type")
            request_id = message.get("request_id")
            if message_type == "step":
                state, action, buttons, timings_ms = pair.step_with_timings()
                conn.send(
                    {
                        "type": "step",
                        "side": side,
                        "request_id": request_id,
                        "state": state,
                        "action": action,
                        "buttons": buttons,
                        "synchronized": pair.synchronized,
                        "timings_ms": timings_ms,
                    }
                )
            elif message_type == "shutdown":
                conn.send({"type": "shutdown_ack", "side": side, "request_id": request_id})
                break
            else:
                raise RuntimeError(f"Unsupported teleop command: {message}")
    except EOFError:
        pass
    except BaseException as exc:
        with contextlib.suppress(Exception):
            conn.send(_error_message("teleop", side, exc))
    finally:
        if pair is not None:
            with contextlib.suppress(Exception):
                pair.close()
        for robot in robots:
            with contextlib.suppress(Exception):
                robot.close()
        with contextlib.suppress(Exception):
            conn.close()


def _writer_worker_main(
    camera_specs: dict[str, SharedCameraRingSpec],
    command_queue: Any,
    status_queue: Any,
    stop_event: Any,
) -> None:
    rings: dict[str, SharedCameraRing] = {}
    writer = None
    written_rows = 0
    episode_dir: Path | None = None
    try:
        rings = {name: SharedCameraRing.open(spec) for name, spec in camera_specs.items()}
        status_queue.put({"type": "writer_ready"})
        while not stop_event.is_set():
            try:
                command = command_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            command_type = command.get("type")
            if command_type == "start_episode":
                if writer is not None:
                    writer.close()
                episode_dir = Path(command["episode_dir"])
                writer = mcap_episode.YamMcapEpisodeWriter(
                    episode_dir,
                    task=str(command["task"]),
                    fps=float(command["fps"]),
                    camera_fps=float(command["camera_fps"]),
                    image_width=int(command["image_width"]),
                    image_height=int(command["image_height"]),
                )
                written_rows = 0
                status_queue.put({"type": "episode_started", "episode_dir": str(episode_dir)})
            elif command_type == "write_row":
                if writer is None:
                    raise RuntimeError("Writer received row before start_episode")
                camera_refs = {
                    name: CameraFrameRef(**payload) for name, payload in dict(command["camera_refs"]).items()
                }
                frames_rgb = {name: rings[name].read_frame(ref) for name, ref in camera_refs.items()}
                writer.write_step(
                    frames_rgb=frames_rgb,
                    camera_metadata=dict(command["camera_metadata"]),
                    state=np.asarray(command["state"], dtype=np.float32),
                    action=np.asarray(command["action"], dtype=np.float32),
                    timestamp_ns=int(command["timestamp_ns"]),
                )
                written_rows += 1
                status_queue.put(
                    {
                        "type": "row_written",
                        "frame_index": int(command["frame_index"]),
                        "written_rows": written_rows,
                        "episode_dir": str(episode_dir) if episode_dir is not None else None,
                    }
                )
            elif command_type == "stop_episode":
                close_start = time.perf_counter()
                if writer is not None:
                    writer.close()
                    writer = None
                status_queue.put(
                    {
                        "type": "episode_stopped",
                        "episode_dir": str(episode_dir) if episode_dir is not None else None,
                        "written_rows": written_rows,
                        "drain_close_ms": 1000.0 * (time.perf_counter() - close_start),
                    }
                )
                episode_dir = None
            elif command_type == "shutdown":
                break
            else:
                raise RuntimeError(f"Unsupported writer command: {command}")
    except BaseException as exc:
        status_queue.put(_error_message("writer", "mcap", exc))
    finally:
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()
        for ring in rings.values():
            with contextlib.suppress(Exception):
                ring.close()
        with contextlib.suppress(Exception):
            status_queue.put({"type": "shutdown_ack"})


def _put_drop_old(message_queue: Any, message: dict[str, Any]) -> None:
    try:
        message_queue.put_nowait(message)
        return
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            message_queue.get_nowait()
        with contextlib.suppress(queue.Full):
            message_queue.put_nowait(message)


def _error_message(worker_type: str, target: str, exc: BaseException) -> dict[str, Any]:
    return {
        "type": "error",
        "worker_type": worker_type,
        "target": target,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _format_worker_error(message: dict[str, Any]) -> str:
    return (
        f"{message.get('worker_type', 'worker')} {message.get('target', '')} failed: "
        f"{message.get('error_type', 'Error')}: {message.get('error', '')}\n{message.get('traceback', '')}"
    )
