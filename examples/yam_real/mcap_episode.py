from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import dataclasses
import json
from pathlib import Path
import time

from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
import numpy as np

from openpi.pistream import mcap as pistream_mcap
from openpi.pistream.proto import frame_pb2

CAMERA_FIELD_MAP = {
    "cam_high": pistream_mcap.FieldKey("base_0_camera", "observation/base_0_camera/rgb/image"),
    "cam_left_wrist": pistream_mcap.FieldKey(
        "left_wrist_0_camera",
        "observation/left_wrist_0_camera/rgb/image",
    ),
    "cam_right_wrist": pistream_mcap.FieldKey(
        "right_wrist_0_camera",
        "observation/right_wrist_0_camera/rgb/image",
    ),
}

LEFT_FOLLOWER = "arx_follow_1"
RIGHT_FOLLOWER = "arx_follow_2"
ACTIVE_AGENT_FIELD = pistream_mcap.FieldKey("system", "system/active_agent/node_id")
ACTION_START_FIELD = pistream_mcap.FieldKey("teleop", "system/active_agent/action_start_timestamp")
ACTION_SNAPSHOT_FIELD = pistream_mcap.FieldKey("teleop", "system/active_agent/action_snapshot")
MIN_TIMESTAMP_STEP_NS = 1_000_000
EXPECTED_SHARD_COUNT = 4


@dataclasses.dataclass(frozen=True)
class FieldValueRef:
    field: pistream_mcap.FieldKey
    sequence_id: int
    timestamp_ns: int


@dataclasses.dataclass(frozen=True)
class VideoUserdata:
    camera_name: str
    field: pistream_mcap.FieldKey
    sequence_id: int
    timestamp_ns: int


@dataclasses.dataclass(frozen=True)
class YamMcapEpisode:
    task: str
    fps: float
    timestamps_ns: np.ndarray
    state: np.ndarray
    action: np.ndarray
    images: dict[str, np.ndarray] | None = None
    camera_timestamps_ns: dict[str, np.ndarray] | None = None
    camera_row_timestamps_ns: dict[str, np.ndarray] | None = None
    camera_frame_reuse: dict[str, dict[str, int]] | None = None


class YamMcapEpisodeWriter:
    def __init__(
        self,
        episode_dir: Path,
        *,
        task: str,
        fps: float,
        camera_fps: float = 30.0,
        image_width: int = 640,
        image_height: int = 480,
        camera_config: dict[str, object] | None = None,
    ) -> None:
        self._episode_dir = episode_dir
        self._fps = fps
        self._camera_fps = camera_fps
        self._field_specs = _build_field_specs(width=image_width, height=image_height, fps=camera_fps)
        self._snapshot_index = _snapshot_index(_snapshot_fields())
        self._writer = pistream_mcap.ShardedMcapWriter(
            episode_dir,
            self._field_specs,
            metadata={
                "task": task,
                "fps": str(fps),
                "row_fps": str(fps),
                "camera_fps": str(camera_fps),
                "created_at": str(time.time()),
                "format": "openpi_yam_pistream_mcap_v2",
                "image_color_space": "rgb24",
                "camera_config": json.dumps(camera_config or {}, default=str, sort_keys=True),
            },
        )
        self._video_encoders = {
            camera_name: pistream_mcap.H264VideoEncoder[VideoUserdata](
                width=image_width,
                height=image_height,
                fps=camera_fps,
            )
            for camera_name in CAMERA_FIELD_MAP
        }
        self._latest_camera_refs: dict[str, FieldValueRef] = {}
        self._last_camera_sequence_index: dict[str, int] = {}
        self._last_timestamp_ns = 0
        self._num_steps = 0
        self._closed = False

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def episode_dir(self) -> Path:
        return self._episode_dir

    def write_step(
        self,
        *,
        camera_snapshot=None,
        frames_rgb: dict[str, np.ndarray] | None = None,
        camera_metadata: dict[str, dict] | None = None,
        state: np.ndarray,
        action: np.ndarray,
        timestamp_ns: int,
    ) -> None:
        if self._closed:
            raise ValueError("Episode writer is closed.")
        timestamp_ns = self._monotonic_timestamp(timestamp_ns)
        if camera_snapshot is not None:
            frames_rgb = camera_snapshot.frames
            camera_metadata = camera_snapshot.metadata
        if frames_rgb is None:
            raise ValueError("write_step requires camera_snapshot or frames_rgb")
        camera_metadata = camera_metadata or {}
        self._write_new_camera_frames(frames_rgb, camera_metadata, fallback_timestamp_ns=timestamp_ns)

        state_l, state_r = _split_bimanual(state)
        action_l, action_r = _split_bimanual(action)

        snapshot_refs: list[FieldValueRef] = [self._latest_camera_ref(camera_name) for camera_name in CAMERA_FIELD_MAP]

        snapshot_refs.extend(self._write_follower_fields(LEFT_FOLLOWER, state_l, action_l, timestamp_ns))
        snapshot_refs.extend(self._write_follower_fields(RIGHT_FOLLOWER, state_r, action_r, timestamp_ns))

        active_agent_sequence = self._writer.next_sequence(ACTIVE_AGENT_FIELD.publisher_id)
        self._writer.write(
            ACTIVE_AGENT_FIELD,
            wrappers_pb2.StringValue(value="teleop"),
            timestamp_ns=timestamp_ns,
            sequence_id=active_agent_sequence,
        )
        snapshot_refs.append(
            FieldValueRef(field=ACTIVE_AGENT_FIELD, sequence_id=active_agent_sequence, timestamp_ns=timestamp_ns)
        )

        teleop_sequence = self._writer.next_sequence(ACTION_START_FIELD.publisher_id)
        self._writer.write(
            ACTION_START_FIELD,
            pistream_mcap.timestamp_from_nanos(timestamp_ns),
            timestamp_ns=timestamp_ns,
            sequence_id=teleop_sequence,
        )
        snapshot_refs.append(
            FieldValueRef(field=ACTION_START_FIELD, sequence_id=teleop_sequence, timestamp_ns=timestamp_ns)
        )

        self._writer.write(
            ACTION_SNAPSHOT_FIELD,
            self._make_snapshot_ref(snapshot_refs),
            timestamp_ns=timestamp_ns,
            sequence_id=teleop_sequence,
        )
        self._num_steps += 1

    def close(self) -> None:
        if self._closed:
            return
        for encoder in self._video_encoders.values():
            for encoded_frame in encoder.finalize():
                self._write_video_frame(encoded_frame)
        self._writer.close()
        self._closed = True

    def _monotonic_timestamp(self, timestamp_ns: int) -> int:
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns <= self._last_timestamp_ns:
            timestamp_ns = self._last_timestamp_ns + MIN_TIMESTAMP_STEP_NS
        self._last_timestamp_ns = timestamp_ns
        return timestamp_ns

    def _write_new_camera_frames(
        self,
        frames_rgb: dict[str, np.ndarray],
        camera_metadata: dict[str, dict],
        *,
        fallback_timestamp_ns: int,
    ) -> None:
        for camera_name, field in CAMERA_FIELD_MAP.items():
            if camera_name not in frames_rgb:
                raise KeyError(f"Missing camera frame {camera_name!r}")
            metadata = camera_metadata.get(camera_name, {})
            sequence_index = int(
                metadata.get("sequence_index", self._last_camera_sequence_index.get(camera_name, -1) + 1)
            )
            if self._last_camera_sequence_index.get(camera_name) == sequence_index:
                continue

            capture_timestamp_ns = int(
                metadata.get(
                    "capture_timestamp_ns",
                    int(float(metadata.get("capture_time_s", fallback_timestamp_ns / 1_000_000_000.0)) * 1e9),
                )
            )
            sequence_id = self._writer.next_sequence(field.publisher_id)
            encoded_frames = self._video_encoders[camera_name].encode_rgb(
                frames_rgb[camera_name],
                userdata=VideoUserdata(
                    camera_name=camera_name,
                    field=field,
                    sequence_id=sequence_id,
                    timestamp_ns=capture_timestamp_ns,
                ),
            )
            for encoded_frame in encoded_frames:
                self._write_video_frame(encoded_frame)
            self._last_camera_sequence_index[camera_name] = sequence_index

    def _latest_camera_ref(self, camera_name: str) -> FieldValueRef:
        if camera_name not in self._latest_camera_refs:
            raise RuntimeError(
                f"No encoded camera frame is available for {camera_name!r}. "
                "The H264 encoder did not emit a frame for the first camera sample."
            )
        return self._latest_camera_refs[camera_name]

    def _write_follower_fields(
        self,
        publisher_id: str,
        state_7d: np.ndarray,
        action_7d: np.ndarray,
        timestamp_ns: int,
    ) -> list[FieldValueRef]:
        sequence_id = self._writer.next_sequence(publisher_id)
        fields = [
            (_observation_joints_field(publisher_id), state_7d[:6]),
            (_observation_gripper_field(publisher_id), state_7d[6:7]),
            (_action_joints_field(publisher_id), action_7d[:6]),
            (_action_gripper_field(publisher_id), action_7d[6:7]),
        ]
        refs: list[FieldValueRef] = []
        for field, value in fields:
            self._writer.write(
                field,
                pistream_mcap.double_list(value),
                timestamp_ns=timestamp_ns,
                sequence_id=sequence_id,
            )
            refs.append(FieldValueRef(field=field, sequence_id=sequence_id, timestamp_ns=timestamp_ns))
        return refs

    def _write_video_frame(self, encoded_frame: pistream_mcap.EncodedVideoFrame[VideoUserdata]) -> None:
        if encoded_frame.userdata is None:
            raise RuntimeError("Encoded video frame is missing userdata.")
        self._writer.write(
            encoded_frame.userdata.field,
            encoded_frame.to_proto(),
            timestamp_ns=encoded_frame.userdata.timestamp_ns,
            sequence_id=encoded_frame.userdata.sequence_id,
        )
        self._latest_camera_refs[encoded_frame.userdata.camera_name] = FieldValueRef(
            field=encoded_frame.userdata.field,
            sequence_id=encoded_frame.userdata.sequence_id,
            timestamp_ns=encoded_frame.userdata.timestamp_ns,
        )

    def _make_snapshot_ref(self, refs: Iterable[FieldValueRef]) -> frame_pb2.PiStreamSnapshotRef:
        snapshot = frame_pb2.PiStreamSnapshotRef()
        for ref in refs:
            snapshot.fields.add(
                sequence_id=ref.sequence_id,
                capture_timestamp=pistream_mcap.timestamp_from_nanos(ref.timestamp_ns),
                index=self._snapshot_index[ref.field],
            )
        return snapshot


def read_episode(episode_dir: Path, *, decode_images: bool = True) -> YamMcapEpisode:
    _validate_expected_shards(episode_dir)
    metadata = pistream_mcap.read_mcap_metadata(episode_dir)
    arrays_by_sequence: dict[pistream_mcap.FieldKey, dict[int, np.ndarray]] = {
        field: {} for field in _required_state_action_fields()
    }
    camera_sequences: dict[str, set[int]] = {camera_name: set() for camera_name in CAMERA_FIELD_MAP}
    images_by_sequence: dict[str, dict[int, np.ndarray]] = {camera_name: {} for camera_name in CAMERA_FIELD_MAP}
    video_pending_sequences: dict[str, deque[tuple[int, int]]] = {
        camera_name: deque() for camera_name in CAMERA_FIELD_MAP
    }
    snapshots: list[tuple[int, dict[pistream_mcap.FieldKey, FieldValueRef]]] = []
    snapshot_field_by_index = {index: field for field, index in _snapshot_index(_snapshot_fields()).items()}
    decoders = (
        {camera_name: pistream_mcap.H264VideoDecoder() for camera_name in CAMERA_FIELD_MAP} if decode_images else {}
    )

    for _, channel, message in pistream_mcap.iter_mcap_messages(episode_dir):
        if channel.topic == pistream_mcap.PI_STREAM_DESCRIPTOR_TOPIC:
            continue
        if not channel.topic.startswith(pistream_mcap.PI_STREAM_FRAME_PREFIX):
            continue
        field = pistream_mcap.parse_frame_topic(channel.topic)
        if field in arrays_by_sequence:
            decoded = frame_pb2.DoubleList()
            decoded.ParseFromString(message.data)
            arrays_by_sequence[field][message.sequence] = pistream_mcap.parse_double_list(decoded, _field_shape(field))
        elif field == ACTION_SNAPSHOT_FIELD:
            snapshot = frame_pb2.PiStreamSnapshotRef()
            snapshot.ParseFromString(message.data)
            refs: dict[pistream_mcap.FieldKey, FieldValueRef] = {}
            for ref in snapshot.fields:
                ref_field = snapshot_field_by_index.get(
                    ref.index,
                    pistream_mcap.FieldKey(ref.publisher_id, ref.key) if ref.publisher_id and ref.key else None,
                )
                if ref_field is None:
                    continue
                refs[ref_field] = FieldValueRef(
                    field=ref_field,
                    sequence_id=ref.sequence_id,
                    timestamp_ns=pistream_mcap.timestamp_to_nanos(ref.capture_timestamp),
                )
            snapshots.append((message.log_time, refs))
        elif field in set(CAMERA_FIELD_MAP.values()):
            video_frame = frame_pb2.VideoFrame()
            video_frame.ParseFromString(message.data)
            camera_name = _camera_name_for_field(field)
            camera_sequences[camera_name].add(message.sequence)
            if decode_images:
                video_pending_sequences[camera_name].append((message.sequence, message.log_time))
                try:
                    decoded_images = decoders[camera_name].decode_to_rgb(video_frame)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to decode {camera_name} video sequence {message.sequence} in {episode_dir}"
                    ) from exc
                for image in decoded_images:
                    sequence_id, _timestamp_ns = video_pending_sequences[camera_name].popleft()
                    _validate_rgb_image(image, episode_dir, camera_name, sequence_id)
                    images_by_sequence[camera_name][sequence_id] = image
    if decode_images:
        _finalize_video_decoders(
            decoders,
            episode_dir=episode_dir,
            pending_sequences=video_pending_sequences,
            images=images_by_sequence,
        )

    snapshots = _validate_snapshots(
        episode_dir,
        snapshots=snapshots,
        arrays=arrays_by_sequence,
        camera_sequences=camera_sequences,
    )
    state = []
    action = []
    images_by_row: dict[str, list[np.ndarray]] = {camera_name: [] for camera_name in CAMERA_FIELD_MAP}
    camera_row_timestamps: dict[str, list[int]] = {camera_name: [] for camera_name in CAMERA_FIELD_MAP}
    for _timestamp_ns, refs in snapshots:
        state.append(
            np.concatenate(
                [
                    _array_for_ref(arrays_by_sequence, refs, _observation_joints_field(LEFT_FOLLOWER)),
                    _array_for_ref(arrays_by_sequence, refs, _observation_gripper_field(LEFT_FOLLOWER)),
                    _array_for_ref(arrays_by_sequence, refs, _observation_joints_field(RIGHT_FOLLOWER)),
                    _array_for_ref(arrays_by_sequence, refs, _observation_gripper_field(RIGHT_FOLLOWER)),
                ]
            ).astype(np.float32)
        )
        action.append(
            np.concatenate(
                [
                    _array_for_ref(arrays_by_sequence, refs, _action_joints_field(LEFT_FOLLOWER)),
                    _array_for_ref(arrays_by_sequence, refs, _action_gripper_field(LEFT_FOLLOWER)),
                    _array_for_ref(arrays_by_sequence, refs, _action_joints_field(RIGHT_FOLLOWER)),
                    _array_for_ref(arrays_by_sequence, refs, _action_gripper_field(RIGHT_FOLLOWER)),
                ]
            ).astype(np.float32)
        )
        for camera_name, field in CAMERA_FIELD_MAP.items():
            ref = _ref_for_field(refs, field)
            camera_row_timestamps[camera_name].append(ref.timestamp_ns)
            if decode_images:
                try:
                    images_by_row[camera_name].append(images_by_sequence[camera_name][ref.sequence_id])
                except KeyError as exc:
                    raise RuntimeError(
                        f"MCAP episode {episode_dir} is missing decoded image for "
                        f"{camera_name} sequence {ref.sequence_id}"
                    ) from exc

    image_arrays = None
    if decode_images:
        image_arrays = {camera_name: np.stack(images, axis=0) for camera_name, images in images_by_row.items()}
    timestamps_ns = np.asarray([timestamp_ns for timestamp_ns, _refs in snapshots], dtype=np.int64)
    camera_timestamps_ns = {
        camera_name: np.asarray(_unique_preserve_order(values), dtype=np.int64)
        for camera_name, values in camera_row_timestamps.items()
    }
    state_array = np.asarray(state, dtype=np.float32)
    action_array = np.asarray(action, dtype=np.float32)
    _validate_episode_arrays(episode_dir, state_array, action_array)
    return YamMcapEpisode(
        task=metadata.get("task", ""),
        fps=_fps_from_metadata_or_timestamps(metadata, timestamps_ns),
        timestamps_ns=timestamps_ns,
        state=state_array,
        action=action_array,
        images=image_arrays,
        camera_timestamps_ns=camera_timestamps_ns,
        camera_row_timestamps_ns={
            camera_name: np.asarray(values, dtype=np.int64) for camera_name, values in camera_row_timestamps.items()
        },
        camera_frame_reuse={
            camera_name: {
                "rows": len(values),
                "unique_frames": len(set(values)),
                "reused_rows": len(values) - len(set(values)),
                "missing_rows": 0,
            }
            for camera_name, values in camera_row_timestamps.items()
        },
    )


def is_mcap_episode(episode_dir: Path) -> bool:
    return episode_dir.is_dir() and all(
        _expected_shard_path(episode_dir, index).is_file() for index in range(EXPECTED_SHARD_COUNT)
    )


def iter_episode_dirs(raw_dir: Path) -> list[Path]:
    episode_dirs = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_dir() or not any(path.glob("episode_part*.mcap")):
            continue
        _validate_expected_shards(path)
        read_episode(path, decode_images=False)
        episode_dirs.append(path)
    return episode_dirs


def _build_field_specs(*, width: int, height: int, fps: float) -> list[pistream_mcap.FieldSpec]:
    specs: list[pistream_mcap.FieldSpec] = []
    for shard, field in enumerate(CAMERA_FIELD_MAP.values(), start=1):
        specs.append(
            pistream_mcap.FieldSpec(
                publisher_id=field.publisher_id,
                key=field.key,
                encoding=pistream_mcap.compressed_video_encoding(width=width, height=height, fps=fps),
                message_type=frame_pb2.VideoFrame,
                shard=shard,
            )
        )
    for publisher_id in (LEFT_FOLLOWER, RIGHT_FOLLOWER):
        specs.extend(
            _ndarray_field_spec(field, shape=(6,))
            for field in (
                _observation_joints_field(publisher_id),
                _action_joints_field(publisher_id),
            )
        )
        specs.extend(
            _ndarray_field_spec(field, shape=(1,))
            for field in (
                _observation_gripper_field(publisher_id),
                _action_gripper_field(publisher_id),
            )
        )
    specs.extend(
        [
            pistream_mcap.FieldSpec(
                publisher_id=ACTIVE_AGENT_FIELD.publisher_id,
                key=ACTIVE_AGENT_FIELD.key,
                encoding=pistream_mcap.string_encoding(),
                message_type=wrappers_pb2.StringValue,
            ),
            pistream_mcap.FieldSpec(
                publisher_id=ACTION_START_FIELD.publisher_id,
                key=ACTION_START_FIELD.key,
                encoding=pistream_mcap.timestamp_encoding(),
                message_type=timestamp_pb2.Timestamp,
            ),
            pistream_mcap.FieldSpec(
                publisher_id=ACTION_SNAPSHOT_FIELD.publisher_id,
                key=ACTION_SNAPSHOT_FIELD.key,
                encoding=pistream_mcap.snapshot_ref_encoding(_snapshot_fields()),
                message_type=frame_pb2.PiStreamSnapshotRef,
            ),
        ]
    )
    return specs


def _ndarray_field_spec(field: pistream_mcap.FieldKey, *, shape: tuple[int, ...]) -> pistream_mcap.FieldSpec:
    return pistream_mcap.FieldSpec(
        publisher_id=field.publisher_id,
        key=field.key,
        encoding=pistream_mcap.ndarray_encoding(shape, "float64"),
        message_type=frame_pb2.DoubleList,
    )


def _snapshot_fields() -> list[pistream_mcap.FieldKey]:
    return [
        *CAMERA_FIELD_MAP.values(),
        *_required_state_action_fields(),
        ACTIVE_AGENT_FIELD,
        ACTION_START_FIELD,
    ]


def _snapshot_index(fields: Iterable[pistream_mcap.FieldKey]) -> dict[pistream_mcap.FieldKey, int]:
    return {
        field: index
        for index, field in enumerate(sorted(set(fields), key=lambda field: (field.key, field.publisher_id)), start=1)
    }


def _required_state_action_fields() -> list[pistream_mcap.FieldKey]:
    return [
        _observation_joints_field(LEFT_FOLLOWER),
        _observation_gripper_field(LEFT_FOLLOWER),
        _observation_joints_field(RIGHT_FOLLOWER),
        _observation_gripper_field(RIGHT_FOLLOWER),
        _action_joints_field(LEFT_FOLLOWER),
        _action_gripper_field(LEFT_FOLLOWER),
        _action_joints_field(RIGHT_FOLLOWER),
        _action_gripper_field(RIGHT_FOLLOWER),
    ]


def _observation_joints_field(publisher_id: str) -> pistream_mcap.FieldKey:
    return pistream_mcap.FieldKey(publisher_id, f"observation/{publisher_id}/joints/position")


def _observation_gripper_field(publisher_id: str) -> pistream_mcap.FieldKey:
    return pistream_mcap.FieldKey(publisher_id, f"observation/{publisher_id}/gripper/position")


def _action_joints_field(publisher_id: str) -> pistream_mcap.FieldKey:
    return pistream_mcap.FieldKey(publisher_id, f"action/{publisher_id}/joints/position")


def _action_gripper_field(publisher_id: str) -> pistream_mcap.FieldKey:
    return pistream_mcap.FieldKey(publisher_id, f"action/{publisher_id}/gripper/position")


def _split_bimanual(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (14,):
        raise ValueError(f"Expected 14D bimanual value, got {value.shape}")
    return value[:7], value[7:]


def _field_shape(field: pistream_mcap.FieldKey) -> tuple[int, ...]:
    return (1,) if field.key.endswith("/gripper/position") else (6,)


def _camera_name_for_field(field: pistream_mcap.FieldKey) -> str:
    for camera_name, camera_field in CAMERA_FIELD_MAP.items():
        if field == camera_field:
            return camera_name
    raise KeyError(field)


def _fps_from_metadata_or_timestamps(metadata: dict[str, str], timestamps_ns: np.ndarray) -> float:
    if metadata.get("row_fps"):
        return float(metadata["row_fps"])
    if metadata.get("fps"):
        return float(metadata["fps"])
    if len(timestamps_ns) < 2:
        return 0.0
    diffs = np.diff(timestamps_ns.astype(np.float64)) / 1_000_000_000.0
    return float(round(1.0 / np.median(diffs)))


def _expected_shard_path(episode_dir: Path, index: int) -> Path:
    return episode_dir / f"episode_part{index}.mcap"


def _validate_expected_shards(episode_dir: Path) -> None:
    missing = [
        _expected_shard_path(episode_dir, index).name
        for index in range(EXPECTED_SHARD_COUNT)
        if not _expected_shard_path(episode_dir, index).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"MCAP episode {episode_dir} is missing shard(s): {missing}")


def _validate_snapshots(
    episode_dir: Path,
    *,
    snapshots: list[tuple[int, dict[pistream_mcap.FieldKey, FieldValueRef]]],
    arrays: dict[pistream_mcap.FieldKey, dict[int, np.ndarray]],
    camera_sequences: dict[str, set[int]],
) -> list[tuple[int, dict[pistream_mcap.FieldKey, FieldValueRef]]]:
    if not snapshots:
        raise RuntimeError(f"No action_snapshot rows found in MCAP episode {episode_dir}")
    snapshots = sorted(snapshots, key=lambda item: item[0])
    required_fields = [*CAMERA_FIELD_MAP.values(), *_required_state_action_fields()]
    for row_index, (_timestamp_ns, refs) in enumerate(snapshots):
        missing_refs = [field for field in required_fields if field not in refs]
        if missing_refs:
            names = [f"{field.publisher_id}/{field.key}" for field in missing_refs]
            raise RuntimeError(f"MCAP episode {episode_dir} row {row_index} is missing snapshot refs: {names}")
        for camera_name, field in CAMERA_FIELD_MAP.items():
            ref = refs[field]
            if ref.sequence_id not in camera_sequences[camera_name]:
                raise RuntimeError(
                    f"MCAP episode {episode_dir} row {row_index} references missing "
                    f"{camera_name} video sequence {ref.sequence_id}"
                )
        for field in _required_state_action_fields():
            ref = refs[field]
            if ref.sequence_id not in arrays[field]:
                raise RuntimeError(
                    f"MCAP episode {episode_dir} row {row_index} references missing "
                    f"{field.publisher_id}/{field.key} sequence {ref.sequence_id}"
                )
    return snapshots


def _validate_episode_arrays(episode_dir: Path, state: np.ndarray, action: np.ndarray) -> None:
    if state.ndim != 2 or action.ndim != 2 or state.shape != action.shape or state.shape[-1] != 14:
        raise RuntimeError(f"MCAP episode {episode_dir} has bad state/action shapes: {state.shape}, {action.shape}")
    if not np.all(np.isfinite(state)):
        raise RuntimeError(f"MCAP episode {episode_dir} state contains non-finite values")
    if not np.all(np.isfinite(action)):
        raise RuntimeError(f"MCAP episode {episode_dir} action contains non-finite values")


def _validate_rgb_image(image: np.ndarray, episode_dir: Path, camera_name: str, sequence_id: int) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise RuntimeError(
            f"MCAP episode {episode_dir} decoded bad RGB image for {camera_name} sequence {sequence_id}: "
            f"shape={image.shape}, dtype={image.dtype}"
        )


def _array_for_ref(
    arrays: dict[pistream_mcap.FieldKey, dict[int, np.ndarray]],
    refs: dict[pistream_mcap.FieldKey, FieldValueRef],
    field: pistream_mcap.FieldKey,
) -> np.ndarray:
    ref = _ref_for_field(refs, field)
    return arrays[field][ref.sequence_id]


def _ref_for_field(
    refs: dict[pistream_mcap.FieldKey, FieldValueRef],
    field: pistream_mcap.FieldKey,
) -> FieldValueRef:
    try:
        return refs[field]
    except KeyError as exc:
        raise RuntimeError(f"Snapshot is missing field {field.publisher_id}/{field.key}") from exc


def _unique_preserve_order(values: Iterable[int]) -> list[int]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _finalize_video_decoders(
    decoders: dict[str, pistream_mcap.H264VideoDecoder],
    *,
    episode_dir: Path,
    pending_sequences: dict[str, deque[tuple[int, int]]],
    images: dict[str, dict[int, np.ndarray]],
) -> None:
    for camera_name, decoder in decoders.items():
        try:
            frames = decoder.finalize()
        except Exception as exc:
            raise RuntimeError(f"Failed to finalize {camera_name} video decoder") from exc
        for frame in frames:
            if not pending_sequences[camera_name]:
                break
            sequence_id, _timestamp_ns = pending_sequences[camera_name].popleft()
            image = frame.to_ndarray(format="rgb24")
            _validate_rgb_image(image, episode_dir, camera_name, sequence_id)
            images[camera_name][sequence_id] = image
        if pending_sequences[camera_name]:
            missing = [sequence_id for sequence_id, _timestamp_ns in pending_sequences[camera_name]]
            raise RuntimeError(f"Video decoder did not produce images for {camera_name} sequence(s): {missing}")
