from __future__ import annotations

from collections.abc import Iterable
import dataclasses
from pathlib import Path
import time

import cv2
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


class YamMcapEpisodeWriter:
    def __init__(
        self,
        episode_dir: Path,
        *,
        task: str,
        fps: float,
        image_width: int = 640,
        image_height: int = 480,
    ) -> None:
        self._episode_dir = episode_dir
        self._fps = fps
        self._field_specs = _build_field_specs(width=image_width, height=image_height, fps=fps)
        self._snapshot_index = _snapshot_index(_snapshot_fields())
        self._writer = pistream_mcap.ShardedMcapWriter(
            episode_dir,
            self._field_specs,
            metadata={
                "task": task,
                "fps": str(fps),
                "created_at": str(time.time()),
                "format": "openpi_yam_pistream_mcap",
            },
        )
        self._video_encoders = {
            camera_name: pistream_mcap.H264VideoEncoder[VideoUserdata](
                width=image_width,
                height=image_height,
                fps=fps,
            )
            for camera_name in CAMERA_FIELD_MAP
        }
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
        frames_bgr: dict[str, np.ndarray],
        state: np.ndarray,
        action: np.ndarray,
        timestamp_ns: int,
    ) -> None:
        if self._closed:
            raise ValueError("Episode writer is closed.")
        timestamp_ns = self._monotonic_timestamp(timestamp_ns)
        state_l, state_r = _split_bimanual(state)
        action_l, action_r = _split_bimanual(action)

        snapshot_refs: list[FieldValueRef] = []
        for camera_name, field in CAMERA_FIELD_MAP.items():
            if camera_name not in frames_bgr:
                raise KeyError(f"Missing camera frame {camera_name!r}")
            sequence_id = self._writer.next_sequence(field.publisher_id)
            snapshot_refs.append(FieldValueRef(field=field, sequence_id=sequence_id, timestamp_ns=timestamp_ns))
            rgb = cv2.cvtColor(frames_bgr[camera_name], cv2.COLOR_BGR2RGB)
            encoded_frames = self._video_encoders[camera_name].encode_rgb(
                rgb,
                userdata=VideoUserdata(field=field, sequence_id=sequence_id, timestamp_ns=timestamp_ns),
            )
            for encoded_frame in encoded_frames:
                self._write_video_frame(encoded_frame)

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
    arrays: dict[pistream_mcap.FieldKey, dict[int, np.ndarray]] = {
        field: {} for field in _required_state_action_fields()
    }
    camera_timestamps: dict[str, set[int]] = {camera_name: set() for camera_name in CAMERA_FIELD_MAP}
    images: dict[str, dict[int, np.ndarray]] = {camera_name: {} for camera_name in CAMERA_FIELD_MAP}
    step_timestamps: set[int] = set()
    decoders = (
        {camera_name: pistream_mcap.H264VideoDecoder() for camera_name in CAMERA_FIELD_MAP} if decode_images else {}
    )

    for _, channel, message in pistream_mcap.iter_mcap_messages(episode_dir):
        if channel.topic == pistream_mcap.PI_STREAM_DESCRIPTOR_TOPIC:
            continue
        if not channel.topic.startswith(pistream_mcap.PI_STREAM_FRAME_PREFIX):
            continue
        field = pistream_mcap.parse_frame_topic(channel.topic)
        if field in arrays:
            decoded = frame_pb2.DoubleList()
            decoded.ParseFromString(message.data)
            arrays[field][message.log_time] = pistream_mcap.parse_double_list(decoded, _field_shape(field))
        elif field == ACTION_START_FIELD:
            step_timestamps.add(message.log_time)
        elif field in set(CAMERA_FIELD_MAP.values()):
            video_frame = frame_pb2.VideoFrame()
            video_frame.ParseFromString(message.data)
            camera_name = _camera_name_for_field(field)
            camera_timestamps[camera_name].add(message.log_time)
            if decode_images:
                for image in decoders[camera_name].decode_to_rgb(video_frame):
                    images[camera_name][message.log_time] = image

    timestamps = _validate_required_timestamps(
        episode_dir,
        step_timestamps=step_timestamps,
        arrays=arrays,
        camera_timestamps=camera_timestamps,
        images=images if decode_images else None,
    )
    if decode_images:
        _finalize_video_decoders(decoders)
    state = []
    action = []
    for timestamp_ns in timestamps:
        state.append(
            np.concatenate(
                [
                    arrays[_observation_joints_field(LEFT_FOLLOWER)][timestamp_ns],
                    arrays[_observation_gripper_field(LEFT_FOLLOWER)][timestamp_ns],
                    arrays[_observation_joints_field(RIGHT_FOLLOWER)][timestamp_ns],
                    arrays[_observation_gripper_field(RIGHT_FOLLOWER)][timestamp_ns],
                ]
            ).astype(np.float32)
        )
        action.append(
            np.concatenate(
                [
                    arrays[_action_joints_field(LEFT_FOLLOWER)][timestamp_ns],
                    arrays[_action_gripper_field(LEFT_FOLLOWER)][timestamp_ns],
                    arrays[_action_joints_field(RIGHT_FOLLOWER)][timestamp_ns],
                    arrays[_action_gripper_field(RIGHT_FOLLOWER)][timestamp_ns],
                ]
            ).astype(np.float32)
        )

    image_arrays = None
    if decode_images:
        image_arrays = {
            camera_name: np.stack([camera_images[timestamp_ns] for timestamp_ns in timestamps], axis=0)
            for camera_name, camera_images in images.items()
        }
    timestamps_ns = np.asarray(timestamps, dtype=np.int64)
    return YamMcapEpisode(
        task=metadata.get("task", ""),
        fps=_fps_from_metadata_or_timestamps(metadata, timestamps_ns),
        timestamps_ns=timestamps_ns,
        state=np.asarray(state, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        images=image_arrays,
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


def _validate_required_timestamps(
    episode_dir: Path,
    *,
    step_timestamps: set[int],
    arrays: dict[pistream_mcap.FieldKey, dict[int, np.ndarray]],
    camera_timestamps: dict[str, set[int]],
    images: dict[str, dict[int, np.ndarray]] | None,
) -> list[int]:
    if not step_timestamps:
        raise RuntimeError(f"No action_start_timestamp steps found in MCAP episode {episode_dir}")

    expected = set(step_timestamps)
    for field, values in arrays.items():
        _validate_timestamp_set(episode_dir, f"{field.publisher_id}/{field.key}", expected, set(values))
    for camera_name, timestamps in camera_timestamps.items():
        _validate_timestamp_set(episode_dir, f"{camera_name} video messages", expected, timestamps)
    if images is not None:
        for camera_name, values in images.items():
            _validate_timestamp_set(episode_dir, f"{camera_name} decoded images", expected, set(values))
    return sorted(expected)


def _validate_timestamp_set(episode_dir: Path, name: str, expected: set[int], actual: set[int]) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"MCAP episode {episode_dir} has inconsistent timestamps for {name}: "
            f"missing={len(missing)}, extra={len(extra)}"
        )


def _finalize_video_decoders(decoders: dict[str, pistream_mcap.H264VideoDecoder]) -> None:
    for decoder in decoders.values():
        decoder.finalize()
