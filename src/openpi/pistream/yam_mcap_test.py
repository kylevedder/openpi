from __future__ import annotations

from pathlib import Path

from mcap.reader import make_reader
from mcap_protobuf.writer import Writer
import numpy as np
import pytest

from examples.yam_real import mcap_episode
from openpi.pistream import mcap as pistream_mcap
from openpi.pistream.proto import descriptor_pb2


def test_yam_mcap_round_trip(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    writer = mcap_episode.YamMcapEpisodeWriter(episode_dir, task="round trip task", fps=20.0)
    for step in range(3):
        writer.write_step(
            frames_bgr=_frames(step),
            state=np.arange(14, dtype=np.float32) + step,
            action=np.arange(14, dtype=np.float32) + 10 + step,
            timestamp_ns=1_700_000_000_000_000_000 + step * 50_000_000,
        )
    writer.close()

    episode = mcap_episode.read_episode(episode_dir, decode_images=True)

    assert sorted(path.name for path in episode_dir.iterdir()) == [
        "episode_part0.mcap",
        "episode_part1.mcap",
        "episode_part2.mcap",
        "episode_part3.mcap",
    ]
    assert episode.task == "round trip task"
    assert episode.fps == 20.0
    assert episode.state.shape == (3, 14)
    assert episode.action.shape == (3, 14)
    np.testing.assert_array_equal(episode.state[0], np.arange(14, dtype=np.float32))
    np.testing.assert_array_equal(episode.action[0], np.arange(14, dtype=np.float32) + 10)
    assert episode.images is not None
    assert {camera: images.shape for camera, images in episode.images.items()} == {
        "cam_high": (3, 480, 640, 3),
        "cam_left_wrist": (3, 480, 640, 3),
        "cam_right_wrist": (3, 480, 640, 3),
    }


def test_yam_mcap_descriptors_are_arx_subset(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    writer = mcap_episode.YamMcapEpisodeWriter(episode_dir, task="descriptor task", fps=20.0)
    writer.write_step(
        frames_bgr=_frames(0),
        state=np.arange(14, dtype=np.float32),
        action=np.arange(14, dtype=np.float32) + 10,
        timestamp_ns=1_700_000_000_000_000_000,
    )
    writer.close()

    metadata = pistream_mcap.read_mcap_metadata(episode_dir)
    assert metadata["task"] == "descriptor task"
    assert metadata["fps"] == "20.0"

    part0_descriptor = _read_descriptor(episode_dir / "episode_part0.mcap")
    part0_fields = {(field.publisher_id, field.key) for field in part0_descriptor.fields}
    assert ("arx_follow_1", "observation/arx_follow_1/joints/position") in part0_fields
    assert ("arx_follow_2", "action/arx_follow_2/gripper/position") in part0_fields
    assert ("system", "system/active_agent/node_id") in part0_fields
    assert ("teleop", "system/active_agent/action_snapshot") in part0_fields
    assert not any("end_effector" in key for _, key in part0_fields)
    assert not any("base_1_camera" in publisher_id or "base_1_camera" in key for publisher_id, key in part0_fields)

    snapshot_field = next(
        field for field in part0_descriptor.fields if field.key == "system/active_agent/action_snapshot"
    )
    indexed_fields = {
        (entry.publisher_id, entry.key): entry.index for entry in snapshot_field.encoding.snapshot_ref.field_index_map
    }
    assert indexed_fields[("base_0_camera", "observation/base_0_camera/rgb/image")] > 0
    assert indexed_fields[("teleop", "system/active_agent/action_start_timestamp")] > 0
    assert len(set(indexed_fields.values())) == len(indexed_fields)

    camera_descriptors = [_read_descriptor(episode_dir / f"episode_part{index}.mcap") for index in range(1, 4)]
    camera_fields = [
        {(field.publisher_id, field.key) for field in descriptor.fields} for descriptor in camera_descriptors
    ]
    assert camera_fields == [
        {("base_0_camera", "observation/base_0_camera/rgb/image")},
        {("left_wrist_0_camera", "observation/left_wrist_0_camera/rgb/image")},
        {("right_wrist_0_camera", "observation/right_wrist_0_camera/rgb/image")},
    ]


def test_yam_mcap_reader_rejects_empty_episode(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    writer = mcap_episode.YamMcapEpisodeWriter(episode_dir, task="empty task", fps=20.0)
    writer.close()

    with pytest.raises(RuntimeError, match="No action_start_timestamp steps"):
        mcap_episode.read_episode(episode_dir, decode_images=False)


def test_yam_mcap_episode_discovery_rejects_partial_shards(tmp_path: Path) -> None:
    episode_dir = tmp_path / "partial"
    episode_dir.mkdir()
    (episode_dir / "episode_part0.mcap").write_bytes(b"partial")

    assert not mcap_episode.is_mcap_episode(episode_dir)
    with pytest.raises(FileNotFoundError, match="missing shard"):
        mcap_episode.iter_episode_dirs(tmp_path)


def test_yam_mcap_episode_discovery_rejects_empty_camera_shards(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    writer = mcap_episode.YamMcapEpisodeWriter(episode_dir, task="missing cameras", fps=20.0)
    writer.write_step(
        frames_bgr=_frames(0),
        state=np.arange(14, dtype=np.float32),
        action=np.arange(14, dtype=np.float32) + 10,
        timestamp_ns=1_700_000_000_000_000_000,
    )
    writer.close()
    for shard_index in range(1, 4):
        _write_empty_descriptor_shard(episode_dir / f"episode_part{shard_index}.mcap")

    with pytest.raises(RuntimeError, match="video messages"):
        mcap_episode.read_episode(episode_dir, decode_images=False)
    with pytest.raises(RuntimeError, match="video messages"):
        mcap_episode.iter_episode_dirs(tmp_path)


def _frames(step: int) -> dict[str, np.ndarray]:
    return {
        "cam_high": np.full((480, 640, 3), step * 20, dtype=np.uint8),
        "cam_left_wrist": np.full((480, 640, 3), 40 + step * 20, dtype=np.uint8),
        "cam_right_wrist": np.full((480, 640, 3), 80 + step * 20, dtype=np.uint8),
    }


def _read_descriptor(path: Path) -> descriptor_pb2.PiStreamDescriptor:
    with path.open("rb") as file:
        reader = make_reader(file)
        for schema, channel, message in reader.iter_messages(log_time_order=False):
            if channel.topic == pistream_mcap.PI_STREAM_DESCRIPTOR_TOPIC:
                assert schema is not None
                assert schema.name == "monopi.pi_stream.PiStreamDescriptor"
                descriptor = descriptor_pb2.PiStreamDescriptor()
                descriptor.ParseFromString(message.data)
                return descriptor
    raise AssertionError(f"No stream descriptor found in {path}")


def _write_empty_descriptor_shard(path: Path) -> None:
    with path.open("wb") as file:
        writer = Writer(file)
        writer.write_message(
            pistream_mcap.PI_STREAM_DESCRIPTOR_TOPIC,
            descriptor_pb2.PiStreamDescriptor(),
            log_time=0,
            publish_time=0,
            sequence=0,
        )
        writer.finish()
