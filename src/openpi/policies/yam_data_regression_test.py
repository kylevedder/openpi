import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from examples.yam_real import convert_yam_data_to_lerobot
from examples.yam_real import data_regression
from examples.yam_real import mcap_episode
from examples.yam_real import read_yam_encoders
from openpi import transforms
from openpi.policies import yam_policy
from openpi.shared import jpeg_transport
from openpi.shared import normalize
from openpi.training import config as training_config


def test_yam_gripper_conversion_is_exactly_one_flip() -> None:
    i2rt = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.2], dtype=np.float32)

    openpi = yam_policy.i2rt_arm_state_to_openpi(i2rt)

    np.testing.assert_allclose(openpi[:6], i2rt[:6])
    assert openpi[6] == np.float32(0.8)
    np.testing.assert_allclose(yam_policy.openpi_arm_state_to_i2rt(openpi), i2rt)


def test_yam_delta_action_mask_keeps_grippers_absolute() -> None:
    state = np.arange(14, dtype=np.float32) / 10.0
    actions = np.stack([state + 1.0, state + 2.0], axis=0).astype(np.float32)
    original_grippers = actions[:, data_regression.GRIPPER_DIMS].copy()

    mask = transforms.make_bool_mask(6, -1, 6, -1)
    transformed = transforms.DeltaActions(mask)({"state": state.copy(), "actions": actions.copy()})

    np.testing.assert_allclose(transformed["actions"][:, :6], actions[:, :6] - state[:6])
    np.testing.assert_allclose(transformed["actions"][:, 7:13], actions[:, 7:13] - state[7:13])
    np.testing.assert_allclose(transformed["actions"][:, data_regression.GRIPPER_DIMS], original_grippers)


def test_synthetic_npz_canonical_round_trip(tmp_path: Path) -> None:
    expected = data_regression.make_synthetic_episode()
    episode_dir = data_regression.write_npz_fixture(expected, tmp_path / "npz_episode")

    actual = data_regression.load_npz_episode(episode_dir)
    diff = data_regression.compare_canonical_episodes(expected, actual)
    summary = data_regression.summarize_episode(actual)

    assert diff == {"max_state_abs_error": 0.0, "max_action_abs_error": 0.0}
    assert summary["warnings"] == []
    assert summary["gripper"]["action"]["left_gripper"]["close_count"] > 0
    assert summary["gripper"]["action"]["right_gripper"]["open_count"] > 0
    assert summary["arm_joints"]["action"]["abs_max"] > 0
    assert "left_waist" in summary["action_stats_by_name"]
    assert summary["timing"]["median_dt_s"] == 0.02


def test_synthetic_mcap_canonical_round_trip(tmp_path: Path) -> None:
    expected = data_regression.make_synthetic_episode()
    episode_dir = data_regression.write_mcap_fixture(expected, tmp_path / "mcap_episode")

    actual = data_regression.load_mcap_episode(episode_dir)
    diff = data_regression.compare_canonical_episodes(expected, actual)
    summary = data_regression.summarize_episode(actual)

    assert diff == {"max_state_abs_error": 0.0, "max_action_abs_error": 0.0}
    assert summary["warnings"] == []


def test_mcap_rows_can_reuse_latest_30hz_camera_frame(tmp_path: Path) -> None:
    episode = data_regression.make_synthetic_episode(num_frames=4)
    episode_dir = tmp_path / "mcap_reuse"
    writer = mcap_episode.YamMcapEpisodeWriter(episode_dir, task=episode.task, fps=50.0, camera_fps=30.0)
    try:
        for row_idx in range(episode.num_frames):
            camera_sequence = 0 if row_idx < 2 else row_idx - 1
            camera_timestamp_ns = int((camera_sequence / 30.0) * 1_000_000_000)
            frames = {
                camera_name: np.full((480, 640, 3), 30 + camera_sequence + camera_idx, dtype=np.uint8)
                for camera_idx, camera_name in enumerate(data_regression.CAMERA_NAMES)
            }
            writer.write_step(
                frames_rgb=frames,
                camera_metadata={
                    camera_name: {
                        "sequence_index": camera_sequence,
                        "capture_timestamp_ns": camera_timestamp_ns,
                    }
                    for camera_name in data_regression.CAMERA_NAMES
                },
                state=episode.state[row_idx],
                action=episode.action[row_idx],
                timestamp_ns=int(episode.timestamps_s[row_idx] * 1_000_000_000),
            )
    finally:
        writer.close()

    actual = mcap_episode.read_episode(episode_dir, decode_images=True)

    assert actual.images is not None
    assert actual.images["cam_high"].shape == (4, 480, 640, 3)
    assert actual.camera_frame_reuse is not None
    assert actual.camera_frame_reuse["cam_high"] == {
        "rows": 4,
        "unique_frames": 3,
        "reused_rows": 1,
        "missing_rows": 0,
    }
    np.testing.assert_array_equal(actual.camera_timestamps_ns["cam_high"], np.asarray([0, 33_333_333, 66_666_666]))


def test_mcap_decode_preserves_rgb_channel_order(tmp_path: Path) -> None:
    episode = data_regression.make_synthetic_episode()
    episode_dir = _write_mcap_fixture_with_rgb_values(
        episode,
        tmp_path / "mcap_rgb",
        {
            "cam_high": np.asarray([20, 90, 170], dtype=np.uint8),
            "cam_left_wrist": np.asarray([40, 110, 190], dtype=np.uint8),
            "cam_right_wrist": np.asarray([60, 130, 210], dtype=np.uint8),
        },
    )

    actual = mcap_episode.read_episode(episode_dir, decode_images=True)

    assert actual.images is not None
    mean_rgb = actual.images["cam_high"][0].mean(axis=(0, 1))
    np.testing.assert_allclose(mean_rgb, np.asarray([20, 90, 170], dtype=np.float32), atol=6)


def test_mcap_reader_rejects_missing_shard(tmp_path: Path) -> None:
    episode = data_regression.make_synthetic_episode()
    episode_dir = data_regression.write_mcap_fixture(episode, tmp_path / "mcap_missing_shard")
    (episode_dir / "episode_part1.mcap").unlink()

    with pytest.raises(FileNotFoundError, match="missing shard"):
        mcap_episode.read_episode(episode_dir, decode_images=False)


def test_mcap_snapshot_validation_reports_missing_snapshot_field(tmp_path: Path) -> None:
    refs = _snapshot_refs()
    refs.pop(mcap_episode.CAMERA_FIELD_MAP["cam_high"])

    with pytest.raises(RuntimeError, match="missing snapshot refs"):
        mcap_episode._validate_snapshots(  # noqa: SLF001
            tmp_path,
            snapshots=[(0, refs)],
            arrays=_array_refs(),
            camera_sequences=_camera_sequences(),
        )


def test_mcap_snapshot_validation_reports_missing_state_sequence(tmp_path: Path) -> None:
    arrays = _array_refs()
    arrays[mcap_episode._action_joints_field(mcap_episode.LEFT_FOLLOWER)] = {}  # noqa: SLF001

    with pytest.raises(RuntimeError, match="references missing"):
        mcap_episode._validate_snapshots(  # noqa: SLF001
            tmp_path,
            snapshots=[(0, _snapshot_refs())],
            arrays=arrays,
            camera_sequences=_camera_sequences(),
        )


def test_mcap_snapshot_validation_reports_missing_camera_sequence(tmp_path: Path) -> None:
    camera_sequences = _camera_sequences()
    camera_sequences["cam_high"] = set()

    with pytest.raises(RuntimeError, match="references missing cam_high video sequence"):
        mcap_episode._validate_snapshots(  # noqa: SLF001
            tmp_path,
            snapshots=[(0, _snapshot_refs())],
            arrays=_array_refs(),
            camera_sequences=camera_sequences,
        )


def test_npz_to_lerobot_conversion_preserves_gripper_values(tmp_path: Path, monkeypatch) -> None:
    expected = data_regression.make_synthetic_episode()
    raw_dir = tmp_path / "raw"
    data_regression.write_npz_fixture(expected, raw_dir / "episode_000")
    lerobot_home = tmp_path / "lerobot"
    monkeypatch.setattr(convert_yam_data_to_lerobot, "HF_LEROBOT_HOME", lerobot_home)

    convert_yam_data_to_lerobot.main(
        convert_yam_data_to_lerobot.Args(
            raw_dir=raw_dir,
            repo_id="local/yam_regression_npz",
            raw_format="npz",
            image_writer_processes=0,
            image_writer_threads=0,
        )
    )

    actual = data_regression.load_lerobot_episode(
        "local/yam_regression_npz",
        root=lerobot_home / "local/yam_regression_npz",
        episode_index=0,
    )
    data_regression.compare_canonical_episodes(expected, actual)


def test_mcap_to_lerobot_conversion_preserves_gripper_values(tmp_path: Path, monkeypatch) -> None:
    expected = data_regression.make_synthetic_episode()
    raw_dir = tmp_path / "raw"
    _write_mcap_fixture_with_rgb_values(
        expected,
        raw_dir / "episode_000",
        {
            "cam_high": np.asarray([20, 90, 170], dtype=np.uint8),
            "cam_left_wrist": np.asarray([40, 110, 190], dtype=np.uint8),
            "cam_right_wrist": np.asarray([60, 130, 210], dtype=np.uint8),
        },
    )
    lerobot_home = tmp_path / "lerobot"
    monkeypatch.setattr(convert_yam_data_to_lerobot, "HF_LEROBOT_HOME", lerobot_home)

    convert_yam_data_to_lerobot.main(
        convert_yam_data_to_lerobot.Args(
            raw_dir=raw_dir,
            repo_id="local/yam_regression_mcap",
            raw_format="auto",
            image_writer_processes=0,
            image_writer_threads=0,
        )
    )

    actual = data_regression.load_lerobot_episode(
        "local/yam_regression_mcap",
        root=lerobot_home / "local/yam_regression_mcap",
        episode_index=0,
    )
    data_regression.compare_canonical_episodes(expected, actual)
    summary_path = lerobot_home / "local/yam_regression_mcap" / "conversion_summary.jsonl"
    summary_records = [json.loads(line) for line in summary_path.read_text().splitlines()]
    assert summary_records[0]["rows"] == expected.num_frames
    assert summary_records[0]["row_fps"] == 50.0
    assert summary_records[0]["task"] == expected.task
    assert set(summary_records[0]["camera_frame_reuse"]) == set(data_regression.CAMERA_NAMES)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/yam_regression_mcap", root=lerobot_home / "local/yam_regression_mcap")
    item = dataset[0]
    cam_high = _to_numpy(item["observation.images.cam_high"])
    assert cam_high.shape == (3, 480, 640)
    np.testing.assert_allclose(
        cam_high.reshape(3, -1).mean(axis=1),
        np.asarray([20, 90, 170], dtype=np.float32) / 255.0,
        atol=6 / 255.0,
    )

    train_config = training_config.get_config("pi05_yam_bimanual_50hz_jpeg_q85")
    data_config = train_config.data.create(Path("/tmp/nonexistent-assets"), train_config.model)
    assert train_config.model.action_horizon == 50
    raw_item = {
        "observation.images.cam_high": _chw_float_image_to_hwc_uint8(cam_high),
        "observation.images.cam_left_wrist": _chw_float_image_to_hwc_uint8(
            _to_numpy(item["observation.images.cam_left_wrist"])
        ),
        "observation.images.cam_right_wrist": _chw_float_image_to_hwc_uint8(
            _to_numpy(item["observation.images.cam_right_wrist"])
        ),
        "observation.state": expected.state[0],
        "action": expected.action[:4].copy(),
        "prompt": expected.task,
    }
    transformed = transforms.compose(
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, *data_config.model_transforms.inputs]
    )(raw_item)
    assert any(isinstance(transform, transforms.JpegRoundTripImages) for transform in data_config.model_transforms.inputs)
    assert transformed["image"]["base_0_rgb"].shape[:2] == jpeg_transport.IMAGE_RESOLUTION
    np.testing.assert_allclose(
        transformed["actions"][:, data_regression.GRIPPER_DIMS],
        expected.action[:4, data_regression.GRIPPER_DIMS],
    )


def test_mcap_episode_manifest_selection_preserves_order_and_rejects_bad_entries(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    episode = data_regression.make_synthetic_episode()
    data_regression.write_mcap_fixture(episode, raw_dir / "episode_a")
    data_regression.write_mcap_fixture(episode, raw_dir / "episode_b")
    manifest = tmp_path / "episodes.txt"
    manifest.write_text("episode_b\n# comment\nepisode_a\n")

    selected = convert_yam_data_to_lerobot._episode_dirs(raw_dir, manifest, "mcap")  # noqa: SLF001

    assert selected == [raw_dir / "episode_b", raw_dir / "episode_a"]

    manifest.write_text("episode_a\nepisode_a\n")
    with pytest.raises(ValueError, match="Duplicate episode"):
        convert_yam_data_to_lerobot._episode_dirs(raw_dir, manifest, "mcap")  # noqa: SLF001

    manifest.write_text("missing\n")
    with pytest.raises(FileNotFoundError, match="Manifest-listed episode"):
        convert_yam_data_to_lerobot._episode_dirs(raw_dir, manifest, "mcap")  # noqa: SLF001


def test_npz_to_lerobot_conversion_rejects_timing_mismatch(tmp_path: Path, monkeypatch) -> None:
    expected = data_regression.make_synthetic_episode()
    bad_episode = data_regression.CanonicalEpisode(
        source_path=expected.source_path,
        source_format=expected.source_format,
        task=expected.task,
        fps=50.0,
        state=expected.state,
        action=expected.action,
        timestamps_s=np.arange(expected.num_frames, dtype=np.float64) * 0.06,
        image_counts=expected.image_counts,
    )
    raw_dir = tmp_path / "raw"
    data_regression.write_npz_fixture(bad_episode, raw_dir / "episode_000")
    monkeypatch.setattr(convert_yam_data_to_lerobot, "HF_LEROBOT_HOME", tmp_path / "lerobot")

    with pytest.raises(RuntimeError, match="Timing mismatch"):
        convert_yam_data_to_lerobot.main(
            convert_yam_data_to_lerobot.Args(
                raw_dir=raw_dir,
                repo_id="local/yam_regression_bad_timing",
                raw_format="npz",
                image_writer_processes=0,
                image_writer_threads=0,
            )
        )


def test_mcap_to_lerobot_conversion_rejects_timing_mismatch_and_allows_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = data_regression.make_synthetic_episode()
    bad_episode = data_regression.CanonicalEpisode(
        source_path=expected.source_path,
        source_format=expected.source_format,
        task=expected.task,
        fps=50.0,
        state=expected.state,
        action=expected.action,
        timestamps_s=np.arange(expected.num_frames, dtype=np.float64) * 0.06,
        image_counts=expected.image_counts,
    )
    raw_dir = tmp_path / "raw"
    data_regression.write_mcap_fixture(bad_episode, raw_dir / "episode_000")
    lerobot_home = tmp_path / "lerobot"
    monkeypatch.setattr(convert_yam_data_to_lerobot, "HF_LEROBOT_HOME", lerobot_home)

    with pytest.raises(RuntimeError, match="Timing mismatch"):
        convert_yam_data_to_lerobot.main(
            convert_yam_data_to_lerobot.Args(
                raw_dir=raw_dir,
                repo_id="local/yam_regression_bad_mcap_timing",
                raw_format="mcap",
                image_writer_processes=0,
                image_writer_threads=0,
            )
        )

    convert_yam_data_to_lerobot.main(
        convert_yam_data_to_lerobot.Args(
            raw_dir=raw_dir,
            repo_id="local/yam_regression_allowed_mcap_timing",
            raw_format="mcap",
            allow_timing_mismatch=True,
            image_writer_processes=0,
            image_writer_threads=0,
        )
    )
    summary_path = lerobot_home / "local/yam_regression_allowed_mcap_timing" / "conversion_summary.jsonl"
    assert "timestamp median dt" in summary_path.read_text()


def test_yam_training_data_config_preserves_absolute_gripper_targets() -> None:
    episode = data_regression.make_synthetic_episode()
    factory = training_config.LeRobotYamDataConfig(
        repo_id="local/yam_regression",
        base_config=training_config.DataConfig(prompt_from_task=True),
    )
    data_config = factory.create(
        Path("/tmp/nonexistent-assets"), training_config.get_config("pi05_yam_bimanual_50hz").model
    )
    raw_item = {
        "observation.images.cam_high": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.state": episode.state[0],
        "action": episode.action[:4].copy(),
        "prompt": "synthetic gripper regression",
    }

    transformed = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])(
        raw_item
    )

    np.testing.assert_allclose(
        transformed["actions"][:, data_regression.GRIPPER_DIMS],
        episode.action[:4, data_regression.GRIPPER_DIMS],
    )
    np.testing.assert_allclose(
        transformed["actions"][:, :6],
        episode.action[:4, :6] - episode.state[0, :6],
    )
    np.testing.assert_allclose(
        transformed["actions"][:, 7:13],
        episode.action[:4, 7:13] - episode.state[0, 7:13],
    )

    stats = normalize.RunningStats()
    stats.update(transformed["actions"])
    action_stats = stats.get_statistics()
    gripper_dims = list(data_regression.GRIPPER_DIMS)
    assert np.all(np.isfinite(action_stats.mean[gripper_dims]))
    assert np.all(action_stats.std[gripper_dims] > 0)


def test_joint_regression_warns_on_large_arm_jump() -> None:
    episode = data_regression.make_synthetic_episode()
    action = episode.action.copy()
    action[2, 0] += 10.0
    bad_episode = data_regression.CanonicalEpisode(
        source_path=episode.source_path,
        source_format=episode.source_format,
        task=episode.task,
        fps=episode.fps,
        state=episode.state,
        action=action,
        timestamps_s=episode.timestamps_s,
        image_counts=episode.image_counts,
    )

    summary = data_regression.summarize_episode(bad_episode)

    assert any("action arm joint abs max" in warning for warning in summary["warnings"])
    assert any("action arm joint step max" in warning for warning in summary["warnings"])


def test_timing_regression_warns_when_manifest_fps_disagrees_with_timestamps() -> None:
    episode = data_regression.make_synthetic_episode()
    slow_timestamps = np.arange(episode.num_frames, dtype=np.float64) * 0.06
    bad_episode = data_regression.CanonicalEpisode(
        source_path=episode.source_path,
        source_format=episode.source_format,
        task=episode.task,
        fps=50.0,
        state=episode.state,
        action=episode.action,
        timestamps_s=slow_timestamps,
        image_counts=episode.image_counts,
    )

    summary = data_regression.summarize_episode(bad_episode)

    assert summary["timing"]["median_dt_s"] == 0.06
    assert 16.0 < summary["timing"]["median_fps"] < 17.0
    assert any("timestamp median dt" in warning for warning in summary["warnings"])


def test_encoder_readout_motor_config_tracks_current_i2rt_config() -> None:
    follower_motors, follower_directions = read_yam_encoders.build_motor_config(
        arm="yam", gripper="linear_4310", role="follower"
    )
    leader_motors, leader_directions = read_yam_encoders.build_motor_config(
        arm="yam", gripper="linear_4310", role="leader"
    )

    assert len(follower_motors) == 7
    assert follower_motors[-1][0] == 0x07
    assert follower_directions.shape == (7,)
    assert len(leader_motors) == 6
    assert leader_directions.shape == (6,)


def _write_mcap_fixture_with_rgb_values(
    episode: data_regression.CanonicalEpisode,
    episode_dir: Path,
    rgb_by_camera: dict[str, np.ndarray],
) -> Path:
    writer = mcap_episode.YamMcapEpisodeWriter(episode_dir, task=episode.task, fps=episode.fps)
    try:
        for frame_idx in range(episode.num_frames):
            frames = {
                camera_name: np.broadcast_to(rgb, (480, 640, 3)).copy()
                for camera_name, rgb in rgb_by_camera.items()
            }
            timestamp_ns = int(episode.timestamps_s[frame_idx] * 1_000_000_000)
            writer.write_step(
                frames_rgb=frames,
                camera_metadata={
                    camera_name: {
                        "sequence_index": frame_idx,
                        "capture_timestamp_ns": timestamp_ns,
                    }
                    for camera_name in data_regression.CAMERA_NAMES
                },
                state=episode.state[frame_idx],
                action=episode.action[frame_idx],
                timestamp_ns=timestamp_ns,
            )
    finally:
        writer.close()
    return episode_dir


def _snapshot_refs(sequence_id: int = 1) -> dict[mcap_episode.pistream_mcap.FieldKey, mcap_episode.FieldValueRef]:
    fields = [*mcap_episode.CAMERA_FIELD_MAP.values(), *mcap_episode._required_state_action_fields()]  # noqa: SLF001
    return {
        field: mcap_episode.FieldValueRef(field=field, sequence_id=sequence_id, timestamp_ns=0)
        for field in fields
    }


def _array_refs(sequence_id: int = 1) -> dict[mcap_episode.pistream_mcap.FieldKey, dict[int, np.ndarray]]:
    return {
        field: {sequence_id: np.zeros(mcap_episode._field_shape(field), dtype=np.float64)}  # noqa: SLF001
        for field in mcap_episode._required_state_action_fields()  # noqa: SLF001
    }


def _camera_sequences(sequence_id: int = 1) -> dict[str, set[int]]:
    return {camera_name: {sequence_id} for camera_name in data_regression.CAMERA_NAMES}


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _chw_float_image_to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype.kind == "f":
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    return np.transpose(image, (1, 2, 0))
