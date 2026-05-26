from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from examples.yam_real import convert_yam_data_to_lerobot
from examples.yam_real import data_regression
from examples.yam_real import read_yam_encoders
from openpi import transforms
from openpi.policies import yam_policy
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
    data_regression.write_mcap_fixture(expected, raw_dir / "episode_000")
    lerobot_home = tmp_path / "lerobot"
    monkeypatch.setattr(convert_yam_data_to_lerobot, "HF_LEROBOT_HOME", lerobot_home)

    convert_yam_data_to_lerobot.main(
        convert_yam_data_to_lerobot.Args(
            raw_dir=raw_dir,
            repo_id="local/yam_regression_mcap",
            raw_format="mcap",
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
