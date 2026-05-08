import numpy as np

from openpi.policies import yam_policy


def test_yam_schema_matches_pi_arx_bimanual_slots():
    assert yam_policy.ARM_JOINT_ORDER == (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    )
    assert yam_policy.STATE_ORDER == (
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    )
    assert yam_policy.ACTION_SPACE == "pi0/arx_bimanual"
    assert yam_policy.GRIPPER_CONVENTION == "0.0=open, 1.0=closed"


def test_yam_i2rt_gripper_conversion_happens_at_hardware_boundary():
    closed_i2rt = np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.float32)
    open_i2rt = np.array([0, 1, 2, 3, 4, 5, 1], dtype=np.float32)

    np.testing.assert_array_equal(
        yam_policy.i2rt_arm_state_to_openpi(closed_i2rt),
        np.array([0, 1, 2, 3, 4, 5, 1], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        yam_policy.i2rt_arm_state_to_openpi(open_i2rt),
        np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        yam_policy.openpi_arm_state_to_i2rt(yam_policy.i2rt_arm_state_to_openpi(open_i2rt)),
        open_i2rt,
    )
