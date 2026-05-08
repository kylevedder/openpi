# YAM Real Robot

This example connects the local YAM leader/follower cell to OpenPI.

## Hardware Names

The setup assumes these durable names exist:

```text
can_leader_l
can_leader_r
can_follower_l
can_follower_r
/dev/yam/cam_left_wrist
/dev/yam/cam_top
/dev/yam/cam_right_wrist
```

The OpenPI observation/action order is:

```text
[left_waist, left_shoulder, left_elbow, left_forearm_roll, left_wrist_angle, left_wrist_rotate, left_gripper,
 right_waist, right_shoulder, right_elbow, right_forearm_roll, right_wrist_angle, right_wrist_rotate, right_gripper]
```

This matches the PI/ARX bimanual action-space slots: left arm joints, left gripper, right arm joints, right gripper.
Left and right follow OpenPI's robot convention: viewed from behind the robot looking toward the workspace.
YAM/i2rt linear grippers report `0.0=closed, 1.0=open`; this example converts at the hardware boundary so all OpenPI
data and policy actions use `0.0=open, 1.0=closed`.

## 0. Hardware Readout

Read motor encoder positions without constructing the robot controller or running gripper calibration:

```bash
uv run python -m examples.yam_real.read_yam_encoders --role follower --side both
```

The shoulder check labels motor id `2`. If the arm is visibly pitched but the shoulder encoder reports near zero,
the motor zero is wrong. If the encoder reports a large angle, software can see the tilt.

The OpenPI hardware scripts default to `--no-use-gravity-comp` and hold follower arms at their current pose with PD on
startup. Only opt into model gravity compensation with `--use-gravity-comp` after validating the sign and frame for the
actual arm mounting.

## 1. Record One Teleop Episode

Run from the `openpi` repo root:

```bash
uv run python -m examples.yam_real.record_episode \
  --output-dir yam_data/raw \
  --task "pick up the object and place it in the target area" \
  --fps 20
```

Controls:

- Press the top button on each leader handle to enable sync for that side.
- Press `r` in the terminal to start or stop recording.
- Press `q` to stop and save.

The recorder writes:

```text
yam_data/raw/<episode_name>/
  manifest.json
  episode.npz
  images/cam_high/*.jpg
  images/cam_left_wrist/*.jpg
  images/cam_right_wrist/*.jpg
```

## 2. Replay the Recorded Actions

Dry-run first:

```bash
uv run python -m examples.yam_real.replay_episode \
  --episode-dir yam_data/raw/<episode_name>
```

Execute on the followers:

```bash
uv run python -m examples.yam_real.replay_episode \
  --episode-dir yam_data/raw/<episode_name> \
  --execute
```

Replay applies per-step delta limits before commanding the followers.

## 3. Convert to LeRobot

```bash
uv run python -m examples.yam_real.convert_yam_data_to_lerobot \
  --raw-dir yam_data/raw \
  --repo-id local/yam_bimanual
```

The OpenPI config `pi05_yam_bimanual` expects this repo id by default.

## 4. Modal GPU Training

Install and authenticate Modal locally, then run:

```bash
uv pip install modal
modal volume create yam-openpi
modal volume put yam-openpi ~/.cache/huggingface/lerobot/local/yam_bimanual /lerobot/local/yam_bimanual -f
modal run examples/yam_real/modal_app.py::compute_norm_stats
modal run examples/yam_real/modal_app.py::train --num-train-steps 100 --overwrite
```

Use the short run first to validate the pipeline. For a real run, omit `--num-train-steps 100` or set a larger value.

Before serving, set `SERVE_CHECKPOINT_STEP` in `examples/yam_real/modal_app.py` to the checkpoint step you want to serve.

```bash
modal deploy examples/yam_real/modal_app.py
```

The `serve_policy` endpoint runs OpenPI's websocket policy server on port `8000` inside Modal.

## 5. Policy Playback

Use dry-run first. The robot host sends observations and prints policy actions but does not command the followers:

```bash
uv run python -m examples.yam_real.run_policy \
  --host wss://<modal-endpoint-host> \
  --prompt "pick up the object and place it in the target area"
```

Execute only after dry-run returns sane 14D actions:

```bash
uv run python -m examples.yam_real.run_policy \
  --host wss://<modal-endpoint-host> \
  --prompt "pick up the object and place it in the target area" \
  --execute
```
