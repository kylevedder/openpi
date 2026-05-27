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

The recorder defaults to gravity compensation on, 50 Hz robot/action rows, and 30 Hz MJPG camera capture at 640x480.
Disable gravity compensation for recording with `--no-use-gravity-comp` if you need to debug torque behavior.

## 1. Record One Teleop Episode

Run from the `openpi` repo root:

```bash
uv run python -m examples.yam_real.record_episode
```

The default command writes to `yam_data/raw`, uses the task text `perform the demonstrated bimanual task`, records
50 Hz robot/action rows, captures 30 Hz 640x480 MJPG camera frames, and saves MCAP episodes. Override only the fields
that intentionally differ from the standard collection setup.

Controls:

- Press the top button on each leader handle to enable sync for that side.
- Press the bottom button on either leader handle, or `r` in the terminal, to start recording.
- Press the bottom button or `r` again to stop recording, save the full episode, and stay ready for the next one.
- Repeat start/stop to record back-to-back episodes without restarting the robot process.
- Press `q` to save any active recording and exit.

Recording status is also echoed on the leader arms as a haptic cue: one short pulse means recording started; two short
pulses means recording stopped. Disable this with `--no-status-haptic-cue` if it is distracting.

The recorder writes:

```text
yam_data/raw/<episode_name>/
  episode_part0.mcap
  episode_part1.mcap
  episode_part2.mcap
  episode_part3.mcap
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

Create an explicit training manifest so conversion uses only the selected demos:

```bash
uv run python -m examples.yam_real.build_episode_manifest \
  --raw-dir yam_data/raw \
  --output yam_data/manifests/pi05_yam_bimanual_50hz_20demo.txt \
  --min-frames 1
```

```bash
uv run python -m examples.yam_real.convert_yam_data_to_lerobot \
  --raw-dir yam_data/raw \
  --episode-manifest yam_data/manifests/pi05_yam_bimanual_50hz_20demo.txt \
  --repo-id local/yam_bimanual
```

The converter defaults to auto-detecting MCAP raw episodes and LeRobot image-frame output. It writes
`conversion_summary.jsonl` in the dataset root with row counts, row FPS, inferred camera FPS, camera-frame reuse, and
warnings. The OpenPI configs `pi05_yam_bimanual_50hz` and `pi05_yam_bimanual_50hz_jpeg_q85` expect this repo id by
default.

## Image Transport

Pi0.5 consumes `224x224` images. The `pi05_yam_bimanual_50hz_jpeg_q85` config resizes training images to `224x224`
and then applies the same OpenCV JPEG Q85 encode/decode round-trip used by inference. At runtime, `run_policy` sends
those three resized camera frames as JPEG bytes and the policy server decodes them without applying a second JPEG pass.

Benchmark the transport on the local recordings:

```bash
uv run python -m examples.yam_real.benchmark_jpeg_transport \
  --raw-dir yam_data/raw \
  --episode-manifest yam_data/manifests/pi05_yam_bimanual_50hz_20demo.txt \
  --max-observations 80
```

On the initial 80-observation sample, raw msgpack was about `452 KB` per request. JPEG Q85 was about `28 KB`, roughly
`16x` smaller, with about `0.34 ms` encode and `0.41 ms` decode overhead for all three cameras. Existing raw recordings
are already stored as camera JPEGs; this transport adds the matched post-resize network JPEG artifact that the model sees
during serving.

## 4. Modal GPU Training

Install and authenticate Modal locally, then run:

```bash
uv pip install modal
modal volume create yam-openpi
modal volume put yam-openpi ~/.cache/huggingface/lerobot/local/yam_bimanual /lerobot/local/yam_bimanual -f
modal run examples/yam_real/modal_app.py::compute_norm_stats --config-name pi05_yam_bimanual_50hz_jpeg_q85
modal run examples/yam_real/modal_app.py::train_fsdp2 \
  --config-name pi05_yam_bimanual_50hz_jpeg_q85 \
  --exp-name yam_bimanual_50hz_jpeg_q85_20demo_v1 \
  --num-train-steps 1000 \
  --batch-size 8 \
  --overwrite
```

Use the short run first to validate the pipeline. For a real run over the full 20k-step config, omit
`--num-train-steps 100`:

```bash
modal run --detach examples/yam_real/modal_app.py::train_fsdp2 \
  --config-name pi05_yam_bimanual_50hz_jpeg_q85 \
  --exp-name yam_bimanual_50hz_jpeg_q85_20demo_v1 \
  --batch-size 8
```

The Modal serving entrypoint is configured to serve checkpoint step `999` from
`yam_bimanual_50hz_jpeg_q85_20demo_v1`.

```bash
modal deploy examples/yam_real/modal_app.py
```

The `serve_policy` endpoint runs OpenPI's websocket policy server on port `8000` inside Modal.
The same deployment also includes `YamQuicPolicyServer`, a direct Modal QUIC portal server used to bypass public
websocket ingress during latency tests and robot playback.

## 5. Policy Playback

Each `run_policy` invocation tees stdout and stderr to a timestamped log under
`yam_data/logs/run_policy/`. The most recent run is always available at:

```bash
yam_data/logs/run_policy/latest.log
```

Use dry-run first. The robot host sends observations and prints policy actions but does not command the followers:

```bash
uv run python -m examples.yam_real.run_policy \
  --host wss://<modal-endpoint-host> \
  --prompt "pick up the object and place it in the target area" \
  --max-steps 40 \
  --fps 50 \
  --action-horizon 50 \
  --use-gravity-comp
```

For the Modal QUIC path, use the deployed class directly instead of a `wss://` host:

```bash
uv run python -m examples.yam_real.run_policy \
  --transport modal-quic \
  --modal-app-name yam-openpi \
  --modal-class-name YamQuicPolicyServer \
  --prompt "pick up the object and place it in the target area" \
  --max-steps 40 \
  --fps 50 \
  --action-horizon 50 \
  --use-gravity-comp
```

By default `run_policy` uses `--image-transport auto`, which follows the policy metadata. The JPEG-trained policy
advertises `jpeg_q85_224_rgb_v1`, so the local robot host sends compressed `224x224` images automatically.
Early action-chunk prefetch is off by default: `--prefetch-remaining-steps 0` means the next observation and inference
start only after the current chunk and any inter-chunk hold finish. Set `--prefetch-remaining-steps N` to start
inference when `N` actions remain in the current chunk; `--no-prefetch-action-chunks` remains a hard off switch.

For slow-motion debugging, keep `--fps 50` so the policy metadata still matches the trained model, and set
`--action-playback-fps` to the slower command rate. `--max-steps` is still an action-row count, not seconds, so
`--action-playback-fps 10 --max-steps 1200` runs for about 2 minutes.
Add `--inter-chunk-delay-s` to insert a hold pause after each completed action chunk; this delay does not count
against `--max-steps`.

```bash
uv run python -m examples.yam_real.run_policy \
  --transport modal-quic \
  --modal-app-name yam-openpi \
  --modal-class-name YamQuicPolicyServer \
  --modal-server-start-timeout-s 1200 \
  --prompt "pick up the object and place it in the target area" \
  --max-steps 1200 \
  --fps 50 \
  --action-horizon 50 \
  --action-playback-fps 10 \
  --inter-chunk-delay-s 1.0 \
  --max-arm-step-rad 0.01 \
  --max-gripper-step 0.01 \
  --use-gravity-comp \
  --execute
```

Run a short bounded execute with an empty workspace only after dry-run returns sane 14D actions:

```bash
uv run python -m examples.yam_real.run_policy \
  --host wss://<modal-endpoint-host> \
  --prompt "pick up the object and place it in the target area" \
  --max-steps 50 \
  --fps 50 \
  --action-horizon 50 \
  --max-arm-step-rad 0.01 \
  --max-gripper-step 0.01 \
  --use-gravity-comp \
  --execute
```

## 6. Latency Diagnostics

The latency probes are robot-safe: they do not connect to i2rt or command the arms. All probe runs write JSONL logs to:

```bash
yam_data/logs/latency/latest.jsonl
```

Local harness sanity check:

```bash
uv run python -m examples.yam_real.latency_probe_server --mode echo --port 8765
uv run python -m examples.yam_real.latency_probe \
  --protocol echo \
  --host localhost \
  --port 8765 \
  --iterations 50 \
  --warmup 5 \
  --payload-bytes 115875 \
  --response-bytes 3000 \
  --label local_echo_jpeg_payload
```

Deploy temporary Modal diagnostic endpoints:

```bash
uv run modal deploy examples/yam_real/modal_latency_app.py
```

Then probe the public Modal ingress paths. Replace the host if Modal prints a different URL:

```bash
uv run python -m examples.yam_real.latency_probe \
  --protocol echo \
  --host wss://kyle-25446--yam-openpi-latency-echo-us-west.modal.run \
  --iterations 50 \
  --warmup 5 \
  --payload-bytes 115875 \
  --response-bytes 3000 \
  --label modal_echo_us_west_jpeg_payload

uv run python -m examples.yam_real.latency_probe \
  --protocol policy \
  --host wss://kyle-25446--yam-openpi-latency-noop-policy-us-west-108ms.modal.run \
  --iterations 50 \
  --warmup 5 \
  --label modal_noop_policy_us_west_108ms

uv run python -m examples.yam_real.latency_probe \
  --protocol policy \
  --host wss://kyle-25446--yam-openpi-serve-policy.modal.run \
  --iterations 50 \
  --warmup 2 \
  --label modal_real_policy_jpeg
```

Probe the direct Modal QUIC portal paths from the same deployed diagnostic app:

```bash
uv run python -m examples.yam_real.latency_probe \
  --transport modal-quic \
  --protocol echo \
  --iterations 50 \
  --warmup 5 \
  --payload-bytes 115875 \
  --response-bytes 3000 \
  --label modal_quic_echo_jpeg_payload

uv run python -m examples.yam_real.latency_probe \
  --transport modal-quic \
  --protocol echo \
  --iterations 50 \
  --warmup 5 \
  --payload-bytes 452000 \
  --response-bytes 3000 \
  --label modal_quic_echo_raw_equiv_payload

uv run python -m examples.yam_real.latency_probe \
  --transport modal-quic \
  --protocol echo \
  --modal-class-name QuicLatencyGpuServer \
  --modal-server-start-timeout-s 300 \
  --iterations 50 \
  --warmup 5 \
  --payload-bytes 115875 \
  --response-bytes 3000 \
  --label modal_quic_gpu_echo_jpeg_payload

uv run python -m examples.yam_real.latency_probe \
  --transport modal-quic \
  --protocol policy \
  --sleep-ms 108 \
  --iterations 50 \
  --warmup 5 \
  --label modal_quic_noop_policy_108ms
```

The decision rule is simple: if Modal echo is already above roughly `150 ms`, the public Modal ingress path is the
latency bottleneck. If public Modal websocket echo is slow but Modal QUIC echo is near `40 ms`, use `--transport
modal-quic` for robot inference. If both Modal paths are slow, keep Modal for training and serve robot inference on a
direct GPU host.

## 7. Gripper Data Regression Checks

Before uploading or training on a new dataset, run the end-to-end gripper regression inspector. It compares the archived
old-good NPZ demos against the current raw demos, checks gripper dims `6` and `13`, and writes JSONL logs under
`yam_data/logs/data_regression/`.

```bash
uv run python -m examples.yam_real.data_regression \
  --old-good-dir yam_data/archive/raw_before_moved_robot_20260525_145623 \
  --current-raw-dir yam_data/raw \
  --source-format auto \
  --strict
```

The important convention is `0.0=open, 1.0=closed`. The inspector warns on flat gripper action ranges, missing close
samples, out-of-range gripper values, and camera/frame count mismatches. It can also inspect a specific MCAP or LeRobot
source by passing the episode path in `--inputs` and selecting `--source-format mcap` or `--source-format lerobot`.

Then run the task with the normal per-step limits:

```bash
uv run python -m examples.yam_real.run_policy \
  --transport modal-quic \
  --modal-app-name yam-openpi \
  --modal-class-name YamQuicPolicyServer \
  --modal-server-start-timeout-s 1200 \
  --prompt "pick up the object and place it in the target area" \
  --max-steps 200 \
  --fps 50 \
  --action-horizon 50 \
  --max-arm-step-rad 0.02 \
  --max-gripper-step 0.02 \
  --use-gravity-comp \
  --execute
```
