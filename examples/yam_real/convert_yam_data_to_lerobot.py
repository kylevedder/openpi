from __future__ import annotations

from concurrent import futures
import contextlib
import ctypes
import dataclasses
import gc
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Literal

import datasets
from lerobot.common.datasets import compute_stats as lerobot_compute_stats
from lerobot.common.datasets import utils as lerobot_dataset_utils
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tqdm
import tyro

from examples.yam_real import common
from examples.yam_real import data_regression


@dataclasses.dataclass
class Args:
    raw_dir: Path = Path("yam_data/raw")
    episode_manifest: Path | None = None
    repo_id: str = "local/yam_bimanual"
    mode: Literal["image", "video"] = "image"
    resume: bool = False
    image_writer_processes: int = 0
    image_writer_threads: int = 0
    image_writer_flush_interval_frames: int = 0
    episode_processes: int = 0
    scratch_dir: Path = Path("/tmp/yam_lerobot_preprocess")
    profile_jsonl: Path | None = None
    push_to_hub: bool = False
    overwrite: bool = True
    max_median_dt_error_ratio: float = 0.25
    max_p95_dt_error_ratio: float = 2.5


@dataclasses.dataclass(frozen=True)
class _FastEpisodeData:
    task: str
    fps: float
    timestamps_ns: np.ndarray
    state: np.ndarray
    action: np.ndarray
    images_by_sequence: dict[str, dict[int, np.ndarray]]
    camera_sequence_by_row: dict[str, np.ndarray]
    camera_timestamps_ns: dict[str, np.ndarray]
    camera_frame_reuse: dict[str, dict[str, int]]


@dataclasses.dataclass(frozen=True)
class _FastEpisodeJob:
    episode_dir: Path
    episode_index: int
    global_start_index: int
    task_index: int
    task: str
    fps: int
    output_root: Path
    summary_record: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class _FastEpisodeResult:
    episode_index: int
    episode_dir: str
    rows: int
    parquet_path: str
    episode_record: dict[str, Any]
    episode_stats: dict[str, dict[str, np.ndarray]]
    summary_record: dict[str, Any]
    profile: dict[str, Any]


def main(args: Args) -> None:
    output_path = HF_LEROBOT_HOME / args.repo_id
    if args.episode_processes > 0:
        _main_parallel(args, output_path)
        return

    resume_from = 0
    if output_path.exists() and args.resume:
        shutil.rmtree(output_path / "images", ignore_errors=True)
        dataset = LeRobotDataset(args.repo_id, root=output_path)
        resume_from = int(dataset.meta.total_episodes)
        print(f"Resuming existing LeRobot dataset at {output_path}: {resume_from} episodes already saved")
    elif output_path.exists():
        if not args.overwrite:
            raise FileExistsError(output_path)
        shutil.rmtree(output_path)
        dataset = None
    else:
        dataset = None

    episode_dirs = _episode_dirs(args.raw_dir, args.episode_manifest)
    if not episode_dirs:
        raise FileNotFoundError(f"No recorded YAM episodes found under {args.raw_dir}")
    if resume_from > len(episode_dirs):
        raise ValueError(f"Cannot resume from {resume_from} episodes; manifest only has {len(episode_dirs)}")
    episode_dirs = episode_dirs[resume_from:]
    if not episode_dirs:
        print(f"No remaining episodes to convert; dataset is already complete at {output_path}")
        return

    first_episode = data_regression.load_canonical_episode(episode_dirs[0], source_format="mcap")
    fps = round(float(first_episode.fps))
    validation_records = [_validate_episode(episode_dir, args) for episode_dir in episode_dirs]
    validation_record_by_path = {record["source_path"]: record for record in validation_records}
    frame_counts = [int(record["rows"]) for record in validation_records]
    print(f"Selected {len(episode_dirs)} YAM episodes")
    print(f"Total frames: {sum(frame_counts)}")

    if dataset is None:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=output_path,
            robot_type="yam_bimanual",
            fps=fps,
            features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["action"],
                },
                "observation.images.cam_high": _image_feature(args.mode),
                "observation.images.cam_left_wrist": _image_feature(args.mode),
                "observation.images.cam_right_wrist": _image_feature(args.mode),
            },
            use_videos=args.mode == "video",
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )

    summary_path = output_path / "conversion_summary.jsonl"
    if not args.resume and summary_path.exists():
        summary_path.unlink()

    for episode_dir in tqdm.tqdm(episode_dirs, desc="Converting YAM episodes"):
        task, states, actions, images_or_paths = _load_episode_for_conversion(episode_dir)
        if states.shape != actions.shape or states.shape[-1] != 14:
            raise RuntimeError(f"Bad state/action shapes in {episode_dir}: {states.shape}, {actions.shape}")
        if not isinstance(images_or_paths, dict):
            raise RuntimeError(f"YAM conversion only supports canonical MCAP decoded images: {episode_dir}")
        for camera_name in common.CAMERA_NAMES:
            if camera_name not in images_or_paths:
                raise RuntimeError(f"Missing images for {camera_name} in {episode_dir}")
            images = images_or_paths[camera_name]
            if len(images) != len(states):
                raise RuntimeError(
                    f"Image count mismatch for {camera_name} in {episode_dir}: {len(images)} vs {len(states)}"
                )

        for idx in range(len(states)):
            frame = {
                "observation.state": states[idx],
                "action": actions[idx],
                "task": task,
            }
            frame.update(_load_images_for_frame(episode_dir, images_or_paths, idx))
            dataset.add_frame(frame)
            if args.image_writer_flush_interval_frames and (idx + 1) % args.image_writer_flush_interval_frames == 0:
                dataset._wait_image_writer()  # noqa: SLF001
        dataset.save_episode()
        _append_conversion_summary(summary_path, validation_record_by_path[str(episode_dir)])
        del states, actions, images_or_paths
        gc.collect()
        _malloc_trim()

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    if args.push_to_hub:
        dataset.push_to_hub()
    print(f"Wrote LeRobot dataset to {output_path}")


def _main_parallel(args: Args, output_path: Path) -> None:
    if args.resume:
        raise ValueError("--resume is not supported with --episode-processes")
    if args.push_to_hub:
        raise ValueError("--push-to-hub is not supported with --episode-processes")
    if args.mode != "image":
        raise ValueError("--episode-processes currently supports --mode image only")
    if args.episode_processes <= 0:
        raise ValueError("--episode-processes must be positive for parallel conversion")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)

    _set_process_worker_env()

    episode_dirs = _episode_dirs(args.raw_dir, args.episode_manifest)
    if not episode_dirs:
        raise FileNotFoundError(f"No recorded YAM episodes found under {args.raw_dir}")

    validation_records = [_validate_episode(episode_dir, args) for episode_dir in episode_dirs]
    validation_record_by_path = {record["source_path"]: record for record in validation_records}
    frame_counts = [int(record["rows"]) for record in validation_records]
    first_episode = data_regression.load_canonical_episode(episode_dirs[0], source_format="mcap")
    fps = round(float(first_episode.fps))
    tasks = _ordered_unique(str(record["task"]) for record in validation_records)
    task_to_index = {task: index for index, task in enumerate(tasks)}

    print(f"Selected {len(episode_dirs)} YAM episodes")
    print(f"Total frames: {sum(frame_counts)}")
    print(f"Parallel conversion: episode_processes={args.episode_processes}")

    run_id = f"run_{int(time.time())}_{os.getpid()}"
    scratch_root = args.scratch_dir / run_id / "dataset"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=False)
    profile_path = args.profile_jsonl or (args.scratch_dir / run_id / "profile.jsonl")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    if profile_path.exists():
        profile_path.unlink()

    jobs = []
    global_start_index = 0
    for episode_index, (episode_dir, frame_count) in enumerate(zip(episode_dirs, frame_counts, strict=True)):
        summary_record = validation_record_by_path[str(episode_dir)]
        task = str(summary_record["task"])
        jobs.append(
            _FastEpisodeJob(
                episode_dir=episode_dir,
                episode_index=episode_index,
                global_start_index=global_start_index,
                task_index=task_to_index[task],
                task=task,
                fps=fps,
                output_root=scratch_root,
                summary_record=summary_record,
            )
        )
        global_start_index += frame_count

    results: list[_FastEpisodeResult] = []
    pool_started = time.perf_counter()
    completed_frames = 0
    context = mp.get_context("spawn")
    with futures.ProcessPoolExecutor(max_workers=args.episode_processes, mp_context=context) as executor:
        future_to_job = {executor.submit(_convert_episode_process, job): job for job in jobs}
        with tqdm.tqdm(total=len(jobs), desc="Converting YAM episodes", unit="episode") as progress:
            for future in futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                except Exception as exc:
                    for pending in future_to_job:
                        pending.cancel()
                    raise RuntimeError(f"Parallel conversion failed for {job.episode_dir}") from exc
                results.append(result)
                completed_frames += result.rows
                elapsed_s = max(time.perf_counter() - pool_started, 1e-9)
                aggregate_frames_per_s = completed_frames / elapsed_s
                remaining_frames = max(sum(frame_counts) - completed_frames, 0)
                result.profile["aggregate_frames_per_s"] = aggregate_frames_per_s
                result.profile["eta_s"] = remaining_frames / aggregate_frames_per_s if aggregate_frames_per_s else 0.0
                _append_jsonl(profile_path, _json_safe(result.profile))
                progress.update(1)
                progress.set_postfix(
                    {
                        "episode": result.episode_index,
                        "rows": result.rows,
                        "fps": f"{aggregate_frames_per_s:.1f}",
                        "sec": f"{result.profile['total_s']:.1f}",
                    }
                )

    results = sorted(results, key=lambda result: result.episode_index)
    if [result.episode_index for result in results] != list(range(len(jobs))):
        raise RuntimeError("Parallel conversion produced missing or duplicate episode indices")

    _write_fast_lerobot_metadata(
        output_root=scratch_root,
        fps=fps,
        tasks=tasks,
        results=results,
        total_frames=sum(frame_counts),
        mode=args.mode,
    )
    shutil.copy2(profile_path, scratch_root / "conversion_profile.jsonl")
    _validate_fast_output(
        repo_id=args.repo_id,
        output_root=scratch_root,
        expected_episodes=len(results),
        expected_frames=sum(frame_counts),
    )
    _publish_parallel_output(scratch_root, output_path, overwrite=args.overwrite)
    print(f"Wrote LeRobot dataset to {output_path}")


def _convert_episode_process(job: _FastEpisodeJob) -> _FastEpisodeResult:
    started = time.perf_counter()
    decode_start = time.perf_counter()
    episode = _read_fast_episode(job.episode_dir)
    decode_s = time.perf_counter() - decode_start
    rows = int(episode.state.shape[0])
    if rows <= 0:
        raise RuntimeError(f"Episode {job.episode_dir} has no rows")
    if episode.task != job.task:
        raise RuntimeError(f"Task mismatch in {job.episode_dir}: {episode.task!r} vs {job.task!r}")

    encode_start = time.perf_counter()
    encoded_images = _encode_unique_camera_images(episode.images_by_sequence)
    encode_s = time.perf_counter() - encode_start

    parquet_start = time.perf_counter()
    parquet_path = _write_episode_parquet(job, episode, encoded_images)
    parquet_s = time.perf_counter() - parquet_start

    stats_start = time.perf_counter()
    episode_stats = _compute_fast_episode_stats(job, episode)
    stats_s = time.perf_counter() - stats_start

    total_s = time.perf_counter() - started
    profile = {
        "worker_pid": os.getpid(),
        "episode_index": job.episode_index,
        "episode_name": job.episode_dir.name,
        "episode_dir": str(job.episode_dir),
        "rows": rows,
        "unique_camera_frames": {
            camera_name: len(images) for camera_name, images in episode.images_by_sequence.items()
        },
        "decode_s": decode_s,
        "image_encode_s": encode_s,
        "parquet_write_s": parquet_s,
        "stats_s": stats_s,
        "total_s": total_s,
        "peak_rss_mb": _peak_rss_mb(),
    }
    return _FastEpisodeResult(
        episode_index=job.episode_index,
        episode_dir=str(job.episode_dir),
        rows=rows,
        parquet_path=str(parquet_path),
        episode_record={
            "episode_index": job.episode_index,
            "tasks": [job.task],
            "length": rows,
        },
        episode_stats=episode_stats,
        summary_record=job.summary_record,
        profile=profile,
    )


def _read_fast_episode(episode_dir: Path) -> _FastEpisodeData:
    from collections import deque

    from examples.yam_real import mcap_episode

    mcap_episode._validate_expected_shards(episode_dir)  # noqa: SLF001
    metadata = mcap_episode.read_episode_metadata(episode_dir)
    mcap_episode._validate_stream_descriptors(episode_dir, metadata)  # noqa: SLF001
    expected_fields = {
        spec.field_key
        for spec in mcap_episode._build_field_specs(  # noqa: SLF001
            width=int(metadata.get("image_width", 640)),
            height=int(metadata.get("image_height", 480)),
            fps=float(metadata.get("camera_fps", 30.0)),
        )
    }
    arrays_by_sequence = {field: {} for field in mcap_episode._required_state_action_fields()}  # noqa: SLF001
    camera_sequences = {camera_name: set() for camera_name in mcap_episode.CAMERA_FIELD_MAP}
    images_by_sequence = {camera_name: {} for camera_name in mcap_episode.CAMERA_FIELD_MAP}
    video_pending_sequences = {camera_name: deque() for camera_name in mcap_episode.CAMERA_FIELD_MAP}
    snapshots = []
    snapshot_field_by_index = {
        index: field
        for field, index in mcap_episode._snapshot_index(mcap_episode._snapshot_fields()).items()  # noqa: SLF001
    }
    decoders = {
        camera_name: mcap_episode.pistream_mcap.H264VideoDecoder()
        for camera_name in mcap_episode.CAMERA_FIELD_MAP
    }

    for _, channel, message in mcap_episode.pistream_mcap.iter_mcap_messages(episode_dir):
        if channel.topic == mcap_episode.pistream_mcap.PI_STREAM_DESCRIPTOR_TOPIC:
            continue
        if not channel.topic.startswith(mcap_episode.pistream_mcap.PI_STREAM_FRAME_PREFIX):
            raise RuntimeError(f"MCAP episode {episode_dir} contains non-canonical topic {channel.topic!r}")
        field = mcap_episode.pistream_mcap.parse_frame_topic(channel.topic)
        if field not in expected_fields:
            raise RuntimeError(
                f"MCAP episode {episode_dir} contains undeclared/non-canonical frame topic {channel.topic!r}"
            )
        if field in arrays_by_sequence:
            decoded = mcap_episode.frame_pb2.DoubleList()
            decoded.ParseFromString(message.data)
            arrays_by_sequence[field][message.sequence] = mcap_episode.pistream_mcap.parse_double_list(
                decoded,
                mcap_episode._field_shape(field),  # noqa: SLF001
            )
        elif field == mcap_episode.ACTION_SNAPSHOT_FIELD:
            snapshot = mcap_episode.frame_pb2.PiStreamSnapshotRef()
            snapshot.ParseFromString(message.data)
            refs = {}
            for ref in snapshot.fields:
                if ref.index not in snapshot_field_by_index:
                    raise RuntimeError(
                        f"MCAP episode {episode_dir} action_snapshot at {message.log_time} contains "
                        f"unsupported snapshot ref index {ref.index}."
                    )
                if ref.publisher_id or ref.key:
                    raise RuntimeError(
                        f"MCAP episode {episode_dir} action_snapshot at {message.log_time} contains "
                        "publisher_id/key snapshot refs."
                    )
                ref_field = snapshot_field_by_index[ref.index]
                refs[ref_field] = mcap_episode.FieldValueRef(
                    field=ref_field,
                    sequence_id=ref.sequence_id,
                    timestamp_ns=mcap_episode.pistream_mcap.timestamp_to_nanos(ref.capture_timestamp),
                )
            snapshots.append((message.log_time, refs))
        elif field in set(mcap_episode.CAMERA_FIELD_MAP.values()):
            video_frame = mcap_episode.frame_pb2.VideoFrame()
            video_frame.ParseFromString(message.data)
            camera_name = mcap_episode._camera_name_for_field(field)  # noqa: SLF001
            camera_sequences[camera_name].add(message.sequence)
            video_pending_sequences[camera_name].append((message.sequence, message.log_time))
            decoded_images = decoders[camera_name].decode_to_rgb(video_frame)
            for image in decoded_images:
                sequence_id, _timestamp_ns = video_pending_sequences[camera_name].popleft()
                mcap_episode._validate_rgb_image(image, episode_dir, camera_name, sequence_id)  # noqa: SLF001
                images_by_sequence[camera_name][sequence_id] = image

    mcap_episode._finalize_video_decoders(  # noqa: SLF001
        decoders,
        episode_dir=episode_dir,
        pending_sequences=video_pending_sequences,
        images=images_by_sequence,
    )
    snapshots = mcap_episode._validate_snapshots(  # noqa: SLF001
        episode_dir,
        snapshots=snapshots,
        arrays=arrays_by_sequence,
        camera_sequences=camera_sequences,
    )

    state = []
    action = []
    camera_row_timestamps = {camera_name: [] for camera_name in mcap_episode.CAMERA_FIELD_MAP}
    camera_sequence_by_row = {camera_name: [] for camera_name in mcap_episode.CAMERA_FIELD_MAP}
    left_observation_joints = mcap_episode._observation_joints_field(mcap_episode.LEFT_FOLLOWER)  # noqa: SLF001
    left_observation_gripper = mcap_episode._observation_gripper_field(mcap_episode.LEFT_FOLLOWER)  # noqa: SLF001
    right_observation_joints = mcap_episode._observation_joints_field(mcap_episode.RIGHT_FOLLOWER)  # noqa: SLF001
    right_observation_gripper = mcap_episode._observation_gripper_field(mcap_episode.RIGHT_FOLLOWER)  # noqa: SLF001
    left_action_joints = mcap_episode._action_joints_field(mcap_episode.LEFT_FOLLOWER)  # noqa: SLF001
    left_action_gripper = mcap_episode._action_gripper_field(mcap_episode.LEFT_FOLLOWER)  # noqa: SLF001
    right_action_joints = mcap_episode._action_joints_field(mcap_episode.RIGHT_FOLLOWER)  # noqa: SLF001
    right_action_gripper = mcap_episode._action_gripper_field(mcap_episode.RIGHT_FOLLOWER)  # noqa: SLF001
    for _timestamp_ns, refs in snapshots:
        state.append(
            np.concatenate(
                [
                    mcap_episode._array_for_ref(arrays_by_sequence, refs, left_observation_joints),  # noqa: SLF001
                    mcap_episode._array_for_ref(arrays_by_sequence, refs, left_observation_gripper),  # noqa: SLF001
                    mcap_episode._array_for_ref(arrays_by_sequence, refs, right_observation_joints),  # noqa: SLF001
                    mcap_episode._array_for_ref(arrays_by_sequence, refs, right_observation_gripper),  # noqa: SLF001
                ]
            ).astype(np.float32)
        )
        action.append(
            np.concatenate(
                [
                    mcap_episode._array_for_ref(arrays_by_sequence, refs, left_action_joints),  # noqa: SLF001
                    mcap_episode._array_for_ref(arrays_by_sequence, refs, left_action_gripper),  # noqa: SLF001
                    mcap_episode._array_for_ref(arrays_by_sequence, refs, right_action_joints),  # noqa: SLF001
                    mcap_episode._array_for_ref(arrays_by_sequence, refs, right_action_gripper),  # noqa: SLF001
                ]
            ).astype(np.float32)
        )
        for camera_name, field in mcap_episode.CAMERA_FIELD_MAP.items():
            ref = mcap_episode._ref_for_field(refs, field)  # noqa: SLF001
            camera_sequence_by_row[camera_name].append(ref.sequence_id)
            camera_row_timestamps[camera_name].append(ref.timestamp_ns)

    timestamps_ns = np.asarray([timestamp_ns for timestamp_ns, _refs in snapshots], dtype=np.int64)
    state_array = np.asarray(state, dtype=np.float32)
    action_array = np.asarray(action, dtype=np.float32)
    mcap_episode._validate_episode_arrays(episode_dir, state_array, action_array)  # noqa: SLF001
    return _FastEpisodeData(
        task=str(metadata.get("task", "")),
        fps=mcap_episode._fps_from_episode_metadata_or_timestamps(metadata, timestamps_ns),  # noqa: SLF001
        timestamps_ns=timestamps_ns,
        state=state_array,
        action=action_array,
        images_by_sequence=images_by_sequence,
        camera_sequence_by_row={
            camera_name: np.asarray(values, dtype=np.int64) for camera_name, values in camera_sequence_by_row.items()
        },
        camera_timestamps_ns={
            camera_name: np.asarray(mcap_episode._unique_preserve_order(values), dtype=np.int64)  # noqa: SLF001
            for camera_name, values in camera_row_timestamps.items()
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


def _encode_unique_camera_images(images_by_sequence: dict[str, dict[int, np.ndarray]]) -> dict[str, dict[int, bytes]]:
    return {
        camera_name: {
            int(sequence_id): _encode_png_bytes(image)
            for sequence_id, image in sorted(camera_images.items(), key=lambda item: item[0])
        }
        for camera_name, camera_images in images_by_sequence.items()
    }


def _encode_png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _write_episode_parquet(
    job: _FastEpisodeJob,
    episode: _FastEpisodeData,
    encoded_images: dict[str, dict[int, bytes]],
) -> Path:
    rows = int(episode.state.shape[0])
    frame_index = np.arange(rows, dtype=np.int64)
    episode_dict: dict[str, Any] = {
        "observation.state": episode.state,
        "action": episode.action,
        "observation.images.cam_high": _image_column(
            encoded_images["cam_high"], episode.camera_sequence_by_row["cam_high"], rows
        ),
        "observation.images.cam_left_wrist": _image_column(
            encoded_images["cam_left_wrist"], episode.camera_sequence_by_row["cam_left_wrist"], rows
        ),
        "observation.images.cam_right_wrist": _image_column(
            encoded_images["cam_right_wrist"], episode.camera_sequence_by_row["cam_right_wrist"], rows
        ),
        "timestamp": (frame_index.astype(np.float32) / np.float32(job.fps)),
        "frame_index": frame_index,
        "episode_index": np.full((rows,), job.episode_index, dtype=np.int64),
        "index": np.arange(job.global_start_index, job.global_start_index + rows, dtype=np.int64),
        "task_index": np.full((rows,), job.task_index, dtype=np.int64),
    }
    parquet_path = _episode_parquet_path(job.output_root, job.episode_index)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.Dataset.from_dict(episode_dict, features=_hf_features("image"), split="train")
    dataset.to_parquet(parquet_path)
    return parquet_path


def _image_column(encoded_images: dict[int, bytes], sequence_by_row: np.ndarray, rows: int) -> list[dict[str, Any]]:
    if len(sequence_by_row) != rows:
        raise RuntimeError(f"Image sequence row count mismatch: {len(sequence_by_row)} vs {rows}")
    return [
        {
            "bytes": encoded_images[int(sequence_id)],
            "path": f"frame_{row_index:06d}.png",
        }
        for row_index, sequence_id in enumerate(sequence_by_row)
    ]


def _compute_fast_episode_stats(job: _FastEpisodeJob, episode: _FastEpisodeData) -> dict[str, dict[str, np.ndarray]]:
    rows = int(episode.state.shape[0])
    frame_index = np.arange(rows, dtype=np.int64)
    stats_inputs = {
        "observation.state": episode.state,
        "action": episode.action,
        "timestamp": frame_index.astype(np.float32) / np.float32(job.fps),
        "frame_index": frame_index,
        "episode_index": np.full((rows,), job.episode_index, dtype=np.int64),
        "index": np.arange(job.global_start_index, job.global_start_index + rows, dtype=np.int64),
        "task_index": np.full((rows,), job.task_index, dtype=np.int64),
    }
    stats = {}
    for key, array in stats_inputs.items():
        stats[key] = lerobot_compute_stats.get_feature_stats(
            np.asarray(array),
            axis=0,
            keepdims=np.asarray(array).ndim == 1,
        )
    for camera_name in common.CAMERA_NAMES:
        image_key = f"observation.images.{camera_name}"
        sampled = _sample_row_images(episode, camera_name)
        camera_stats = lerobot_compute_stats.get_feature_stats(
            sampled,
            axis=(0, 2, 3),
            keepdims=True,
        )
        stats[image_key] = {
            key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
            for key, value in camera_stats.items()
        }
    return stats


def _sample_row_images(episode: _FastEpisodeData, camera_name: str) -> np.ndarray:
    sampled_indices = lerobot_compute_stats.sample_indices(len(episode.state))
    sampled_images = None
    sequence_by_row = episode.camera_sequence_by_row[camera_name]
    for output_index, row_index in enumerate(sampled_indices):
        sequence_id = int(sequence_by_row[row_index])
        image = episode.images_by_sequence[camera_name][sequence_id].transpose(2, 0, 1)
        image = lerobot_compute_stats.auto_downsample_height_width(image)
        if sampled_images is None:
            sampled_images = np.empty((len(sampled_indices), *image.shape), dtype=np.uint8)
        sampled_images[output_index] = image
    if sampled_images is None:
        raise RuntimeError(f"No sampled images for {camera_name}")
    return sampled_images


def _write_fast_lerobot_metadata(
    *,
    output_root: Path,
    fps: int,
    tasks: list[str],
    results: list[_FastEpisodeResult],
    total_frames: int,
    mode: str,
) -> None:
    features = _lerobot_features_with_defaults(mode)
    info = lerobot_dataset_utils.create_empty_dataset_info(
        CODEBASE_VERSION,
        fps,
        "yam_bimanual",
        features,
        use_videos=False,
    )
    info["total_episodes"] = len(results)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = len(tasks)
    info["total_chunks"] = 1 if results else 0
    info["splits"] = {"train": f"0:{len(results)}"}
    lerobot_dataset_utils.write_info(info, output_root)

    tasks_path = output_root / lerobot_dataset_utils.TASKS_PATH
    episodes_path = output_root / lerobot_dataset_utils.EPISODES_PATH
    episodes_stats_path = output_root / lerobot_dataset_utils.EPISODES_STATS_PATH
    for path in (tasks_path, episodes_path, episodes_stats_path):
        if path.exists():
            path.unlink()
    for task_index, task in enumerate(tasks):
        _append_jsonl(tasks_path, {"task_index": task_index, "task": task})
    for result in results:
        _append_jsonl(episodes_path, result.episode_record)
        lerobot_dataset_utils.write_episode_stats(result.episode_index, result.episode_stats, output_root)
    summary_path = output_root / "conversion_summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()
    for result in results:
        _append_jsonl(summary_path, result.summary_record)


def _publish_parallel_output(scratch_root: Path, output_path: Path, *, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.name}.tmp_{os.getpid()}_{int(time.time())}")
    if tmp_output_path.exists():
        shutil.rmtree(tmp_output_path)
    shutil.copytree(scratch_root, tmp_output_path)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(output_path)
        shutil.rmtree(output_path)
    tmp_output_path.rename(output_path)


def _validate_fast_output(*, repo_id: str, output_root: Path, expected_episodes: int, expected_frames: int) -> None:
    dataset = LeRobotDataset(repo_id, root=output_root)
    if dataset.meta.total_episodes != expected_episodes:
        raise RuntimeError(
            f"Fast conversion produced {dataset.meta.total_episodes} episodes; expected {expected_episodes}"
        )
    if dataset.meta.total_frames != expected_frames:
        raise RuntimeError(f"Fast conversion produced {dataset.meta.total_frames} frames; expected {expected_frames}")
    for episode_index in range(expected_episodes):
        parquet_path = _episode_parquet_path(output_root, episode_index)
        if not parquet_path.exists():
            raise RuntimeError(f"Fast conversion missing episode parquet: {parquet_path}")


def _episode_parquet_path(output_root: Path, episode_index: int) -> Path:
    return output_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def _hf_features(mode: str) -> datasets.Features:
    return lerobot_dataset_utils.get_hf_features_from_features(_lerobot_features_with_defaults(mode))


def _lerobot_features_with_defaults(mode: str) -> dict:
    return {**_yam_lerobot_features(mode), **lerobot_dataset_utils.DEFAULT_FEATURES}


def _yam_lerobot_features(mode: str) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["action"],
        },
        "observation.images.cam_high": _image_feature(mode),
        "observation.images.cam_left_wrist": _image_feature(mode),
        "observation.images.cam_right_wrist": _image_feature(mode),
    }


def _ordered_unique(values) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _set_process_worker_env() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _peak_rss_mb() -> float:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss_kb) / 1024.0


def _malloc_trim() -> None:
    if not hasattr(ctypes, "CDLL"):
        return
    with contextlib.suppress(Exception):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def _episode_dirs(raw_dir: Path, episode_manifest: Path | None) -> list[Path]:
    if episode_manifest is None:
        return data_regression.discover_episodes(raw_dir, source_format="mcap")

    if not episode_manifest.exists():
        raise FileNotFoundError(episode_manifest)

    episode_dirs = []
    seen = set()
    for line_number, line in enumerate(episode_manifest.read_text().splitlines(), start=1):
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        rel_path = Path(entry)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"Invalid episode path on line {line_number}: {entry}")
        if entry in seen:
            raise ValueError(f"Duplicate episode on line {line_number}: {entry}")
        seen.add(entry)
        episode_dir = raw_dir / rel_path
        if not data_regression.discover_episodes(episode_dir, source_format="mcap"):
            raise FileNotFoundError(f"Manifest-listed episode is not a canonical MCAP YAM episode: {episode_dir}")
        episode_dirs.append(episode_dir)
    return episode_dirs


def _validate_episode(episode_dir: Path, args: Args) -> dict:
    episode = data_regression.load_canonical_episode(episode_dir, source_format="mcap")
    states = episode.state
    actions = episode.action
    if states.shape != actions.shape or states.shape[-1] != 14:
        raise RuntimeError(f"Bad state/action shapes in {episode_dir}: {states.shape}, {actions.shape}")
    if episode.fps <= 0:
        raise RuntimeError(f"Bad row fps in {episode_dir}: {episode.fps}")
    for camera_name in common.CAMERA_NAMES:
        if episode.image_counts.get(camera_name, 0) != len(states):
            raise RuntimeError(
                f"Image count mismatch for {camera_name} in {episode_dir}: "
                f"{episode.image_counts.get(camera_name, 0)} vs {len(states)}"
            )
    summary = data_regression.summarize_episode(
        episode,
        max_median_dt_error_ratio=args.max_median_dt_error_ratio,
        max_p95_dt_error_ratio=args.max_p95_dt_error_ratio,
    )
    timing_warnings = [warning for warning in summary["warnings"] if warning.startswith("timestamp ")]
    if timing_warnings:
        detail = "; ".join(timing_warnings)
        raise RuntimeError(f"Timing mismatch in {episode_dir}: {detail}. Fix collection FPS before conversion.")
    fatal_warnings = [
        warning
        for warning in summary["warnings"]
        if "contains non-finite" in warning or "gripper values outside [0, 1]" in warning
    ]
    if fatal_warnings:
        detail = "; ".join(fatal_warnings)
        raise RuntimeError(f"Invalid values in {episode_dir}: {detail}")
    return _conversion_summary_record(episode, summary)


def _load_episode_for_conversion(episode_dir: Path) -> tuple[str, np.ndarray, np.ndarray, object]:
    from examples.yam_real import mcap_episode

    episode = mcap_episode.read_episode(episode_dir, decode_images=True)
    if episode.images is None:
        raise RuntimeError(f"MCAP episode did not decode images: {episode_dir}")
    _validate_image_arrays(episode_dir, episode.images, rows=len(episode.state))
    return (
        episode.task,
        np.asarray(episode.state, dtype=np.float32),
        np.asarray(episode.action, dtype=np.float32),
        episode.images,
    )


def _load_images_for_frame(episode_dir: Path, images_or_paths: object, idx: int) -> dict[str, np.ndarray]:
    if not isinstance(images_or_paths, dict):
        raise RuntimeError(f"YAM conversion only supports canonical MCAP decoded images: {episode_dir}")
    return {
        "observation.images.cam_high": _hwc_rgb_to_chw(images_or_paths["cam_high"][idx]),
        "observation.images.cam_left_wrist": _hwc_rgb_to_chw(images_or_paths["cam_left_wrist"][idx]),
        "observation.images.cam_right_wrist": _hwc_rgb_to_chw(images_or_paths["cam_right_wrist"][idx]),
    }


def _image_feature(mode: str) -> dict:
    return {
        "dtype": mode,
        "shape": (3, 480, 640),
        "names": ["channel", "height", "width"],
    }


def _hwc_rgb_to_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise RuntimeError(f"Expected HWC uint8 RGB image, got shape={image.shape}, dtype={image.dtype}")
    return np.transpose(image, (2, 0, 1))


def _validate_image_arrays(episode_dir: Path, images: dict[str, np.ndarray], *, rows: int) -> None:
    for camera_name in common.CAMERA_NAMES:
        if camera_name not in images:
            raise RuntimeError(f"MCAP episode {episode_dir} is missing decoded images for {camera_name}")
        camera_images = np.asarray(images[camera_name])
        if camera_images.shape[:1] != (rows,):
            raise RuntimeError(
                f"MCAP image row count mismatch for {camera_name} in {episode_dir}: "
                f"{camera_images.shape[:1]} vs {(rows,)}"
            )
        if camera_images.dtype != np.uint8 or camera_images.ndim != 4 or camera_images.shape[-1] != 3:
            raise RuntimeError(
                f"MCAP images for {camera_name} in {episode_dir} must be NHWC uint8 RGB, "
                f"got shape={camera_images.shape}, dtype={camera_images.dtype}"
            )


def _conversion_summary_record(episode: data_regression.CanonicalEpisode, summary: dict) -> dict:
    return {
        "source_path": str(episode.source_path),
        "source_format": episode.source_format,
        "task": episode.task,
        "rows": episode.num_frames,
        "row_fps": episode.row_fps or episode.fps,
        "camera_fps": {
            camera_name: camera_summary.get("median_fps", 0.0)
            for camera_name, camera_summary in summary.get("camera_timing", {}).items()
        },
        "camera_frame_reuse": episode.camera_frame_reuse,
        "warnings": summary["warnings"],
    }


def _append_conversion_summary(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
