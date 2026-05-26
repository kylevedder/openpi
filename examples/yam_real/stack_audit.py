from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import cv2
import numpy as np
import tyro

from examples.yam_real import common
from examples.yam_real import data_regression
from openpi.shared import jpeg_transport
from openpi.training import config as training_config

CONFIG_NAME = "pi05_yam_bimanual_50hz_jpeg_q85"
DEFAULT_MANIFESTS = (
    Path("yam_data/manifests/pi05_yam_bimanual_50hz_20demo.txt"),
    Path("yam_data/manifests/pi05_yam_bimanual_50hz_moved_robot_20demo_20260525.txt"),
    Path("yam_data/manifests/pi05_yam_bimanual_50hz_recollect_20260525.txt"),
)


@dataclasses.dataclass(frozen=True)
class Args:
    old_good_dir: Path = Path("yam_data/archive/raw_before_moved_robot_20260525_145623")
    moved_robot_dir: Path = Path("yam_data/archive/raw_before_recollect_20260525_172213")
    current_raw_dir: Path = Path("yam_data/raw")
    manifests: tuple[Path, ...] = DEFAULT_MANIFESTS
    output_dir: Path = Path("yam_data/logs/stack_audit")
    config_name: str = CONFIG_NAME
    strict: bool = True
    sample_images: bool = True
    image_sample_episodes_per_group: int = 2
    include_live_device_checks: bool = False
    max_median_dt_error_ratio: float = 0.25
    max_p95_dt_error_ratio: float = 2.5
    max_arm_abs_rad: float = 3.2
    max_arm_step_rad: float = 1.25


def main(args: Args) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parent
    groups = {
        "old_good_20260515": args.old_good_dir,
        "moved_robot_20260525": args.moved_robot_dir,
        "current_recollect_20260525": args.current_raw_dir,
    }

    issues: list[dict[str, Any]] = []
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "provenance": {
            "openpi": _git_provenance(repo_root),
            "i2rt": _git_provenance(workspace_root / "i2rt"),
        },
        "training_config": _training_config_summary(args.config_name, issues),
        "modal_app": _modal_app_summary(repo_root / "examples" / "yam_real" / "modal_app.py", issues),
        "udev": _udev_summary(workspace_root / "yam_udev", issues, include_live=args.include_live_device_checks),
        "episode_groups": {},
        "manifests": [],
        "commands": _recommended_commands(args.config_name),
        "issues": issues,
    }

    for group_name, group_dir in groups.items():
        group_report = _episode_group_summary(group_name, group_dir, args, issues)
        if args.sample_images:
            group_report["image_samples"] = _sample_group_images(
                group_dir,
                max_episodes=args.image_sample_episodes_per_group,
                issues=issues,
                group_name=group_name,
            )
        report["episode_groups"][group_name] = group_report

    for manifest_path in args.manifests:
        report["manifests"].append(_manifest_summary(manifest_path, groups, issues))

    _write_report(report, args.output_dir)
    blocker_count = sum(1 for issue in issues if issue["severity"] == "blocker")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    print(f"YAM stack audit complete: {blocker_count} blocker(s), {warning_count} warning(s)")
    print(f"Report directory: {args.output_dir}")
    for issue in issues:
        print(f"[{issue['severity']}] {issue['component']}: {issue['message']}")

    if args.strict and blocker_count:
        raise RuntimeError(f"YAM stack audit found {blocker_count} blocker(s); see {args.output_dir}")


def _episode_group_summary(group_name: str, group_dir: Path, args: Args, issues: list[dict[str, Any]]) -> dict:
    episode_dirs = data_regression.discover_episodes(group_dir)
    if not episode_dirs:
        _add_issue(issues, "blocker", group_name, f"no YAM episodes found under {group_dir}")
        return {"path": str(group_dir), "episodes": 0, "records": [], "aggregate": {}}

    records = []
    for episode_dir in episode_dirs:
        episode = data_regression.load_canonical_episode(episode_dir)
        record = data_regression.summarize_episode(
            episode,
            max_arm_abs_rad=args.max_arm_abs_rad,
            max_arm_step_rad=args.max_arm_step_rad,
            max_median_dt_error_ratio=args.max_median_dt_error_ratio,
            max_p95_dt_error_ratio=args.max_p95_dt_error_ratio,
        )
        records.append(record)
        for warning in record["warnings"]:
            severity = "blocker" if warning.startswith("timestamp median dt") else "warning"
            _add_issue(issues, severity, group_name, f"{Path(record['source_path']).name}: {warning}")

    return {
        "path": str(group_dir),
        "episodes": len(records),
        "records": records,
        "aggregate": _aggregate_records(records),
    }


def _aggregate_records(records: list[dict]) -> dict:
    if not records:
        return {}
    frame_counts = np.asarray([record["num_frames"] for record in records], dtype=np.int64)
    median_dt = np.asarray([record["timing"]["median_dt_s"] for record in records], dtype=np.float64)
    median_fps = np.asarray([record["timing"]["median_fps"] for record in records], dtype=np.float64)
    left_ranges = np.asarray(
        [record["gripper"]["action"]["left_gripper"]["range"] for record in records], dtype=np.float64
    )
    right_ranges = np.asarray(
        [record["gripper"]["action"]["right_gripper"]["range"] for record in records], dtype=np.float64
    )
    return {
        "frames_total": int(np.sum(frame_counts)),
        "frame_min": int(np.min(frame_counts)),
        "frame_median": int(np.median(frame_counts)),
        "frame_max": int(np.max(frame_counts)),
        "median_dt_s_median": float(np.median(median_dt)),
        "median_fps_median": float(np.median(median_fps)),
        "left_action_gripper_range_min": float(np.min(left_ranges)),
        "left_action_gripper_range_median": float(np.median(left_ranges)),
        "right_action_gripper_range_min": float(np.min(right_ranges)),
        "right_action_gripper_range_median": float(np.median(right_ranges)),
        "warning_count": sum(len(record["warnings"]) for record in records),
    }


def _manifest_summary(manifest_path: Path, groups: dict[str, Path], issues: list[dict[str, Any]]) -> dict:
    if not manifest_path.exists():
        _add_issue(issues, "blocker", "manifest", f"missing manifest {manifest_path}")
        return {"path": str(manifest_path), "exists": False}

    header, entries = _read_manifest(manifest_path)
    resolved = []
    missing = []
    frames_total = 0
    for entry in entries:
        episode_dir = _resolve_manifest_entry(entry, groups)
        if episode_dir is None:
            missing.append(entry)
            continue
        episode = data_regression.load_canonical_episode(episode_dir)
        frames_total += episode.num_frames
        resolved.append({"entry": entry, "path": str(episode_dir), "frames": episode.num_frames})

    expected_episodes = _optional_int(header.get("selected_episodes"))
    expected_frames = _optional_int(header.get("selected_frames"))
    if missing:
        _add_issue(issues, "blocker", "manifest", f"{manifest_path} has {len(missing)} missing episode(s)")
    if expected_episodes is not None and expected_episodes != len(entries):
        _add_issue(
            issues,
            "blocker",
            "manifest",
            f"{manifest_path} header selected_episodes={expected_episodes} but lists {len(entries)} entries",
        )
    if expected_frames is not None and expected_frames != frames_total:
        _add_issue(
            issues,
            "blocker",
            "manifest",
            f"{manifest_path} header selected_frames={expected_frames} but resolved frames={frames_total}",
        )
    return {
        "path": str(manifest_path),
        "exists": True,
        "header": header,
        "entries": entries,
        "resolved": resolved,
        "missing": missing,
        "resolved_frames": frames_total,
    }


def _read_manifest(path: Path) -> tuple[dict[str, str], list[str]]:
    header = {}
    entries = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            match = re.match(r"#\s*([^:]+):\s*(.*)$", stripped)
            if match:
                header[match.group(1).strip()] = match.group(2).strip()
            continue
        entries.append(stripped)
    return header, entries


def _resolve_manifest_entry(entry: str, groups: dict[str, Path]) -> Path | None:
    rel_path = Path(entry)
    for group_dir in groups.values():
        candidate = group_dir / rel_path
        if (candidate / "episode.npz").exists() or any(candidate.glob("episode_part*.mcap")):
            return candidate
    return None


def _sample_group_images(
    group_dir: Path,
    *,
    max_episodes: int,
    issues: list[dict[str, Any]],
    group_name: str,
) -> list[dict]:
    samples = []
    for episode_dir in data_regression.discover_episodes(group_dir)[:max_episodes]:
        manifest, arrays = common.load_episode(episode_dir)
        image_paths = arrays.get("image_paths")
        if image_paths is None or len(image_paths) == 0:
            _add_issue(issues, "warning", group_name, f"{episode_dir.name}: no image_paths to sample")
            continue
        indexes = sorted({0, len(image_paths) // 2, len(image_paths) - 1})
        for idx in indexes:
            paths = image_paths[idx].item() if hasattr(image_paths[idx], "item") else image_paths[idx]
            for camera_name in common.CAMERA_NAMES:
                image_path = episode_dir / paths[camera_name]
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    _add_issue(issues, "blocker", group_name, f"failed to read {image_path}")
                    continue
                stats = {
                    "episode": episode_dir.name,
                    "frame_index": idx,
                    "camera": camera_name,
                    "path": str(image_path),
                    "shape": list(image.shape),
                    "mean_bgr": np.mean(image, axis=(0, 1)).astype(float).tolist(),
                    "std": float(np.std(image)),
                }
                if image.ndim != 3 or image.shape[2] != 3:
                    _add_issue(issues, "blocker", group_name, f"{image_path} has bad image shape {image.shape}")
                if stats["std"] < 1.0:
                    _add_issue(issues, "warning", group_name, f"{image_path} appears nearly blank")
                samples.append(stats)
        _ = manifest
    return samples


def _training_config_summary(config_name: str, issues: list[dict[str, Any]]) -> dict:
    train_config = training_config.get_config(config_name)
    metadata = dict(train_config.policy_metadata or {})
    expected_metadata = {
        "action_space": common.ACTION_SPACE,
        "gripper_convention": common.GRIPPER_CONVENTION,
        "state_order": list(common.STATE_ORDER),
        "action_horizon": 50,
        "fps": 50.0,
        "image_transport": jpeg_transport.IMAGE_TRANSPORT,
        "jpeg_quality": jpeg_transport.JPEG_QUALITY,
        "image_resolution": list(jpeg_transport.IMAGE_RESOLUTION),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            _add_issue(
                issues,
                "blocker",
                "training_config",
                f"{config_name} policy_metadata[{key!r}]={metadata.get(key)!r}, expected {expected!r}",
            )

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.repo_id != "local/yam_bimanual":
        _add_issue(
            issues,
            "blocker",
            "training_config",
            f"{config_name} repo_id={data_config.repo_id!r}, expected 'local/yam_bimanual'",
        )

    return {
        "name": train_config.name,
        "model_action_horizon": train_config.model.action_horizon,
        "model_action_dim": train_config.model.action_dim,
        "batch_size": train_config.batch_size,
        "num_train_steps": train_config.num_train_steps,
        "data_repo_id": data_config.repo_id,
        "assets_dirs": str(train_config.assets_dirs),
        "policy_metadata": metadata,
    }


def _modal_app_summary(path: Path, issues: list[dict[str, Any]]) -> dict:
    if not path.exists():
        _add_issue(issues, "blocker", "modal_app", f"missing {path}")
        return {"path": str(path), "exists": False}
    constants = _literal_assignments(path)
    required = ("APP_NAME", "CONFIG_NAME", "DEFAULT_EXP_NAME", "SERVE_CHECKPOINT_STEP")
    for name in required:
        if name not in constants:
            _add_issue(issues, "blocker", "modal_app", f"{path} does not define {name}")
    if constants.get("CONFIG_NAME") != CONFIG_NAME:
        _add_issue(
            issues,
            "blocker",
            "modal_app",
            f"CONFIG_NAME={constants.get('CONFIG_NAME')!r}, expected {CONFIG_NAME!r}",
        )
    checkpoint_dir = None
    if all(name in constants for name in ("CONFIG_NAME", "DEFAULT_EXP_NAME", "SERVE_CHECKPOINT_STEP")):
        checkpoint_dir = (
            Path("checkpoints")
            / str(constants["CONFIG_NAME"])
            / str(constants["DEFAULT_EXP_NAME"])
            / str(constants["SERVE_CHECKPOINT_STEP"])
        )
    return {"path": str(path), "exists": True, "constants": constants, "checkpoint_dir": str(checkpoint_dir)}


def _udev_summary(root: Path, issues: list[dict[str, Any]], *, include_live: bool) -> dict:
    camera_rules = root / "99-yam-cameras.rules"
    can_rules = root / "99-yam-can.rules"
    cameras = _parse_udev_names(camera_rules, key="SYMLINK", pattern=r'SYMLINK\+="yam/([^"]+)"')
    can = _parse_udev_names(can_rules, key="NAME", pattern=r'NAME="([^"]+)"')

    expected_cameras = {Path(path).name for path in common.CAMERA_PATHS.values()}
    actual_cameras = set(cameras)
    if actual_cameras != expected_cameras:
        _add_issue(
            issues,
            "blocker",
            "udev",
            f"camera udev names {sorted(actual_cameras)} do not match expected {sorted(expected_cameras)}",
        )

    expected_can = set(common.LEADER_CHANNELS.values()) | set(common.FOLLOWER_CHANNELS.values())
    actual_can = set(can)
    if actual_can != expected_can:
        _add_issue(
            issues,
            "blocker",
            "udev",
            f"CAN udev names {sorted(actual_can)} do not match expected {sorted(expected_can)}",
        )

    live = {}
    if include_live:
        live = _live_device_summary(expected_cameras=expected_cameras, expected_can=expected_can, issues=issues)
    return {
        "root": str(root),
        "camera_rules": str(camera_rules),
        "can_rules": str(can_rules),
        "cameras": cameras,
        "can": can,
        "live": live,
    }


def _parse_udev_names(path: Path, *, key: str, pattern: str) -> dict[str, str]:
    if not path.exists():
        return {}
    names = {}
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        name_match = re.search(pattern, line)
        serial_match = re.search(r'ATTRS?\{serial\}=="([^"]+)"', line)
        if name_match:
            names[name_match.group(1)] = serial_match.group(1) if serial_match else ""
        _ = key
    return names


def _live_device_summary(
    *,
    expected_cameras: set[str],
    expected_can: set[str],
    issues: list[dict[str, Any]],
) -> dict:
    camera_exists = {name: (Path("/dev/yam") / name).exists() for name in sorted(expected_cameras)}
    for name, exists in camera_exists.items():
        if not exists:
            _add_issue(issues, "blocker", "live_devices", f"missing /dev/yam/{name}")

    proc = _run(["ip", "-details", "-br", "link", "show", "type", "can"], cwd=Path.cwd())
    can_output = proc["stdout"]
    present_can = {line.split()[0] for line in can_output.splitlines() if line.split()}
    for name in sorted(expected_can - present_can):
        _add_issue(issues, "blocker", "live_devices", f"missing CAN interface {name}")
    return {"camera_exists": camera_exists, "can_interfaces": sorted(present_can), "ip_output": can_output}


def _git_provenance(path: Path) -> dict:
    return {
        "path": str(path),
        "head": _run(["git", "rev-parse", "HEAD"], cwd=path)["stdout"].strip(),
        "branch": _run(["git", "branch", "--show-current"], cwd=path)["stdout"].strip(),
        "status_short": _run(["git", "status", "--short"], cwd=path)["stdout"].splitlines(),
    }


def _literal_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return values


def _write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"yam_stack_audit_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    latest_json = output_dir / "latest.json"
    latest_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        "# YAM Stack Audit",
        "",
        f"- Created: {report['created_at']}",
        f"- OpenPI HEAD: {report['provenance']['openpi']['head']}",
        f"- i2rt HEAD: {report['provenance']['i2rt']['head']}",
        f"- Issues: {len(report['issues'])}",
        "",
        "## Episode Groups",
    ]
    for name, group in report["episode_groups"].items():
        aggregate = group.get("aggregate", {})
        lines.append(
            f"- {name}: episodes={group.get('episodes', 0)}, frames={aggregate.get('frames_total', 0)}, "
            f"median_fps={aggregate.get('median_fps_median', 0):.2f}, warnings={aggregate.get('warning_count', 0)}"
        )
    lines.extend(["", "## Issues"])
    lines.extend(f"- [{issue['severity']}] {issue['component']}: {issue['message']}" for issue in report["issues"])
    markdown_path = output_dir / "latest.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def _recommended_commands(config_name: str) -> list[str]:
    return [
        "uv run pytest src/openpi/policies/yam_policy_test.py "
        "src/openpi/policies/yam_data_regression_test.py "
        "src/openpi/shared/jpeg_transport_test.py "
        "src/openpi/shared/image_tools_test.py "
        "src/openpi/shared/normalize_test.py "
        "src/openpi/training/data_loader_test.py",
        "cd /home/pi-sj/code/i2rt && uv run pytest i2rt/robots/tests/test_robot_variants.py "
        "i2rt/robots/tests/test_gravity_comp.py",
        "uv run python -m examples.yam_real.record_episode --startup-check-only --fps 50",
        "uv run python -m examples.yam_real.read_yam_encoders --role follower --side both",
        f"uv run python scripts/compute_norm_stats.py --config-name {config_name} --max-frames 512",
        "uv run python -m examples.yam_real.run_policy --transport modal-quic --modal-app-name yam-openpi "
        '--modal-class-name YamQuicPolicyServer --prompt "pick up the object and place it in the target area" '
        "--max-steps 40 --fps 50 --action-horizon 50",
    ]


def _run(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    component: str,
    message: str,
    **extra: Any,
) -> None:
    issues.append({"severity": severity, "component": component, "message": message, **extra})


if __name__ == "__main__":
    main(tyro.cli(Args))
