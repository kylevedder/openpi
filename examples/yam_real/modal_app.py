from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys

import modal

APP_NAME = "yam-openpi"
OPENPI_ROOT = "/root/openpi"
VOLUME_ROOT = "/mnt/yam"
CONFIG_NAME = "pi05_yam_bimanual_50hz_jpeg_q85"
DEFAULT_EXP_NAME = "yam_bimanual_50hz_jpeg_q85_latest"
SERVE_CHECKPOINT_STEP = 2999
QUIC_REGIONS = ["us-west-1", "westus"]
QUIC_SCALEDOWN_WINDOW_S = 20 * 60
NORM_STATS_CPUS = 16.0
NORM_STATS_NUM_WORKERS = 16
H100_TRAIN_GPUS = "H100:8"
H100_TRAIN_CPUS = 64.0
H100_TRAIN_NUM_WORKERS = 32
H100_TRAIN_FSDP_DEVICES = 8
A100_TRAIN_GPUS = "A100-80GB:8"
A100_TRAIN_CPUS = H100_TRAIN_CPUS
A100_TRAIN_NUM_WORKERS = H100_TRAIN_NUM_WORKERS
A100_TRAIN_FSDP_DEVICES = H100_TRAIN_FSDP_DEVICES
A100_FALLBACK_GPUS = "A100-80GB:4"
A100_FALLBACK_CPUS = 32.0
A100_FALLBACK_NUM_WORKERS = 16
A100_FALLBACK_FSDP_DEVICES = 4
DEFAULT_MODAL_RAW_DIR = f"{VOLUME_ROOT}/raw/yam_data/raw"
DEFAULT_MODAL_EPISODE_MANIFEST = (
    f"{VOLUME_ROOT}/raw/yam_data/manifests/pi05_yam_bimanual_50hz_latest.txt"
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("yam-openpi", create_if_missing=True)


def _openpi_source_root() -> Path:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "openpi").is_dir():
            return candidate
    container_root = Path(OPENPI_ROOT)
    if container_root.exists():
        return container_root
    raise RuntimeError("Could not locate the OpenPI repo root")


def _ensure_openpi_importable() -> None:
    venv_site_packages = (
        Path(OPENPI_ROOT)
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    import_paths = [
        Path(OPENPI_ROOT),
        Path(OPENPI_ROOT) / "src",
        Path(OPENPI_ROOT) / "packages" / "openpi-client" / "src",
        venv_site_packages,
    ]
    for import_path in reversed(import_paths):
        if import_path.exists() and str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .env({"HF_LEROBOT_HOME": f"{VOLUME_ROOT}/lerobot", "OPENPI_DATA_HOME": f"{VOLUME_ROOT}/openpi_cache"})
    .add_local_dir(
        _openpi_source_root(),
        remote_path=OPENPI_ROOT,
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            ".ruff_cache/**",
            ".venv",
            ".venv/**",
            "__pycache__",
            "**/__pycache__/**",
            "assets/**",
            "camera_dumps/**",
            "checkpoints/**",
            "yam_data/**",
        ],
    )
    .run_commands(f"cd {OPENPI_ROOT} && GIT_LFS_SKIP_SMUDGE=1 uv sync")
)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=NORM_STATS_CPUS,
    memory=65536,
    ephemeral_disk=524288,
    timeout=60 * 60,
)
def compute_norm_stats(
    config_name: str = CONFIG_NAME,
    max_frames: int | None = None,
    batch_size: int | None = None,
    num_workers: int = NORM_STATS_NUM_WORKERS,
    log_system_stats: bool = True,  # noqa: FBT001, FBT002
    system_stats_interval_s: float = 10.0,
) -> None:
    _prepare_volume_paths()
    cmd = ["uv", "run", "scripts/compute_norm_stats.py", "--config-name", config_name]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    cmd.extend(["--num-workers", str(num_workers)])
    if log_system_stats:
        cmd.append("--log-system-stats")
        cmd.extend(["--system-stats-interval-s", str(system_stats_interval_s)])
    subprocess.run(cmd, cwd=OPENPI_ROOT, check=True)
    volume.commit()


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=16.0,
    memory=65536,
    ephemeral_disk=524288,
    timeout=6 * 60 * 60,
)
def preprocess_yam_mcap_to_lerobot(
    raw_dir: str = DEFAULT_MODAL_RAW_DIR,
    episode_manifest: str = DEFAULT_MODAL_EPISODE_MANIFEST,
    repo_id: str = "local/yam_bimanual",
    mode: str = "image",
    episode_processes: int = 8,
    scratch_dir: str = "/tmp/yam_lerobot_preprocess",
    profile_jsonl: str = "/tmp/yam_lerobot_preprocess/profile.jsonl",
    overwrite: bool = True,  # noqa: FBT001, FBT002
    resume: bool = False,  # noqa: FBT001, FBT002
) -> None:
    _prepare_volume_paths()
    cmd = _preprocess_yam_mcap_to_lerobot_cmd(
        raw_dir=raw_dir,
        episode_manifest=episode_manifest,
        repo_id=repo_id,
        mode=mode,
        episode_processes=episode_processes,
        scratch_dir=scratch_dir,
        profile_jsonl=profile_jsonl,
        overwrite=overwrite,
        resume=resume,
    )
    subprocess.run(cmd, cwd=OPENPI_ROOT, check=True)
    volume.commit()


@app.function(image=image, gpu="A100-80GB", volumes={VOLUME_ROOT: volume}, timeout=24 * 60 * 60)
def train(
    config_name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    num_train_steps: int | None = None,
    batch_size: int | None = None,
    tracking_backend: str = "trackio",
    overwrite: bool = False,  # noqa: FBT001, FBT002
    wandb_enabled: bool = False,  # noqa: FBT001, FBT002
) -> None:
    _prepare_volume_paths()
    cmd = _train_cmd(
        config_name=config_name,
        exp_name=exp_name,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        tracking_backend=tracking_backend,
        overwrite=overwrite,
        wandb_enabled=wandb_enabled,
    )
    subprocess.run(cmd, cwd=OPENPI_ROOT, check=True)
    volume.commit()


@app.function(image=image, gpu="A100-80GB:2", volumes={VOLUME_ROOT: volume}, timeout=24 * 60 * 60)
def train_fsdp2(
    config_name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    num_train_steps: int | None = None,
    batch_size: int | None = None,
    tracking_backend: str = "trackio",
    overwrite: bool = False,  # noqa: FBT001, FBT002
    wandb_enabled: bool = False,  # noqa: FBT001, FBT002
) -> None:
    _prepare_volume_paths()
    cmd = _train_cmd(
        config_name=config_name,
        exp_name=exp_name,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        tracking_backend=tracking_backend,
        overwrite=overwrite,
        wandb_enabled=wandb_enabled,
        fsdp_devices=2,
    )
    subprocess.run(cmd, cwd=OPENPI_ROOT, check=True)
    volume.commit()


@app.function(image=image, gpu=H100_TRAIN_GPUS, cpu=H100_TRAIN_CPUS, volumes={VOLUME_ROOT: volume}, timeout=24 * 60 * 60)
def train_fsdp8_h100(
    config_name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    num_train_steps: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = H100_TRAIN_NUM_WORKERS,
    tracking_backend: str = "trackio",
    overwrite: bool = False,  # noqa: FBT001, FBT002
    wandb_enabled: bool = False,  # noqa: FBT001, FBT002
) -> None:
    _prepare_volume_paths()
    cmd = _train_cmd(
        config_name=config_name,
        exp_name=exp_name,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        tracking_backend=tracking_backend,
        overwrite=overwrite,
        wandb_enabled=wandb_enabled,
        fsdp_devices=H100_TRAIN_FSDP_DEVICES,
    )
    subprocess.run(cmd, cwd=OPENPI_ROOT, check=True)
    volume.commit()


@app.function(image=image, gpu=A100_TRAIN_GPUS, cpu=A100_TRAIN_CPUS, volumes={VOLUME_ROOT: volume}, timeout=24 * 60 * 60)
def train_fsdp8_a100(
    config_name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    num_train_steps: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = A100_TRAIN_NUM_WORKERS,
    tracking_backend: str = "trackio",
    overwrite: bool = False,  # noqa: FBT001, FBT002
    wandb_enabled: bool = False,  # noqa: FBT001, FBT002
) -> None:
    _prepare_volume_paths()
    cmd = _train_cmd(
        config_name=config_name,
        exp_name=exp_name,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        tracking_backend=tracking_backend,
        overwrite=overwrite,
        wandb_enabled=wandb_enabled,
        fsdp_devices=A100_TRAIN_FSDP_DEVICES,
    )
    subprocess.run(cmd, cwd=OPENPI_ROOT, check=True)
    volume.commit()


@app.function(
    image=image,
    gpu=A100_FALLBACK_GPUS,
    cpu=A100_FALLBACK_CPUS,
    volumes={VOLUME_ROOT: volume},
    timeout=24 * 60 * 60,
)
def train_fsdp4_a100(
    config_name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    num_train_steps: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = A100_FALLBACK_NUM_WORKERS,
    tracking_backend: str = "trackio",
    overwrite: bool = False,  # noqa: FBT001, FBT002
    wandb_enabled: bool = False,  # noqa: FBT001, FBT002
) -> None:
    _prepare_volume_paths()
    cmd = _train_cmd(
        config_name=config_name,
        exp_name=exp_name,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        tracking_backend=tracking_backend,
        overwrite=overwrite,
        wandb_enabled=wandb_enabled,
        fsdp_devices=A100_FALLBACK_FSDP_DEVICES,
    )
    subprocess.run(cmd, cwd=OPENPI_ROOT, check=True)
    volume.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={VOLUME_ROOT: volume},
    timeout=24 * 60 * 60,
)
@modal.web_server(8000, startup_timeout=30 * 60)
def serve_policy() -> None:
    _prepare_volume_paths()
    checkpoint_dir = f"checkpoints/{CONFIG_NAME}/{DEFAULT_EXP_NAME}/{SERVE_CHECKPOINT_STEP}"
    checkpoint_path = Path(OPENPI_ROOT) / checkpoint_dir
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
    if not (checkpoint_path / "params").exists():
        raise FileNotFoundError(f"Checkpoint params directory does not exist: {checkpoint_path / 'params'}")
    cmd = [
        "uv",
        "run",
        "scripts/serve_policy.py",
        "--port=8000",
        "policy:checkpoint",
        f"--policy.config={CONFIG_NAME}",
        f"--policy.dir={checkpoint_dir}",
    ]
    subprocess.Popen(cmd, cwd=OPENPI_ROOT)


@app.cls(
    image=image,
    gpu="A100-40GB",
    volumes={VOLUME_ROOT: volume},
    timeout=24 * 60 * 60,
    scaledown_window=QUIC_SCALEDOWN_WINDOW_S,
    region=QUIC_REGIONS,
    experimental_options={"region_ranking_enabled": True},
    max_containers=3,
)
class YamQuicPolicyServer:
    @modal.enter()
    def enter(self) -> None:
        logging.basicConfig(level=logging.INFO, force=True)
        _ensure_openpi_importable()
        _prepare_volume_paths()
        checkpoint_dir = f"checkpoints/{CONFIG_NAME}/{DEFAULT_EXP_NAME}/{SERVE_CHECKPOINT_STEP}"
        checkpoint_path = Path(OPENPI_ROOT) / checkpoint_dir
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
        if not (checkpoint_path / "params").exists():
            raise FileNotFoundError(f"Checkpoint params directory does not exist: {checkpoint_path / 'params'}")

        from openpi.policies import policy_config
        from openpi.training import config as training_config

        logging.info("Loading YAM QUIC policy from %s", checkpoint_dir)
        self.policy = policy_config.create_trained_policy(
            training_config.get_config(CONFIG_NAME),
            str(checkpoint_path),
        )
        self.metadata = dict(self.policy.metadata)
        self.metadata.update(
            {
                "transport": "modal-quic",
                "modal_app": APP_NAME,
                "modal_class": type(self).__name__,
                "regions": QUIC_REGIONS,
            }
        )
        logging.info("YAM QUIC policy loaded")

    @modal.method()
    def serve(self, rendezvous: modal.Dict) -> None:
        _ensure_openpi_importable()
        from openpi.serving import modal_quic

        modal_quic.serve_policy(
            rendezvous=rendezvous,
            policy=self.policy,
            metadata=self.metadata,
        )


def _train_cmd(
    *,
    config_name: str,
    exp_name: str,
    num_train_steps: int | None,
    batch_size: int | None,
    tracking_backend: str,
    overwrite: bool,
    wandb_enabled: bool,
    fsdp_devices: int | None = None,
    num_workers: int | None = None,
) -> list[str]:
    cmd = ["uv", "run", "scripts/train.py", config_name, f"--exp-name={exp_name}"]
    cmd.extend(["--tracking-backend", tracking_backend])
    if overwrite:
        cmd.append("--overwrite")
    if not wandb_enabled:
        cmd.append("--no-wandb-enabled")
    if num_train_steps is not None:
        cmd.extend(["--num-train-steps", str(num_train_steps)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if fsdp_devices is not None:
        cmd.extend(["--fsdp-devices", str(fsdp_devices)])
    if num_workers is not None:
        cmd.extend(["--num-workers", str(num_workers)])
    return cmd


def _preprocess_yam_mcap_to_lerobot_cmd(
    *,
    raw_dir: str,
    episode_manifest: str | None,
    repo_id: str,
    mode: str,
    episode_processes: int,
    scratch_dir: str,
    profile_jsonl: str,
    overwrite: bool,
    resume: bool = False,
) -> list[str]:
    if mode not in {"image", "video"}:
        raise ValueError(f"Unsupported conversion mode: {mode!r}")
    if episode_processes < 1:
        raise ValueError("preprocess_yam_mcap_to_lerobot requires at least one episode process")
    if overwrite and resume:
        raise ValueError("preprocess_yam_mcap_to_lerobot cannot use overwrite and resume together")

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "examples.yam_real.convert_yam_data_to_lerobot",
        "--raw-dir",
        raw_dir,
        "--repo-id",
        repo_id,
        "--mode",
        mode,
        "--episode-processes",
        str(episode_processes),
        "--scratch-dir",
        scratch_dir,
        "--profile-jsonl",
        profile_jsonl,
    ]
    if episode_manifest:
        cmd.extend(["--episode-manifest", episode_manifest])
    if overwrite:
        cmd.append("--overwrite")
    if resume:
        cmd.append("--resume")
    return cmd


def _prepare_volume_paths() -> None:
    for name in ("assets", "checkpoints"):
        target = Path(VOLUME_ROOT) / name
        link = Path(OPENPI_ROOT) / name
        target.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            continue
        if link.exists():
            raise RuntimeError(f"{link} exists and is not a symlink; refusing to hide it")
        link.symlink_to(target, target_is_directory=True)
    (Path(VOLUME_ROOT) / "lerobot").mkdir(parents=True, exist_ok=True)
    (Path(VOLUME_ROOT) / "openpi_cache").mkdir(parents=True, exist_ok=True)
