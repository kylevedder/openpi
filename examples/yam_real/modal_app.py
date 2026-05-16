from __future__ import annotations

from pathlib import Path
import subprocess

import modal

APP_NAME = "yam-openpi"
OPENPI_ROOT = "/root/openpi"
VOLUME_ROOT = "/mnt/yam"
CONFIG_NAME = "pi05_yam_bimanual_50hz_jpeg_q85"
DEFAULT_EXP_NAME = "yam_bimanual_50hz_jpeg_q85_20demo_v1"
SERVE_CHECKPOINT_STEP = 999

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


@app.function(image=image, volumes={VOLUME_ROOT: volume}, timeout=60 * 60)
def compute_norm_stats(config_name: str = CONFIG_NAME, max_frames: int | None = None) -> None:
    _prepare_volume_paths()
    cmd = ["uv", "run", "scripts/compute_norm_stats.py", "--config-name", config_name]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    subprocess.run(cmd, cwd=OPENPI_ROOT, check=True)
    volume.commit()


@app.function(image=image, gpu="A100-80GB", volumes={VOLUME_ROOT: volume}, timeout=24 * 60 * 60)
def train(
    config_name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    num_train_steps: int | None = None,
    batch_size: int | None = None,
    overwrite: bool = False,  # noqa: FBT001, FBT002
    wandb_enabled: bool = False,  # noqa: FBT001, FBT002
) -> None:
    _prepare_volume_paths()
    cmd = _train_cmd(
        config_name=config_name,
        exp_name=exp_name,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
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
    overwrite: bool = False,  # noqa: FBT001, FBT002
    wandb_enabled: bool = False,  # noqa: FBT001, FBT002
) -> None:
    _prepare_volume_paths()
    cmd = _train_cmd(
        config_name=config_name,
        exp_name=exp_name,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        overwrite=overwrite,
        wandb_enabled=wandb_enabled,
        fsdp_devices=2,
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


def _train_cmd(
    *,
    config_name: str,
    exp_name: str,
    num_train_steps: int | None,
    batch_size: int | None,
    overwrite: bool,
    wandb_enabled: bool,
    fsdp_devices: int | None = None,
) -> list[str]:
    cmd = ["uv", "run", "scripts/train.py", config_name, f"--exp-name={exp_name}"]
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
