from __future__ import annotations

from pathlib import Path
import subprocess

import modal

APP_NAME = "yam-openpi"
OPENPI_ROOT = "/root/openpi"
VOLUME_ROOT = "/mnt/yam"
CONFIG_NAME = "pi05_yam_bimanual"
DEFAULT_EXP_NAME = "yam_bimanual_v1"
SERVE_CHECKPOINT_STEP = 19999

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("yam-openpi", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .env({"HF_LEROBOT_HOME": f"{VOLUME_ROOT}/lerobot", "OPENPI_DATA_HOME": f"{VOLUME_ROOT}/openpi_cache"})
    .add_local_dir(
        Path(__file__).resolve().parents[2],
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
    cmd = ["uv", "run", "scripts/compute_norm_stats.py", config_name]
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
) -> None:
    _prepare_volume_paths()
    cmd = ["uv", "run", "scripts/train.py", config_name, f"--exp-name={exp_name}"]
    if overwrite:
        cmd.append("--overwrite")
    if num_train_steps is not None:
        cmd.extend(["--num-train-steps", str(num_train_steps)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    subprocess.run(cmd, cwd=OPENPI_ROOT, check=True)
    volume.commit()


@app.function(image=image, gpu="A100-40GB", volumes={VOLUME_ROOT: volume}, timeout=24 * 60 * 60)
@modal.web_server(8000)
def serve_policy() -> None:
    _prepare_volume_paths()
    checkpoint_dir = f"checkpoints/{CONFIG_NAME}/{DEFAULT_EXP_NAME}/{SERVE_CHECKPOINT_STEP}"
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
