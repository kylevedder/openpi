from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys

import modal

APP_NAME = "yam-openpi-latency"
OPENPI_ROOT = "/root/openpi"
PORT = 8765
SCALEDOWN_WINDOW_S = 20 * 60
QUIC_REGIONS = ["us-west-1", "westus"]

app = modal.App(APP_NAME)


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


@app.function(image=image, timeout=60 * 60, min_containers=1, scaledown_window=SCALEDOWN_WINDOW_S)
@modal.web_server(PORT, startup_timeout=120)
def echo_default() -> None:
    _serve("echo")


@app.function(image=image, timeout=60 * 60, min_containers=1, scaledown_window=SCALEDOWN_WINDOW_S, region="us")
@modal.web_server(PORT, startup_timeout=120)
def echo_us() -> None:
    _serve("echo")


@app.function(image=image, timeout=60 * 60, min_containers=1, scaledown_window=SCALEDOWN_WINDOW_S, region="us-west")
@modal.web_server(PORT, startup_timeout=120)
def echo_us_west() -> None:
    _serve("echo")


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    min_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_S,
    region="us-west",
)
@modal.web_server(PORT, startup_timeout=120)
def echo_gpu_us_west() -> None:
    _serve("echo")


@app.function(image=image, timeout=60 * 60, min_containers=1, scaledown_window=SCALEDOWN_WINDOW_S, region="us-west")
@modal.web_server(PORT, startup_timeout=120)
def noop_policy_us_west_0ms() -> None:
    _serve("noop-policy", fixed_sleep_ms=0.0)


@app.function(image=image, timeout=60 * 60, min_containers=1, scaledown_window=SCALEDOWN_WINDOW_S, region="us-west")
@modal.web_server(PORT, startup_timeout=120)
def noop_policy_us_west_108ms() -> None:
    _serve("noop-policy", fixed_sleep_ms=108.0)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    min_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_S,
    region="us-west",
)
@modal.web_server(PORT, startup_timeout=120)
def noop_policy_gpu_us_west_108ms() -> None:
    _serve("noop-policy", fixed_sleep_ms=108.0)


@app.cls(
    image=image,
    timeout=60 * 60,
    min_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_S,
    region=QUIC_REGIONS,
    experimental_options={"region_ranking_enabled": True},
    max_containers=10,
)
class QuicLatencyServer:
    @modal.method()
    def serve(
        self,
        rendezvous: modal.Dict,
        mode: str = "echo",
        fixed_sleep_ms: float = 0.0,
        response_bytes: int = 3_000,
        action_horizon: int = 50,
        action_dim: int = 14,
    ) -> None:
        _serve_quic(
            rendezvous=rendezvous,
            mode=mode,
            fixed_sleep_ms=fixed_sleep_ms,
            response_bytes=response_bytes,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )


@app.cls(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    min_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_S,
    region=QUIC_REGIONS,
    experimental_options={"region_ranking_enabled": True},
    max_containers=10,
)
class QuicLatencyGpuServer:
    @modal.method()
    def serve(
        self,
        rendezvous: modal.Dict,
        mode: str = "echo",
        fixed_sleep_ms: float = 0.0,
        response_bytes: int = 3_000,
        action_horizon: int = 50,
        action_dim: int = 14,
    ) -> None:
        _serve_quic(
            rendezvous=rendezvous,
            mode=mode,
            fixed_sleep_ms=fixed_sleep_ms,
            response_bytes=response_bytes,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )


def _serve(mode: str, *, fixed_sleep_ms: float = 0.0) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "examples.yam_real.latency_probe_server",
        "--mode",
        mode,
        "--port",
        str(PORT),
        "--fixed-sleep-ms",
        str(fixed_sleep_ms),
    ]
    subprocess.Popen(cmd, cwd=OPENPI_ROOT)


def _serve_quic(
    *,
    rendezvous: modal.Dict,
    mode: str,
    fixed_sleep_ms: float,
    response_bytes: int,
    action_horizon: int,
    action_dim: int,
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    _ensure_openpi_importable()
    from examples.yam_real import latency_probe_server
    from openpi.serving import modal_quic

    if mode == "echo":
        metadata = {
            "protocol": "yam_latency_echo_quic_v1",
            "transport": "modal-quic",
            "fixed_sleep_ms": fixed_sleep_ms,
            "default_response_bytes": response_bytes,
            "regions": QUIC_REGIONS,
        }
        modal_quic.serve_echo(
            rendezvous=rendezvous,
            metadata=metadata,
            fixed_sleep_ms=fixed_sleep_ms,
            default_response_bytes=response_bytes,
        )
        return

    if mode == "noop-policy":
        policy = latency_probe_server.NoopPolicy(
            fixed_sleep_ms=fixed_sleep_ms,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        metadata = dict(policy.metadata)
        metadata.update(
            {
                "protocol": "yam_latency_noop_policy_quic_v1",
                "transport": "modal-quic",
                "regions": QUIC_REGIONS,
            }
        )
        modal_quic.serve_policy(
            rendezvous=rendezvous,
            policy=policy,
            metadata=metadata,
        )
        return

    raise ValueError(f"Unsupported QUIC latency mode: {mode}")
