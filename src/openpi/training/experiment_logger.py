"""Experiment tracking helpers for local and remote training logs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import importlib
import logging
import os
import pathlib
from types import ModuleType
from typing import Any

import numpy as np

import openpi.training.config as _config

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ExperimentLogger:
    """Thin adapter over Trackio, WandB, or a disabled logger."""

    backend: _config.TrackingBackend
    module: ModuleType | Any | None = None
    run: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.backend != "none" and self.module is not None

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        if not self.enabled:
            return
        self.module.log(self._prepare_metrics(metrics), step=step)

    def image(self, value: Any, *, caption: str | None = None) -> Any:
        if not self.enabled:
            return value
        image = to_rgb_uint8(value)
        if caption is None:
            return self.module.Image(image)
        return self.module.Image(image, caption=caption)

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self.module.finish()
        except RuntimeError:
            logger.debug("Experiment logger was already finished.", exc_info=True)

    def _prepare_metrics(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        if self.backend != "trackio":
            return dict(metrics)

        prepared = {}
        for key, value in metrics.items():
            if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
                media_values = [_is_trackio_media(item, self.module) for item in value]
                if media_values and all(media_values):
                    prepared.update({f"{key}/{i}": item for i, item in enumerate(value)})
                    continue
            prepared[key] = value
        return prepared


def disabled() -> ExperimentLogger:
    return ExperimentLogger(backend="none")


def init_logger(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    enabled: bool = True,
    log_code: bool = False,
) -> ExperimentLogger:
    """Initialize the configured experiment logger.

    Trackio is local-first and is now the default backend. WandB remains available
    for explicit opt-in compatibility.
    """

    if not enabled or config.tracking_backend == "none":
        return disabled()

    if config.tracking_backend == "trackio":
        return _init_trackio(config, resuming=resuming)

    if config.tracking_backend == "wandb":
        if not config.wandb_enabled:
            return disabled()
        return _init_wandb(config, resuming=resuming, log_code=log_code)

    raise ValueError(f"Unsupported tracking backend: {config.tracking_backend!r}")


def _init_trackio(config: _config.TrainConfig, *, resuming: bool) -> ExperimentLogger:
    _set_trackio_dir(config)
    trackio = importlib.import_module("trackio")

    run = trackio.init(
        project=config.project_name,
        name=_run_name(config),
        group=config.name,
        config=serialize_config(config),
        resume="allow" if resuming else "never",
    )
    return ExperimentLogger(backend="trackio", module=trackio, run=run)


def _init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool) -> ExperimentLogger:
    wandb = importlib.import_module("wandb")

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming and (ckpt_dir / "wandb_id.txt").exists():
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        run = wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        run = wandb.init(
            name=config.exp_name,
            config=serialize_config(config),
            project=config.project_name,
        )
        if getattr(wandb, "run", None) is not None and getattr(wandb.run, "id", None) is not None:
            (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code and getattr(wandb, "run", None) is not None:
        wandb.run.log_code(pathlib.Path(__file__).parents[3])

    return ExperimentLogger(backend="wandb", module=wandb, run=run)


def _run_name(config: _config.TrainConfig) -> str:
    return f"{config.name}__{config.exp_name}"


def _set_trackio_dir(config: _config.TrainConfig) -> None:
    if config.tracking_dir:
        os.environ["TRACKIO_DIR"] = str(pathlib.Path(config.tracking_dir).expanduser().resolve())
        return

    if os.environ.get("TRACKIO_DIR"):
        return

    os.environ["TRACKIO_DIR"] = str((pathlib.Path(config.checkpoint_base_dir) / "trackio").resolve())


def _is_trackio_media(value: Any, module: ModuleType | Any | None) -> bool:
    media_module = getattr(module, "media", None)
    media_cls = getattr(media_module, "TrackioMedia", None)
    if media_cls is not None and isinstance(value, media_cls):
        return True
    return type(value).__module__.startswith("trackio.media")


def serialize_config(value: Any) -> Any:
    """Convert nested config objects into JSON-safe Python values."""

    return _to_json_safe(value)


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path | os.PathLike):
        return os.fspath(value)
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(_to_json_safe(k)): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_json_safe(item) for item in value]
    if callable(value):
        return getattr(value, "__qualname__", repr(value))
    return repr(value)


def to_rgb_uint8(value: Any) -> np.ndarray:
    """Convert image-like arrays to HWC RGB uint8 for Trackio-compatible logging."""

    image = np.asarray(value)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] in (1, 3) and (
        image.shape[-1] not in (1, 3) or image.shape[1] == image.shape[-1]
    ):
        image = np.moveaxis(image, 0, -1)

    if image.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got shape {image.shape}")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] > 3:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 image channels, got shape {image.shape}")

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    image = image.astype(np.float32)
    if np.nanmin(image) >= -1.0 and np.nanmax(image) <= 1.0:
        image = (image + 1.0) * 127.5 if np.nanmin(image) < 0.0 else image * 255.0

    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
