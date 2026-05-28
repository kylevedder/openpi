from __future__ import annotations

import dataclasses
import os
import pathlib
import types

import numpy as np

from openpi.training import experiment_logger


@dataclasses.dataclass
class _Nested:
    path: pathlib.Path
    array_value: np.float32


@dataclasses.dataclass
class _FakeConfig:
    checkpoint_base_dir: str
    checkpoint_dir: pathlib.Path
    tracking_backend: str = "trackio"
    tracking_dir: str | None = None
    wandb_enabled: bool = True
    name: str = "debug"
    exp_name: str = "test"
    project_name: str = "openpi"
    nested: _Nested | None = None


class _FakeImage:
    def __init__(self, value, caption=None):
        self.value = value
        self.caption = caption


class _FakeTrackioMedia(_FakeImage):
    pass


_FakeTrackioMedia.__module__ = "trackio.media.image"


def _fake_trackio(calls: list[tuple[str, dict]]):
    module = types.SimpleNamespace()
    module.Image = _FakeImage

    def init(**kwargs):
        calls.append(("init", kwargs))
        return types.SimpleNamespace(name=kwargs["name"])

    def log(metrics, step=None):
        calls.append(("log", {"metrics": metrics, "step": step}))

    def finish():
        calls.append(("finish", {}))

    module.init = init
    module.log = log
    module.finish = finish
    return module


def _fake_wandb(calls: list[tuple[str, dict]]):
    module = types.SimpleNamespace()
    module.Image = _FakeImage
    module.run = types.SimpleNamespace(id="wandb-run-id", log_code=lambda path: calls.append(("log_code", {"path": path})))

    def init(**kwargs):
        calls.append(("init", kwargs))
        return module.run

    def log(metrics, step=None):
        calls.append(("log", {"metrics": metrics, "step": step}))

    def finish():
        calls.append(("finish", {}))

    module.init = init
    module.log = log
    module.finish = finish
    return module


def test_trackio_backend_defaults_to_checkpoint_tracking_dir(tmp_path, monkeypatch):
    calls = []
    fake_module = _fake_trackio(calls)
    monkeypatch.delenv("TRACKIO_DIR", raising=False)
    monkeypatch.setattr(experiment_logger.importlib, "import_module", lambda name: fake_module)
    config = _FakeConfig(checkpoint_base_dir=str(tmp_path / "checkpoints"), checkpoint_dir=tmp_path / "ckpt")

    logger = experiment_logger.init_logger(config, resuming=True)
    logger.log({"train/loss": 1.0}, step=7)
    logger.finish()

    assert logger.enabled
    assert calls[0][0] == "init"
    assert calls[0][1]["project"] == "openpi"
    assert calls[0][1]["name"] == "debug__test"
    assert calls[0][1]["group"] == "debug"
    assert calls[0][1]["resume"] == "allow"
    assert calls[1] == ("log", {"metrics": {"train/loss": 1.0}, "step": 7})
    assert calls[2] == ("finish", {})
    assert pathlib.Path(os.environ["TRACKIO_DIR"]) == (tmp_path / "checkpoints" / "trackio").resolve()


def test_wandb_backend_uses_legacy_run_id(tmp_path, monkeypatch):
    calls = []
    fake_module = _fake_wandb(calls)
    monkeypatch.setattr(experiment_logger.importlib, "import_module", lambda name: fake_module)
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "wandb_id.txt").write_text("old-run")
    config = _FakeConfig(
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        checkpoint_dir=checkpoint_dir,
        tracking_backend="wandb",
    )

    logger = experiment_logger.init_logger(config, resuming=True)
    logger.finish()

    assert logger.enabled
    assert calls[0] == ("init", {"id": "old-run", "resume": "must", "project": "openpi"})
    assert calls[1] == ("finish", {})


def test_disabled_backend_does_not_import(monkeypatch, tmp_path):
    def fail_import(name):
        raise AssertionError(f"imported {name}")

    monkeypatch.setattr(experiment_logger.importlib, "import_module", fail_import)
    config = _FakeConfig(
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        checkpoint_dir=tmp_path / "ckpt",
        tracking_backend="none",
    )

    logger = experiment_logger.init_logger(config, resuming=False)

    assert not logger.enabled
    logger.log({"train/loss": 1.0}, step=1)
    logger.finish()


def test_serialize_config_is_json_safe(tmp_path):
    config = _FakeConfig(
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        checkpoint_dir=tmp_path / "ckpt",
        nested=_Nested(path=tmp_path / "data", array_value=np.float32(1.25)),
    )

    serialized = experiment_logger.serialize_config(config)

    assert serialized["checkpoint_dir"] == str(tmp_path / "ckpt")
    assert serialized["nested"]["path"] == str(tmp_path / "data")
    assert serialized["nested"]["array_value"] == 1.25


def test_image_conversion_produces_rgb_uint8(tmp_path):
    logger = experiment_logger.ExperimentLogger(backend="trackio", module=_fake_trackio([]))

    image = logger.image(np.zeros((1, 4, 4), dtype=np.float32), caption="sample")

    assert image.value.shape == (4, 4, 3)
    assert image.value.dtype == np.uint8
    assert image.caption == "sample"


def test_trackio_media_lists_are_logged_as_top_level_keys():
    calls = []
    logger = experiment_logger.ExperimentLogger(backend="trackio", module=_fake_trackio(calls))
    first = _FakeTrackioMedia("a")
    second = _FakeTrackioMedia("b")

    logger.log({"data/camera_views": [first, second]}, step=0)

    assert calls == [
        (
            "log",
            {
                "metrics": {
                    "data/camera_views/0": first,
                    "data/camera_views/1": second,
                },
                "step": 0,
            },
        )
    ]
