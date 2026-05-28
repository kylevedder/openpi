from openpi.training import checkpoints as _checkpoints


def test_initialize_checkpoint_dir_uses_configured_max_to_keep(tmp_path, monkeypatch):
    captured = {}

    class FakeCheckpointManager:
        def __init__(self, directory, item_handlers, options):
            del item_handlers
            captured["directory"] = directory
            captured["options"] = options

        def all_steps(self):
            return ()

    monkeypatch.setattr(_checkpoints.ocp, "CheckpointManager", FakeCheckpointManager)

    manager, resuming = _checkpoints.initialize_checkpoint_dir(
        tmp_path / "ckpt",
        keep_period=5000,
        max_to_keep=None,
        overwrite=False,
        resume=False,
    )

    assert isinstance(manager, FakeCheckpointManager)
    assert resuming is False
    assert captured["directory"] == tmp_path / "ckpt"
    assert captured["options"].max_to_keep is None
