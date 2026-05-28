"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import contextlib
import os
import numpy as np
import threading
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _start_system_stats_logger(enabled: bool, interval_s: float) -> contextlib.AbstractContextManager[None]:
    if not enabled:
        return contextlib.nullcontext()

    try:
        import psutil
    except ImportError:
        print("[norm-stats-system] psutil is not installed; system stats logging is disabled.", flush=True)
        return contextlib.nullcontext()

    stop_event = threading.Event()

    @contextlib.contextmanager
    def monitor():
        process = psutil.Process()
        print(
            f"[norm-stats-system] logical_cpus={os.cpu_count()} "
            f"interval_s={interval_s:.1f} main_pid={process.pid}",
            flush=True,
        )
        psutil.cpu_percent(interval=None)
        process.cpu_percent(interval=None)

        def log_loop() -> None:
            while not stop_event.wait(interval_s):
                children = process.children(recursive=True)
                child_cpu = 0.0
                for child in children:
                    with contextlib.suppress(psutil.Error):
                        child_cpu += child.cpu_percent(interval=None)
                process_cpu = 0.0
                with contextlib.suppress(psutil.Error):
                    process_cpu = process.cpu_percent(interval=None)
                loadavg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
                print(
                    "[norm-stats-system] "
                    f"cpu_total={psutil.cpu_percent(interval=None):.1f}% "
                    f"main_cpu={process_cpu:.1f}% "
                    f"child_cpu={child_cpu:.1f}% "
                    f"child_processes={len(children)} "
                    f"loadavg={loadavg[0]:.2f},{loadavg[1]:.2f},{loadavg[2]:.2f}",
                    flush=True,
                )

        thread = threading.Thread(target=log_loop, name="norm-stats-system-monitor", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=interval_s)

    return monitor()


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    log_system_stats: bool = False,
    system_stats_interval_s: float = 10.0,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    batch_size = batch_size or config.batch_size
    num_workers = config.num_workers if num_workers is None else num_workers
    print(
        "Norm stats settings: "
        f"config_name={config_name} batch_size={batch_size} num_workers={num_workers} "
        f"max_frames={max_frames} logical_cpus={os.cpu_count()}",
        flush=True,
    )

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, batch_size, config.model, num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    with _start_system_stats_logger(log_system_stats, system_stats_interval_s):
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
