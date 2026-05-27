from __future__ import annotations

import dataclasses
from pathlib import Path
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import msgpack_numpy
import tyro

from examples.yam_real import common
from examples.yam_real import data_regression
from examples.yam_real import mcap_episode
from openpi.shared import jpeg_transport


@dataclasses.dataclass
class Args:
    raw_dir: Path = Path("yam_data/raw")
    episode_manifest: Path | None = None
    max_observations: int = 80
    qualities: tuple[int, ...] = (50, 60, 70, 75, 80, 85, 90, 95)


def main(args: Args) -> None:
    observations = list(_load_observations(args.raw_dir, args.episode_manifest, args.max_observations))
    if not observations:
        raise FileNotFoundError(f"No observations found under {args.raw_dir}")

    packer = msgpack_numpy.Packer()
    state = np.zeros((14,), dtype=np.float32)
    prompt = "benchmark"
    raw_payload_sizes = [
        len(
            packer.pack(
                {
                    "state": state,
                    "images": {name: np.transpose(image, (2, 0, 1)) for name, image in observation.items()},
                    "prompt": prompt,
                }
            )
        )
        for observation in observations
    ]

    print(
        f"observations={len(observations)} cameras={len(common.CAMERA_NAMES)} resolution={jpeg_transport.IMAGE_RESOLUTION}"
    )
    print(f"raw_msgpack_bytes mean={_mean(raw_payload_sizes):.0f} p95={_p95(raw_payload_sizes):.0f}")
    print("quality  jpeg_bytes_mean  jpeg_bytes_p95  shrink  encode_ms_mean  encode_ms_p95  decode_ms_mean  psnr_mean")
    for quality in args.qualities:
        metrics = _measure_quality(observations, packer, state, prompt, quality)
        print(
            f"{quality:7d}  "
            f"{_mean(metrics.payload_sizes):15.0f}  "
            f"{_p95(metrics.payload_sizes):14.0f}  "
            f"{_mean(raw_payload_sizes) / _mean(metrics.payload_sizes):6.1f}  "
            f"{_mean(metrics.encode_ms):14.2f}  "
            f"{_p95(metrics.encode_ms):13.2f}  "
            f"{_mean(metrics.decode_ms):14.2f}  "
            f"{_mean(metrics.psnr):9.1f}"
        )


@dataclasses.dataclass
class QualityMetrics:
    payload_sizes: list[int]
    encode_ms: list[float]
    decode_ms: list[float]
    psnr: list[float]


def _measure_quality(
    observations: list[dict[str, np.ndarray]],
    packer: msgpack_numpy.Packer,
    state: np.ndarray,
    prompt: str,
    quality: int,
) -> QualityMetrics:
    payload_sizes = []
    encode_ms = []
    decode_ms = []
    psnr = []
    for observation in observations:
        encode_start = time.monotonic()
        encoded = {name: jpeg_transport.encode_rgb_jpeg(image, quality=quality) for name, image in observation.items()}
        encode_ms.append(1000.0 * (time.monotonic() - encode_start))

        decode_start = time.monotonic()
        decoded = {name: jpeg_transport.decode_rgb_jpeg(image) for name, image in encoded.items()}
        decode_ms.append(1000.0 * (time.monotonic() - decode_start))

        payload_sizes.append(len(packer.pack({"state": state, "images": encoded, "prompt": prompt})))
        psnr.extend(_psnr(observation[name], decoded[name]) for name in observation)

    return QualityMetrics(payload_sizes=payload_sizes, encode_ms=encode_ms, decode_ms=decode_ms, psnr=psnr)


def _load_observations(raw_dir: Path, episode_manifest: Path | None, max_observations: int):
    for episode_dir in _episode_dirs(raw_dir, episode_manifest):
        episode = mcap_episode.read_episode(episode_dir, decode_images=True)
        if episode.images is None:
            continue
        rows = int(episode.state.shape[0])
        for idx in range(rows):
            yield {
                camera_name: _resize_rgb(episode.images[camera_name][idx]) for camera_name in common.CAMERA_NAMES
            }
            max_observations -= 1
            if max_observations <= 0:
                return


def _episode_dirs(raw_dir: Path, episode_manifest: Path | None) -> list[Path]:
    if episode_manifest is None:
        return data_regression.discover_episodes(raw_dir, source_format="mcap")

    if not episode_manifest.exists():
        raise FileNotFoundError(episode_manifest)

    episode_dirs = []
    for line_number, line in enumerate(episode_manifest.read_text().splitlines(), start=1):
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        rel_path = Path(entry)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"Invalid episode path on line {line_number}: {entry}")
        episode_dir = raw_dir / rel_path
        if not mcap_episode.is_mcap_episode(episode_dir):
            raise FileNotFoundError(f"Manifest-listed episode is not a canonical MCAP YAM episode: {episode_dir}")
        episode_dirs.append(episode_dir)
    return episode_dirs


def _resize_rgb(image: np.ndarray) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, *jpeg_transport.IMAGE_RESOLUTION))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * float(np.log10(255.0 / np.sqrt(mse)))


def _mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values))


def _p95(values: list[float] | list[int]) -> float:
    return float(np.percentile(values, 95))


if __name__ == "__main__":
    main(tyro.cli(Args))
