from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from examples.yam_real import run_policy


def test_resolve_playback_timing_defaults_prefetch_to_zero() -> None:
    timing = run_policy._resolve_playback_timing(run_policy.Args())  # noqa: SLF001

    assert timing.prefetch_remaining_steps == 0


def test_resolve_playback_timing_preserves_explicit_prefetch_remaining_steps() -> None:
    timing = run_policy._resolve_playback_timing(  # noqa: SLF001
        run_policy.Args(
            fps=50.0,
            action_playback_fps=25.0,
            action_horizon=50,
            prefetch_remaining_steps=30,
        )
    )

    assert timing.prefetch_remaining_steps == 30


@pytest.mark.parametrize("prefetch_remaining_steps", [-1, 50])
def test_resolve_playback_timing_rejects_out_of_range_prefetch_remaining_steps(
    prefetch_remaining_steps: int,
) -> None:
    with pytest.raises(ValueError, match="--prefetch-remaining-steps"):
        run_policy._resolve_playback_timing(  # noqa: SLF001
            run_policy.Args(action_horizon=50, prefetch_remaining_steps=prefetch_remaining_steps)
        )


def test_prefetch_remaining_steps_zero_does_not_schedule_early_prefetch() -> None:
    timing = run_policy._resolve_playback_timing(  # noqa: SLF001
        run_policy.Args(action_horizon=50, inter_chunk_delay_s=2.0, prefetch_remaining_steps=0)
    )

    assert not run_policy._should_prefetch_action_chunk(  # noqa: SLF001
        prefetch_action_chunks=True,
        pending_chunk=None,
        remaining_steps=0,
        playback_timing=timing,
    )


def test_prefetch_fires_when_requested_remaining_steps_are_reached() -> None:
    timing = run_policy._resolve_playback_timing(  # noqa: SLF001
        run_policy.Args(action_horizon=50, prefetch_remaining_steps=30)
    )

    fired_action_steps = [
        action_step
        for action_step in range(50)
        if run_policy._should_prefetch_action_chunk(  # noqa: SLF001
            prefetch_action_chunks=True,
            pending_chunk=None,
            remaining_steps=50 - (action_step + 1),
            playback_timing=timing,
        )
    ]

    assert fired_action_steps == [19]
