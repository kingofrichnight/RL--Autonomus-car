from __future__ import annotations

import pytest

from scripts.train_ppo import callback_frequency, training_seed_offsets


def test_training_seed_offsets_are_deterministic_and_separated() -> None:
    assert training_seed_offsets(4, 1_000) == [0, 1_000, 2_000, 3_000]


@pytest.mark.parametrize(("n_envs", "stride"), [(0, 1_000), (4, 0), (-1, 1_000)])
def test_training_seed_offsets_reject_invalid_values(n_envs: int, stride: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        training_seed_offsets(n_envs, stride)


def test_callback_frequency_accounts_for_vectorized_steps() -> None:
    assert callback_frequency(10_000, 4) == 2_500
    assert callback_frequency(3, 4) == 1


@pytest.mark.parametrize(("timesteps", "n_envs"), [(0, 4), (10_000, 0)])
def test_callback_frequency_rejects_invalid_values(timesteps: int, n_envs: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        callback_frequency(timesteps, n_envs)
