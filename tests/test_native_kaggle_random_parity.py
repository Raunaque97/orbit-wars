from __future__ import annotations

import pytest

kaggle_environments = pytest.importorskip("kaggle_environments")

from tools.native_kaggle_random_parity import run_random_trajectory_parity  # noqa: E402


def test_native_matches_kaggle_for_random_legal_trajectories():
    stats = run_random_trajectory_parity(
        seeds=2,
        ticks=80,
        action_seed=20260607,
        launch_prob=0.45,
        min_ships=5,
    )

    assert stats.ticks_checked > 0
    assert stats.moves_launched > 0
