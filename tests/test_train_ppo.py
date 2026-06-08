import pytest

from rl.train_ppo import (
    OpponentEntry,
    _choose_opponent_index,
    _opponent_sampling_score,
    _prune_opponent_population,
    _random_v2_policy_prob,
    _rolling_win_rate,
)


class FixedRandom:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def test_rolling_win_rate_handles_empty_and_nonempty_windows():
    assert _rolling_win_rate([]) == 0.0
    assert _rolling_win_rate([1, 0, 1, 1]) == 0.75


def test_opponent_population_selection_uses_scores():
    population = [
        OpponentEntry("bc_opening", None, 0.20, fixed=True),
        OpponentEntry("bc_complete", None, 0.20, fixed=True),
        OpponentEntry("lagged_self_play", None, 0.35, fixed=True),
    ]

    assert _choose_opponent_index(FixedRandom(0.10), population) == 0
    assert _choose_opponent_index(FixedRandom(0.50), population) == 1
    assert _choose_opponent_index(FixedRandom(0.95), population) == 2


def test_opponent_sampling_keeps_hard_opponents_but_samples_them_less():
    hard = OpponentEntry("bc_complete", None, 0.20, fixed=True, games=64, learner_wins=0)
    near_even = OpponentEntry(
        "bc_opening", None, 0.20, fixed=True, games=64, learner_wins=32
    )

    assert _opponent_sampling_score(hard) > 0.0
    assert _opponent_sampling_score(hard) < _opponent_sampling_score(near_even)


def test_prune_opponent_population_removes_easy_snapshots_only():
    fixed = OpponentEntry("bc_opening", None, 0.20, fixed=True, games=100, learner_wins=100)
    easy = OpponentEntry("snapshot_easy", None, 0.25, games=100, learner_wins=90)
    useful = OpponentEntry("snapshot_useful", None, 0.25, games=100, learner_wins=55)
    population = [fixed, easy, useful]

    assert _prune_opponent_population(population) == ["snapshot_easy"]
    assert [entry.name for entry in population] == ["bc_opening", "snapshot_useful"]


def test_random_v2_policy_prob_decays_when_learner_farms_it():
    early = [OpponentEntry("random_v2", None, 0.15, fixed=True, games=10, learner_wins=10)]
    hard = [OpponentEntry("random_v2", None, 0.15, fixed=True, games=100, learner_wins=30)]
    easy = [OpponentEntry("random_v2", None, 0.15, fixed=True, games=100, learner_wins=90)]

    assert _random_v2_policy_prob(early) == pytest.approx(0.9)
    assert _random_v2_policy_prob(hard) == pytest.approx(0.9)
    assert _random_v2_policy_prob(easy) == pytest.approx(0.0)
