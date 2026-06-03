import pytest

import orbit_rl_native

kaggle_environments = pytest.importorskip("kaggle_environments")


def _noop(_obs):
    return []


def test_rl_features_mark_real_kaggle_comets_missing_after_expiration():
    env = kaggle_environments.make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([_noop, _noop, _noop, _noop])

    base = env.steps[82][0].observation
    assert list(base.comet_planet_ids) == [20, 21, 22, 23]
    assert base.comets[0].path_index == 32
    assert all(len(path) == 34 for path in base.comets[0].paths)
    assert list(env.steps[84][0].observation.comet_planet_ids) == []

    batch = orbit_rl_native.FeatureEngine().compute(base, horizon=4)
    ids = list(batch["planet_ids"])
    comet_rows = [ids.index(comet_id) for comet_id in base.comet_planet_ids]

    assert batch["stats"]["active_comets"] == 4
    assert batch["stats"]["expiring_comets_within_horizon"] == 4
    assert batch["stats"]["next_comet_spawn_step"] == 150
    assert batch["stats"]["turns_until_next_comet_spawn"] == 68
    for row in comet_rows:
        assert batch["garrisons"][row, 0, 1] != -2
        assert batch["garrisons"][row, 1, 1] != -2
        assert batch["garrisons"][row, 2].tolist() == [0, -2]
        assert batch["garrisons"][row, 3].tolist() == [0, -2]
