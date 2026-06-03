import math

import orbit_rl_native


def _obs():
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 10, 2],
        [1, -1, 20.0, 10.0, 2.0, 5, 1],
    ]
    return {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.0,
        "planets": planets,
        "initial_planets": planets,
        "fleets": [],
        "comet_planet_ids": [],
        "comets": [],
    }


def test_rl_feature_shapes_and_basic_forecast():
    batch = orbit_rl_native.FeatureEngine().compute(_obs(), horizon=4)

    assert batch["planet_ids"] == [0, 1]
    assert batch["ship_buckets"] == [5, 10, 20, 40, 80, 160]
    assert batch["garrisons"].shape == (2, 4, 2)
    assert batch["delays"].shape == (6, 2, 2)
    assert batch["angles"].shape == (6, 2, 2)
    assert batch["garrisons"][0, :, 0].tolist() == [10, 12, 14, 16]
    assert batch["garrisons"][0, :, 1].tolist() == [0, 0, 0, 0]
    assert batch["garrisons"][1, :, 0].tolist() == [5, 5, 5, 5]
    assert batch["garrisons"][1, :, 1].tolist() == [-1, -1, -1, -1]
    assert 0 < batch["delays"][0, 0, 1] < 200
    assert math.isclose(batch["angles"][0, 0, 1], 0.0)


def test_rl_garrison_forecast_applies_visible_fleet_capture():
    obs = _obs()
    obs["fleets"] = [[99, 0, 17.0, 10.0, 0.0, 0, 6]]

    batch = orbit_rl_native.FeatureEngine().compute(obs, horizon=3)

    assert batch["stats"]["predicted_arrivals"] == 1
    assert batch["garrisons"][1, 0].tolist() == [5, -1]
    assert batch["garrisons"][1, 1].tolist() == [1, 0]
    assert batch["garrisons"][1, 2].tolist() == [2, 0]
