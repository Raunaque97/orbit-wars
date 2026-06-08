from __future__ import annotations

import math

import pytest

from rl.native_env import NativeOrbitEnv

kaggle_environments = pytest.importorskip("kaggle_environments")


def _obs_to_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {
        name: getattr(obs, name)
        for name in (
            "step",
            "angular_velocity",
            "planets",
            "initial_planets",
            "fleets",
            "next_fleet_id",
            "comets",
            "comet_planet_ids",
        )
        if hasattr(obs, name)
    }


def _assert_planets_equal(native, kaggle):
    assert [int(p[0]) for p in native] == [int(p[0]) for p in kaggle]
    for np, kp in zip(native, kaggle):
        assert int(np[0]) == int(kp[0])
        assert int(np[1]) == int(kp[1])
        assert math.isclose(float(np[2]), float(kp[2]), abs_tol=1e-8)
        assert math.isclose(float(np[3]), float(kp[3]), abs_tol=1e-8)
        assert math.isclose(float(np[4]), float(kp[4]), abs_tol=1e-8)
        assert int(np[5]) == int(kp[5])
        assert int(np[6]) == int(kp[6])


def _assert_fleets_equal(native, kaggle):
    assert len(native) == len(kaggle)
    for nf, kf in zip(native, kaggle):
        assert int(nf[0]) == int(kf[0])
        assert int(nf[1]) == int(kf[1])
        assert math.isclose(float(nf[2]), float(kf[2]), abs_tol=1e-8)
        assert math.isclose(float(nf[3]), float(kf[3]), abs_tol=1e-8)
        assert math.isclose(float(nf[4]), float(kf[4]), abs_tol=1e-8)
        assert int(nf[5]) == int(kf[5])
        assert int(nf[6]) == int(kf[6])


def _assert_comets_equal(native, kaggle):
    assert [int(pid) for pid in native.get("comet_planet_ids", [])] == [
        int(pid) for pid in kaggle.get("comet_planet_ids", [])
    ]
    assert len(native.get("comets", [])) == len(kaggle.get("comets", []))
    for ng, kg in zip(native.get("comets", []), kaggle.get("comets", [])):
        assert [int(pid) for pid in ng["planet_ids"]] == [int(pid) for pid in kg["planet_ids"]]
        assert int(ng["path_index"]) == int(kg["path_index"])
        assert len(ng["paths"]) == len(kg["paths"])


def _assert_obs_equal(native_obs, kaggle_obs):
    native = _obs_to_dict(native_obs)
    kaggle = _obs_to_dict(kaggle_obs)
    assert int(native["step"]) == int(kaggle["step"])
    assert math.isclose(
        float(native["angular_velocity"]), float(kaggle["angular_velocity"]), abs_tol=1e-12
    )
    assert int(native.get("next_fleet_id", 0)) == int(kaggle.get("next_fleet_id", 0))
    _assert_planets_equal(native["planets"], kaggle["planets"])
    _assert_fleets_equal(native["fleets"], kaggle["fleets"])
    _assert_comets_equal(native, kaggle)


def test_native_reset_matches_kaggle_initial_state():
    seed = 17
    env = kaggle_environments.make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.reset()
    native = NativeOrbitEnv(seed=seed)

    _assert_obs_equal(native.state[0].observation, env.state[0].observation)


def test_native_noop_matches_kaggle_through_first_comet_spawn():
    seed = 23
    env = kaggle_environments.make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.reset()
    native = NativeOrbitEnv(seed=seed)

    for _ in range(60):
        env.step([[], []])
        native.step([[], []])
        _assert_obs_equal(native.state[0].observation, env.state[0].observation)
        assert native.state[0].status == env.state[0].status
        assert native.state[0].reward == env.state[0].reward


def test_native_deterministic_launches_match_kaggle():
    seed = 31
    env = kaggle_environments.make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.reset()
    native = NativeOrbitEnv(seed=seed)

    for tick in range(20):
        actions = [[], []]
        for player in (0, 1):
            obs = _obs_to_dict(env.state[player].observation)
            source = next(p for p in obs["planets"] if int(p[1]) == player)
            if tick in {0, 7, 14} and int(source[5]) >= 5:
                angle = 0.0 if player == 0 else math.pi
                actions[player] = [[int(source[0]), angle, 5]]
        env.step(actions)
        native.step(actions)
        _assert_obs_equal(native.state[0].observation, env.state[0].observation)
