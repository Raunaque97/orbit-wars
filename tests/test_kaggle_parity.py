import math

import pytest

import orbit_native

kaggle_environments = pytest.importorskip("kaggle_environments")


def test_native_agent_runs_inside_kaggle_env_one_game():
    env = kaggle_environments.make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run(["main.py", "random"])

    final = env.steps[-1]
    assert all(step.status in {"DONE", "ACTIVE", "INACTIVE"} for step in final)


def test_speed_formula_matches_documented_points():
    assert orbit_native.fleet_speed(1) == 1.0
    assert math.isclose(orbit_native.fleet_speed(1000), 6.0)
