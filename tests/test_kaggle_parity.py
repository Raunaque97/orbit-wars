import math

import pytest

import orbit_native

kaggle_environments = pytest.importorskip("kaggle_environments")


def test_native_agent_runs_inside_kaggle_env_one_game(monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_TIME_BUDGET_MS", "25")
    env = kaggle_environments.make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run(["main.py", "random"])

    final = env.steps[-1]
    assert all(step.status in {"DONE", "ACTIVE", "INACTIVE"} for step in final)


def test_speed_formula_matches_documented_points():
    assert orbit_native.fleet_speed(1) == 1.0
    assert math.isclose(orbit_native.fleet_speed(1000), 6.0)


def test_forecast_planets_matches_kaggle_noop_future_with_fleets(monkeypatch):
    import agent_v2

    monkeypatch.setenv("ORBIT_WARS_TIME_BUDGET_MS", "25")
    cutoff = 35
    horizon = 12

    def cutoff_agent(obs):
        step = obs.get("step", 0) if isinstance(obs, dict) else obs.step
        if step >= cutoff:
            return []
        return agent_v2.agent(obs)

    env = kaggle_environments.make("orbit_wars", configuration={"seed": 3}, debug=True)
    env.run([cutoff_agent, cutoff_agent])

    base = env.steps[cutoff][0].observation
    assert len(base.fleets) > 0
    base_obs = {
        name: getattr(base, name)
        for name in (
            "player",
            "step",
            "planets",
            "initial_planets",
            "fleets",
            "angular_velocity",
            "next_fleet_id",
            "comets",
            "comet_planet_ids",
        )
        if hasattr(base, name)
    }

    forecast = orbit_native.forecast_planets(base_obs, horizon)

    for dt in range(horizon + 1):
        actual = {
            planet[0]: (planet[1], planet[5], planet[6])
            for planet in env.steps[cutoff + dt][0].observation.planets
        }
        predicted = {
            row[dt][0]: (row[dt][1], row[dt][5], row[dt][6])
            for row in forecast
        }
        assert predicted == actual
