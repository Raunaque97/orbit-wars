import pytest

kaggle_environments = pytest.importorskip("kaggle_environments")

from tools.check_action_safety import first_collision  # noqa: E402


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, "get"):
        return obj.get(name, default)
    return getattr(obj, name, default)


def test_agent_actions_hit_planets_before_sun_or_bounds(monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_TIME_BUDGET_MS", "25")
    for seed in range(2):
        env = kaggle_environments.make("orbit_wars", configuration={"seed": seed}, debug=True)
        env.run(["main.py", "main.py"])
        synthetic_steps = [-1, -1]

        for step_index in range(1, len(env.steps)):
            previous = env.steps[step_index - 1]
            current = env.steps[step_index]
            for player, state in enumerate(current):
                obs = previous[player].observation
                raw_step = _get(obs, "step", None)
                if raw_step is None:
                    synthetic_steps[player] += 1
                else:
                    synthetic_steps[player] = raw_step

                for move in state.action or []:
                    collision = first_collision(obs, move, synthetic_steps[player])
                    assert collision[0] == "planet"
