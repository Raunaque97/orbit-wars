import math

import random_v2_agent
from agent_common import load_orbit_native, native_obs


def _obs():
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 50, 1],
        [1, -1, 20.0, 10.0, 2.0, 5, 3],
    ]
    return {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.0,
        "planets": planets,
        "initial_planets": planets,
        "fleets": [],
    }


def test_random_v2_forced_random_returns_reachable_move(monkeypatch):
    monkeypatch.setattr(random_v2_agent, "RANDOM_POLICY_PROB", 1.0)
    monkeypatch.setattr(random_v2_agent, "MAX_RANDOM_LAUNCH_PROB", 1.0)
    monkeypatch.setattr(random_v2_agent, "RANDOM_SEED", 1)
    random_v2_agent._SEARCH_ENGINES.clear()
    random_v2_agent._ROUTE_ENGINES.clear()
    random_v2_agent._RNGS.clear()

    obs = _obs()
    move = random_v2_agent.agent(obs)[0]

    assert move[0] == 0
    assert 5 <= move[2] <= 50
    orbit_native = load_orbit_native()
    parsed = native_obs(obs)
    engine = orbit_native.Engine()
    engine.initialize(parsed)
    route = engine.query_route(int(move[0]), 1, int(move[2]), parsed["step"])
    assert route["reachable"]
    assert math.isclose(float(move[1]), route["angle"])


def test_random_v2_forced_search_uses_v2_budget(monkeypatch):
    monkeypatch.setattr(random_v2_agent, "RANDOM_POLICY_PROB", 0.0)
    monkeypatch.setattr(random_v2_agent, "SEARCH_BUDGET_MS", 100)
    random_v2_agent._SEARCH_ENGINES.clear()
    random_v2_agent._ROUTE_ENGINES.clear()
    random_v2_agent._RNGS.clear()

    assert random_v2_agent.agent(_obs()) == [[0, 0.0, 6]]


def test_random_v2_launch_probability_exp_decay():
    assert random_v2_agent._launch_probability(4) == 0.0
    assert random_v2_agent._launch_probability(5) == 0.0
    assert random_v2_agent._launch_probability(9) < 0.03
    assert random_v2_agent._launch_probability(50) == random_v2_agent.MAX_RANDOM_LAUNCH_PROB
    assert random_v2_agent._launch_probability(80) == random_v2_agent.MAX_RANDOM_LAUNCH_PROB


def test_random_v2_random_policy_can_noop_without_falling_back_to_search(monkeypatch):
    monkeypatch.setattr(random_v2_agent, "RANDOM_POLICY_PROB", 1.0)
    monkeypatch.setattr(random_v2_agent, "MAX_RANDOM_LAUNCH_PROB", 0.0)
    random_v2_agent._SEARCH_ENGINES.clear()
    random_v2_agent._ROUTE_ENGINES.clear()
    random_v2_agent._RNGS.clear()

    assert random_v2_agent.agent(_obs()) == []
