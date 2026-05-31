import math
from pathlib import Path
import json

import orbit_native


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


def test_query_and_act():
    engine = orbit_native.Engine()
    engine.initialize(_obs())

    route = engine.query_route(0, 1, 6, 0)
    moves = engine.act(_obs())

    assert route["reachable"]
    assert route["angle"] == 0.0
    assert moves == [[0, 0.0, 6]]


def test_simulate_step_launches_and_produces():
    state = orbit_native.simulate_step(_obs(), [[[0, 0.0, 6]]])

    assert state["step"] == 1
    assert state["planets"][0][5] == 45
    assert len(state["fleets"]) == 1
    assert math.isclose(state["fleets"][0][4], 0.0)


def test_simulate_step_uses_kaggle_observed_orbit_timing():
    planets = [
        [0, 0, 50.0, 20.0, 1.0, 10, 1],
        [1, 1, 90.0, 90.0, 2.0, 10, 1],
    ]
    obs = {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.1,
        "planets": planets,
        "initial_planets": planets,
        "fleets": [],
    }

    state1 = orbit_native.simulate_step(obs, [[], []])
    state2 = orbit_native.simulate_step(state1, [[], []])

    assert math.isclose(state1["planets"][0][2], planets[0][2])
    assert math.isclose(state1["planets"][0][3], planets[0][3])
    assert not math.isclose(state2["planets"][0][2], planets[0][2])
    assert not math.isclose(state2["planets"][0][3], planets[0][3])


def test_search_reports_stats():
    engine = orbit_native.Engine()
    result = engine.search(_obs(), 50)

    assert result["moves"] == [[0, 0.0, 6]]
    assert result["stats"]["states_considered"] >= 1
    assert 1 <= result["stats"]["route_queries"] <= result["stats"]["states_considered"]
    assert result["stats"]["elapsed_ms"] >= 0.0


def test_search_v2_reports_stats():
    engine = orbit_native.Engine()
    result = engine.search_v2(_obs(), 50)

    assert result["moves"] == [[0, 0.0, 6]]
    assert result["stats"]["states_considered"] >= 1
    assert 1 <= result["stats"]["route_queries"] <= result["stats"]["states_considered"]
    assert result["stats"]["elapsed_ms"] >= 0.0


def test_agent_does_not_resend_to_planet_already_captured_by_fleet():
    obs = {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 50, 1],
            [1, -1, 20.0, 10.0, 2.0, 20, 3],
        ],
        "initial_planets": [
            [0, 0, 10.0, 10.0, 2.0, 50, 1],
            [1, -1, 20.0, 10.0, 2.0, 20, 3],
        ],
        "fleets": [
            [10, 0, 17.0, 10.0, 0.0, 0, 25],
        ],
    }

    assert orbit_native.Engine().act(obs) == []


def test_replay_regression_moving_target_angle_not_one_tick_ahead():
    replay_path = Path("replays/episode-78246037-replay.json")
    if not replay_path.exists():
        return

    replay = json.loads(replay_path.read_text())
    obs = replay["steps"][341][0]["observation"]
    engine = orbit_native.Engine()
    engine.initialize(obs)
    route = engine.query_route(16, 12, 13, obs["step"])

    assert route["reachable"]
    assert route["arrival_tick"] == 345
    assert math.isclose(route["angle"], 1.5406296453576427)
    assert abs(route["angle"] - 1.5281103354705565) > 0.01
