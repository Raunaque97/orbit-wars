import math

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


def test_search_reports_stats():
    engine = orbit_native.Engine()
    result = engine.search(_obs(), 50)

    assert result["moves"] == [[0, 0.0, 6]]
    assert result["stats"]["states_considered"] >= 1
    assert result["stats"]["route_queries"] == result["stats"]["states_considered"]
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
