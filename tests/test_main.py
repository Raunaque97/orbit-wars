import main


def test_agent_uses_native_module():
    obs = {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 50, 1],
            [1, -1, 20.0, 10.0, 2.0, 5, 3],
        ],
        "initial_planets": [
            [0, 0, 10.0, 10.0, 2.0, 50, 1],
            [1, -1, 20.0, 10.0, 2.0, 5, 3],
        ],
    }

    assert main.agent(obs) == [[0, 0.0, 6]]
