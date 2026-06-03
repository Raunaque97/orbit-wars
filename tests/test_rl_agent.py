from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from rl.agent import OrbitWarsRLAgent
from rl.model import FeatureSpec, make_model


def _obs():
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 50, 2],
        [1, -1, 20.0, 10.0, 2.0, 5, 1],
        [2, 1, 80.0, 80.0, 2.0, 20, 3],
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


def test_rl_agent_harness_returns_legal_moves(tmp_path: Path):
    spec = FeatureSpec()
    model = make_model(spec)
    checkpoint = tmp_path / "policy.pt"
    torch.save({"model": model.state_dict(), "spec": spec, "epoch": 0, "loss": 0.0}, checkpoint)

    agent = OrbitWarsRLAgent(checkpoint, max_actions=3)
    moves = agent.act(_obs())

    assert len(moves) <= 3
    for source_id, _angle, ships in moves:
        assert source_id == 0
        assert 0 < ships <= 50
