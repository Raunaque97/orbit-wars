from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

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


class _FixedPolicy(nn.Module):
    def __init__(self, stop_logit: float, *, amount_index: int = 0) -> None:
        super().__init__()
        self.stop_logit = stop_logit
        self.amount_index = amount_index

    def forward(self, planet_features, edge_features, planet_mask):
        n = int(planet_features.shape[0])
        edge_logits = torch.full((n, n), -10.0)
        amount_logits = torch.full((n, n, 6), -10.0)
        stop_logits = torch.full((n,), 10.0)
        stop_logits[0] = self.stop_logit
        edge_logits[0, 1] = 10.0
        amount_logits[0, 1, self.amount_index] = 10.0
        return {
            "planet_embeddings": torch.zeros((n, 8)),
            "edge_logits": edge_logits,
            "amount_logits": amount_logits,
            "stop_logits": stop_logits,
            "value": torch.tensor(0.0),
        }


def _checkpoint(tmp_path: Path) -> Path:
    spec = FeatureSpec()
    model = make_model(spec)
    checkpoint = tmp_path / "policy.pt"
    torch.save({"model": model.state_dict(), "spec": spec, "epoch": 0, "loss": 0.0}, checkpoint)
    return checkpoint


def test_rl_agent_launches_one_move_per_source_when_stop_probability_is_low(
    tmp_path: Path,
):
    agent = OrbitWarsRLAgent(_checkpoint(tmp_path))
    agent.model = _FixedPolicy(stop_logit=-10.0)
    moves = agent.act(_obs())

    assert len(moves) == 1
    source_id, _angle, ships = moves[0]
    assert source_id == 0
    assert ships == 7


def test_rl_agent_saves_ships_when_stop_probability_is_high(tmp_path: Path):
    agent = OrbitWarsRLAgent(_checkpoint(tmp_path))
    agent.model = _FixedPolicy(stop_logit=10.0)

    assert agent.act(_obs()) == []


def test_rl_agent_allows_underpowered_non_mincapture_fleets(tmp_path: Path):
    obs = _obs()
    obs["planets"][1][5] = 100
    agent = OrbitWarsRLAgent(_checkpoint(tmp_path))
    agent.model = _FixedPolicy(stop_logit=-10.0, amount_index=5)

    moves = agent.act(obs)

    assert len(moves) == 1
    source_id, _angle, ships = moves[0]
    assert source_id == 0
    assert ships == 50
