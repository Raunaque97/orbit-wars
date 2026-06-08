import numpy as np
import pytest

torch = pytest.importorskip("torch")

import orbit_rl_native
from rl.model import FeatureSpec, build_graph_inputs, make_model, pad_graph_batch
from rl.model import amount_bin_for_move, amount_bin_ship_counts, forecast_surplus_for_planet


def _obs(extra_planet=False):
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 30, 2],
        [1, -1, 20.0, 10.0, 2.0, 5, 1],
    ]
    if extra_planet:
        planets.append([2, 1, 75.0, 80.0, 2.0, 15, 3])
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


def test_build_graph_inputs_shapes_and_owner_encoding():
    obs = _obs()
    batch = orbit_rl_native.FeatureEngine().compute(obs, horizon=50)
    graph = build_graph_inputs(obs, batch)
    spec = graph["spec"]

    assert graph["planet_features"].shape == (2, spec.planet_dim)
    assert graph["edge_features"].shape == (2, 2, spec.edge_dim)
    assert graph["planet_mask"].tolist() == [True, True]
    assert graph["planet_features"][0, :3].tolist() == [-1.0, -1.0, -1.0]
    assert graph["planet_features"][1, :3].tolist() == [0.0, 0.0, 0.0]
    assert np.isclose(graph["edge_features"][0, 1, 6].item(), 1.0)
    assert np.isclose(graph["edge_features"][0, 1, 8].item(), 2.0)


def test_graph_policy_forward_single_and_padded_batch():
    torch.manual_seed(7)
    spec = FeatureSpec()
    graphs = []
    for extra in (False, True):
        obs = _obs(extra_planet=extra)
        batch = orbit_rl_native.FeatureEngine().compute(obs, horizon=spec.horizon)
        graphs.append(build_graph_inputs(obs, batch, spec=spec))

    model = make_model(spec, hidden_dim=64, num_layers=2, num_heads=4)
    single = model(
        graphs[0]["planet_features"],
        graphs[0]["edge_features"],
        graphs[0]["planet_mask"],
    )
    assert single["planet_embeddings"].shape == (2, 64)
    assert single["edge_logits"].shape == (2, 2)
    assert single["amount_logits"].shape == (2, 2, 6)
    assert single["stop_logits"].shape == (2,)
    assert single["value"].shape == ()

    padded = pad_graph_batch(graphs)
    out = model(padded["planet_features"], padded["edge_features"], padded["planet_mask"])
    assert out["planet_embeddings"].shape == (2, 3, 64)
    assert out["edge_logits"].shape == (2, 3, 3)
    assert out["amount_logits"].shape == (2, 3, 3, 6)
    assert out["stop_logits"].shape == (2, 3)
    assert out["value"].shape == (2,)
    assert torch.isfinite(out["edge_logits"][0, :2, :2]).all()
    assert out["edge_logits"][0, 2, 0].item() < -1e20
    assert out["stop_logits"][0, 2].item() < -1e20


def test_amount_bin_mapping_uses_forecast_surplus():
    obs = _obs()
    batch = orbit_rl_native.FeatureEngine().compute(obs, horizon=50)

    surplus = forecast_surplus_for_planet(batch, planet_id=0, owner=0)
    assert surplus == 30
    assert amount_bin_ship_counts(source_ships=30, surplus=30, minimum_to_capture=6) == [
        7,
        6,
        15,
        24,
        30,
        30,
    ]
    assert amount_bin_for_move(23, source_ships=30, surplus=30, minimum_to_capture=6) == 3
