from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rl.model import FeatureSpec, make_model
from rl.native_rollout import (
    collect_native_rollout,
    compute_native_gae,
    export_torchscript_model,
    ppo_update_native,
)


def _checkpoint_model():
    payload = torch.load(
        "rl/checkpoints/bc_opening_v1/best.pt", map_location="cpu", weights_only=False
    )
    spec = payload.get("spec", FeatureSpec())
    model = make_model(spec)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, spec


def test_native_rollout_collects_tensor_batch(tmp_path: Path):
    model, spec = _checkpoint_model()
    model_path = export_torchscript_model(model, spec, tmp_path / "learner.ts.pt", batch_size=2)

    batch = collect_native_rollout(
        learner_model_path=model_path,
        seeds=[0, 1],
        opponents=[
            {"name": "random_v2", "model_path": None, "population_index": 0},
            {"name": "random_v2", "model_path": None, "population_index": 0},
        ],
        random_v2_policy_prob=1.0,
        max_steps=3,
        seed=11,
        early_ship_share=0.8,
        early_production_share=0.7,
        delay_cache_dir=tmp_path / "delay_cache",
        worker_threads=2,
    )

    assert batch["planet_features"].shape[0] == 6
    assert batch["planet_features"].shape[-1] == spec.planet_dim
    assert batch["edge_features"].shape[-1] == spec.edge_dim
    assert batch["decision_offsets"].shape[0] == batch["reward"].shape[0] + 1
    assert batch["episode_lengths"].tolist() == [3, 3]


def test_native_rollout_batch_replays_through_ppo_update(tmp_path: Path):
    model, spec = _checkpoint_model()
    model_path = export_torchscript_model(model, spec, tmp_path / "learner.ts.pt", batch_size=2)
    batch = collect_native_rollout(
        learner_model_path=model_path,
        seeds=[2, 3],
        opponents=[
            {"name": "random_v2", "model_path": None, "population_index": 0},
            {"name": "random_v2", "model_path": None, "population_index": 0},
        ],
        random_v2_policy_prob=1.0,
        max_steps=4,
        seed=12,
        early_ship_share=0.8,
        early_production_share=0.7,
        delay_cache_dir=tmp_path / "delay_cache",
        worker_threads=2,
    )
    advantages, returns = compute_native_gae(batch, 0.995, 0.95)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    config = SimpleNamespace(
        device="cpu",
        seed=7,
        update_epochs=1,
        minibatch_size=4,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.001,
        max_grad_norm=1.0,
    )

    stats = ppo_update_native(model, optimizer, batch, advantages, returns, config)

    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()
    assert stats["loss"] == pytest.approx(stats["loss"])
    assert "approx_kl" in stats
