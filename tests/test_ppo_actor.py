import pytest

torch = pytest.importorskip("torch")
from torch import nn

from rl.ppo_actor import PPOActor, evaluate_transition, evaluate_transitions_batch


class _LaunchPolicy(nn.Module):
    def forward(self, planet_features, edge_features, planet_mask):
        squeeze = planet_features.dim() == 2
        if squeeze:
            planet_features = planet_features.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
            planet_mask = planet_mask.unsqueeze(0)
        bsz, n, _dim = planet_features.shape
        edge_logits = torch.full((bsz, n, n), -20.0)
        amount_logits = torch.full((bsz, n, n, 6), -20.0)
        stop_logits = torch.full((bsz, n), 20.0)
        stop_logits[:, 0] = -20.0
        edge_logits[:, 0, 1] = 20.0
        amount_logits[:, 0, 1, 5] = 20.0
        value = torch.full((bsz,), 0.5)
        out = {
            "planet_embeddings": torch.zeros((bsz, n, 8)),
            "edge_logits": edge_logits,
            "amount_logits": amount_logits,
            "stop_logits": stop_logits,
            "value": value,
        }
        if squeeze:
            out = {key: value.squeeze(0) for key, value in out.items()}
        return {
            **out,
        }


def _obs():
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 50, 2],
        [1, -1, 20.0, 10.0, 2.0, 100, 1],
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


def test_ppo_actor_samples_moves_and_replays_logprob():
    torch.manual_seed(1)
    model = _LaunchPolicy()
    actor = PPOActor(model)

    transition = actor.sample(_obs())

    assert transition.moves
    assert transition.moves[0][0] == 0
    assert transition.moves[0][2] == 50
    assert transition.value == 0.5
    assert transition.decisions[0].stop_action == 0
    logprob, entropy, value = evaluate_transition(model, transition)
    assert torch.isfinite(logprob)
    assert torch.isfinite(entropy)
    assert value.item() == 0.5


def test_batched_transition_eval_matches_single_eval():
    torch.manual_seed(1)
    model = _LaunchPolicy()
    actor = PPOActor(model)
    transition = actor.sample(_obs())

    single_logprob, single_entropy, single_value = evaluate_transition(model, transition)
    batch_logprobs, batch_entropies, batch_values = evaluate_transitions_batch(
        model, [transition]
    )

    assert batch_logprobs.shape == (1,)
    assert batch_entropies.shape == (1,)
    assert batch_values.shape == (1,)
    assert batch_logprobs[0].item() == pytest.approx(single_logprob.item())
    assert batch_entropies[0].item() == pytest.approx(single_entropy.item())
    assert batch_values[0].item() == pytest.approx(single_value.item())
