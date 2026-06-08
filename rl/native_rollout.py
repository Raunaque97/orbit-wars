from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Categorical

import orbit_rollout_native
from rl import orbit_map as ow
from rl.model import FeatureSpec, make_model
from rl.native_env import NativeOrbitEnv, _spawn_comets


NATIVE_MODEL_DIR = "native_models"
NATIVE_DELAY_CACHE_DIR = Path("rl/data/native_delay_cache")


def export_torchscript_model(
    model: nn.Module,
    spec: FeatureSpec,
    path: Path,
    *,
    batch_size: int = 16,
    planets: int = 32,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    example = (
        torch.randn(batch_size, planets, spec.planet_dim, dtype=torch.float32),
        torch.randn(batch_size, planets, planets, spec.edge_dim, dtype=torch.float32),
        torch.ones(batch_size, planets, dtype=torch.bool),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(model.cpu(), example, strict=False)
        traced = torch.jit.freeze(traced)
        traced = torch.jit.optimize_for_inference(traced)
        traced(*example)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    traced.save(str(tmp_path))
    os.replace(tmp_path, path)
    if was_training:
        model.train()
    return path


def export_torchscript_state(
    state: dict[str, torch.Tensor],
    spec: FeatureSpec,
    path: Path,
) -> Path:
    model = make_model(spec)
    model.load_state_dict(state, strict=False)
    return export_torchscript_model(model, spec, path)


def _spawn_event_for(initial_state: dict[str, Any], seed: int, spawn_step: int) -> dict[str, Any] | None:
    state = copy.deepcopy(initial_state)
    state["step"] = spawn_step - 1
    state["comets"] = []
    state["comet_planet_ids"] = []
    base_planet_count = len(state.get("planets", []))
    base_initial_count = len(state.get("initial_planets", []))
    _spawn_comets(state, seed=seed, comet_speed=4.0)
    if len(state.get("planets", [])) == base_planet_count:
        return None
    return {
        "step": spawn_step,
        "planets": state["planets"][base_planet_count:],
        "initial_planets": state["initial_planets"][base_initial_count:],
        "comet_planet_ids": state.get("comet_planet_ids", []),
        "comets": state.get("comets", []),
    }


def initial_state_and_spawns(seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = NativeOrbitEnv(seed=seed, num_agents=2)
    initial = copy.deepcopy(env._state)
    spawns = []
    for spawn_step in ow.COMET_SPAWN_STEPS:
        event = _spawn_event_for(initial, seed, spawn_step)
        if event is not None:
            spawns.append(event)
    return initial, spawns


def collect_native_rollout(
    *,
    learner_model_path: Path,
    seeds: list[int],
    opponents: list[dict[str, Any]],
    random_v2_policy_prob: float,
    max_steps: int,
    seed: int,
    early_ship_share: float,
    early_production_share: float,
    delay_cache_dir: Path = NATIVE_DELAY_CACHE_DIR,
    torch_threads: int = 4,
    worker_threads: int = 1,
) -> dict[str, Any]:
    initial_states = []
    spawn_events = []
    for episode_seed in seeds:
        initial, spawns = initial_state_and_spawns(int(episode_seed))
        initial_states.append(initial)
        spawn_events.append(spawns)
    return orbit_rollout_native.collect_rollout(
        initial_states,
        spawn_events,
        opponents,
        str(learner_model_path),
        float(random_v2_policy_prob),
        int(max_steps),
        int(seed),
        float(early_ship_share),
        float(early_production_share),
        str(delay_cache_dir),
        int(torch_threads),
        int(worker_threads),
    )


def concat_native_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("concat_native_batches needs at least one batch")
    if len(batches) == 1:
        return batches[0]

    max_n = max(int(batch["planet_features"].shape[1]) for batch in batches)
    planet_dim = int(batches[0]["planet_features"].shape[-1])
    edge_dim = int(batches[0]["edge_features"].shape[-1])
    total_t = sum(int(batch["planet_features"].shape[0]) for batch in batches)
    planet_features = torch.zeros((total_t, max_n, planet_dim), dtype=torch.float32)
    edge_features = torch.zeros((total_t, max_n, max_n, edge_dim), dtype=torch.float32)
    planet_mask = torch.zeros((total_t, max_n), dtype=torch.bool)

    tensor_keys = [
        "old_logprob",
        "old_entropy",
        "value",
        "reward",
        "done",
        "action_terms",
    ]
    merged: dict[str, Any] = {
        key: torch.cat([batch[key] for batch in batches], dim=0) for key in tensor_keys
    }

    transition_cursor = 0
    decision_cursor = 0
    episode_cursor = 0
    decision_offsets = [0]
    episode_offsets = [0]
    decision_parts = {key: [] for key in ("source_idx", "stop_action", "target_idx", "amount_idx", "amount_mask")}
    episode_lengths = []
    final_rewards = []
    invalid_counts: dict[str, int] = {}
    stats = {"feature_calls": 0, "delay_cache_hits": 0, "feature_ms": 0.0}

    for batch in batches:
        t = int(batch["planet_features"].shape[0])
        n = int(batch["planet_features"].shape[1])
        planet_features[transition_cursor : transition_cursor + t, :n] = batch["planet_features"]
        edge_features[transition_cursor : transition_cursor + t, :n, :n] = batch["edge_features"]
        planet_mask[transition_cursor : transition_cursor + t, :n] = batch["planet_mask"]

        local_decision_offsets = batch["decision_offsets"].to(torch.long).tolist()
        for offset in local_decision_offsets[1:]:
            decision_offsets.append(decision_cursor + int(offset))
        for key in decision_parts:
            decision_parts[key].append(batch[key])
        decision_cursor += int(local_decision_offsets[-1])

        local_episode_offsets = batch["episode_offsets"].to(torch.long).tolist()
        for offset in local_episode_offsets[1:]:
            episode_offsets.append(transition_cursor + int(offset))
        episode_lengths.append(batch["episode_lengths"])
        final_rewards.append(batch["final_rewards"])

        for key, value in dict(batch.get("invalid_counts", {})).items():
            invalid_counts[str(key)] = invalid_counts.get(str(key), 0) + int(value)
        batch_stats = dict(batch.get("stats", {}))
        calls = int(batch_stats.get("feature_calls", 0))
        stats["feature_ms"] += float(batch_stats.get("feature_ms", 0.0)) * calls
        stats["feature_calls"] += calls
        stats["delay_cache_hits"] += int(batch_stats.get("delay_cache_hits", 0))
        transition_cursor += t
        episode_cursor += len(local_episode_offsets) - 1

    merged["planet_features"] = planet_features
    merged["edge_features"] = edge_features
    merged["planet_mask"] = planet_mask
    merged["decision_offsets"] = torch.tensor(decision_offsets, dtype=torch.long)
    merged["episode_offsets"] = torch.tensor(episode_offsets, dtype=torch.long)
    for key, parts in decision_parts.items():
        merged[key] = torch.cat(parts, dim=0) if parts else batches[0][key][:0]
    merged["episode_lengths"] = torch.cat(episode_lengths, dim=0)
    merged["final_rewards"] = torch.cat(final_rewards, dim=0)
    merged["invalid_counts"] = invalid_counts
    stats["feature_ms"] = stats["feature_ms"] / max(1, int(stats["feature_calls"]))
    merged["stats"] = stats
    return merged


def compute_native_gae(
    batch: dict[str, Any], gamma: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = batch["reward"].to(torch.float32)
    values = batch["value"].to(torch.float32)
    done = batch["done"].to(torch.bool)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    offsets = batch["episode_offsets"].to(torch.long).tolist()
    for start, end in zip(offsets[:-1], offsets[1:]):
        gae = 0.0
        next_value = 0.0
        for idx in range(int(end) - 1, int(start) - 1, -1):
            mask = 0.0 if bool(done[idx]) else 1.0
            delta = float(rewards[idx]) + gamma * next_value * mask - float(values[idx])
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[idx] = gae
            returns[idx] = gae + float(values[idx])
            next_value = float(values[idx])
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    return advantages, returns


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    returns_var = torch.var(returns, unbiased=False)
    if float(returns_var.detach().cpu()) <= 1e-12:
        return values.new_tensor(0.0)
    residual_var = torch.var(returns - values, unbiased=False)
    return 1.0 - residual_var / returns_var


def evaluate_native_batch(
    model: nn.Module,
    batch: dict[str, Any],
    indices: list[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    out = model(
        batch["planet_features"].index_select(0, idx_tensor).to(device),
        batch["edge_features"].index_select(0, idx_tensor).to(device),
        batch["planet_mask"].index_select(0, idx_tensor).to(device),
    )
    logprobs = []
    entropies = []
    decision_offsets = batch["decision_offsets"].to(torch.long)
    source_idx = batch["source_idx"].to(torch.long)
    stop_action = batch["stop_action"].to(torch.long)
    target_idx = batch["target_idx"].to(torch.long)
    amount_idx = batch["amount_idx"].to(torch.long)
    amount_mask = batch["amount_mask"].to(torch.bool)

    for local_idx, transition_idx in enumerate(indices):
        logprob = out["value"].new_zeros(())
        entropy = out["value"].new_zeros(())
        start = int(decision_offsets[transition_idx])
        end = int(decision_offsets[transition_idx + 1])
        for decision_idx in range(start, end):
            src = int(source_idx[decision_idx])
            stop_dist = Bernoulli(logits=out["stop_logits"][local_idx, src])
            stop_value = torch.tensor(
                float(stop_action[decision_idx]), dtype=out["value"].dtype, device=device
            )
            logprob = logprob + stop_dist.log_prob(stop_value)
            entropy = entropy + stop_dist.entropy()
            if int(stop_action[decision_idx]) == 1:
                continue
            tgt = int(target_idx[decision_idx])
            amt = int(amount_idx[decision_idx])
            if tgt < 0 or amt < 0:
                continue
            target_logits = out["edge_logits"][local_idx, src].clone()
            target_logits[src] = float("-inf")
            target_dist = Categorical(logits=target_logits)
            target_value = torch.tensor(tgt, dtype=torch.long, device=device)
            logprob = logprob + target_dist.log_prob(target_value)
            entropy = entropy + target_dist.entropy()

            mask = amount_mask[decision_idx].to(device)
            amount_logits = out["amount_logits"][local_idx, src, tgt].masked_fill(
                ~mask, float("-inf")
            )
            amount_dist = Categorical(logits=amount_logits)
            amount_value = torch.tensor(amt, dtype=torch.long, device=device)
            logprob = logprob + amount_dist.log_prob(amount_value)
            entropy = entropy + amount_dist.entropy()
        logprobs.append(logprob)
        entropies.append(entropy)
    return torch.stack(logprobs), torch.stack(entropies), out["value"]


def ppo_update_native(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: Any,
) -> dict[str, float]:
    device = torch.device(config.device)
    indices = list(range(int(batch["reward"].shape[0])))
    rng = torch.Generator().manual_seed(int(config.seed))
    last_stats = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "approx_kl_per_decision": 0.0,
        "clip_fraction_per_decision": 0.0,
        "mean_action_terms": 0.0,
        "explained_variance": 0.0,
    }
    metric_sums = {key: 0.0 for key in last_stats if key not in {"loss", "policy_loss", "value_loss", "entropy"}}
    metric_weight = 0
    old_all = batch["old_logprob"].to(torch.float32)
    terms_all = batch["action_terms"].to(torch.float32).clamp_min(1.0)

    for _epoch in range(config.update_epochs):
        perm = torch.randperm(len(indices), generator=rng).tolist()
        for start in range(0, len(perm), config.minibatch_size):
            batch_indices = perm[start : start + config.minibatch_size]
            logprobs, entropies, values = evaluate_native_batch(
                model, batch, batch_indices, device=device
            )
            old_logprobs = old_all[batch_indices].to(device)
            action_terms = terms_all[batch_indices].to(device)
            batch_advantages = advantages[batch_indices].to(device)
            target_returns = returns[batch_indices].to(device)
            logratio = logprobs - old_logprobs
            ratios = torch.exp(logratio)
            unclipped = ratios * batch_advantages
            clipped = torch.clamp(
                ratios, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
            ) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, target_returns)
            entropy_bonus = entropies.mean()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy_bonus
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratios - 1.0) - logratio).mean()
                clip_fraction = ((ratios - 1.0).abs() > config.clip_ratio).float().mean()
                logratio_per_decision = logratio / action_terms
                ratios_per_decision = torch.exp(logratio_per_decision)
                approx_kl_per_decision = (
                    (ratios_per_decision - 1.0) - logratio_per_decision
                ).mean()
                clip_fraction_per_decision = (
                    (ratios_per_decision - 1.0).abs() > config.clip_ratio
                ).float().mean()
                explained_variance = _explained_variance(values, target_returns)
                weight = len(batch_indices)
                metric_sums["approx_kl"] += float(approx_kl.cpu()) * weight
                metric_sums["clip_fraction"] += float(clip_fraction.cpu()) * weight
                metric_sums["approx_kl_per_decision"] += float(approx_kl_per_decision.cpu()) * weight
                metric_sums["clip_fraction_per_decision"] += float(clip_fraction_per_decision.cpu()) * weight
                metric_sums["mean_action_terms"] += float(action_terms.mean().cpu()) * weight
                metric_sums["explained_variance"] += float(explained_variance.cpu()) * weight
                metric_weight += weight
            last_stats = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy_bonus.detach().cpu()),
                "approx_kl": metric_sums["approx_kl"] / max(1, metric_weight),
                "clip_fraction": metric_sums["clip_fraction"] / max(1, metric_weight),
                "approx_kl_per_decision": metric_sums["approx_kl_per_decision"] / max(1, metric_weight),
                "clip_fraction_per_decision": metric_sums["clip_fraction_per_decision"] / max(1, metric_weight),
                "mean_action_terms": metric_sums["mean_action_terms"] / max(1, metric_weight),
                "explained_variance": metric_sums["explained_variance"] / max(1, metric_weight),
            }
    return last_stats
