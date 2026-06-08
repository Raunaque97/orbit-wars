from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.distributions import Bernoulli, Categorical

import orbit_rl_native
from rl.model import (
    ALLY_DELAY_BUCKET,
    BLOCKED_DELAY,
    FeatureSpec,
    ROUTE_TIMEOUT,
    amount_bin_ship_counts,
    build_graph_inputs,
    forecast_surplus_for_planet,
    minimum_to_capture_at_arrival,
    pad_graph_batch,
)


INVALID_PENALTIES = {
    "mincapture_unaffordable": -0.05,
    "bad_amount": -0.05,
    "route_sun": -0.15,
    "route_planet": -0.10,
    "route_bounds": -0.10,
    "route_timeout": -0.10,
    "route_wrong_planet": -0.10,
    "route_blocked": -0.10,
}


@dataclass
class PPODecision:
    source_idx: int
    stop_action: int
    target_idx: int | None = None
    amount_idx: int | None = None
    amount_mask: list[bool] | None = None
    invalid_reason: str | None = None


@dataclass
class PPOTransition:
    graph: dict[str, Any]
    decisions: list[PPODecision]
    old_logprob: float
    old_entropy: float
    value: float
    reward: float = 0.0
    done: bool = False
    moves: list[list[float | int]] = field(default_factory=list)
    invalid_counts: dict[str, int] = field(default_factory=dict)
    feature_stats: dict[str, float | int] = field(default_factory=dict)
    action_terms: int = 0


def _obs_get(obs: Any, name: str, fallback: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, fallback)
    if hasattr(obs, "get"):
        value = obs.get(name, fallback)
        return fallback if value is None else value
    return getattr(obs, name, fallback)


def _planet_table(obs: Any) -> list[list[Any]]:
    return list(_obs_get(obs, "planets", []) or [])


def _detach_graph(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "planet_features": graph["planet_features"].detach().cpu(),
        "edge_features": graph["edge_features"].detach().cpu(),
        "planet_mask": graph["planet_mask"].detach().cpu(),
        "planet_ids": list(graph["planet_ids"]),
    }


def _graph_to_device(graph: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "planet_features": graph["planet_features"].to(device),
        "edge_features": graph["edge_features"].to(device),
        "planet_mask": graph["planet_mask"].to(device),
        "planet_ids": list(graph["planet_ids"]),
    }


def _route_invalid_reason(blocked_by: str) -> str:
    if blocked_by in {"sun", "planet", "bounds", "timeout", "wrong_planet"}:
        return f"route_{blocked_by}"
    return "route_blocked"


def _add_invalid(transition: PPOTransition, reason: str) -> None:
    transition.invalid_counts[reason] = transition.invalid_counts.get(reason, 0) + 1
    transition.reward += INVALID_PENALTIES.get(reason, INVALID_PENALTIES["route_blocked"])


def _valid_amount_mask(
    *,
    candidates: list[int],
    amount_logits: torch.Tensor,
    amount_idx_zero_affordable: bool,
) -> torch.Tensor:
    mask = torch.tensor(
        [
            ships > 0 and (idx != 0 or amount_idx_zero_affordable)
            for idx, ships in enumerate(candidates)
        ],
        dtype=torch.bool,
        device=amount_logits.device,
    )
    if not bool(mask.any()):
        mask[-1] = True
    return mask


class PPOActor:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        spec: FeatureSpec | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.spec = spec or FeatureSpec()
        self.device = torch.device(device)
        self.engine = orbit_rl_native.FeatureEngine()

    def sample(
        self,
        obs: Any,
        *,
        feature_batch: dict[str, Any] | None = None,
        route_engine: orbit_rl_native.FeatureEngine | None = None,
    ) -> PPOTransition:
        engine = route_engine or self.engine
        batch = feature_batch if feature_batch is not None else engine.compute(obs, self.spec.horizon)
        graph = self.build_graph(obs, batch)

        with torch.no_grad():
            out = self.model(
                graph["planet_features"], graph["edge_features"], graph["planet_mask"]
            )

        return self.sample_from_output(
            obs,
            batch,
            graph,
            out,
            route_engine=engine,
        )

    def build_graph(self, obs: Any, feature_batch: dict[str, Any]) -> dict[str, Any]:
        return build_graph_inputs(obs, feature_batch, spec=self.spec, device=self.device)

    def sample_from_output(
        self,
        obs: Any,
        feature_batch: dict[str, Any],
        graph: dict[str, Any],
        out: dict[str, torch.Tensor],
        *,
        route_engine: orbit_rl_native.FeatureEngine | None = None,
    ) -> PPOTransition:
        player = int(_obs_get(obs, "player", 0))
        planets = _planet_table(obs)
        engine = route_engine or self.engine
        batch = feature_batch
        planet_ids = [int(pid) for pid in graph["planet_ids"]]
        planets_by_id = {int(p[0]): p for p in planets}
        ship_buckets = [int(v) for v in batch["ship_buckets"]]
        delay_bucket_index = ship_buckets.index(ALLY_DELAY_BUCKET)

        transition = PPOTransition(
            graph=_detach_graph(graph),
            decisions=[],
            old_logprob=0.0,
            old_entropy=0.0,
            value=float(out["value"].detach().cpu()),
        )
        if len(planet_ids) <= 1:
            return transition

        logprob = out["value"].new_zeros(())
        entropy = out["value"].new_zeros(())
        action_terms = 0
        moves: list[list[float | int]] = []

        for src_idx, source_id in enumerate(planet_ids):
            source = planets_by_id.get(source_id)
            if source is None or int(source[1]) != player or int(source[5]) <= 0:
                continue

            stop_dist = Bernoulli(logits=out["stop_logits"][src_idx])
            stop_sample = stop_dist.sample()
            stop_action = int(stop_sample.item())
            logprob = logprob + stop_dist.log_prob(stop_sample)
            entropy = entropy + stop_dist.entropy()
            action_terms += 1
            decision = PPODecision(source_idx=src_idx, stop_action=stop_action)
            transition.decisions.append(decision)

            if stop_action == 1:
                continue

            target_logits = out["edge_logits"][src_idx].clone()
            target_logits[src_idx] = float("-inf")
            if not torch.isfinite(target_logits).any():
                decision.invalid_reason = "route_blocked"
                _add_invalid(transition, decision.invalid_reason)
                continue
            target_dist = Categorical(logits=target_logits)
            target_sample = target_dist.sample()
            target_idx = int(target_sample.item())
            logprob = logprob + target_dist.log_prob(target_sample)
            entropy = entropy + target_dist.entropy()
            action_terms += 1
            decision.target_idx = target_idx

            target_id = planet_ids[target_idx]
            available = int(source[5])
            delay = int(batch["delays"][delay_bucket_index, src_idx, target_idx])
            coarse_arrival_delay = delay if delay < BLOCKED_DELAY else ROUTE_TIMEOUT
            surplus = forecast_surplus_for_planet(
                batch, source_id, player, self.spec.horizon
            )
            minimum_to_capture = minimum_to_capture_at_arrival(
                batch, target_id, player, coarse_arrival_delay
            )
            candidates = amount_bin_ship_counts(
                source_ships=available,
                surplus=surplus,
                minimum_to_capture=minimum_to_capture,
            )
            amount_logits = out["amount_logits"][src_idx, target_idx].clone()
            amount_mask = _valid_amount_mask(
                candidates=candidates,
                amount_logits=amount_logits,
                amount_idx_zero_affordable=minimum_to_capture <= available,
            )
            amount_logits = amount_logits.masked_fill(~amount_mask, float("-inf"))
            amount_dist = Categorical(logits=amount_logits)
            amount_sample = amount_dist.sample()
            amount_idx = int(amount_sample.item())
            logprob = logprob + amount_dist.log_prob(amount_sample)
            entropy = entropy + amount_dist.entropy()
            action_terms += 1
            decision.amount_idx = amount_idx
            decision.amount_mask = [bool(v) for v in amount_mask.detach().cpu().tolist()]

            ships = int(candidates[amount_idx])
            if ships <= 0 or ships > available:
                decision.invalid_reason = "bad_amount"
                _add_invalid(transition, decision.invalid_reason)
                continue

            exact_route = engine.query_route(obs, source_id, target_id, ships)
            if not exact_route["reachable"]:
                decision.invalid_reason = _route_invalid_reason(
                    str(exact_route.get("blocked_by", "blocked"))
                )
                _add_invalid(transition, decision.invalid_reason)
                continue

            moves.append([source_id, float(exact_route["angle"]), ships])

        transition.old_logprob = float(logprob.detach().cpu())
        transition.old_entropy = float(entropy.detach().cpu())
        transition.moves = moves
        transition.action_terms = action_terms
        return transition


def evaluate_transition(
    model: torch.nn.Module,
    transition: PPOTransition,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    graph = _graph_to_device(transition.graph, torch.device(device))
    out = model(graph["planet_features"], graph["edge_features"], graph["planet_mask"])

    logprob = out["value"].new_zeros(())
    entropy = out["value"].new_zeros(())
    for decision in transition.decisions:
        stop_dist = Bernoulli(logits=out["stop_logits"][decision.source_idx])
        stop_value = torch.tensor(
            float(decision.stop_action), dtype=out["value"].dtype, device=out["value"].device
        )
        logprob = logprob + stop_dist.log_prob(stop_value)
        entropy = entropy + stop_dist.entropy()

        if decision.stop_action == 1:
            continue
        if decision.target_idx is None or decision.amount_idx is None:
            continue

        target_logits = out["edge_logits"][decision.source_idx].clone()
        target_logits[decision.source_idx] = float("-inf")
        target_dist = Categorical(logits=target_logits)
        target_value = torch.tensor(
            decision.target_idx, dtype=torch.long, device=out["value"].device
        )
        logprob = logprob + target_dist.log_prob(target_value)
        entropy = entropy + target_dist.entropy()

        amount_dist = Categorical(
            logits=out["amount_logits"][decision.source_idx, decision.target_idx].masked_fill(
                ~torch.tensor(
                    decision.amount_mask or [True] * out["amount_logits"].shape[-1],
                    dtype=torch.bool,
                    device=out["value"].device,
                ),
                float("-inf"),
            )
        )
        amount_value = torch.tensor(
            decision.amount_idx, dtype=torch.long, device=out["value"].device
        )
        logprob = logprob + amount_dist.log_prob(amount_value)
        entropy = entropy + amount_dist.entropy()

    return logprob, entropy, out["value"]


def evaluate_transitions_batch(
    model: torch.nn.Module,
    transitions: list[PPOTransition],
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not transitions:
        raise ValueError("evaluate_transitions_batch needs at least one transition")

    target_device = torch.device(device)
    graphs = [_graph_to_device(transition.graph, target_device) for transition in transitions]
    graph = pad_graph_batch(graphs)
    out = model(graph["planet_features"], graph["edge_features"], graph["planet_mask"])

    logprobs = []
    entropies = []
    for batch_idx, transition in enumerate(transitions):
        logprob = out["value"].new_zeros(())
        entropy = out["value"].new_zeros(())
        for decision in transition.decisions:
            stop_dist = Bernoulli(logits=out["stop_logits"][batch_idx, decision.source_idx])
            stop_value = torch.tensor(
                float(decision.stop_action),
                dtype=out["value"].dtype,
                device=out["value"].device,
            )
            logprob = logprob + stop_dist.log_prob(stop_value)
            entropy = entropy + stop_dist.entropy()

            if decision.stop_action == 1:
                continue
            if decision.target_idx is None or decision.amount_idx is None:
                continue

            target_logits = out["edge_logits"][batch_idx, decision.source_idx].clone()
            target_logits[decision.source_idx] = float("-inf")
            target_dist = Categorical(logits=target_logits)
            target_value = torch.tensor(
                decision.target_idx, dtype=torch.long, device=out["value"].device
            )
            logprob = logprob + target_dist.log_prob(target_value)
            entropy = entropy + target_dist.entropy()

            amount_mask = torch.tensor(
                decision.amount_mask
                or [True] * out["amount_logits"].shape[-1],
                dtype=torch.bool,
                device=out["value"].device,
            )
            amount_logits = out["amount_logits"][
                batch_idx, decision.source_idx, decision.target_idx
            ].masked_fill(~amount_mask, float("-inf"))
            amount_dist = Categorical(logits=amount_logits)
            amount_value = torch.tensor(
                decision.amount_idx, dtype=torch.long, device=out["value"].device
            )
            logprob = logprob + amount_dist.log_prob(amount_value)
            entropy = entropy + amount_dist.entropy()

        logprobs.append(logprob)
        entropies.append(entropy)

    return torch.stack(logprobs), torch.stack(entropies), out["value"]
