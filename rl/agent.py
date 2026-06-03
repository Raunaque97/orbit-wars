from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

import orbit_rl_native
from rl.model import (
    ALLY_DELAY_BUCKET,
    BLOCKED_DELAY,
    FeatureSpec,
    amount_bin_ship_counts,
    build_graph_inputs,
    forecast_surplus_for_planet,
    make_model,
    minimum_to_capture_at_arrival,
)


DEFAULT_CHECKPOINT = Path("rl/checkpoints/bc_v1/best.pt")


def _obs_get(obs: Any, name: str, fallback: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, fallback)
    if hasattr(obs, "get"):
        value = obs.get(name, fallback)
        return fallback if value is None else value
    return getattr(obs, name, fallback)


def _planet_table(obs: Any) -> list[list[Any]]:
    return list(_obs_get(obs, "planets", []) or [])


class OrbitWarsRLAgent:
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        max_actions: int = 6,
        allow_comet_launches: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint = Path(
            checkpoint or os.environ.get("ORBIT_RL_CHECKPOINT", DEFAULT_CHECKPOINT)
        )
        self.max_actions = max_actions
        self.allow_comet_launches = allow_comet_launches
        self.device = torch.device(device)
        self.engine = orbit_rl_native.FeatureEngine()
        self.spec = FeatureSpec()
        self.model = make_model(self.spec).to(self.device)
        self._load_checkpoint(self.checkpoint)
        self.model.eval()

    def _load_checkpoint(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.spec = payload.get("spec", FeatureSpec())
        self.model = make_model(self.spec).to(self.device)
        self.model.load_state_dict(payload["model"])

    def act(self, obs: Any) -> list[list[float | int]]:
        player = int(_obs_get(obs, "player", 0))
        planets = _planet_table(obs)
        if not planets:
            return []

        batch = self.engine.compute(obs, self.spec.horizon)
        graph = build_graph_inputs(obs, batch, spec=self.spec, device=self.device)
        planet_ids = [int(pid) for pid in graph["planet_ids"]]
        planets_by_id = {int(p[0]): p for p in planets}
        ship_buckets = [int(v) for v in batch["ship_buckets"]]
        delay_bucket_index = ship_buckets.index(ALLY_DELAY_BUCKET)

        with torch.inference_mode():
            out = self.model(
                graph["planet_features"], graph["edge_features"], graph["planet_mask"]
            )
        edge_logits = out["edge_logits"].detach().cpu()
        amount_logits = out["amount_logits"].detach().cpu()

        flat_order = torch.argsort(edge_logits.reshape(-1), descending=True).tolist()
        spent_by_source: dict[int, int] = {}
        moves: list[list[float | int]] = []

        n = len(planet_ids)
        for flat_index in flat_order:
            if len(moves) >= self.max_actions:
                break
            src_idx = flat_index // n
            target_idx = flat_index % n
            if src_idx == target_idx:
                continue

            source_id = planet_ids[src_idx]
            target_id = planet_ids[target_idx]
            source = planets_by_id.get(source_id)
            if source is None or int(source[1]) != player:
                continue

            current_ships = int(source[5])
            already_spent = spent_by_source.get(source_id, 0)
            available = current_ships - already_spent
            if available <= 0:
                continue

            delay = int(batch["delays"][delay_bucket_index, src_idx, target_idx])
            if delay >= BLOCKED_DELAY:
                continue

            surplus = max(
                0,
                forecast_surplus_for_planet(batch, source_id, player, self.spec.horizon)
                - already_spent,
            )
            minimum_to_capture = minimum_to_capture_at_arrival(
                batch, target_id, player, delay
            )
            amount_idx = int(torch.argmax(amount_logits[src_idx, target_idx]).item())
            if amount_idx == 0 and minimum_to_capture > available:
                continue

            candidates = amount_bin_ship_counts(
                source_ships=available,
                surplus=surplus,
                minimum_to_capture=minimum_to_capture,
            )
            ships = int(candidates[amount_idx])
            if ships <= 0 or ships > available:
                continue

            exact_route = self.engine.query_route(obs, source_id, target_id, ships)
            if not exact_route["reachable"]:
                continue
            exact_minimum_to_capture = minimum_to_capture_at_arrival(
                batch, target_id, player, int(exact_route["delay"])
            )
            if exact_minimum_to_capture > 0 and ships < exact_minimum_to_capture:
                continue

            spent_by_source[source_id] = already_spent + ships
            moves.append([source_id, float(exact_route["angle"]), ships])

        return moves


_GLOBAL_AGENT: OrbitWarsRLAgent | None = None


def agent(obs: Any) -> list[list[float | int]]:
    global _GLOBAL_AGENT
    if _GLOBAL_AGENT is None:
        _GLOBAL_AGENT = OrbitWarsRLAgent()
    return _GLOBAL_AGENT.act(obs)
