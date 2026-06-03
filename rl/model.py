from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


CENTER_X = 50.0
CENTER_Y = 50.0
MAX_STEPS = 500
MISSING_OWNER = -2
BLOCKED_DELAY = 200
ROUTE_TIMEOUT = 141
EDGE_DELAY_BUCKETS = (5, 10, 20, 40, 80, 160)
ALLY_DELAY_BUCKET = 20


@dataclass(frozen=True)
class FeatureSpec:
    horizon: int = 50
    nearest_allies: int = 3

    @property
    def planet_dim(self) -> int:
        return 12 + self.horizon * 4 + self.nearest_allies * 2

    @property
    def edge_dim(self) -> int:
        return 11


def _obs_get(obs: Any, name: str, fallback: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, fallback)
    if hasattr(obs, "get"):
        value = obs.get(name, fallback)
        return fallback if value is None else value
    return getattr(obs, name, fallback)


def _owner_vec(owner: int, player: int) -> list[float]:
    if owner == MISSING_OWNER or owner == -1:
        return [0.0, 0.0, 0.0]
    if owner == player:
        return [-1.0, -1.0, -1.0]
    enemies = [pid for pid in range(4) if pid != player]
    out = [0.0, 0.0, 0.0]
    if owner in enemies:
        out[enemies.index(owner)] = 1.0
    return out


def _is_orbiting(planet: np.ndarray) -> bool:
    dx = float(planet[2]) - CENTER_X
    dy = float(planet[3]) - CENTER_Y
    return math.hypot(dx, dy) + float(planet[4]) < 50.0


def _segment_circle_intersects(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float = CENTER_X,
    cy: float = CENTER_Y,
    radius: float = 10.0,
) -> bool:
    abx = bx - ax
    aby = by - ay
    length_sq = abx * abx + aby * aby
    if length_sq <= 1e-12:
        return (ax - cx) ** 2 + (ay - cy) ** 2 <= radius * radius
    t = max(0.0, min(1.0, ((cx - ax) * abx + (cy - ay) * aby) / length_sq))
    px = ax + abx * t
    py = ay + aby * t
    return (px - cx) ** 2 + (py - cy) ** 2 <= radius * radius


def _comet_remaining_by_id(obs: Any) -> dict[int, int]:
    remaining: dict[int, int] = {}
    for group in _obs_get(obs, "comets", []) or []:
        if isinstance(group, dict):
            planet_ids = group.get("planet_ids", [])
            paths = group.get("paths", [])
            path_index = int(group.get("path_index", 0))
        else:
            planet_ids = getattr(group, "planet_ids", [])
            paths = getattr(group, "paths", [])
            path_index = int(getattr(group, "path_index", 0))
        for planet_id, path in zip(planet_ids, paths):
            remaining[int(planet_id)] = max(0, len(path) - path_index)
    return remaining


def _future_owner_ship_features(
    garrisons: np.ndarray, player: int, horizon: int
) -> np.ndarray:
    n = garrisons.shape[0]
    out = np.zeros((n, horizon, 4), dtype=np.float32)
    limited = min(horizon, garrisons.shape[1])
    for i in range(n):
        for t in range(limited):
            ships = float(garrisons[i, t, 0])
            owner = int(garrisons[i, t, 1])
            out[i, t, :3] = _owner_vec(owner, player)
            out[i, t, 3] = ships
    return out.reshape(n, horizon * 4)


def build_graph_inputs(
    obs: Any,
    feature_batch: dict[str, Any],
    *,
    spec: FeatureSpec | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    spec = spec or FeatureSpec()
    player = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    raw_planets = np.asarray(_obs_get(obs, "planets", []), dtype=np.float32)
    if raw_planets.ndim != 2 or raw_planets.shape[1] != 7:
        raise ValueError("obs.planets must be an Nx7 table")

    planet_ids = [int(pid) for pid in feature_batch["planet_ids"]]
    by_id = {int(row[0]): row for row in raw_planets}
    planets = np.asarray([by_id[pid] for pid in planet_ids], dtype=np.float32)
    n = len(planet_ids)
    comet_ids = {int(pid) for pid in _obs_get(obs, "comet_planet_ids", []) or []}
    comet_remaining = _comet_remaining_by_id(obs)

    ship_buckets = [int(v) for v in feature_batch["ship_buckets"]]
    delays = np.asarray(feature_batch["delays"], dtype=np.float32)
    garrisons = np.asarray(feature_batch["garrisons"], dtype=np.float32)

    owner_total_prod: dict[int, float] = {}
    owner_total_ships: dict[int, float] = {}
    for planet in planets:
        owner = int(planet[1])
        owner_total_prod[owner] = owner_total_prod.get(owner, 0.0) + float(planet[6])
        owner_total_ships[owner] = owner_total_ships.get(owner, 0.0) + float(planet[5])

    future_flat = _future_owner_ship_features(garrisons, player, spec.horizon)
    ally_bucket_index = ship_buckets.index(ALLY_DELAY_BUCKET)

    planet_features = np.zeros((n, spec.planet_dim), dtype=np.float32)
    for i, planet in enumerate(planets):
        planet_id = planet_ids[i]
        owner = int(planet[1])
        total_ships = owner_total_ships.get(owner, 0.0)
        dx = float(planet[2]) - CENTER_X
        dy = float(planet[3]) - CENTER_Y
        is_comet = planet_id in comet_ids
        time_remaining = (
            float(comet_remaining.get(planet_id, 0)) if is_comet else float(MAX_STEPS - step)
        )

        same_owner: list[tuple[float, float]] = []
        for j, other in enumerate(planets):
            if i == j or int(other[1]) != owner:
                continue
            delay = float(delays[ally_bucket_index, i, j])
            same_owner.append((delay, float(other[5])))
        same_owner.sort(key=lambda item: item[0])

        ally_delays = [ROUTE_TIMEOUT] * spec.nearest_allies
        ally_ships = [0.0] * spec.nearest_allies
        for k, (delay, ships) in enumerate(same_owner[: spec.nearest_allies]):
            ally_delays[k] = min(delay, float(ROUTE_TIMEOUT))
            ally_ships[k] = ships

        values: list[float] = []
        values.extend(_owner_vec(owner, player))
        values.extend(
            [
                owner_total_prod.get(owner, 0.0),
                total_ships,
                float(planet[4]),
                math.hypot(dx, dy),
                1.0 if is_comet else 0.0,
                time_remaining,
                1.0 if _is_orbiting(planet) and not is_comet else 0.0,
                float(planet[6]),
                float(planet[5]) / max(1.0, total_ships),
            ]
        )
        values.extend(future_flat[i].tolist())
        values.extend(ally_delays)
        values.extend(ally_ships)
        planet_features[i] = np.asarray(values, dtype=np.float32)

    delay_indices = [ship_buckets.index(bucket) for bucket in EDGE_DELAY_BUCKETS]
    edge_features = np.zeros((n, n, spec.edge_dim), dtype=np.float32)
    edge_features[:, :, : len(delay_indices)] = delays[delay_indices].transpose(1, 2, 0)
    for i in range(n):
        for j in range(n):
            edge_features[i, j, 6] = planets[j, 6]
            edge_features[i, j, 7] = planets[j, 5]
            edge_features[i, j, 8] = planets[i, 6]
            edge_features[i, j, 9] = planets[i, 5]
            edge_features[i, j, 10] = (
                1.0
                if _segment_circle_intersects(
                    float(planets[i, 2]),
                    float(planets[i, 3]),
                    float(planets[j, 2]),
                    float(planets[j, 3]),
                )
                else 0.0
            )

    return {
        "planet_features": torch.as_tensor(planet_features, dtype=torch.float32, device=device),
        "edge_features": torch.as_tensor(edge_features, dtype=torch.float32, device=device),
        "planet_mask": torch.ones(n, dtype=torch.bool, device=device),
        "planet_ids": planet_ids,
        "spec": spec,
    }


def pad_graph_batch(graphs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(graphs)
    if not items:
        raise ValueError("pad_graph_batch needs at least one graph")

    max_n = max(int(item["planet_features"].shape[0]) for item in items)
    planet_dim = int(items[0]["planet_features"].shape[-1])
    edge_dim = int(items[0]["edge_features"].shape[-1])
    device = items[0]["planet_features"].device

    planets = torch.zeros((len(items), max_n, planet_dim), dtype=torch.float32, device=device)
    edges = torch.zeros((len(items), max_n, max_n, edge_dim), dtype=torch.float32, device=device)
    mask = torch.zeros((len(items), max_n), dtype=torch.bool, device=device)
    planet_ids: list[list[int]] = []

    for b, item in enumerate(items):
        n = int(item["planet_features"].shape[0])
        planets[b, :n] = item["planet_features"]
        edges[b, :n, :n] = item["edge_features"]
        mask[b, :n] = item.get("planet_mask", torch.ones(n, dtype=torch.bool, device=device))
        planet_ids.append(list(item["planet_ids"]))

    return {
        "planet_features": planets,
        "edge_features": edges,
        "planet_mask": mask,
        "planet_ids": planet_ids,
    }


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        *,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )

    def forward(
        self, h: torch.Tensor, edge_features: torch.Tensor, planet_mask: torch.Tensor
    ) -> torch.Tensor:
        bsz, n, _ = h.shape
        q = self.q(h).view(bsz, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(bsz, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(bsz, n, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge_features).permute(0, 3, 1, 2)
        key_mask = planet_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        message = torch.einsum("bhij,bhjd->bhid", attn, v)
        message = message.transpose(1, 2).contiguous().view(bsz, n, self.hidden_dim)
        h = self.norm_attn(h + self.dropout(self.out(message)))
        h = self.norm_ff(h + self.dropout(self.ff(h)))
        return h * planet_mask[:, :, None].to(h.dtype)


class OrbitWarsGraphPolicy(nn.Module):
    def __init__(
        self,
        planet_dim: int,
        edge_dim: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        amount_bins: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.planet_encoder = nn.Sequential(
            nn.Linear(planet_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim,
                    edge_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        pair_dim = hidden_dim * 2 + edge_dim
        self.edge_policy = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.amount_policy = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, amount_bins),
        )
        self.stop_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        planet_features: torch.Tensor,
        edge_features: torch.Tensor,
        planet_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        squeeze = False
        if planet_features.dim() == 2:
            planet_features = planet_features.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
            squeeze = True
        if planet_mask is None:
            planet_mask = torch.ones(
                planet_features.shape[:2], dtype=torch.bool, device=planet_features.device
            )
        elif planet_mask.dim() == 1:
            planet_mask = planet_mask.unsqueeze(0)

        h = self.planet_encoder(planet_features)
        h = h * planet_mask[:, :, None].to(h.dtype)
        for layer in self.layers:
            h = layer(h, edge_features, planet_mask)

        bsz, n, hidden_dim = h.shape
        src = h[:, :, None, :].expand(bsz, n, n, hidden_dim)
        dst = h[:, None, :, :].expand(bsz, n, n, hidden_dim)
        pair = torch.cat([src, dst, edge_features], dim=-1)
        edge_logits = self.edge_policy(pair).squeeze(-1)
        amount_logits = self.amount_policy(pair)
        stop_logits = self.stop_policy(h).squeeze(-1)

        pair_mask = planet_mask[:, :, None] & planet_mask[:, None, :]
        edge_logits = edge_logits.masked_fill(~pair_mask, torch.finfo(edge_logits.dtype).min)
        amount_logits = amount_logits.masked_fill(
            ~pair_mask[:, :, :, None], torch.finfo(amount_logits.dtype).min
        )
        stop_logits = stop_logits.masked_fill(~planet_mask, torch.finfo(stop_logits.dtype).min)

        out = {
            "planet_embeddings": h,
            "edge_logits": edge_logits,
            "amount_logits": amount_logits,
            "stop_logits": stop_logits,
        }
        if squeeze:
            out = {key: value.squeeze(0) for key, value in out.items()}
        return out


def make_model(spec: FeatureSpec | None = None, **kwargs: Any) -> OrbitWarsGraphPolicy:
    spec = spec or FeatureSpec()
    return OrbitWarsGraphPolicy(spec.planet_dim, spec.edge_dim, **kwargs)
