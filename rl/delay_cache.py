from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

import orbit_rl_native


DEFAULT_DELAY_CACHE_DIR = Path("rl/data/delay_cache")
MAX_ROUTE_DELAY = 141


def _obs_get(obs: Any, name: str, fallback: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, fallback)
    if hasattr(obs, "get"):
        value = obs.get(name, fallback)
        return fallback if value is None else value
    return getattr(obs, name, fallback)


class DelayMatrixCache:
    def __init__(self, root: Path = DEFAULT_DELAY_CACHE_DIR) -> None:
        self.root = Path(root)

    def path_for(self, seed: int, step: int) -> Path:
        return self.root / f"seed_{int(seed):06d}" / f"step_{int(step):04d}.npz"

    def load(
        self, *, seed: int, step: int, planet_ids: list[int], ship_buckets: list[int]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        path = self.path_for(seed, step)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                cached_planet_ids = data["planet_ids"].astype(np.int64).tolist()
                cached_ship_buckets = data["ship_buckets"].astype(np.int64).tolist()
                if cached_planet_ids != [int(pid) for pid in planet_ids]:
                    return None
                if cached_ship_buckets != [int(bucket) for bucket in ship_buckets]:
                    return None
                return data["delays"].copy(), data["angles"].copy()
        except (OSError, ValueError, KeyError):
            return None

    def save(self, *, seed: int, step: int, batch: dict[str, Any]) -> None:
        path = self.path_for(seed, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with tmp_path.open("wb") as handle:
            np.savez(
                handle,
                planet_ids=np.asarray(batch["planet_ids"], dtype=np.int32),
                ship_buckets=np.asarray(batch["ship_buckets"], dtype=np.int16),
                delays=np.asarray(batch["delays"], dtype=np.int16),
                angles=np.asarray(batch["angles"], dtype=np.float32),
            )
        os.replace(tmp_path, path)


class CachedFeatureEngine:
    def __init__(
        self,
        *,
        cache: DelayMatrixCache | None = None,
        enabled: bool = True,
        readonly: bool = False,
    ) -> None:
        self.engine = orbit_rl_native.FeatureEngine()
        self.cache = cache or DelayMatrixCache()
        self.enabled = enabled
        self.readonly = readonly

    def compute(
        self,
        obs: Any,
        horizon: int = 50,
        max_route_delay: int = MAX_ROUTE_DELAY,
    ) -> dict[str, Any]:
        seed = _obs_get(obs, "seed", None)
        step = int(_obs_get(obs, "step", 0))
        if not self.enabled or seed is None:
            batch = self.engine.compute(obs, horizon, max_route_delay)
            batch["stats"]["delay_cache_hit"] = 0
            return batch

        batch = self.engine.compute(obs, horizon, max_route_delay, False)
        cached = self.cache.load(
            seed=int(seed),
            step=step,
            planet_ids=[int(pid) for pid in batch["planet_ids"]],
            ship_buckets=[int(bucket) for bucket in batch["ship_buckets"]],
        )
        if cached is not None:
            delays, angles = cached
            batch["delays"] = delays
            batch["angles"] = angles
            batch["stats"]["delay_cache_hit"] = 1
            return batch

        full_batch = self.engine.compute(obs, horizon, max_route_delay, True)
        full_batch["stats"]["delay_cache_hit"] = 0
        if not self.readonly:
            self.cache.save(seed=int(seed), step=step, batch=full_batch)
        return full_batch

    def query_route(
        self,
        obs: Any,
        src_id: int,
        target_id: int,
        ships: int,
        max_route_delay: int = MAX_ROUTE_DELAY,
    ) -> dict[str, Any]:
        return self.engine.query_route(obs, src_id, target_id, ships, max_route_delay)
