from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orbit_rl_native
from rl.delay_cache import DEFAULT_DELAY_CACHE_DIR, DelayMatrixCache
from rl.native_env import NativeOrbitEnv


DEFAULT_SEED_START = 0
DEFAULT_SEEDS = 128
DEFAULT_STEPS = 500
DEFAULT_HORIZON = 50
MAX_ROUTE_DELAY = 141


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_DELAY_CACHE_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache = DelayMatrixCache(args.cache_dir)
    total_started = time.perf_counter()
    computed = 0
    skipped = 0
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        env = NativeOrbitEnv(seed=seed, num_agents=2)
        engine = orbit_rl_native.FeatureEngine()
        seed_started = time.perf_counter()
        seed_computed = 0
        seed_skipped = 0
        for _ in range(args.steps):
            obs = dict(env.state[0].observation)
            obs["player"] = 0
            step = int(obs.get("step", 0))
            path = cache.path_for(seed, step)
            if path.exists() and not args.force:
                skipped += 1
                seed_skipped += 1
            else:
                batch = engine.compute(obs, args.horizon, MAX_ROUTE_DELAY, True)
                cache.save(seed=seed, step=step, batch=batch)
                computed += 1
                seed_computed += 1
            if step >= args.steps - 1:
                break
            env.step([[], []])
        print(
            f"seed={seed} computed={seed_computed} skipped={seed_skipped} "
            f"elapsed_sec={time.perf_counter() - seed_started:.2f}",
            flush=True,
        )
    print(
        f"done seeds={args.seeds} computed={computed} skipped={skipped} "
        f"elapsed_sec={time.perf_counter() - total_started:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
