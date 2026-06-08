from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.train_ppo import (  # noqa: E402
    BC_COMPLETE_OPPONENT_CHECKPOINT,
    BC_OPENING_OPPONENT_CHECKPOINT,
    DEFAULT_INIT_CHECKPOINT,
    DEFAULT_LR,
    DEFAULT_MINIBATCH_SIZE,
    OPPONENT_BC_COMPLETE_WEIGHT,
    OPPONENT_BC_OPENING_WEIGHT,
    OPPONENT_LAGGED_WEIGHT,
    OPPONENT_RANDOM_V2_WEIGHT,
    OpponentChoice,
    OpponentEntry,
    PPOConfig,
    TRAIN_SEED_POOL_SIZE,
    TRAIN_SEED_POOL_START,
    _choose_opponent_index,
    _load_checkpoint_state,
    _load_model,
    _model_state_cpu,
    _opponent_population_summary,
    _random_v2_policy_prob,
)
from rl.native_rollout import (  # noqa: E402
    NATIVE_DELAY_CACHE_DIR,
    collect_native_rollout,
    compute_native_gae,
    export_torchscript_model,
    export_torchscript_state,
    ppo_update_native,
)


DEFAULT_EPISODE_COUNTS = "4,8,16"
DEFAULT_UPDATE_EPOCHS = 1
DEFAULT_SEED = 700
DEFAULT_WORKER_THREADS = 4


def _rss_mb() -> float:
    proc = psutil.Process()
    total = proc.memory_info().rss
    try:
        children = proc.children(recursive=True)
    except (PermissionError, psutil.AccessDenied):
        children = []
    for child in children:
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            pass
    return total / (1024.0 * 1024.0)


class MemoryMonitor:
    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.peak_mb = _rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "MemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, _rss_mb())

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, _rss_mb())
            self._stop.wait(self.interval_sec)


def _parse_counts(raw: str) -> list[int]:
    counts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not counts:
        raise ValueError("at least one episode count is required")
    return counts


def _opponents_for_batch(
    *,
    count: int,
    seed: int,
    lagged_model_state: dict[str, torch.Tensor],
    bc_opening_state: dict[str, torch.Tensor],
    bc_complete_state: dict[str, torch.Tensor],
) -> tuple[list[OpponentChoice], dict[str, int], list[OpponentEntry]]:
    rng = random.Random(seed)
    population = [
        OpponentEntry(
            name="bc_opening",
            state=bc_opening_state,
            base_weight=OPPONENT_BC_OPENING_WEIGHT,
            fixed=True,
        ),
        OpponentEntry(
            name="bc_complete",
            state=bc_complete_state,
            base_weight=OPPONENT_BC_COMPLETE_WEIGHT,
            fixed=True,
        ),
        OpponentEntry(
            name="lagged_self_play",
            state=lagged_model_state,
            base_weight=OPPONENT_LAGGED_WEIGHT,
            fixed=True,
        ),
        OpponentEntry(
            name="random_v2",
            state=None,
            base_weight=OPPONENT_RANDOM_V2_WEIGHT,
            fixed=True,
        ),
    ]
    opponents: list[OpponentChoice] = []
    counts: dict[str, int] = {}
    model_dir = Path("rl/checkpoints/native_bench_models")
    for _ in range(count):
        opponent_index = _choose_opponent_index(rng, population)
        entry = population[opponent_index]
        if entry.state is not None and entry.torchscript_path is None:
            entry.torchscript_path = str(model_dir / f"{entry.name}.ts.pt")
        opponents.append(
            OpponentChoice(
                name=entry.name,
                state=entry.state,
                population_index=opponent_index,
                torchscript_path=entry.torchscript_path,
            )
        )
        counts[entry.name] = counts.get(entry.name, 0) + 1
    return opponents, counts, population


def _seeds_for_case(*, count: int, seed: int) -> list[int]:
    pool = [
        TRAIN_SEED_POOL_START + offset
        for offset in range(max(1, TRAIN_SEED_POOL_SIZE))
    ]
    rng = random.Random(seed)
    if count <= len(pool):
        return rng.sample(pool, count)
    return [rng.choice(pool) for _ in range(count)]


def _run_case(
    *,
    episodes: int,
    seed: int,
    config_base: PPOConfig,
    include_update: bool,
    worker_threads: int,
) -> dict[str, Any]:
    config = PPOConfig(**vars(config_base))
    config.episodes_per_iter = episodes

    model, spec, metadata = _load_model(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lagged_model_state = _model_state_cpu(model)
    device = torch.device(config.device)
    bc_opening_state = _load_checkpoint_state(BC_OPENING_OPPONENT_CHECKPOINT, device)
    bc_complete_state = _load_checkpoint_state(BC_COMPLETE_OPPONENT_CHECKPOINT, device)
    opponents, opponent_counts, population = _opponents_for_batch(
        count=episodes,
        seed=seed,
        lagged_model_state=lagged_model_state,
        bc_opening_state=bc_opening_state,
        bc_complete_state=bc_complete_state,
    )
    seeds = _seeds_for_case(count=episodes, seed=seed)
    model_dir = Path("rl/checkpoints/native_bench_models")
    spec = metadata.get("spec", spec)
    learner_path = export_torchscript_model(model, spec, model_dir / "learner.ts.pt")
    for entry in population:
        if entry.state is not None and entry.torchscript_path is not None:
            export_torchscript_state(entry.state, spec, Path(entry.torchscript_path))

    gc.collect()
    rss_before = _rss_mb()
    with MemoryMonitor() as monitor:
        started = time.perf_counter()
        rollout_batch = collect_native_rollout(
            learner_model_path=learner_path,
            seeds=seeds,
            opponents=[
                {
                    "name": opponent.name,
                    "model_path": opponent.torchscript_path,
                    "population_index": opponent.population_index,
                }
                for opponent in opponents
            ],
            random_v2_policy_prob=_random_v2_policy_prob(population),
            max_steps=config.max_steps,
            seed=seed,
            early_ship_share=config.early_stop_ship_share,
            early_production_share=config.early_stop_production_share,
            delay_cache_dir=NATIVE_DELAY_CACHE_DIR,
            torch_threads=max(1, torch.get_num_threads()),
            worker_threads=max(1, worker_threads),
        )
        rollout_sec = time.perf_counter() - started
        rss_after_rollout = _rss_mb()

        lengths = rollout_batch["episode_lengths"].to(torch.long).tolist()
        rewards = rollout_batch["final_rewards"].to(torch.float32).tolist()
        transitions = int(rollout_batch["reward"].shape[0])
        rollout_stats = dict(rollout_batch.get("stats", {}))
        feature_calls = int(rollout_stats.get("feature_calls", transitions))
        delay_cache_hits = int(rollout_stats.get("delay_cache_hits", 0))
        feature_ms = float(rollout_stats.get("feature_ms", 0.0))
        update_sec = 0.0
        update_stats: dict[str, float] = {}
        if include_update and transitions:
            advantages, returns = compute_native_gae(
                rollout_batch, config.gamma, config.gae_lambda
            )
            update_started = time.perf_counter()
            update_stats = ppo_update_native(
                model, optimizer, rollout_batch, advantages, returns, config
            )
            update_sec = time.perf_counter() - update_started
        peak_rss_mb = monitor.peak_mb

    total_steps = sum(lengths)
    return {
        "episodes": episodes,
        "seed": seed,
        "seed_min": min(seeds) if seeds else None,
        "seed_max": max(seeds) if seeds else None,
        "init_epoch": metadata.get("epoch"),
        "opponent_counts": opponent_counts,
        "opponent_population": _opponent_population_summary(population),
        "rollout_sec": rollout_sec,
        "update_sec": update_sec,
        "total_sec": rollout_sec + update_sec,
        "games_per_sec_rollout": episodes / rollout_sec if rollout_sec > 0 else 0.0,
        "games_per_sec_total": episodes / (rollout_sec + update_sec)
        if rollout_sec + update_sec > 0
        else 0.0,
        "steps": total_steps,
        "steps_per_sec_rollout": total_steps / rollout_sec if rollout_sec > 0 else 0.0,
        "mean_length": total_steps / max(1, episodes),
        "transitions": transitions,
        "transitions_per_episode": transitions / max(1, episodes),
        "win_rate": sum(1 for reward in rewards if reward > 0) / max(1, len(rewards)),
        "delay_cache_hits": delay_cache_hits,
        "feature_calls": feature_calls,
        "delay_cache_hit_rate": delay_cache_hits / max(1, feature_calls),
        "feature_ms": feature_ms,
        "feature_delay_ms": 0.0,
        "model_load_ms": rollout_stats.get("model_load_ms"),
        "init_ms": rollout_stats.get("init_ms"),
        "graph_wall_ms": rollout_stats.get("graph_wall_ms"),
        "learner_forward_ms": rollout_stats.get("learner_forward_ms"),
        "learner_sample_ms": rollout_stats.get("learner_sample_ms"),
        "opponent_ms": rollout_stats.get("opponent_ms"),
        "opponent_graph_ms": rollout_stats.get("opponent_graph_ms"),
        "opponent_forward_ms": rollout_stats.get("opponent_forward_ms"),
        "opponent_sample_ms": rollout_stats.get("opponent_sample_ms"),
        "opponent_random_ms": rollout_stats.get("opponent_random_ms"),
        "sim_step_ms": rollout_stats.get("sim_step_ms"),
        "pack_ms": rollout_stats.get("pack_ms"),
        "collect_total_ms": rollout_stats.get("collect_total_ms"),
        "native_torch_threads": rollout_stats.get("torch_threads"),
        "native_worker_threads": rollout_stats.get("worker_threads"),
        "rss_before_mb": rss_before,
        "rss_after_rollout_mb": rss_after_rollout,
        "peak_rss_mb": peak_rss_mb,
        "loss": update_stats.get("loss"),
        "policy_loss": update_stats.get("policy_loss"),
        "value_loss": update_stats.get("value_loss"),
        "entropy": update_stats.get("entropy"),
        "approx_kl": update_stats.get("approx_kl"),
        "clip_fraction": update_stats.get("clip_fraction"),
        "explained_variance": update_stats.get("explained_variance"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-counts", default=DEFAULT_EPISODE_COUNTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--update-epochs", type=int, default=DEFAULT_UPDATE_EPOCHS)
    parser.add_argument("--minibatch-size", type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=DEFAULT_WORKER_THREADS)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--jsonl", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    counts = _parse_counts(args.episode_counts)
    config = PPOConfig(
        init_checkpoint=args.init_checkpoint,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        device=args.device,
    )
    print(
        f"torch_threads={torch.get_num_threads()} worker_threads={args.worker_threads} "
        "opponent_population="
        "bc_opening,bc_complete,lagged_self_play,random_v2",
        flush=True,
    )
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)

    for idx, episodes in enumerate(counts):
        result = _run_case(
            episodes=episodes,
            seed=args.seed + idx * 10000,
            config_base=config,
            include_update=not args.collect_only,
            worker_threads=args.worker_threads,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.jsonl is not None:
            with args.jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
