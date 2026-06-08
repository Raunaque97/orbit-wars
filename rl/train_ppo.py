from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rl.model import FeatureSpec, make_model
from rl.native_rollout import (
    NATIVE_DELAY_CACHE_DIR,
    NATIVE_MODEL_DIR,
    collect_native_rollout,
    compute_native_gae,
    concat_native_batches,
    export_torchscript_model,
    export_torchscript_state,
    ppo_update_native,
)


# Edit these defaults before long experiments if you want the config visible in diffs.
DEFAULT_INIT_CHECKPOINT = Path("rl/checkpoints/bc_opening_v1/best.pt")
BC_OPENING_OPPONENT_CHECKPOINT = Path("rl/checkpoints/bc_opening_v1/best.pt")
BC_COMPLETE_OPPONENT_CHECKPOINT = Path("rl/checkpoints/bc_complete_v1/best.pt")
DEFAULT_RUN = "ppo_v1"
DEFAULT_SEED = 7
ENTROPY_COEF = 0.001
MAX_EPISODE_STEPS = 500
PARALLEL_ENVS = 64
ROLLOUT_BATCHES_PER_ITER = 2
DEFAULT_EPISODES_PER_ITER = PARALLEL_ENVS * ROLLOUT_BATCHES_PER_ITER
DEFAULT_UPDATE_EPOCHS = 2
DEFAULT_MINIBATCH_SIZE = 512
DEFAULT_LR = 3e-5
TORCH_NUM_THREADS = 4
NATIVE_WORKER_THREADS = 4
AUTO_RESUME = True
RUN_METADATA_FILENAME = "metadata.json"
RUN_METRICS_FILENAME = "metrics.jsonl"

OPPONENT_LAG_ITERATIONS = 10
PSRO_SNAPSHOT_EVERY = 10
PSRO_MAX_SNAPSHOTS = 8
PSRO_PRUNE_MIN_GAMES = 64
PSRO_PRUNE_LEARNER_WIN_RATE = 0.85
OPPONENT_LAGGED_WEIGHT = 0.35
OPPONENT_BC_OPENING_WEIGHT = 0.20
OPPONENT_BC_COMPLETE_WEIGHT = 0.10
OPPONENT_RANDOM_V2_WEIGHT = 0.15
OPPONENT_SNAPSHOT_WEIGHT = 0.25
RANDOM_V2_POLICY_RANDOM_PROB_MAX = 0.90
RANDOM_V2_POLICY_RANDOM_PROB_MIN = 0.00
RANDOM_V2_DECAY_START_WIN_RATE = 0.60
RANDOM_V2_DECAY_END_WIN_RATE = 0.85
TRAIN_SEED_POOL_START = 0
TRAIN_SEED_POOL_SIZE = 100
TRAINING_WIN_RATE_WINDOW = DEFAULT_EPISODES_PER_ITER
EARLY_STOP_SHIP_SHARE = 0.80
EARLY_STOP_PRODUCTION_SHARE = 0.70


@dataclass
class PPOConfig:
    run: str = DEFAULT_RUN
    checkpoint_root: Path = Path("rl/checkpoints")
    init_checkpoint: Path = DEFAULT_INIT_CHECKPOINT
    seed: int = DEFAULT_SEED
    iterations: int = 1
    episodes_per_iter: int = DEFAULT_EPISODES_PER_ITER
    max_steps: int = MAX_EPISODE_STEPS
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    lr: float = DEFAULT_LR
    update_epochs: int = DEFAULT_UPDATE_EPOCHS
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE
    value_coef: float = 0.50
    entropy_coef: float = ENTROPY_COEF
    max_grad_norm: float = 1.0
    save_every: int = 1
    early_stop_ship_share: float = EARLY_STOP_SHIP_SHARE
    early_stop_production_share: float = EARLY_STOP_PRODUCTION_SHARE
    device: str = "cpu"


@dataclass
class OpponentEntry:
    name: str
    state: dict[str, torch.Tensor] | None
    base_weight: float
    fixed: bool = False
    added_iteration: int = 0
    games: int = 0
    learner_wins: int = 0
    torchscript_path: str | None = None

    @property
    def learner_win_rate(self) -> float:
        if self.games <= 0:
            return 0.5
        return self.learner_wins / self.games


@dataclass
class OpponentChoice:
    name: str
    state: dict[str, torch.Tensor] | None
    population_index: int | None = None
    torchscript_path: str | None = None


def _rolling_win_rate(results: list[int]) -> float:
    return sum(results) / max(1, len(results))


def _random_v2_policy_prob(population: list[OpponentEntry]) -> float:
    entry = next((item for item in population if item.name == "random_v2"), None)
    if entry is None or entry.games < PSRO_PRUNE_MIN_GAMES:
        return RANDOM_V2_POLICY_RANDOM_PROB_MAX
    win_rate = entry.learner_win_rate
    if win_rate <= RANDOM_V2_DECAY_START_WIN_RATE:
        return RANDOM_V2_POLICY_RANDOM_PROB_MAX
    if win_rate >= RANDOM_V2_DECAY_END_WIN_RATE:
        return RANDOM_V2_POLICY_RANDOM_PROB_MIN
    progress = (win_rate - RANDOM_V2_DECAY_START_WIN_RATE) / max(
        1e-9, RANDOM_V2_DECAY_END_WIN_RATE - RANDOM_V2_DECAY_START_WIN_RATE
    )
    return (
        RANDOM_V2_POLICY_RANDOM_PROB_MAX
        + progress * (RANDOM_V2_POLICY_RANDOM_PROB_MIN - RANDOM_V2_POLICY_RANDOM_PROB_MAX)
    )


def _load_model(config: PPOConfig) -> tuple[nn.Module, FeatureSpec, dict[str, Any]]:
    device = torch.device(config.device)
    spec = FeatureSpec()
    model = make_model(spec).to(device)
    metadata: dict[str, Any] = {}
    if config.init_checkpoint.exists():
        payload = torch.load(config.init_checkpoint, map_location=device, weights_only=False)
        spec = payload.get("spec", spec)
        model = make_model(spec).to(device)
        model.load_state_dict(payload["model"], strict=False)
        metadata = {key: value for key, value in payload.items() if key != "model"}
    return model, spec, metadata


def _state_dict_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state.items()}


def _load_checkpoint_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device, weights_only=False)
    return _state_dict_cpu(payload["model"])


def _opponent_sampling_score(entry: OpponentEntry) -> float:
    win_rate = entry.learner_win_rate
    if entry.games < 16:
        multiplier = 1.25
    elif win_rate < 0.15:
        multiplier = 0.45
    elif win_rate < 0.35:
        multiplier = 0.75
    elif win_rate < 0.45:
        multiplier = 1.00
    elif win_rate <= 0.65:
        multiplier = 1.50
    elif win_rate < PSRO_PRUNE_LEARNER_WIN_RATE:
        multiplier = 1.00
    else:
        multiplier = 0.35
    score = entry.base_weight * multiplier
    if (
        not entry.fixed
        and entry.games >= PSRO_PRUNE_MIN_GAMES
        and win_rate >= PSRO_PRUNE_LEARNER_WIN_RATE
    ):
        score *= 0.25
    return max(1e-6, score)


def _choose_opponent_index(
    rng: random.Random, population: list[OpponentEntry]
) -> int:
    if not population:
        raise ValueError("opponent population cannot be empty")
    weights = [_opponent_sampling_score(entry) for entry in population]
    total = sum(weights)
    draw = rng.random() * total
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if draw <= cumulative:
            return idx
    return len(population) - 1


def _opponent_population_summary(
    population: list[OpponentEntry],
) -> dict[str, dict[str, float | int | bool]]:
    return {
        entry.name: {
            "games": entry.games,
            "learner_win_rate": round(entry.learner_win_rate, 3),
            "weight": round(_opponent_sampling_score(entry), 4),
            "fixed": entry.fixed,
        }
        for entry in population
    }


def _serialize_population(
    population: list[OpponentEntry],
) -> list[dict[str, Any]]:
    return [
        {
            "name": entry.name,
            "state": _state_dict_cpu(entry.state) if entry.state is not None else None,
            "base_weight": entry.base_weight,
            "fixed": entry.fixed,
            "added_iteration": entry.added_iteration,
            "games": entry.games,
            "learner_wins": entry.learner_wins,
            "torchscript_path": entry.torchscript_path,
        }
        for entry in population
    ]


def _restore_population(raw_population: list[dict[str, Any]]) -> list[OpponentEntry]:
    population: list[OpponentEntry] = []
    for raw in raw_population:
        state = raw.get("state")
        population.append(
            OpponentEntry(
                name=str(raw["name"]),
                state=_state_dict_cpu(state) if state is not None else None,
                base_weight=float(raw["base_weight"]),
                fixed=bool(raw.get("fixed", False)),
                added_iteration=int(raw.get("added_iteration", 0)),
                games=int(raw.get("games", 0)),
                learner_wins=int(raw.get("learner_wins", 0)),
                torchscript_path=raw.get("torchscript_path"),
            )
        )
    return population


def _sync_fixed_opponent_weights(population: list[OpponentEntry]) -> None:
    weights = {
        "bc_opening": OPPONENT_BC_OPENING_WEIGHT,
        "bc_complete": OPPONENT_BC_COMPLETE_WEIGHT,
        "lagged_self_play": OPPONENT_LAGGED_WEIGHT,
        "random_v2": OPPONENT_RANDOM_V2_WEIGHT,
    }
    for entry in population:
        if entry.fixed and entry.name in weights:
            entry.base_weight = weights[entry.name]


def _metadata_population(
    population: list[OpponentEntry],
) -> list[dict[str, float | int | bool | str]]:
    return [
        {
            "name": entry.name,
            "base_weight": entry.base_weight,
            "sampling_weight": round(_opponent_sampling_score(entry), 6),
            "fixed": entry.fixed,
            "added_iteration": entry.added_iteration,
            "games": entry.games,
            "learner_wins": entry.learner_wins,
            "learner_win_rate": round(entry.learner_win_rate, 6),
            "has_state": entry.state is not None,
            "torchscript_path": entry.torchscript_path or "",
        }
        for entry in population
    ]


def _prune_opponent_population(population: list[OpponentEntry]) -> list[str]:
    removed: list[str] = []
    while True:
        snapshots = [entry for entry in population if not entry.fixed]
        too_many = len(snapshots) > PSRO_MAX_SNAPSHOTS
        easy = [
            entry
            for entry in snapshots
            if entry.games >= PSRO_PRUNE_MIN_GAMES
            and entry.learner_win_rate >= PSRO_PRUNE_LEARNER_WIN_RATE
        ]
        if not too_many and not easy:
            break
        if easy:
            victim = max(easy, key=lambda entry: (entry.learner_win_rate, entry.games))
        else:
            victim = min(
                snapshots,
                key=lambda entry: (
                    _opponent_sampling_score(entry),
                    entry.learner_win_rate,
                    -entry.added_iteration,
                ),
            )
        population.remove(victim)
        removed.append(victim.name)
    return removed


def _episode_seeds_for_iteration(config: PPOConfig, iteration: int) -> list[int]:
    pool = [
        TRAIN_SEED_POOL_START + offset
        for offset in range(max(1, TRAIN_SEED_POOL_SIZE))
    ]
    rng = random.Random(config.seed + iteration * 9973)
    if config.episodes_per_iter <= len(pool):
        return rng.sample(pool, config.episodes_per_iter)
    return [rng.choice(pool) for _ in range(config.episodes_per_iter)]


def _model_state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _safe_model_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def _opponent_torchscript_path(run_dir: Path, entry: OpponentEntry) -> Path:
    suffix = f"{_safe_model_name(entry.name)}_{int(entry.added_iteration):04d}.ts.pt"
    return run_dir / NATIVE_MODEL_DIR / suffix


def _ensure_opponent_torchscript(
    run_dir: Path, spec: FeatureSpec, entry: OpponentEntry
) -> str | None:
    if entry.state is None:
        entry.torchscript_path = None
        return None
    path = Path(entry.torchscript_path) if entry.torchscript_path else _opponent_torchscript_path(run_dir, entry)
    if not path.exists():
        export_torchscript_state(entry.state, spec, path)
    entry.torchscript_path = str(path)
    return entry.torchscript_path


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: FeatureSpec,
    *,
    iteration: int,
    config: PPOConfig,
    stats: dict[str, Any],
    population: list[OpponentEntry],
    recent_training_results: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "spec": spec,
        "iteration": iteration,
        "config": vars(config),
        "stats": stats,
        "opponent_population": _serialize_population(population),
        "recent_training_results": list(recent_training_results),
    }
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_run_metadata(
    run_dir: Path,
    *,
    iteration: int,
    config: PPOConfig,
    stats: dict[str, Any],
    population: list[OpponentEntry],
    checkpoint_path: Path,
) -> None:
    payload = {
        "run": config.run,
        "iteration": iteration,
        "checkpoint": str(checkpoint_path),
        "init_checkpoint": str(config.init_checkpoint),
        "config": _jsonable(vars(config)),
        "stats": _jsonable(stats),
        "opponent_population": _metadata_population(population),
        "parallel_envs": PARALLEL_ENVS,
        "rollout_batches_per_iter": ROLLOUT_BATCHES_PER_ITER,
        "native_worker_threads": NATIVE_WORKER_THREADS,
        "seed_pool": {
            "start": TRAIN_SEED_POOL_START,
            "size": TRAIN_SEED_POOL_SIZE,
        },
    }
    path = run_dir / RUN_METADATA_FILENAME
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def append_iteration_metrics(
    run_dir: Path,
    *,
    iteration: int,
    config: PPOConfig,
    stats: dict[str, Any],
    population: list[OpponentEntry],
) -> None:
    payload = {
        "run": config.run,
        "iteration": iteration,
        "stats": _jsonable(stats),
        "opponent_population": _metadata_population(population),
    }
    path = run_dir / RUN_METRICS_FILENAME
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def train(config: PPOConfig) -> None:
    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    run_dir = config.checkpoint_root / config.run
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = run_dir / "latest.pt"
    if AUTO_RESUME and resume_path.exists() and config.init_checkpoint == DEFAULT_INIT_CHECKPOINT:
        config.init_checkpoint = resume_path

    model, spec, init_metadata = _load_model(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if "optimizer" in init_metadata:
        optimizer.load_state_dict(init_metadata["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = config.lr
    start_iteration = int(init_metadata.get("iteration", 0))

    lagged_model_state = _model_state_cpu(model)
    device = torch.device(config.device)
    if "opponent_population" in init_metadata:
        population = _restore_population(init_metadata["opponent_population"])
    else:
        population = [
            OpponentEntry(
                name="bc_opening",
                state=_load_checkpoint_state(BC_OPENING_OPPONENT_CHECKPOINT, device),
                base_weight=OPPONENT_BC_OPENING_WEIGHT,
                fixed=True,
            ),
            OpponentEntry(
                name="bc_complete",
                state=_load_checkpoint_state(BC_COMPLETE_OPPONENT_CHECKPOINT, device),
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
    _sync_fixed_opponent_weights(population)
    lagged_population_index = next(
        (idx for idx, entry in enumerate(population) if entry.name == "lagged_self_play"),
        None,
    )
    if lagged_population_index is None:
        population.append(
            OpponentEntry(
                name="lagged_self_play",
                state=lagged_model_state,
                base_weight=OPPONENT_LAGGED_WEIGHT,
                fixed=True,
            )
        )
        lagged_population_index = len(population) - 1
    if population[lagged_population_index].state is None:
        population[lagged_population_index].state = lagged_model_state
    recent_training_results = [
        int(value)
        for value in init_metadata.get("recent_training_results", [])
    ][-TRAINING_WIN_RATE_WINDOW:]
    print(f"run_dir={run_dir}")
    print(f"init_checkpoint={config.init_checkpoint} init_epoch={init_metadata.get('epoch')}")
    print(f"resume_iteration={start_iteration}", flush=True)
    print(
        f"torch_threads={torch.get_num_threads()} parallel_envs={PARALLEL_ENVS} "
        f"worker_threads={NATIVE_WORKER_THREADS} "
        f"rollout_batches_per_iter={ROLLOUT_BATCHES_PER_ITER} "
        f"episodes_per_iter={config.episodes_per_iter} "
        f"update_epochs={config.update_epochs} minibatch_size={config.minibatch_size}",
        flush=True,
    )
    print(f"opponent_population={_opponent_population_summary(population)}", flush=True)

    for iteration in range(start_iteration + 1, config.iterations + 1):
        rewards: list[float] = []
        lengths: list[int] = []
        native_batches: list[dict[str, Any]] = []
        random_v2_policy_prob = _random_v2_policy_prob(population)
        learner_ts_path = run_dir / NATIVE_MODEL_DIR / f"learner_iter_{iteration:04d}.ts.pt"
        export_torchscript_model(model, spec, learner_ts_path)
        for entry in population:
            _ensure_opponent_torchscript(run_dir, spec, entry)
        episode_seeds = _episode_seeds_for_iteration(config, iteration)
        opponent_rng = random.Random(config.seed + iteration * 100003)
        opponents: list[OpponentChoice] = []
        opponent_counts: dict[str, int] = {}
        for _seed in episode_seeds:
            opponent_index = _choose_opponent_index(opponent_rng, population)
            opponent_entry = population[opponent_index]
            opponents.append(
                OpponentChoice(
                    name=opponent_entry.name,
                    state=opponent_entry.state,
                    population_index=opponent_index,
                    torchscript_path=opponent_entry.torchscript_path,
                )
            )
            opponent_counts[opponent_entry.name] = (
                opponent_counts.get(opponent_entry.name, 0) + 1
            )
        for batch_start in range(0, len(episode_seeds), PARALLEL_ENVS):
            batch_seeds = episode_seeds[batch_start : batch_start + PARALLEL_ENVS]
            batch_opponents = opponents[batch_start : batch_start + PARALLEL_ENVS]
            native_batch = collect_native_rollout(
                learner_model_path=learner_ts_path,
                seeds=batch_seeds,
                opponents=[
                    {
                        "name": opponent.name,
                        "model_path": opponent.torchscript_path,
                        "population_index": opponent.population_index,
                    }
                    for opponent in batch_opponents
                ],
                random_v2_policy_prob=random_v2_policy_prob,
                max_steps=config.max_steps,
                seed=config.seed + iteration * 100003 + batch_start,
                early_ship_share=config.early_stop_ship_share,
                early_production_share=config.early_stop_production_share,
                delay_cache_dir=NATIVE_DELAY_CACHE_DIR,
                torch_threads=TORCH_NUM_THREADS,
                worker_threads=NATIVE_WORKER_THREADS,
            )
            native_batches.append(native_batch)
            batch_rewards = native_batch["final_rewards"].to(torch.float32).tolist()
            batch_lengths = native_batch["episode_lengths"].to(torch.long).tolist()
            for local_idx, (final_reward, length) in enumerate(zip(batch_rewards, batch_lengths)):
                opponent_choice = batch_opponents[local_idx]
                if opponent_choice.population_index is not None:
                    opponent_entry = population[opponent_choice.population_index]
                    opponent_entry.games += 1
                    if final_reward > 0:
                        opponent_entry.learner_wins += 1
                rewards.append(final_reward)
                recent_training_results.append(1 if final_reward > 0 else 0)
                if len(recent_training_results) > TRAINING_WIN_RATE_WINDOW:
                    recent_training_results = recent_training_results[
                        -TRAINING_WIN_RATE_WINDOW :
                    ]
                lengths.append(length)

        rollout_batch = concat_native_batches(native_batches)
        invalid_counts = dict(rollout_batch.get("invalid_counts", {}))
        rollout_stats = dict(rollout_batch.get("stats", {}))
        feature_calls = int(rollout_stats.get("feature_calls", 0))
        feature_cache_hits = int(rollout_stats.get("delay_cache_hits", 0))
        feature_ms = float(rollout_stats.get("feature_ms", 0.0))
        advantages, returns = compute_native_gae(
            rollout_batch, config.gamma, config.gae_lambda
        )
        update_stats = ppo_update_native(
            model, optimizer, rollout_batch, advantages, returns, config
        )
        win_rate = sum(1 for reward in rewards if reward > 0) / max(1, len(rewards))
        rolling_train_win_rate = _rolling_win_rate(recent_training_results)
        stats = {
            **update_stats,
            "iteration": iteration,
            "episodes": len(rewards),
            "mean_reward": sum(rewards) / max(1, len(rewards)),
            "win_rate": win_rate,
            "rolling_train_win_rate": rolling_train_win_rate,
            "rolling_train_games": len(recent_training_results),
            "mean_length": sum(lengths) / max(1, len(lengths)),
            "opponent_counts": opponent_counts,
            "invalid_counts": invalid_counts,
            "transitions": int(rollout_batch["reward"].shape[0]),
            "parallel_envs": PARALLEL_ENVS,
            "rollout_batches_per_iter": ROLLOUT_BATCHES_PER_ITER,
            "native_worker_threads": NATIVE_WORKER_THREADS,
            "opponent_population": _opponent_population_summary(population),
            "random_v2_policy_prob": random_v2_policy_prob,
            "delay_cache_hits": feature_cache_hits,
            "feature_calls": feature_calls,
            "feature_ms": feature_ms,
            "feature_delay_ms": 0.0,
        }
        print(
            "iter={iteration} episodes={episodes} win_rate={win_rate:.3f} "
            "rolling_win_rate={rolling_train_win_rate:.3f}/{rolling_train_games} "
            "mean_reward={mean_reward:.3f} mean_length={mean_length:.1f} "
            "loss={loss:.4f} policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
            "entropy={entropy:.4f} ev={explained_variance:.4f} "
            "kl={approx_kl:.5f} clip_frac={clip_fraction:.3f} "
            "kl_dec={approx_kl_per_decision:.5f} "
            "clip_dec={clip_fraction_per_decision:.3f} "
            "terms={mean_action_terms:.1f} "
            "opponents={opponent_counts} "
            "population={opponent_population} "
            "random_v2_p={random_v2_policy_prob:.3f} "
            "delay_cache={delay_cache_hits}/{feature_calls} "
            "feature_ms={feature_ms:.3f} feature_delay_ms={feature_delay_ms:.3f} "
            "invalid={invalid_counts}".format(**stats),
            flush=True,
        )

        if iteration % PSRO_SNAPSHOT_EVERY == 0:
            population.append(
                OpponentEntry(
                    name=f"snapshot_{iteration:04d}",
                    state=_model_state_cpu(model),
                    base_weight=OPPONENT_SNAPSHOT_WEIGHT,
                    fixed=False,
                    added_iteration=iteration,
                    torchscript_path=None,
                )
            )
            removed = _prune_opponent_population(population)
            print(
                f"psro_snapshot_added_at={iteration} population_size={len(population)} "
                f"removed={removed} population={_opponent_population_summary(population)}",
                flush=True,
            )
        if iteration % OPPONENT_LAG_ITERATIONS == 0:
            lagged_model_state = _model_state_cpu(model)
            population[lagged_population_index].state = lagged_model_state
            population[lagged_population_index].games = 0
            population[lagged_population_index].learner_wins = 0
            population[lagged_population_index].added_iteration = iteration
            population[lagged_population_index].torchscript_path = None
            print(f"lagged_self_play_opponent_refreshed_at={iteration}", flush=True)

        latest_path = run_dir / "latest.pt"
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            spec,
            iteration=iteration,
            config=config,
            stats=stats,
            population=population,
            recent_training_results=recent_training_results,
        )
        if iteration % max(1, config.save_every) == 0:
            iter_path = run_dir / f"iter_{iteration:04d}.pt"
            save_checkpoint(
                iter_path,
                model,
                optimizer,
                spec,
                iteration=iteration,
                config=config,
                stats=stats,
                population=population,
                recent_training_results=recent_training_results,
            )
        write_run_metadata(
            run_dir,
            iteration=iteration,
            config=config,
            stats=stats,
            population=population,
            checkpoint_path=latest_path,
        )
        append_iteration_metrics(
            run_dir,
            iteration=iteration,
            config=config,
            stats=stats,
            population=population,
        )


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("rl/checkpoints"))
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--episodes-per-iter", type=int, default=DEFAULT_EPISODES_PER_ITER)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--update-epochs", type=int, default=DEFAULT_UPDATE_EPOCHS)
    parser.add_argument("--minibatch-size", type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--early-stop-ship-share", type=float, default=EARLY_STOP_SHIP_SHARE)
    parser.add_argument(
        "--early-stop-production-share", type=float, default=EARLY_STOP_PRODUCTION_SHARE
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    return PPOConfig(**vars(args))


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
