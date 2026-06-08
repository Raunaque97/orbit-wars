from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.model import FeatureSpec, make_model
from rl.native_rollout import (
    NATIVE_DELAY_CACHE_DIR,
    collect_native_rollout,
    compute_native_gae,
    concat_native_batches,
    export_torchscript_model,
    export_torchscript_state,
)
from rl.train_ppo import (
    EARLY_STOP_PRODUCTION_SHARE,
    EARLY_STOP_SHIP_SHARE,
    NATIVE_WORKER_THREADS,
    OPPONENT_BC_COMPLETE_WEIGHT,
    OPPONENT_BC_OPENING_WEIGHT,
    OPPONENT_LAGGED_WEIGHT,
    OPPONENT_RANDOM_V2_WEIGHT,
    PPOConfig,
    TORCH_NUM_THREADS,
    OpponentChoice,
    OpponentEntry,
    _choose_opponent_index,
    _model_state_cpu,
    _opponent_torchscript_path,
    _random_v2_policy_prob,
    _restore_population,
    _safe_model_name,
    _state_dict_cpu,
    _sync_fixed_opponent_weights,
)


DEFAULT_CHECKPOINT = Path("rl/checkpoints/ppo_psro_v1/latest.pt")
DEFAULT_RUN_DIR = Path("rl/checkpoints/ppo_psro_v1")
DEFAULT_EPISODES = 128
DEFAULT_PARALLEL_ENVS = 64
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LR = 3e-4


def _load_checkpoint(path: Path) -> tuple[nn.Module, FeatureSpec, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = payload.get("spec", FeatureSpec())
    model = make_model(spec)
    model.load_state_dict(payload["model"], strict=False)
    metadata = {key: value for key, value in payload.items() if key != "model"}
    return model, spec, metadata


def _ensure_torchscript(run_dir: Path, spec: FeatureSpec, entry: OpponentEntry) -> str | None:
    if entry.state is None:
        entry.torchscript_path = None
        return None
    path = (
        Path(entry.torchscript_path)
        if entry.torchscript_path
        else _opponent_torchscript_path(run_dir, entry)
    )
    if not path.exists():
        export_torchscript_state(entry.state, spec, path)
    entry.torchscript_path = str(path)
    return entry.torchscript_path


def _population_from_checkpoint(
    metadata: dict[str, Any], model: nn.Module
) -> list[OpponentEntry]:
    raw = metadata.get("opponent_population")
    if raw:
        population = _restore_population(raw)
        _sync_fixed_opponent_weights(population)
        return population
    state = _model_state_cpu(model)
    return [
        OpponentEntry("lagged_self_play", state, OPPONENT_LAGGED_WEIGHT, fixed=True),
        OpponentEntry("random_v2", None, OPPONENT_RANDOM_V2_WEIGHT, fixed=True),
    ]


def _seeds(count: int, seed: int, pool_size: int) -> list[int]:
    rng = random.Random(seed)
    pool = list(range(max(1, pool_size)))
    if count <= len(pool):
        return rng.sample(pool, count)
    return [rng.choice(pool) for _ in range(count)]


def _select_opponents(
    population: list[OpponentEntry], count: int, seed: int
) -> list[OpponentChoice]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        idx = _choose_opponent_index(rng, population)
        entry = population[idx]
        out.append(
            OpponentChoice(
                name=entry.name,
                state=entry.state,
                population_index=idx,
                torchscript_path=entry.torchscript_path,
            )
        )
    return out


def collect_fixed_batch(
    *,
    model: nn.Module,
    spec: FeatureSpec,
    metadata: dict[str, Any],
    checkpoint: Path,
    run_dir: Path,
    episodes: int,
    parallel_envs: int,
    max_steps: int,
    seed: int,
    seed_pool_size: int,
    torch_threads: int,
    worker_threads: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    model.eval()
    population = _population_from_checkpoint(metadata, model)
    for entry in population:
        _ensure_torchscript(run_dir, spec, entry)
    random_v2_p = _random_v2_policy_prob(population)

    learner_ts = run_dir / "diagnostics" / f"value_overfit_{_safe_model_name(checkpoint.stem)}.ts.pt"
    export_torchscript_model(model, spec, learner_ts)
    episode_seeds = _seeds(episodes, seed, seed_pool_size)
    opponents = _select_opponents(population, episodes, seed + 100003)

    batches = []
    started = time.perf_counter()
    for start in range(0, episodes, parallel_envs):
        batch_seeds = episode_seeds[start : start + parallel_envs]
        batch_opponents = opponents[start : start + parallel_envs]
        batch = collect_native_rollout(
            learner_model_path=learner_ts,
            seeds=batch_seeds,
            opponents=[
                {
                    "name": opponent.name,
                    "model_path": opponent.torchscript_path,
                    "population_index": opponent.population_index,
                }
                for opponent in batch_opponents
            ],
            random_v2_policy_prob=random_v2_p,
            max_steps=max_steps,
            seed=seed + start,
            early_ship_share=EARLY_STOP_SHIP_SHARE,
            early_production_share=EARLY_STOP_PRODUCTION_SHARE,
            delay_cache_dir=NATIVE_DELAY_CACHE_DIR,
            torch_threads=torch_threads,
            worker_threads=worker_threads,
        )
        batches.append(batch)
    batch = concat_native_batches(batches)
    _advantages, returns = compute_native_gae(batch, gamma=0.995, gae_lambda=0.95)
    elapsed = time.perf_counter() - started
    stats = dict(batch.get("stats", {}))
    print(
        "collected "
        f"episodes={episodes} transitions={int(batch['reward'].shape[0])} "
        f"sec={elapsed:.1f} cache={stats.get('delay_cache_hits', 0)}/{stats.get('feature_calls', 0)} "
        f"random_v2_p={random_v2_p:.3f}",
        flush=True,
    )
    return batch, returns, torch.tensor(episode_seeds, dtype=torch.long)


@torch.no_grad()
def evaluate_value(
    model: nn.Module,
    batch: dict[str, Any],
    returns: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    values = []
    n = int(returns.shape[0])
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        out = model(
            batch["planet_features"][start:end].to(device),
            batch["edge_features"][start:end].to(device),
            batch["planet_mask"][start:end].to(device),
        )
        values.append(out["value"].detach().cpu())
    value = torch.cat(values, dim=0).to(torch.float32)
    target = returns.to(torch.float32)
    residual = target - value
    target_var = torch.var(target, unbiased=False)
    ev = 0.0
    if float(target_var) > 1e-12:
        ev = float(1.0 - torch.var(residual, unbiased=False) / target_var)
    return {
        "ev": ev,
        "mse": float(F.mse_loss(value, target)),
        "value_mean": float(value.mean()),
        "return_mean": float(target.mean()),
        "return_std": float(target.std(unbiased=False)),
    }


def train_value_overfit(
    model: nn.Module,
    batch: dict[str, Any],
    returns: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    freeze_encoder: bool,
    log_every: int,
) -> list[dict[str, float]]:
    model.to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    if freeze_encoder:
        for name, param in model.named_parameters():
            if not name.startswith("value_head"):
                param.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad], lr=lr
    )
    generator = torch.Generator().manual_seed(seed)
    n = int(returns.shape[0])
    history = []
    initial = evaluate_value(model, batch, returns, batch_size=batch_size, device=device)
    initial["epoch"] = 0
    history.append(initial)
    print(
        f"epoch=0 ev={initial['ev']:.4f} mse={initial['mse']:.4f} "
        f"return_std={initial['return_std']:.4f}",
        flush=True,
    )

    target_cpu = returns.to(torch.float32)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=generator)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            out = model(
                batch["planet_features"].index_select(0, idx).to(device),
                batch["edge_features"].index_select(0, idx).to(device),
                batch["planet_mask"].index_select(0, idx).to(device),
            )
            target = target_cpu.index_select(0, idx).to(device)
            loss = F.mse_loss(out["value"], target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == epochs or epoch % max(1, log_every) == 0:
            metrics = evaluate_value(model, batch, returns, batch_size=batch_size, device=device)
            metrics["epoch"] = epoch
            metrics["train_mse"] = sum(losses) / max(1, len(losses))
            history.append(metrics)
            print(
                f"epoch={epoch} ev={metrics['ev']:.4f} mse={metrics['mse']:.4f} "
                f"train_mse={metrics['train_mse']:.4f}",
                flush=True,
            )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--parallel-envs", type=int, default=DEFAULT_PARALLEL_ENVS)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--seed-pool-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=TORCH_NUM_THREADS)
    parser.add_argument("--worker-threads", type=int, default=NATIVE_WORKER_THREADS)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--jsonl", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device(args.device)
    model, spec, metadata = _load_checkpoint(args.checkpoint)
    batch, returns, episode_seeds = collect_fixed_batch(
        model=model,
        spec=spec,
        metadata=metadata,
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        episodes=args.episodes,
        parallel_envs=args.parallel_envs,
        max_steps=args.max_steps,
        seed=args.seed,
        seed_pool_size=args.seed_pool_size,
        torch_threads=args.torch_threads,
        worker_threads=args.worker_threads,
    )
    history = train_value_overfit(
        model,
        batch,
        returns,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        freeze_encoder=args.freeze_encoder,
        log_every=args.log_every,
    )
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("a", encoding="utf-8") as fh:
            for row in history:
                payload = {
                    **row,
                    "checkpoint": str(args.checkpoint),
                    "episodes": args.episodes,
                    "transitions": int(batch["reward"].shape[0]),
                    "seed": args.seed,
                    "episode_seed_min": int(episode_seeds.min()),
                    "episode_seed_max": int(episode_seeds.max()),
                    "freeze_encoder": args.freeze_encoder,
                }
                fh.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
