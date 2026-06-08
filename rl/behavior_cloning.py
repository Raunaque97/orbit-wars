from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

import orbit_rl_native
from rl.model import (
    ALLY_DELAY_BUCKET,
    FeatureSpec,
    amount_bin_for_move,
    build_graph_inputs,
    forecast_surplus_for_planet,
    make_model,
    minimum_to_capture_at_arrival,
    pad_graph_batch,
)

SAMPLE_CACHE_VERSION = 1


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _with_step(obs: dict[str, Any], step: int) -> dict[str, Any]:
    out = dict(obs)
    out["step"] = step
    return out


def _sanitize_obs(obs: dict[str, Any]) -> dict[str, Any]:
    out = dict(obs)
    out["player"] = int(out.get("player", 0))
    out["step"] = int(out.get("step", 0))
    out["planets"] = [
        [
            int(p[0]),
            int(p[1]),
            float(p[2]),
            float(p[3]),
            float(p[4]),
            int(p[5]),
            int(p[6]),
        ]
        for p in out.get("planets", [])
    ]
    out["initial_planets"] = [
        [
            int(p[0]),
            int(p[1]),
            float(p[2]),
            float(p[3]),
            float(p[4]),
            int(p[5]),
            int(p[6]),
        ]
        for p in out.get("initial_planets", out["planets"])
    ]
    out["fleets"] = [
        [
            int(f[0]),
            int(f[1]),
            float(f[2]),
            float(f[3]),
            float(f[4]),
            int(f[5]),
            int(f[6]),
        ]
        for f in out.get("fleets", [])
    ]
    out["comet_planet_ids"] = [int(pid) for pid in out.get("comet_planet_ids", [])]
    return out


def _winning_agent_indices(replay: dict[str, Any]) -> list[int]:
    rewards = replay.get("rewards") or []
    if not rewards:
        return list(range(len(replay.get("steps", [[]])[0])))
    best = max(rewards)
    return [idx for idx, reward in enumerate(rewards) if reward == best]


def _infer_target_index(
    graph: dict[str, Any],
    batch: dict[str, Any],
    source_id: int,
    action_angle: float,
    ships_sent: int,
) -> tuple[int, int]:
    planet_ids = [int(pid) for pid in batch["planet_ids"]]
    source_index = planet_ids.index(source_id)
    ship_buckets = [int(v) for v in batch["ship_buckets"]]
    bucket_index = min(
        range(len(ship_buckets)), key=lambda idx: abs(ship_buckets[idx] - ships_sent)
    )
    delays = batch["delays"]
    angles = batch["angles"]

    best_target = source_index
    best_score = 1e100
    for target_index, target_id in enumerate(planet_ids):
        if target_index == source_index:
            continue
        delay = int(delays[bucket_index, source_index, target_index])
        if delay >= 200:
            continue
        diff = _angle_diff(float(angles[bucket_index, source_index, target_index]), action_angle)
        score = diff + 0.001 * delay
        if score < best_score:
            best_score = score
            best_target = target_index
    return source_index, best_target


def iter_replay_samples(
    replay_paths: list[Path],
    *,
    spec: FeatureSpec,
    winner_only: bool = True,
    max_samples: int | None = None,
    min_step: int = 1,
    max_step: int | None = None,
) -> Iterator[dict[str, Any]]:
    yielded = 0
    for path in replay_paths:
        replay = json.loads(path.read_text())
        agent_indices = _winning_agent_indices(replay) if winner_only else list(
            range(len(replay["steps"][0]))
        )
        engines = {agent_idx: orbit_rl_native.FeatureEngine() for agent_idx in agent_indices}

        steps = replay["steps"]
        for action_step_index in range(1, len(steps)):
            if action_step_index < min_step:
                continue
            if max_step is not None and action_step_index >= max_step:
                break
            action_step = steps[action_step_index]
            obs_step = steps[action_step_index - 1]
            for agent_idx in agent_indices:
                agent_step = action_step[agent_idx]
                action = agent_step.get("action") or []
                obs = _sanitize_obs(
                    _with_step(obs_step[agent_idx]["observation"], action_step_index - 1)
                )
                batch = engines[agent_idx].compute(obs, spec.horizon)
                graph = build_graph_inputs(obs, batch, spec=spec)
                planet_ids = [int(pid) for pid in graph["planet_ids"]]
                ship_buckets = [int(v) for v in batch["ship_buckets"]]
                delay_bucket_index = ship_buckets.index(ALLY_DELAY_BUCKET)
                planets_by_id = {int(row[0]): row for row in obs["planets"]}
                player = int(obs.get("player", agent_idx))
                source_actions: dict[int, tuple[int, int]] = {}

                for raw_move in action:
                    source_id = int(raw_move[0])
                    if source_id not in planet_ids or source_id not in planets_by_id:
                        continue
                    source_planet = planets_by_id[source_id]
                    if int(source_planet[1]) != player or source_id in source_actions:
                        continue
                    action_angle = float(raw_move[1])
                    ships_sent = int(raw_move[2])
                    source_index, target_index = _infer_target_index(
                        graph, batch, source_id, action_angle, ships_sent
                    )
                    if target_index == source_index:
                        continue
                    target_id = planet_ids[target_index]
                    delay = int(batch["delays"][delay_bucket_index, source_index, target_index])
                    surplus = forecast_surplus_for_planet(
                        batch, source_id, int(source_planet[1]), spec.horizon
                    )
                    minimum_to_capture = minimum_to_capture_at_arrival(
                        batch, target_id, player, delay
                    )
                    amount_label = amount_bin_for_move(
                        ships_sent,
                        source_ships=int(source_planet[5]),
                        surplus=surplus,
                        minimum_to_capture=minimum_to_capture,
                    )
                    source_actions[source_id] = (target_index, amount_label)

                for source_index, source_id in enumerate(planet_ids):
                    source_planet = planets_by_id.get(source_id)
                    if source_planet is None:
                        continue
                    if int(source_planet[1]) != player or int(source_planet[5]) <= 0:
                        continue
                    action_label = source_actions.get(source_id)
                    target_index = action_label[0] if action_label is not None else -1
                    amount_label = action_label[1] if action_label is not None else -1
                    yield {
                        "graph": graph,
                        "source_index": source_index,
                        "target_index": target_index,
                        "amount_label": amount_label,
                        "stop_label": 0.0 if action_label is not None else 1.0,
                        "has_launch": action_label is not None,
                    }
                    yielded += 1
                    if max_samples is not None and yielded >= max_samples:
                        return


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sample_cache_metadata(
    replay_paths: list[Path],
    *,
    spec: FeatureSpec,
    winner_only: bool,
    min_step: int,
    max_step: int | None,
) -> dict[str, Any]:
    replays = []
    for path in replay_paths:
        stat = path.stat()
        replays.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "version": SAMPLE_CACHE_VERSION,
        "spec": asdict(spec),
        "winner_only": winner_only,
        "min_step": min_step,
        "max_step": max_step,
        "replays": replays,
    }


def _sample_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return cache_dir / f"bc_samples_{digest}.pt"


def load_or_extract_samples(
    replay_paths: list[Path],
    *,
    spec: FeatureSpec,
    winner_only: bool = True,
    min_step: int = 1,
    max_step: int | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    metadata = _sample_cache_metadata(
        replay_paths,
        spec=spec,
        winner_only=winner_only,
        min_step=min_step,
        max_step=max_step,
    )
    cache_path = _sample_cache_path(cache_dir, metadata) if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        payload = _torch_load(cache_path)
        if payload.get("metadata") == metadata:
            samples = payload.get("samples", [])
            print(f"loaded sample cache path={cache_path} samples={len(samples)}", flush=True)
            return samples, True

    if cache_path is not None:
        print(f"building sample cache path={cache_path}", flush=True)
    samples: list[dict[str, Any]] = []
    for replay_index, replay_path in enumerate(replay_paths, start=1):
        before = len(samples)
        samples.extend(
            iter_replay_samples(
                [replay_path],
                spec=spec,
                winner_only=winner_only,
                max_samples=None,
                min_step=min_step,
                max_step=max_step,
            )
        )
        print(
            f"vectorized replay {replay_index}/{len(replay_paths)} "
            f"path={replay_path.name} samples=+{len(samples) - before} total={len(samples)}",
            flush=True,
        )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
        try:
            torch.save({"metadata": metadata, "samples": samples}, tmp_path)
            os.replace(tmp_path, cache_path)
            print(f"saved sample cache path={cache_path} samples={len(samples)}", flush=True)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    return samples, False


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = pad_graph_batch([sample["graph"] for sample in samples])
    source = torch.tensor([sample["source_index"] for sample in samples], dtype=torch.long)
    target = torch.tensor([sample["target_index"] for sample in samples], dtype=torch.long)
    amount = torch.tensor([sample["amount_label"] for sample in samples], dtype=torch.long)
    stop = torch.tensor([sample["stop_label"] for sample in samples], dtype=torch.float32)
    has_launch = torch.tensor([sample["has_launch"] for sample in samples], dtype=torch.bool)
    return {
        **batch,
        "source": source,
        "target": target,
        "amount": amount,
        "stop": stop,
        "has_launch": has_launch,
    }


def train_bc(
    replay_dir: Path,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    max_samples: int | None = 256,
    min_step: int = 1,
    max_step: int | None = 50,
    launch_stop_weight: float = 0.0,
    seed: int = 7,
    lr: float = 3e-4,
    run_dir: Path | None = None,
    save_every: int = 10,
    log_every: int = 0,
    checkpoint: Path | None = None,
    sample_cache_dir: Path | None = None,
    no_sample_cache: bool = False,
) -> dict[str, float]:
    spec = FeatureSpec()
    replay_paths = sorted(replay_dir.glob("episode-*-replay.json"))
    if not replay_paths:
        raise FileNotFoundError(f"No replays found in {replay_dir}")

    model = make_model(spec)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    if sample_cache_dir is None and not no_sample_cache:
        sample_cache_dir = replay_dir / ".bc_sample_cache"
    samples, loaded_from_cache = load_or_extract_samples(
        replay_paths,
        spec=spec,
        winner_only=True,
        min_step=min_step,
        max_step=max_step,
        cache_dir=None if no_sample_cache else sample_cache_dir,
    )
    if not samples:
        raise RuntimeError("No owned-source samples could be extracted from replays")
    total_samples = len(samples)
    if max_samples is not None and len(samples) > max_samples:
        rng.shuffle(samples)
        samples = samples[:max_samples]

    best_loss = float("inf")
    launch_samples = sum(1 for sample in samples if sample["has_launch"])
    no_op_samples = len(samples) - launch_samples
    if launch_stop_weight <= 0.0:
        launch_stop_weight = no_op_samples / max(1, launch_samples)
    print(
        f"samples={len(samples)} total_samples={total_samples} launch_samples={launch_samples} "
        f"no_op_samples={no_op_samples} launch_stop_weight={launch_stop_weight:.4f}",
        flush=True,
    )
    last_loss = 0.0
    last_edge_loss = 0.0
    last_amount_loss = 0.0
    last_stop_loss = 0.0
    run_dir = run_dir
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path: Path, epoch: int, loss_value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "spec": spec,
            "epoch": epoch,
            "loss": loss_value,
            "samples": len(samples),
            "total_samples": total_samples,
            "launch_samples": launch_samples,
            "no_op_samples": no_op_samples,
            "launch_stop_weight": launch_stop_weight,
            "min_step": min_step,
            "max_step": max_step,
            "loaded_from_sample_cache": loaded_from_cache,
        }
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    for epoch in range(1, epochs + 1):
        rng.shuffle(samples)
        epoch_losses: list[float] = []
        epoch_edge_losses: list[float] = []
        epoch_amount_losses: list[float] = []
        epoch_stop_losses: list[float] = []
        batch_index = 0
        for start in range(0, len(samples), batch_size):
            batch_index += 1
            batch = _collate(samples[start : start + batch_size])
            out = model(batch["planet_features"], batch["edge_features"], batch["planet_mask"])
            batch_rows = torch.arange(len(batch["source"]))
            stop_logits = out["stop_logits"][batch_rows, batch["source"]]
            stop_weights = torch.where(
                batch["has_launch"],
                torch.full_like(batch["stop"], float(launch_stop_weight)),
                torch.ones_like(batch["stop"]),
            )
            stop_loss = F.binary_cross_entropy_with_logits(
                stop_logits, batch["stop"], weight=stop_weights
            )
            launch_rows = batch["has_launch"]
            if launch_rows.any():
                launch_indices = torch.nonzero(launch_rows, as_tuple=False).squeeze(-1)
                source = batch["source"][launch_indices]
                target = batch["target"][launch_indices]
                edge_logits = out["edge_logits"][launch_indices, source].clone()
                edge_logits[torch.arange(len(launch_indices)), source] = float("-inf")
                edge_loss = F.cross_entropy(edge_logits, target)
                amount_logits = out["amount_logits"][launch_indices, source, target]
                amount_loss = F.cross_entropy(amount_logits, batch["amount"][launch_indices])
            else:
                edge_loss = stop_loss.new_zeros(())
                amount_loss = stop_loss.new_zeros(())
            loss = stop_loss + edge_loss + amount_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach())
            last_edge_loss = float(edge_loss.detach())
            last_amount_loss = float(amount_loss.detach())
            last_stop_loss = float(stop_loss.detach())
            epoch_losses.append(last_loss)
            epoch_edge_losses.append(last_edge_loss)
            epoch_amount_losses.append(last_amount_loss)
            epoch_stop_losses.append(last_stop_loss)
            if log_every > 0 and batch_index % log_every == 0:
                print(
                    f"epoch={epoch} batch={batch_index} "
                    f"loss={last_loss:.4f} edge_loss={last_edge_loss:.4f} "
                    f"amount_loss={last_amount_loss:.4f} stop_loss={last_stop_loss:.4f}",
                    flush=True,
                )

        epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        epoch_edge_loss = sum(epoch_edge_losses) / max(1, len(epoch_edge_losses))
        epoch_amount_loss = sum(epoch_amount_losses) / max(1, len(epoch_amount_losses))
        epoch_stop_loss = sum(epoch_stop_losses) / max(1, len(epoch_stop_losses))
        last_loss = epoch_loss
        last_stop_loss = epoch_stop_loss
        print(
            f"epoch={epoch} loss={epoch_loss:.4f} edge_loss={epoch_edge_loss:.4f} "
            f"amount_loss={epoch_amount_loss:.4f} stop_loss={epoch_stop_loss:.4f}",
            flush=True,
        )
        if run_dir is not None:
            save_checkpoint(run_dir / "latest.pt", epoch, epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(run_dir / "best.pt", epoch, epoch_loss)
            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", epoch, epoch_loss)

    if checkpoint is not None:
        save_checkpoint(checkpoint, epochs, last_loss)

    return {
        "samples": float(len(samples)),
        "loss": last_loss,
        "edge_loss": last_edge_loss,
        "amount_loss": last_amount_loss,
        "stop_loss": last_stop_loss,
        "best_loss": best_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=Path("rl/data/replays/53204000"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument(
        "--min-step",
        type=int,
        default=1,
        help="Skip replay rows before this step; row 0 is the initial no-action row.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=50,
        help="Only use replay ticks before this step; default trains on opening rows 1-49.",
    )
    parser.add_argument(
        "--launch-stop-weight",
        type=float,
        default=0.0,
        help="Weight stop-loss launch labels; <=0 auto-balances no-op/launch labels.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--run", default="bc_smoke")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("rl/checkpoints"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print per-head losses every N batches; 0 prints epoch summaries only.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--sample-cache-dir",
        type=Path,
        default=None,
        help="Directory for cached vectorized replay samples; defaults to replay-dir/.bc_sample_cache.",
    )
    parser.add_argument(
        "--no-sample-cache",
        action="store_true",
        help="Disable loading and saving vectorized replay samples.",
    )
    args = parser.parse_args()
    run_dir = args.checkpoint_root / args.run

    stats = train_bc(
        args.replay_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        min_step=args.min_step,
        max_step=args.max_step,
        launch_stop_weight=args.launch_stop_weight,
        seed=args.seed,
        lr=args.lr,
        run_dir=run_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        checkpoint=args.checkpoint,
        sample_cache_dir=args.sample_cache_dir,
        no_sample_cache=args.no_sample_cache,
    )
    print(
        f"samples={int(stats['samples'])} loss={stats['loss']:.4f} "
        f"best_loss={stats['best_loss']:.4f} stop_loss={stats['stop_loss']:.4f}"
    )
    print(f"run_dir={run_dir}")
    if args.checkpoint is not None:
        print(f"checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
