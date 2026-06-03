from __future__ import annotations

import argparse
import json
import math
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


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _with_step(obs: dict[str, Any], step: int) -> dict[str, Any]:
    if "step" in obs:
        return obs
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
) -> Iterator[dict[str, Any]]:
    yielded = 0
    for path in replay_paths:
        replay = json.loads(path.read_text())
        agent_indices = _winning_agent_indices(replay) if winner_only else list(
            range(len(replay["steps"][0]))
        )
        engines = {agent_idx: orbit_rl_native.FeatureEngine() for agent_idx in agent_indices}

        for step_index, step in enumerate(replay["steps"]):
            for agent_idx in agent_indices:
                agent_step = step[agent_idx]
                action = agent_step.get("action") or []
                if not action:
                    continue
                obs = _sanitize_obs(_with_step(agent_step["observation"], step_index))
                batch = engines[agent_idx].compute(obs, spec.horizon)
                graph = build_graph_inputs(obs, batch, spec=spec)
                planet_ids = [int(pid) for pid in graph["planet_ids"]]
                ship_buckets = [int(v) for v in batch["ship_buckets"]]
                delay_bucket_index = ship_buckets.index(ALLY_DELAY_BUCKET)
                planets_by_id = {int(row[0]): row for row in obs["planets"]}

                for raw_move in action:
                    source_id = int(raw_move[0])
                    if source_id not in planet_ids or source_id not in planets_by_id:
                        continue
                    action_angle = float(raw_move[1])
                    ships_sent = int(raw_move[2])
                    source_index, target_index = _infer_target_index(
                        graph, batch, source_id, action_angle, ships_sent
                    )
                    target_id = planet_ids[target_index]
                    source_planet = planets_by_id[source_id]
                    delay = int(batch["delays"][delay_bucket_index, source_index, target_index])
                    surplus = forecast_surplus_for_planet(
                        batch, source_id, int(source_planet[1]), spec.horizon
                    )
                    minimum_to_capture = minimum_to_capture_at_arrival(
                        batch, target_id, int(obs.get("player", agent_idx)), delay
                    )
                    amount_label = amount_bin_for_move(
                        ships_sent,
                        source_ships=int(source_planet[5]),
                        surplus=surplus,
                        minimum_to_capture=minimum_to_capture,
                    )

                    yield {
                        "graph": graph,
                        "source_index": source_index,
                        "target_index": target_index,
                        "amount_label": amount_label,
                    }
                    yielded += 1
                    if max_samples is not None and yielded >= max_samples:
                        return


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = pad_graph_batch([sample["graph"] for sample in samples])
    source = torch.tensor([sample["source_index"] for sample in samples], dtype=torch.long)
    target = torch.tensor([sample["target_index"] for sample in samples], dtype=torch.long)
    amount = torch.tensor([sample["amount_label"] for sample in samples], dtype=torch.long)
    return {**batch, "source": source, "target": target, "amount": amount}


def train_bc(
    replay_dir: Path,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    max_samples: int | None = 256,
    lr: float = 3e-4,
    run_dir: Path | None = None,
    save_every: int = 10,
    log_every: int = 0,
    checkpoint: Path | None = None,
) -> dict[str, float]:
    spec = FeatureSpec()
    replay_paths = sorted(replay_dir.glob("episode-*-replay.json"))
    if not replay_paths:
        raise FileNotFoundError(f"No replays found in {replay_dir}")

    model = make_model(spec)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    samples = list(
        iter_replay_samples(replay_paths, spec=spec, winner_only=True, max_samples=max_samples)
    )
    if not samples:
        raise RuntimeError("No move samples could be extracted from replays")

    best_loss = float("inf")
    last_loss = 0.0
    last_edge_loss = 0.0
    last_amount_loss = 0.0
    run_dir = run_dir
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path: Path, epoch: int, loss_value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "spec": spec,
                "epoch": epoch,
                "loss": loss_value,
                "samples": len(samples),
            },
            path,
        )

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        epoch_edge_losses: list[float] = []
        epoch_amount_losses: list[float] = []
        batch_index = 0
        for start in range(0, len(samples), batch_size):
            batch_index += 1
            batch = _collate(samples[start : start + batch_size])
            out = model(batch["planet_features"], batch["edge_features"], batch["planet_mask"])
            n = out["edge_logits"].shape[-1]
            edge_target = batch["source"] * n + batch["target"]
            edge_loss = F.cross_entropy(
                out["edge_logits"].reshape(len(edge_target), -1), edge_target
            )
            amount_logits = out["amount_logits"][
                torch.arange(len(edge_target)), batch["source"], batch["target"]
            ]
            amount_loss = F.cross_entropy(amount_logits, batch["amount"])
            loss = edge_loss + amount_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach())
            last_edge_loss = float(edge_loss.detach())
            last_amount_loss = float(amount_loss.detach())
            epoch_losses.append(last_loss)
            epoch_edge_losses.append(last_edge_loss)
            epoch_amount_losses.append(last_amount_loss)
            if log_every > 0 and batch_index % log_every == 0:
                print(
                    f"epoch={epoch} batch={batch_index} "
                    f"loss={last_loss:.4f} edge_loss={last_edge_loss:.4f} "
                    f"amount_loss={last_amount_loss:.4f}",
                    flush=True,
                )

        epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        epoch_edge_loss = sum(epoch_edge_losses) / max(1, len(epoch_edge_losses))
        epoch_amount_loss = sum(epoch_amount_losses) / max(1, len(epoch_amount_losses))
        last_loss = epoch_loss
        print(
            f"epoch={epoch} loss={epoch_loss:.4f} edge_loss={epoch_edge_loss:.4f} "
            f"amount_loss={epoch_amount_loss:.4f}",
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
        "best_loss": best_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=Path("rl/data/replays/53204000"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=256)
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
    args = parser.parse_args()
    run_dir = args.checkpoint_root / args.run

    stats = train_bc(
        args.replay_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        lr=args.lr,
        run_dir=run_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        checkpoint=args.checkpoint,
    )
    print(
        f"samples={int(stats['samples'])} loss={stats['loss']:.4f} "
        f"best_loss={stats['best_loss']:.4f}"
    )
    print(f"run_dir={run_dir}")
    if args.checkpoint is not None:
        print(f"checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
