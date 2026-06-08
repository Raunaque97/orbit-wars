from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orbit_rl_native
from rl.model import FeatureSpec, build_graph_inputs, make_model, pad_graph_batch
from rl.native_env import NativeOrbitEnv


DEFAULT_CHECKPOINT = Path("rl/checkpoints/bc_opening_v1/best.pt")
DEFAULT_BATCH_SIZES = "1,2,4,8,16,32,64,128,256"
DEFAULT_SAMPLES = 256
DEFAULT_ITERS = 100
DEFAULT_WARMUP = 10
DEFAULT_SEED = 7


def _random_actions(obs: dict[str, Any], rng: random.Random) -> list[list[int | float]]:
    moves: list[list[int | float]] = []
    player = int(obs["player"])
    for planet in obs.get("planets", []) or []:
        if int(planet[1]) != player:
            continue
        ships = int(planet[5])
        if ships < 5 or rng.random() > 0.08:
            continue
        moves.append([int(planet[0]), rng.random() * 6.283185307179586, min(ships, rng.randint(5, 25))])
    return moves


def _parse_batch_sizes(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one batch size is required")
    return values


def _load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, FeatureSpec]:
    spec = FeatureSpec()
    model = make_model(spec).to(device)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        spec = payload.get("spec", spec)
        model = make_model(spec).to(device)
        model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, spec


def _make_graphs(samples: int, spec: FeatureSpec, device: torch.device, seed: int) -> list[dict[str, Any]]:
    graphs: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for idx in range(samples):
        env = NativeOrbitEnv(seed=seed + idx, num_agents=2)
        # Move some states away from only-openings, but keep this outside measured inference.
        for step in range(rng.randrange(0, 12)):
            obs0 = dict(env.state[0].observation)
            obs0["player"] = 0
            obs1 = dict(env.state[1].observation)
            obs1["player"] = 1
            env.step([_random_actions(obs0, rng), _random_actions(obs1, rng)])
            if env.done:
                break
        for player in (0, 1):
            obs = dict(env.state[player].observation)
            obs["player"] = player
            engine = orbit_rl_native.FeatureEngine()
            batch = engine.compute(obs, spec.horizon)
            graphs.append(build_graph_inputs(obs, batch, spec=spec, device=device))
            if len(graphs) >= samples:
                return graphs
    return graphs


def _time_case(
    model: torch.nn.Module,
    graphs: list[dict[str, Any]],
    batch_size: int,
    *,
    iters: int,
    warmup: int,
) -> dict[str, float | int]:
    if batch_size > len(graphs):
        raise ValueError("batch_size cannot exceed number of graphs")

    batches = [
        [graphs[(start + offset) % len(graphs)] for offset in range(batch_size)]
        for start in range(max(iters + warmup, 1))
    ]

    pad_times: list[float] = []
    forward_times: list[float] = []
    total_times: list[float] = []
    with torch.inference_mode():
        for iter_idx, items in enumerate(batches):
            started_total = time.perf_counter()
            started_pad = time.perf_counter()
            batch = pad_graph_batch(items)
            pad_sec = time.perf_counter() - started_pad
            started_forward = time.perf_counter()
            model(batch["planet_features"], batch["edge_features"], batch["planet_mask"])
            forward_sec = time.perf_counter() - started_forward
            total_sec = time.perf_counter() - started_total
            if iter_idx >= warmup:
                pad_times.append(pad_sec)
                forward_times.append(forward_sec)
                total_times.append(total_sec)

    mean_total = statistics.mean(total_times)
    mean_forward = statistics.mean(forward_times)
    mean_pad = statistics.mean(pad_times)
    return {
        "batch_size": batch_size,
        "iters": iters,
        "pad_ms": mean_pad * 1000.0,
        "forward_ms": mean_forward * 1000.0,
        "total_ms": mean_total * 1000.0,
        "states_per_sec_forward": batch_size / mean_forward,
        "states_per_sec_total": batch_size / mean_total,
        "ms_per_state_total": mean_total * 1000.0 / batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device(args.device)
    model, spec = _load_model(args.checkpoint, device)
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    samples = max(args.samples, max(batch_sizes))
    graphs = _make_graphs(samples, spec, device, args.seed)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "graphs": len(graphs),
                "torch_threads": torch.get_num_threads(),
                "device": str(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for batch_size in batch_sizes:
        print(
            json.dumps(
                _time_case(
                    model,
                    graphs,
                    batch_size,
                    iters=args.iters,
                    warmup=args.warmup,
                ),
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
