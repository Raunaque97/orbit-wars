from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.model import FeatureSpec, make_model


DEFAULT_CHECKPOINT = Path("rl/checkpoints/ppo_psro_v1/iter_0100.pt")
DEFAULT_TORCHSCRIPT = Path("rl/checkpoints/ppo_psro_v1/model_ts.pt")
DEFAULT_BATCH_SIZES = "1,4,8,16,32,64"
DEFAULT_PLANETS = 32
DEFAULT_ITERS = 300
DEFAULT_WARMUP = 30


def _parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one value is required")
    return values


def _load_eager(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, FeatureSpec]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    spec = payload.get("spec", FeatureSpec())
    model = make_model(spec).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, spec


def _load_torchscript(path: Path, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def _time_forward(
    model: torch.nn.Module,
    *,
    spec: FeatureSpec,
    batch_size: int,
    planets: int,
    iters: int,
    warmup: int,
    device: torch.device,
) -> dict[str, float | int]:
    planet_features = torch.randn(
        batch_size, planets, spec.planet_dim, dtype=torch.float32, device=device
    )
    edge_features = torch.randn(
        batch_size, planets, planets, spec.edge_dim, dtype=torch.float32, device=device
    )
    planet_mask = torch.ones(batch_size, planets, dtype=torch.bool, device=device)

    times: list[float] = []
    with torch.inference_mode():
        for idx in range(iters + warmup):
            started = time.perf_counter()
            out = model(planet_features, edge_features, planet_mask)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            if idx >= warmup:
                times.append(elapsed)
    mean_sec = sum(times) / max(1, len(times))
    return {
        "batch_size": batch_size,
        "planets": planets,
        "iters": iters,
        "mean_ms": mean_sec * 1000.0,
        "states_per_sec": batch_size / mean_sec if mean_sec > 0 else 0.0,
        "ms_per_state": mean_sec * 1000.0 / max(1, batch_size),
        "last_value_mean": float(out["value"].mean().detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "torchscript"], default="eager")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--torchscript", type=Path, default=DEFAULT_TORCHSCRIPT)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--planets", type=int, default=DEFAULT_PLANETS)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device(args.device)
    if args.mode == "eager":
        model, spec = _load_eager(args.checkpoint, device)
    else:
        spec = FeatureSpec()
        model = _load_torchscript(args.torchscript, device)

    print(
        json.dumps(
            {
                "mode": args.mode,
                "device": str(device),
                "torch_threads": torch.get_num_threads(),
                "planets": args.planets,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for batch_size in _parse_ints(args.batch_sizes):
        print(
            json.dumps(
                _time_forward(
                    model,
                    spec=spec,
                    batch_size=batch_size,
                    planets=args.planets,
                    iters=args.iters,
                    warmup=args.warmup,
                    device=device,
                ),
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
