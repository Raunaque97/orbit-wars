from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.model import FeatureSpec, make_model


DEFAULT_CHECKPOINT = Path("rl/checkpoints/ppo_psro_v1/iter_0100.pt")
DEFAULT_OUTPUT = Path("rl/checkpoints/ppo_psro_v1/model_ts.pt")
DEFAULT_BATCH_SIZE = 16
DEFAULT_PLANETS = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--planets", type=int, default=DEFAULT_PLANETS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    spec = payload.get("spec", FeatureSpec())
    model = make_model(spec).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()

    bsz = max(1, int(args.batch_size))
    n = max(1, int(args.planets))
    example = (
        torch.randn(bsz, n, spec.planet_dim, dtype=torch.float32, device=device),
        torch.randn(bsz, n, n, spec.edge_dim, dtype=torch.float32, device=device),
        torch.ones(bsz, n, dtype=torch.bool, device=device),
    )

    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
        traced = torch.jit.freeze(traced)
        traced = torch.jit.optimize_for_inference(traced)
        traced(*example)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    print(
        f"saved={args.output} checkpoint={args.checkpoint} "
        f"batch_size={bsz} planets={n} planet_dim={spec.planet_dim} edge_dim={spec.edge_dim}"
    )


if __name__ == "__main__":
    main()
