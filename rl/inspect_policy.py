from __future__ import annotations

import argparse
from pathlib import Path

import torch
from kaggle_environments import make

from rl.agent import OrbitWarsRLAgent
from rl.model import build_graph_inputs


def inspect_checkpoint(checkpoint: Path, *, seeds: int) -> None:
    agent = OrbitWarsRLAgent(checkpoint)
    launch_count = 0
    for seed in range(seeds):
        env = make("orbit_wars", configuration={"seed": seed}, debug=True)
        env.reset()
        obs = dict(env.state[0].observation)
        obs["player"] = 0
        obs["step"] = 0

        batch = agent.engine.compute(obs, agent.spec.horizon)
        graph = build_graph_inputs(obs, batch, spec=agent.spec, device=agent.device)
        with torch.inference_mode():
            out = agent.model(
                graph["planet_features"], graph["edge_features"], graph["planet_mask"]
            )

        owned_stop = []
        for idx, planet_id in enumerate(graph["planet_ids"]):
            planet = next(p for p in obs["planets"] if int(p[0]) == int(planet_id))
            if int(planet[1]) == 0:
                owned_stop.append(
                    (int(planet_id), float(torch.sigmoid(out["stop_logits"][idx]).cpu()))
                )

        moves = agent.act(obs)
        launch_count += int(bool(moves))
        stop_text = ", ".join(f"{pid}:{prob:.3f}" for pid, prob in owned_stop)
        print(f"seed={seed} stop=[{stop_text}] moves={moves}", flush=True)

    print(f"opening_launch_seeds={launch_count}/{seeds}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint, seeds=args.seeds)


if __name__ == "__main__":
    main()
