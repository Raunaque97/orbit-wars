from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kaggle_environments import make

from rl.agent import OrbitWarsRLAgent


def _resolve_checkpoint(path: Path) -> Path:
    if path.exists():
        return path
    fallback = path.with_name("best_eval.pt")
    if path.name == "best.pt" and fallback.exists():
        print(f"checkpoint {path} not found; using {fallback}")
        return fallback
    raise FileNotFoundError(path)


def _obs_for_player(env: Any, player: int, step: int) -> dict[str, Any]:
    obs = dict(env.state[player].observation)
    obs["player"] = player
    obs["step"] = step
    return obs


def run_game(
    checkpoint_a: Path,
    checkpoint_b: Path,
    *,
    seed: int,
    max_steps: int,
    swap_sides: bool,
    debug: bool,
) -> tuple[dict[str, Any], Any]:
    env = make("orbit_wars", configuration={"seed": seed}, debug=debug)
    env.reset()
    agent_a = OrbitWarsRLAgent(checkpoint_a)
    agent_b = OrbitWarsRLAgent(checkpoint_b)
    agents = [agent_a, agent_b]
    labels = ["a", "b"]
    if swap_sides:
        agents = [agent_b, agent_a]
        labels = ["b", "a"]

    steps_played = 0
    for step in range(max_steps):
        actions = [
            agents[0].act(_obs_for_player(env, 0, step)),
            agents[1].act(_obs_for_player(env, 1, step)),
        ]
        env.step(actions)
        steps_played = step + 1
        if any(state.status != "ACTIVE" for state in env.state):
            break

    final = env.state
    rewards_by_label = {
        labels[player]: float(final[player].reward or 0.0) for player in (0, 1)
    }
    statuses_by_label = {labels[player]: final[player].status for player in (0, 1)}
    winner = "draw"
    if rewards_by_label["a"] > rewards_by_label["b"]:
        winner = "a"
    elif rewards_by_label["b"] > rewards_by_label["a"]:
        winner = "b"

    return (
        {
            "seed": seed,
            "swap_sides": swap_sides,
            "steps": steps_played,
            "winner": winner,
            "rewards": rewards_by_label,
            "statuses": statuses_by_label,
        },
        env,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--swap-sides", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--replay-output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    checkpoint_a = _resolve_checkpoint(args.checkpoint_a)
    checkpoint_b = _resolve_checkpoint(args.checkpoint_b)

    results: list[dict[str, Any]] = []
    last_env = None
    for game_idx in range(args.games):
        seed = args.seed_start + game_idx
        side_orders = [False, True] if args.swap_sides else [False]
        for swap in side_orders:
            result, env = run_game(
                checkpoint_a,
                checkpoint_b,
                seed=seed,
                max_steps=args.max_steps,
                swap_sides=swap,
                debug=args.debug,
            )
            results.append(result)
            last_env = env
            print(
                f"seed={seed} swap={swap} winner={result['winner']} "
                f"steps={result['steps']} rewards={result['rewards']}",
                flush=True,
            )

    wins_a = sum(1 for result in results if result["winner"] == "a")
    wins_b = sum(1 for result in results if result["winner"] == "b")
    draws = sum(1 for result in results if result["winner"] == "draw")
    summary = {
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "games": len(results),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": wins_a / max(1, len(results)),
        "win_rate_b": wins_b / max(1, len(results)),
        "results": results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json.resolve()}")

    if args.replay_output is not None and last_env is not None:
        args.replay_output.parent.mkdir(parents=True, exist_ok=True)
        html = last_env.render(mode="html", width=args.width, height=args.height)
        args.replay_output.write_text(html, encoding="utf-8")
        print(f"wrote {args.replay_output.resolve()}")


if __name__ == "__main__":
    main()
