#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from kaggle_environments import make


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, "get"):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _agent_ref(name):
    path = Path(name)
    if path.suffix == ".py" or path.exists():
        return str(path)
    py_path = Path(f"{name}.py")
    if py_path.exists():
        return str(py_path)
    return name


def _totals(obs, player):
    planets = _get(obs, "planets", []) or []
    fleets = _get(obs, "fleets", []) or []
    production = sum(int(p[6]) for p in planets if int(p[1]) == player)
    ships = sum(int(p[5]) for p in planets if int(p[1]) == player)
    ships += sum(int(f[6]) for f in fleets if int(f[1]) == player)
    return production, ships


def _winner(rewards, p1_player, p2_player):
    if rewards[p1_player] > rewards[p2_player]:
        return "p1"
    if rewards[p2_player] > rewards[p1_player]:
        return "p2"
    return "draw"


def run_match(seed, p1_agent, p2_agent, alternate):
    if alternate and seed % 2 == 1:
        agents = [p2_agent, p1_agent]
        p1_player, p2_player = 1, 0
    else:
        agents = [p1_agent, p2_agent]
        p1_player, p2_player = 0, 1

    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run(agents)
    rewards = [state.reward for state in env.steps[-1]]

    curve = []
    for tick, step in enumerate(env.steps[:-1]):
        obs = step[p2_player].observation
        p1_prod, p1_ships = _totals(obs, p1_player)
        p2_prod, p2_ships = _totals(obs, p2_player)
        curve.append(
            {
                "tick": tick,
                "prod_diff_p2_minus_p1": p2_prod - p1_prod,
                "ship_diff_p2_minus_p1": p2_ships - p1_ships,
                "p1_prod": p1_prod,
                "p2_prod": p2_prod,
                "p1_ships": p1_ships,
                "p2_ships": p2_ships,
            }
        )

    early = curve[:51]
    avg_prod_0_50 = (
        sum(item["prod_diff_p2_minus_p1"] for item in early) / len(early)
        if early
        else 0.0
    )
    min_prod_0_50 = (
        min(item["prod_diff_p2_minus_p1"] for item in early) if early else 0
    )
    max_prod_0_50 = (
        max(item["prod_diff_p2_minus_p1"] for item in early) if early else 0
    )
    t50 = curve[min(50, len(curve) - 1)] if curve else {}

    return {
        "seed": seed,
        "agents": agents,
        "p1_player": p1_player,
        "p2_player": p2_player,
        "winner": _winner(rewards, p1_player, p2_player),
        "rewards": rewards,
        "steps": len(env.steps) - 1,
        "avg_prod_diff_p2_minus_p1_0_50": avg_prod_0_50,
        "min_prod_diff_p2_minus_p1_0_50": min_prod_0_50,
        "max_prod_diff_p2_minus_p1_0_50": max_prod_0_50,
        "t50": t50,
        "curve": curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", required=True, help="First agent, e.g. agent_v1")
    parser.add_argument("--p2", required=True, help="Second agent, e.g. agent_v2")
    parser.add_argument("--n", type=int, default=10, help="Number of matches")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget-ms", type=int, default=None)
    parser.add_argument("--no-alternate", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.budget_ms is not None:
        os.environ["ORBIT_WARS_TIME_BUDGET_MS"] = str(args.budget_ms)

    p1_agent = _agent_ref(args.p1)
    p2_agent = _agent_ref(args.p2)
    results = []
    score = {"p1": 0, "p2": 0, "draw": 0}

    for offset in range(args.n):
        seed = args.seed_start + offset
        result = run_match(seed, p1_agent, p2_agent, not args.no_alternate)
        results.append(result)
        score[result["winner"]] += 1
        t50_diff = result["t50"].get("prod_diff_p2_minus_p1", 0)
        print(
            f"seed={seed} winner={result['winner']} "
            f"p1_player={result['p1_player']} p2_player={result['p2_player']} "
            f"rewards={result['rewards']} steps={result['steps']} "
            f"avgProdDiffP2-P1_0-50={result['avg_prod_diff_p2_minus_p1_0_50']:+.2f} "
            f"t50={t50_diff:+d} "
            f"min0-50={result['min_prod_diff_p2_minus_p1_0_50']:+d} "
            f"max0-50={result['max_prod_diff_p2_minus_p1_0_50']:+d}",
            flush=True,
        )

    print(f"score {score}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "p1": p1_agent,
                    "p2": p2_agent,
                    "alternate_sides": not args.no_alternate,
                    "score": score,
                    "results": results,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
