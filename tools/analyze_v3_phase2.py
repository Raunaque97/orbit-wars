import argparse
import json
import os
from pathlib import Path

from kaggle_environments import make

import agent_v2
import agent_v3
from agent_common import native_obs
from agent_common import load_orbit_native


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, "get"):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _totals(obs, player):
    planets = _get(obs, "planets", []) or []
    fleets = _get(obs, "fleets", []) or []
    production = sum(int(p[6]) for p in planets if int(p[1]) == player)
    ships = sum(int(p[5]) for p in planets if int(p[1]) == player)
    ships += sum(int(f[6]) for f in fleets if int(f[1]) == player)
    return production, ships


def _same_moves(a, b):
    if len(a) != len(b):
        return False
    norm_a = sorted((int(x[0]), round(float(x[1]), 6), int(x[2])) for x in a)
    norm_b = sorted((int(x[0]), round(float(x[1]), 6), int(x[2])) for x in b)
    return norm_a == norm_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-ms", type=int, default=950)
    parser.add_argument("--v3-player", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ["ORBIT_WARS_TIME_BUDGET_MS"] = str(args.budget_ms)

    agents = [agent_v2.agent, agent_v3.agent]
    if args.v3_player == 0:
        agents = [agent_v3.agent, agent_v2.agent]

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=True)
    env.run(agents)

    orbit_native = load_orbit_native()
    v2_shadow = orbit_native.Engine()
    v3_shadow = orbit_native.Engine()
    v3_player = args.v3_player
    opp = 1 - v3_player

    rows = []
    phase_counts = {}
    diff_count = 0
    for tick, step in enumerate(env.steps[:-1]):
        obs_raw = dict(step[v3_player].observation)
        obs_raw["player"] = v3_player
        obs_raw["step"] = tick
        obs_raw["time_budget_ms"] = args.budget_ms
        parsed = native_obs(obs_raw)

        v3_result = v3_shadow.search_v3(parsed, args.budget_ms)
        v2_result = v2_shadow.search_v2(parsed, args.budget_ms)
        phase = int(v3_result["stats"].get("phase", 0))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        same = _same_moves(v3_result["moves"], v2_result["moves"])
        if not same:
            diff_count += 1

        my_prod, my_ships = _totals(obs_raw, v3_player)
        opp_prod, opp_ships = _totals(obs_raw, opp)
        row = {
            "tick": tick,
            "phase": phase,
            "same_as_v2": same,
            "v3_moves": v3_result["moves"],
            "v2_moves": v2_result["moves"],
            "stats": v3_result["stats"],
            "prod_diff": my_prod - opp_prod,
            "ship_diff": my_ships - opp_ships,
            "actual_action": step[v3_player].get("action") or [],
        }
        rows.append(row)

    final = env.steps[-1]
    summary = {
        "seed": args.seed,
        "v3_player": v3_player,
        "rewards": [state.reward for state in final],
        "statuses": [state.status for state in final],
        "steps": len(env.steps) - 1,
        "phase_counts": phase_counts,
        "diff_count": diff_count,
        "phase2_first_tick": next((r["tick"] for r in rows if r["phase"] == 2), None),
        "phase2_diff_count": sum(
            1 for r in rows if r["phase"] == 2 and not r["same_as_v2"]
        ),
        "rows": rows,
    }

    interesting = [
        r
        for r in rows
        if r["phase"] == 2 and (not r["same_as_v2"] or r["v3_moves"] or r["v2_moves"])
    ][:25]
    print(
        f"seed={args.seed} rewards={summary['rewards']} steps={summary['steps']} "
        f"phase_counts={phase_counts} phase2_first={summary['phase2_first_tick']} "
        f"diffs={diff_count} phase2_diffs={summary['phase2_diff_count']}"
    )
    for r in interesting:
        stats = r["stats"]
        print(
            f"tick={r['tick']} prodDiff={r['prod_diff']:+} shipDiff={r['ship_diff']:+} "
            f"same={r['same_as_v2']} v3moves={len(r['v3_moves'])} v2moves={len(r['v2_moves'])} "
            f"eval={stats.get('action_sets_evaluated')} routes={stats.get('route_queries')} "
            f"elapsed={stats.get('elapsed_ms'):.1f} "
            f"v3={r['v3_moves']} v2={r['v2_moves']} actual={r['actual_action']}"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
