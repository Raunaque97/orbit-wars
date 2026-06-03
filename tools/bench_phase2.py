import argparse
import math
import random
import time

import orbit_native


def make_phase2_obs(planets, owned, seed):
    rng = random.Random(seed)
    raw_planets = []
    center = (50.0, 50.0)

    for i in range(planets):
        angle = 2.0 * math.pi * i / planets
        ring = 18.0 + (i % 4) * 10.0
        x = center[0] + math.cos(angle) * ring
        y = center[1] + math.sin(angle) * ring
        x = max(4.0, min(96.0, x))
        y = max(4.0, min(96.0, y))
        production = 1 + (i * 3) % 5
        radius = 1.0 + math.log(production)
        if i < owned:
            owner = 0
            ships = 55 + rng.randrange(70)
        elif i < owned * 2:
            owner = 1
            ships = 45 + rng.randrange(70)
        else:
            owner = -1
            ships = 15 + rng.randrange(45)
        raw_planets.append([i, owner, x, y, radius, ships, production])

    enemy_source = owned if owned < planets else planets - 1
    fleets = []
    for idx in range(min(4, owned)):
        target = raw_planets[idx]
        radius = float(target[4])
        ships = 35 + idx * 10
        fleets.append(
            [
                1000 + idx,
                1,
                float(target[2]) + radius + 0.25,
                float(target[3]),
                math.pi,
                enemy_source,
                ships,
            ]
        )

    return {
        "player": 0,
        "step": 80,
        "time_budget_ms": 950,
        "angular_velocity": 0.035,
        "planets": raw_planets,
        "initial_planets": raw_planets,
        "fleets": fleets,
    }


def print_stats(label, result, wall_ms):
    stats = result["stats"]
    print(
        f"{label}: phase={stats.get('phase')}, "
        f"min_event_gap={stats.get('min_event_gap')}, "
        f"action_sets={stats.get('candidates_generated'):,}, "
        f"evaluated={stats.get('action_sets_evaluated'):,}, "
        f"rollout_ticks={stats.get('rollout_ticks'):,}, "
        f"states={stats.get('states_considered'):,}, "
        f"routes={stats.get('route_queries'):,}, "
        f"own_response_moves={stats.get('own_response_moves'):,}, "
        f"enemy_response_moves={stats.get('enemy_response_moves'):,}, "
        f"elapsed={stats.get('elapsed_ms'):.2f} ms, "
        f"wall={wall_ms:.2f} ms, "
        f"states/s={stats.get('states_per_second'):,.0f}, "
        f"timed_out={stats.get('timed_out')}, "
        f"moves={len(result['moves'])}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planets", type=int, default=60)
    parser.add_argument("--owned", type=int, default=18)
    parser.add_argument("--budget-ms", type=int, default=950)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warm-runs", type=int, default=5)
    args = parser.parse_args()

    obs = make_phase2_obs(args.planets, args.owned, args.seed)
    engine = orbit_native.Engine()
    engine.initialize(obs)

    started = time.perf_counter()
    result = engine.search_v3(obs, args.budget_ms)
    print_stats("cold", result, (time.perf_counter() - started) * 1000.0)

    total_ticks = 0
    total_wall_ms = 0.0
    for run in range(args.warm_runs):
        started = time.perf_counter()
        result = engine.search_v3(obs, args.budget_ms)
        wall_ms = (time.perf_counter() - started) * 1000.0
        total_ticks += result["stats"].get("rollout_ticks", 0)
        total_wall_ms += wall_ms
        print_stats(f"warm {run + 1}", result, wall_ms)

    if total_wall_ms > 0:
        print(
            f"warm aggregate rollout ticks/s: "
            f"{total_ticks * 1000.0 / total_wall_ms:,.0f}"
        )


if __name__ == "__main__":
    main()
