import argparse
import math
import random
import time

import orbit_native


def make_benchmark_obs(planets, owned, seed):
    rng = random.Random(seed)
    raw_planets = []
    center = (50.0, 50.0)

    for i in range(planets):
        angle = 2.0 * math.pi * i / planets
        ring = 18.0 + (i % 4) * 11.0
        x = center[0] + math.cos(angle) * ring
        y = center[1] + math.sin(angle) * ring
        x = max(3.0, min(97.0, x))
        y = max(3.0, min(97.0, y))
        production = 1 + (i * 3) % 5
        radius = 1.0 + math.log(production)
        owner = 0 if i < owned else (-1 if i % 3 else 1)
        ships = 25 + rng.randrange(45)
        if owner == 0:
            ships = 140 + rng.randrange(160)
        raw_planets.append([i, owner, x, y, radius, ships, production])

    return {
        "player": 0,
        "step": 0,
        "time_budget_ms": 950,
        "angular_velocity": 0.035,
        "planets": raw_planets,
        "initial_planets": raw_planets,
        "fleets": [],
    }


def print_stats(label, result):
    stats = result["stats"]
    moves = result["moves"]
    print(
        f"{label}: states={stats['states_considered']:,}, "
        f"candidates={stats['candidates_generated']:,}, "
        f"routes={stats['route_queries']:,}, "
        f"elapsed={stats['elapsed_ms']:.2f} ms, "
        f"states/s={stats['states_per_second']:,.0f}, "
        f"timed_out={stats['timed_out']}, moves={len(moves)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planets", type=int, default=60)
    parser.add_argument("--owned", type=int, default=20)
    parser.add_argument("--budget-ms", type=int, default=950)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warm-runs", type=int, default=5)
    args = parser.parse_args()

    obs = make_benchmark_obs(args.planets, args.owned, args.seed)
    engine = orbit_native.Engine()
    engine.initialize(obs)

    cold_start = time.perf_counter()
    cold = engine.search(obs, args.budget_ms)
    cold_wall_ms = (time.perf_counter() - cold_start) * 1000.0
    print_stats("cold", cold)
    print(f"cold wall time: {cold_wall_ms:.2f} ms")

    total_states = 0
    total_ms = 0.0
    for i in range(args.warm_runs):
        started = time.perf_counter()
        result = engine.search(obs, args.budget_ms)
        wall_ms = (time.perf_counter() - started) * 1000.0
        total_states += result["stats"]["states_considered"]
        total_ms += wall_ms
        print_stats(f"warm {i + 1}", result)

    if total_ms > 0.0:
        print(f"warm aggregate states/s: {total_states * 1000.0 / total_ms:,.0f}")


if __name__ == "__main__":
    main()
