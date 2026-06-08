import argparse
import math
import random
import statistics
import time

import orbit_rl_native

MAX_ROUTE_DELAY = 141
TIMING_KEYS = (
    "cache_update_ms",
    "comet_stats_ms",
    "predict_arrivals_ms",
    "garrison_forecast_ms",
    "delay_matrix_ms",
    "delay_estimate_ms",
    "delay_proxy_ms",
)


def make_synthetic_obs(planets=40, fleets=20, seed=7, step=0):
    rng = random.Random(seed)
    raw_planets = []
    for i in range(planets):
        angle = 2.0 * math.pi * i / planets
        ring = 16.0 + (i % 5) * 8.5
        x = max(3.0, min(97.0, 50.0 + math.cos(angle) * ring))
        y = max(3.0, min(97.0, 50.0 + math.sin(angle) * ring))
        production = 1 + (i * 3) % 5
        radius = 1.0 + math.log(production)
        owner = 0 if i % 4 == 0 else (1 + i % 3 if i % 5 == 0 else -1)
        ships = 10 + rng.randrange(80)
        raw_planets.append([i, owner, x, y, radius, ships, production])

    raw_fleets = []
    for i in range(fleets):
        src = raw_planets[rng.randrange(planets)]
        angle = rng.random() * 2.0 * math.pi
        raw_fleets.append(
            [
                i,
                rng.randrange(4),
                src[2] + math.cos(angle) * (src[4] + 0.2),
                src[3] + math.sin(angle) * (src[4] + 0.2),
                angle,
                src[0],
                5 + rng.randrange(120),
            ]
        )

    return {
        "player": 0,
        "step": step,
        "angular_velocity": 0.035,
        "planets": raw_planets,
        "initial_planets": raw_planets,
        "fleets": raw_fleets,
        "comet_planet_ids": [],
        "comets": [],
    }


def collect_kaggle_observations(limit):
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run(["random", "random", "random", "random"])
    observations = []
    for step in env.steps[:limit]:
        observations.append(step[0].observation)
    return observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "kaggle"], default="synthetic")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--planets", type=int, default=40)
    parser.add_argument("--fleets", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--same-map",
        action="store_true",
        help="Reuse one synthetic map across samples to approximate repeated turns in a game.",
    )
    args = parser.parse_args()

    if args.mode == "kaggle":
        observations = collect_kaggle_observations(args.samples)
    else:
        if args.same_map:
            base = make_synthetic_obs(args.planets, args.fleets, seed=7, step=0)
            observations = []
            for i in range(args.samples):
                obs = dict(base)
                obs["step"] = i
                observations.append(obs)
        else:
            observations = [
                make_synthetic_obs(args.planets, args.fleets, seed=7 + i, step=i)
                for i in range(args.samples)
            ]

    engine = orbit_rl_native.FeatureEngine()
    engine.initialize(observations[0])

    wall_times = []
    native_times = []
    timing_values = {key: [] for key in TIMING_KEYS}
    last = None
    for obs in observations:
        started = time.perf_counter()
        last = engine.compute(obs, args.horizon, MAX_ROUTE_DELAY)
        wall_times.append((time.perf_counter() - started) * 1000.0)
        native_times.append(last["stats"]["elapsed_ms"])
        for key in TIMING_KEYS:
            timing_values[key].append(float(last["stats"].get(key, 0.0)))

    print(
        f"samples={len(observations)} planets={last['stats']['planets']} "
        f"horizon={args.horizon} route_queries={last['stats']['route_queries']} "
        f"route_proxy_simulations={last['stats']['route_proxy_simulations']} "
        f"route_sim_ticks={last['stats']['route_sim_ticks']}"
    )
    print(
        f"wall_ms avg={statistics.mean(wall_times):.3f} "
        f"p50={statistics.median(wall_times):.3f} max={max(wall_times):.3f}"
    )
    print(
        f"native_ms avg={statistics.mean(native_times):.3f} "
        f"p50={statistics.median(native_times):.3f} max={max(native_times):.3f}"
    )
    if len(wall_times) > 1:
        warm_wall = wall_times[1:]
        warm_native = native_times[1:]
        print(
            f"steady_wall_ms avg={statistics.mean(warm_wall):.3f} "
            f"p50={statistics.median(warm_wall):.3f} max={max(warm_wall):.3f}"
        )
        print(
            f"steady_native_ms avg={statistics.mean(warm_native):.3f} "
            f"p50={statistics.median(warm_native):.3f} max={max(warm_native):.3f}"
        )
    print("timing_breakdown_ms:")
    for key in TIMING_KEYS:
        values = timing_values[key]
        print(
            f"  {key} avg={statistics.mean(values):.3f} "
            f"p50={statistics.median(values):.3f} max={max(values):.3f}"
        )


if __name__ == "__main__":
    main()
