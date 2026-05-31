import random
import time

import orbit_native


def main():
    obs = {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.03,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 100, 1],
            [1, -1, 20.0, 10.0, 2.0, 5, 3],
            [2, -1, 75.0, 20.0, 2.0, 8, 5],
            [3, -1, 70.0, 70.0, 2.0, 8, 4],
        ],
    }
    obs["initial_planets"] = obs["planets"]
    engine = orbit_native.Engine()
    engine.initialize(obs)

    requests = [
        (0, random.choice([1, 2, 3]), random.randint(2, 80), random.randint(0, 250))
        for _ in range(100_000)
    ]
    start = time.perf_counter()
    routes = engine.batch_query_routes(requests)
    elapsed = time.perf_counter() - start
    reachable = sum(1 for route in routes if route["reachable"])
    print(f"{len(requests) / elapsed:,.0f} route queries/s, reachable={reachable}")


if __name__ == "__main__":
    main()
