"""V2 search agent with occasional valid random exploration moves."""

import math
import random

from agent_common import load_orbit_native, native_obs


# Edit these constants before running experiments. Keeping them in the file makes
# replay settings visible in git diffs and avoids long env-var command prefixes.
RANDOM_POLICY_PROB = 0.8
SEARCH_BUDGET_MS = 100
RANDOM_SEED = 1729

MIN_RANDOM_FLEET_SIZE = 5
MAX_RANDOM_LAUNCH_PROB = 0.10
LAUNCH_PROB_MAX_SHIPS = 50
LAUNCH_PROB_DECAY_SHIPS = 15.0

ROUTE_ATTEMPTS_PER_SOURCE = 32

_SEARCH_ENGINES = {}
_ROUTE_ENGINES = {}
_RNGS = {}


def _rng_for(player):
    rng = _RNGS.get(player)
    if rng is None:
        seed = RANDOM_SEED + player * 1009
        rng = random.Random(seed)
        _RNGS[player] = rng
    return rng


def _search_moves(engine, parsed, budget_ms):
    return engine.search_v2(parsed, budget_ms)["moves"]


def _launch_probability(ship_count):
    if ship_count < MIN_RANDOM_FLEET_SIZE:
        return 0.0
    usable = max(0.0, float(ship_count - MIN_RANDOM_FLEET_SIZE))
    max_usable = max(1.0, float(LAUNCH_PROB_MAX_SHIPS - MIN_RANDOM_FLEET_SIZE))
    decay = max(1e-9, float(LAUNCH_PROB_DECAY_SHIPS))
    raw = 1.0 - math.exp(-usable / decay)
    normalizer = 1.0 - math.exp(-max_usable / decay)
    probability = MAX_RANDOM_LAUNCH_PROB * raw / max(1e-9, normalizer)
    return max(0.0, min(float(MAX_RANDOM_LAUNCH_PROB), probability))


def _valid_random_moves(orbit_native, parsed, rng):
    player = parsed["player"]
    step = parsed["step"]

    my_planets = [
        p
        for p in parsed["planets"]
        if int(p[1]) == player and int(p[5]) >= MIN_RANDOM_FLEET_SIZE
    ]
    targets = [p for p in parsed["planets"] if int(p[1]) != player]
    if not my_planets or not targets:
        return []

    route_engine = _ROUTE_ENGINES.get(player)
    if route_engine is None or step == 0:
        route_engine = orbit_native.Engine()
        _ROUTE_ENGINES[player] = route_engine
    route_engine.initialize(parsed)

    moves = []
    rng.shuffle(my_planets)
    for src in my_planets:
        source_ships = int(src[5])
        if rng.random() >= _launch_probability(source_ships):
            continue

        for _ in range(ROUTE_ATTEMPTS_PER_SOURCE):
            target = rng.choice(targets)
            ships = rng.randint(MIN_RANDOM_FLEET_SIZE, source_ships)
            route = route_engine.query_route(int(src[0]), int(target[0]), ships, step)
            if not route["reachable"]:
                continue
            moves.append([int(src[0]), float(route["angle"]), ships])
            break

    return moves


def agent(obs):
    orbit_native = load_orbit_native()
    parsed = native_obs(obs)
    player = parsed["player"]
    rng = _rng_for(player)

    search_engine = _SEARCH_ENGINES.get(player)
    if search_engine is None or parsed["step"] == 0:
        search_engine = orbit_native.Engine()
        _SEARCH_ENGINES[player] = search_engine

    random_policy_prob = max(0.0, min(1.0, float(RANDOM_POLICY_PROB)))
    budget_ms = max(1, min(950, int(SEARCH_BUDGET_MS)))

    if rng.random() < random_policy_prob:
        return _valid_random_moves(orbit_native, parsed, rng)

    return _search_moves(search_engine, parsed, budget_ms)
