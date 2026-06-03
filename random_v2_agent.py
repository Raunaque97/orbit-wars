"""V2 search agent with occasional valid random exploration moves."""

import os
import random

from agent_common import load_orbit_native, native_obs


_SEARCH_ENGINES = {}
_ROUTE_ENGINES = {}
_RNGS = {}


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _rng_for(player):
    rng = _RNGS.get(player)
    if rng is None:
        seed = _env_int("RANDOM_V2_SEED", 1729) + player * 1009
        rng = random.Random(seed)
        _RNGS[player] = rng
    return rng


def _search_moves(engine, parsed, budget_ms):
    return engine.search_v2(parsed, budget_ms)["moves"]


def _valid_random_move(orbit_native, parsed, rng):
    player = parsed["player"]
    step = parsed["step"]
    attempts = _env_int("RANDOM_V2_MAX_ATTEMPTS", 128)

    my_planets = [
        p for p in parsed["planets"] if int(p[1]) == player and int(p[5]) > 0
    ]
    targets = [p for p in parsed["planets"] if int(p[1]) != player]
    if not my_planets or not targets:
        return []

    route_engine = _ROUTE_ENGINES.get(player)
    if route_engine is None or step == 0:
        route_engine = orbit_native.Engine()
        _ROUTE_ENGINES[player] = route_engine
    route_engine.initialize(parsed)

    for _ in range(attempts):
        src = rng.choice(my_planets)
        target = rng.choice(targets)
        source_ships = int(src[5])
        if source_ships <= 0:
            continue

        ships = rng.randint(1, source_ships)
        route = route_engine.query_route(int(src[0]), int(target[0]), ships, step)
        if not route["reachable"]:
            continue
        return [[int(src[0]), float(route["angle"]), ships]]

    return []


def agent(obs):
    orbit_native = load_orbit_native()
    parsed = native_obs(obs)
    player = parsed["player"]
    rng = _rng_for(player)

    search_engine = _SEARCH_ENGINES.get(player)
    if search_engine is None or parsed["step"] == 0:
        search_engine = orbit_native.Engine()
        _SEARCH_ENGINES[player] = search_engine

    prob = max(0.0, min(1.0, _env_float("RANDOM_V2_PROB", 0.10)))
    budget_ms = max(1, min(950, _env_int("RANDOM_V2_SEARCH_BUDGET_MS", 100)))

    if rng.random() < prob:
        moves = _valid_random_move(orbit_native, parsed, rng)
        if moves:
            return moves

    return _search_moves(search_engine, parsed, budget_ms)
