from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orbit_native
from rl.native_env import NativeOrbitEnv


DEFAULT_GAMES = 200
DEFAULT_STEPS = 499
DEFAULT_SEED = 7
DEFAULT_POLICY = "random"
DEFAULT_MODE = "env"


def _random_actions(
    obs: dict[str, Any],
    rng: random.Random,
    *,
    launch_prob: float = 0.15,
    min_fleet: int = 5,
    max_fleet: int = 30,
) -> list[list[int | float]]:
    moves: list[list[int | float]] = []
    player = int(obs.get("player", 0))
    for planet in obs.get("planets", []) or []:
        if int(planet[1]) != player:
            continue
        ships = int(planet[5])
        if ships < min_fleet or rng.random() > launch_prob:
            continue
        fleet_size = min(ships, rng.randint(min_fleet, max_fleet))
        moves.append([int(planet[0]), rng.random() * 2.0 * math.pi, fleet_size])
    return moves


def _actions_for_state(
    state: dict[str, Any], rng: random.Random, policy: str
) -> list[list[list[int | float]]]:
    if policy == "noop":
        return [[], []]
    actions = []
    for player in (0, 1):
        obs = dict(state)
        obs["player"] = player
        actions.append(_random_actions(obs, rng))
    return actions


def _bench_raw(games: int, steps: int, seed: int, policy: str) -> tuple[int, float]:
    total_steps = 0
    started = time.perf_counter()
    for game_idx in range(games):
        env = NativeOrbitEnv(seed=seed + game_idx, num_agents=2)
        state = env._state
        rng = random.Random(seed * 1000003 + game_idx)
        for _ in range(steps):
            actions = _actions_for_state(state, rng, policy)
            state = dict(orbit_native.simulate_step(state, actions))
            total_steps += 1
    elapsed = time.perf_counter() - started
    return total_steps, elapsed


def _bench_env(games: int, steps: int, seed: int, policy: str) -> tuple[int, float]:
    total_steps = 0
    started = time.perf_counter()
    for game_idx in range(games):
        env = NativeOrbitEnv(seed=seed + game_idx, num_agents=2)
        rng = random.Random(seed * 1000003 + game_idx)
        for _ in range(steps):
            actions = _actions_for_state(env._state, rng, policy)
            env.step(actions)
            total_steps += 1
            if env.done:
                break
    elapsed = time.perf_counter() - started
    return total_steps, elapsed


def _bench_simulator(games: int, steps: int, seed: int, policy: str) -> tuple[int, float]:
    total_steps = 0
    started = time.perf_counter()
    for game_idx in range(games):
        env = NativeOrbitEnv(seed=seed + game_idx, num_agents=2)
        sim = orbit_native.Simulator(env._state)
        rng = random.Random(seed * 1000003 + game_idx)
        for _ in range(steps):
            if policy == "noop":
                actions = [[], []]
            else:
                actions = _actions_for_state(dict(sim.state()), rng, policy)
            sim.step(actions)
            total_steps += 1
    elapsed = time.perf_counter() - started
    return total_steps, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--policy", choices=["noop", "random"], default=DEFAULT_POLICY)
    parser.add_argument("--mode", choices=["raw", "env", "simulator"], default=DEFAULT_MODE)
    args = parser.parse_args()

    if args.mode == "raw":
        total_steps, elapsed = _bench_raw(args.games, args.steps, args.seed, args.policy)
    elif args.mode == "simulator":
        total_steps, elapsed = _bench_simulator(args.games, args.steps, args.seed, args.policy)
    else:
        total_steps, elapsed = _bench_env(args.games, args.steps, args.seed, args.policy)

    full_game_steps = max(1, args.steps)
    equivalent_games = total_steps / full_game_steps
    print(f"mode={args.mode} policy={args.policy} games_requested={args.games} steps_cap={args.steps}")
    print(f"steps={total_steps} elapsed_sec={elapsed:.3f}")
    print(f"steps_per_sec={total_steps / elapsed:.1f}")
    print(f"equivalent_{full_game_steps}_step_games_per_sec={equivalent_games / elapsed:.2f}")
    print(f"ms_per_equivalent_game={elapsed * 1000.0 / max(1.0, equivalent_games):.2f}")


if __name__ == "__main__":
    main()
