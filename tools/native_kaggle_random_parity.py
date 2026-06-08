from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Any

import orbit_rl_native
from kaggle_environments import make

from rl.native_env import NativeOrbitEnv


MAX_ROUTE_DELAY = 141


class ParityError(AssertionError):
    pass


@dataclass
class ParityStats:
    seeds: int
    ticks_requested: int
    ticks_checked: int = 0
    moves_launched: int = 0
    games_finished: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "seeds": self.seeds,
            "ticks_requested": self.ticks_requested,
            "ticks_checked": self.ticks_checked,
            "moves_launched": self.moves_launched,
            "games_finished": self.games_finished,
        }


def _obs_to_dict(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    return {
        name: getattr(obs, name)
        for name in (
            "step",
            "angular_velocity",
            "planets",
            "initial_planets",
            "fleets",
            "next_fleet_id",
            "comets",
            "comet_planet_ids",
            "player",
        )
        if hasattr(obs, name)
    }


def _assert_close(a: float, b: float, *, label: str, seed: int, tick: int) -> None:
    if not math.isclose(float(a), float(b), abs_tol=1e-8):
        raise ParityError(f"{label} diverged seed={seed} tick={tick}: native={a} kaggle={b}")


def _assert_planets_equal(
    native: list[list[Any]], kaggle: list[list[Any]], *, seed: int, tick: int
) -> None:
    if [int(p[0]) for p in native] != [int(p[0]) for p in kaggle]:
        raise ParityError(
            f"planet ids diverged seed={seed} tick={tick}: "
            f"native={[int(p[0]) for p in native]} kaggle={[int(p[0]) for p in kaggle]}"
        )
    for idx, (np, kp) in enumerate(zip(native, kaggle)):
        for col in (0, 1, 5, 6):
            if int(np[col]) != int(kp[col]):
                raise ParityError(
                    f"planet[{idx}][{col}] diverged seed={seed} tick={tick}: "
                    f"native={np} kaggle={kp}"
                )
        for col in (2, 3, 4):
            _assert_close(
                float(np[col]), float(kp[col]), label=f"planet[{idx}][{col}]", seed=seed, tick=tick
            )


def _assert_fleets_equal(
    native: list[list[Any]], kaggle: list[list[Any]], *, seed: int, tick: int
) -> None:
    if len(native) != len(kaggle):
        raise ParityError(
            f"fleet count diverged seed={seed} tick={tick}: "
            f"native={len(native)} kaggle={len(kaggle)}"
        )
    for idx, (nf, kf) in enumerate(zip(native, kaggle)):
        for col in (0, 1, 5, 6):
            if int(nf[col]) != int(kf[col]):
                raise ParityError(
                    f"fleet[{idx}][{col}] diverged seed={seed} tick={tick}: "
                    f"native={nf} kaggle={kf}"
                )
        for col in (2, 3, 4):
            _assert_close(
                float(nf[col]), float(kf[col]), label=f"fleet[{idx}][{col}]", seed=seed, tick=tick
            )


def _assert_comets_equal(native: dict[str, Any], kaggle: dict[str, Any], *, seed: int, tick: int) -> None:
    native_ids = [int(pid) for pid in native.get("comet_planet_ids", [])]
    kaggle_ids = [int(pid) for pid in kaggle.get("comet_planet_ids", [])]
    if native_ids != kaggle_ids:
        raise ParityError(
            f"comet ids diverged seed={seed} tick={tick}: native={native_ids} kaggle={kaggle_ids}"
        )
    native_comets = native.get("comets", []) or []
    kaggle_comets = kaggle.get("comets", []) or []
    if len(native_comets) != len(kaggle_comets):
        raise ParityError(
            f"comet group count diverged seed={seed} tick={tick}: "
            f"native={len(native_comets)} kaggle={len(kaggle_comets)}"
        )
    for idx, (ng, kg) in enumerate(zip(native_comets, kaggle_comets)):
        if [int(pid) for pid in ng["planet_ids"]] != [int(pid) for pid in kg["planet_ids"]]:
            raise ParityError(
                f"comet[{idx}] planet ids diverged seed={seed} tick={tick}: "
                f"native={ng['planet_ids']} kaggle={kg['planet_ids']}"
            )
        if int(ng["path_index"]) != int(kg["path_index"]):
            raise ParityError(
                f"comet[{idx}] path index diverged seed={seed} tick={tick}: "
                f"native={ng['path_index']} kaggle={kg['path_index']}"
            )


def assert_observations_equal(native_obs: Any, kaggle_obs: Any, *, seed: int, tick: int) -> None:
    native = _obs_to_dict(native_obs)
    kaggle = _obs_to_dict(kaggle_obs)
    if int(native["step"]) != int(kaggle["step"]):
        raise ParityError(
            f"step diverged seed={seed} tick={tick}: native={native['step']} kaggle={kaggle['step']}"
        )
    _assert_close(
        float(native["angular_velocity"]),
        float(kaggle["angular_velocity"]),
        label="angular_velocity",
        seed=seed,
        tick=tick,
    )
    if int(native.get("next_fleet_id", 0)) != int(kaggle.get("next_fleet_id", 0)):
        raise ParityError(
            f"next_fleet_id diverged seed={seed} tick={tick}: "
            f"native={native.get('next_fleet_id')} kaggle={kaggle.get('next_fleet_id')}"
        )
    _assert_planets_equal(native["planets"], kaggle["planets"], seed=seed, tick=tick)
    _assert_fleets_equal(native["fleets"], kaggle["fleets"], seed=seed, tick=tick)
    _assert_comets_equal(native, kaggle, seed=seed, tick=tick)


def _random_reachable_move(
    obs: Any,
    *,
    player: int,
    rng: random.Random,
    min_ships: int,
    max_fraction: float,
    attempts: int,
) -> list[float | int] | None:
    data = _obs_to_dict(obs)
    planets = [list(p) for p in data.get("planets", [])]
    sources = [p for p in planets if int(p[1]) == player and int(p[5]) >= min_ships]
    if not sources:
        return None

    engine = orbit_rl_native.FeatureEngine()
    for _ in range(attempts):
        source = rng.choice(sources)
        available = int(source[5])
        max_ships = max(min_ships, int(available * max_fraction))
        max_ships = min(available, max_ships)
        if max_ships < min_ships:
            continue
        target_candidates = [p for p in planets if int(p[0]) != int(source[0])]
        rng.shuffle(target_candidates)
        ships = rng.randint(min_ships, max_ships)
        for target in target_candidates[: min(8, len(target_candidates))]:
            route = engine.query_route(
                data, int(source[0]), int(target[0]), int(ships), MAX_ROUTE_DELAY
            )
            if route.get("reachable") and str(route.get("blocked_by", "none")) == "none":
                angle = float(route["angle"])
                if math.isfinite(angle):
                    return [int(source[0]), angle, int(ships)]
    return None


def random_legal_actions(
    env_state: list[Any],
    *,
    rng: random.Random,
    launch_prob: float,
    min_ships: int,
    max_fraction: float,
    route_attempts: int,
) -> list[list[list[float | int]]]:
    actions: list[list[list[float | int]]] = []
    for player, state in enumerate(env_state):
        moves: list[list[float | int]] = []
        if rng.random() < launch_prob:
            move = _random_reachable_move(
                state.observation,
                player=player,
                rng=rng,
                min_ships=min_ships,
                max_fraction=max_fraction,
                attempts=route_attempts,
            )
            if move is not None:
                moves.append(move)
        actions.append(moves)
    return actions


def run_random_trajectory_parity(
    *,
    seeds: int,
    ticks: int,
    seed_start: int = 0,
    action_seed: int = 12345,
    launch_prob: float = 0.35,
    min_ships: int = 5,
    max_fraction: float = 0.50,
    route_attempts: int = 24,
) -> ParityStats:
    stats = ParityStats(seeds=seeds, ticks_requested=ticks)
    for offset in range(seeds):
        seed = seed_start + offset
        rng = random.Random(action_seed + seed * 1009)
        kaggle = make("orbit_wars", configuration={"seed": seed}, debug=True)
        kaggle.reset()
        native = NativeOrbitEnv(seed=seed)
        assert_observations_equal(native.state[0].observation, kaggle.state[0].observation, seed=seed, tick=0)

        for tick in range(ticks):
            if native.done or any(state.status != "ACTIVE" for state in kaggle.state):
                stats.games_finished += 1
                break
            actions = random_legal_actions(
                kaggle.state,
                rng=rng,
                launch_prob=launch_prob,
                min_ships=min_ships,
                max_fraction=max_fraction,
                route_attempts=route_attempts,
            )
            kaggle.step(actions)
            native.step(actions)
            stats.moves_launched += sum(len(moves) for moves in actions)
            stats.ticks_checked += 1
            try:
                assert_observations_equal(
                    native.state[0].observation,
                    kaggle.state[0].observation,
                    seed=seed,
                    tick=tick + 1,
                )
                for player, (native_state, kaggle_state) in enumerate(zip(native.state, kaggle.state)):
                    if native_state.status != kaggle_state.status:
                        raise ParityError(
                            f"status diverged seed={seed} tick={tick + 1} player={player}: "
                            f"native={native_state.status} kaggle={kaggle_state.status}"
                        )
                    if native_state.reward != kaggle_state.reward:
                        raise ParityError(
                            f"reward diverged seed={seed} tick={tick + 1} player={player}: "
                            f"native={native_state.reward} kaggle={kaggle_state.reward}"
                        )
            except ParityError as exc:
                raise ParityError(
                    f"{exc}\nactions={json.dumps(actions)}"
                ) from exc
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--action-seed", type=int, default=12345)
    parser.add_argument("--launch-prob", type=float, default=0.35)
    parser.add_argument("--min-ships", type=int, default=5)
    parser.add_argument("--max-fraction", type=float, default=0.50)
    parser.add_argument("--route-attempts", type=int, default=24)
    args = parser.parse_args()

    stats = run_random_trajectory_parity(
        seeds=args.seeds,
        seed_start=args.seed_start,
        ticks=args.ticks,
        action_seed=args.action_seed,
        launch_prob=args.launch_prob,
        min_ships=args.min_ships,
        max_fraction=args.max_fraction,
        route_attempts=args.route_attempts,
    )
    print(json.dumps(stats.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
