from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import orbit_native
from rl import orbit_map as ow


MAX_EPISODE_STEPS = 500


class NativePlayerState:
    def __init__(
        self,
        *,
        env: NativeOrbitEnv | None = None,
        player: int = 0,
        observation: dict[str, Any] | None = None,
        status: str = "ACTIVE",
        reward: float | None = None,
    ) -> None:
        self._env = env
        self._player = int(player)
        self._observation = observation
        self.status = status
        self.reward = reward

    @property
    def observation(self) -> dict[str, Any]:
        if self._env is not None and self._env._sim is not None:
            obs = dict(self._env._sim.observation(self._player))
            obs["seed"] = self._env.seed
            return obs
        if self._observation is None:
            return {}
        return dict(self._observation)


def _copy_obs_for_player(state: dict[str, Any], player: int) -> dict[str, Any]:
    obs = dict(state)
    obs["player"] = player
    return obs


def _remove_expired_comets(state: dict[str, Any]) -> None:
    expired: list[int] = []
    for group in state.get("comets", []) or []:
        idx = int(group.get("path_index", 0))
        for offset, pid in enumerate(group.get("planet_ids", []) or []):
            paths = group.get("paths", []) or []
            if offset >= len(paths) or idx >= len(paths[offset]):
                expired.append(int(pid))
    if not expired:
        return

    expired_set = set(expired)
    state["planets"] = [p for p in state.get("planets", []) if int(p[0]) not in expired_set]
    state["initial_planets"] = [
        p for p in state.get("initial_planets", []) if int(p[0]) not in expired_set
    ]
    state["comet_planet_ids"] = [
        pid for pid in state.get("comet_planet_ids", []) if int(pid) not in expired_set
    ]
    kept_groups = []
    for group in state.get("comets", []) or []:
        kept_ids = []
        kept_paths = []
        for idx, pid in enumerate(group.get("planet_ids", []) or []):
            if int(pid) not in expired_set:
                kept_ids.append(pid)
                kept_paths.append(group["paths"][idx])
        if kept_ids:
            group = dict(group)
            group["planet_ids"] = kept_ids
            group["paths"] = kept_paths
            kept_groups.append(group)
    state["comets"] = kept_groups


def _spawn_comets(state: dict[str, Any], *, seed: int, comet_speed: float) -> None:
    step = int(state.get("step", 0))
    spawn_step = step + 1
    if spawn_step not in ow.COMET_SPAWN_STEPS:
        return

    comet_rng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
    comet_paths = ow.generate_comet_paths(
        state["initial_planets"],
        float(state["angular_velocity"]),
        spawn_step,
        state.get("comet_planet_ids", []),
        comet_speed,
        rng=comet_rng,
    )
    if not comet_paths:
        return

    next_id = max(int(p[0]) for p in state["planets"]) + 1
    comet_ships = min(
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
    )
    group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
    for i, _path in enumerate(comet_paths):
        pid = next_id + i
        group["planet_ids"].append(pid)
        state.setdefault("comet_planet_ids", []).append(pid)
        planet = [
            pid,
            -1,
            -99.0,
            -99.0,
            ow.COMET_RADIUS,
            comet_ships,
            ow.COMET_PRODUCTION,
        ]
        state["planets"].append(planet)
        state["initial_planets"].append(planet[:])
    state.setdefault("comets", []).append(group)


def _alive_players(state: dict[str, Any]) -> set[int]:
    alive = set()
    for planet in state.get("planets", []) or []:
        owner = int(planet[1])
        if owner != -1:
            alive.add(owner)
    for fleet in state.get("fleets", []) or []:
        alive.add(int(fleet[1]))
    return alive


def _scores(state: dict[str, Any], num_agents: int) -> list[int]:
    scores = [0] * num_agents
    for planet in state.get("planets", []) or []:
        owner = int(planet[1])
        if owner != -1 and 0 <= owner < num_agents:
            scores[owner] += int(planet[5])
    for fleet in state.get("fleets", []) or []:
        owner = int(fleet[1])
        if 0 <= owner < num_agents:
            scores[owner] += int(fleet[6])
    return scores


class NativeOrbitEnv:
    def __init__(
        self,
        *,
        seed: int,
        num_agents: int = 2,
        comet_speed: float = 4.0,
    ) -> None:
        self.seed = int(seed)
        self.num_agents = int(num_agents)
        self.comet_speed = float(comet_speed)
        self.done = False
        self._state: dict[str, Any] = {}
        self._sim: orbit_native.Simulator | None = None
        self.state: list[NativePlayerState] = []
        self.reset()

    def reset(self) -> list[NativePlayerState]:
        rng = random.Random(self.seed)
        angular_velocity = rng.uniform(0.025, 0.05)
        planets = ow.generate_planets(rng)
        initial_planets = [p.copy() for p in planets]

        num_groups = len(planets) // 4
        if num_groups > 0:
            home_group = rng.randint(0, num_groups - 1)
            base = home_group * 4
            if self.num_agents == 2:
                planets[base][1] = 0
                planets[base][5] = 10
                planets[base + 3][1] = 1
                planets[base + 3][5] = 10
            elif self.num_agents == 4:
                for player in range(4):
                    planets[base + player][1] = player
                    planets[base + player][5] = 10

        self.done = False
        self._state = {
            "seed": self.seed,
            "step": 0,
            "angular_velocity": angular_velocity,
            "planets": planets,
            "initial_planets": initial_planets,
            "fleets": [],
            "next_fleet_id": 0,
            "comets": [],
            "comet_planet_ids": [],
        }
        self._sim = orbit_native.Simulator(self._state)
        self._sync_player_states(status="ACTIVE", rewards=[0.0] * self.num_agents)
        return self.state

    def step(self, actions: list[list[list[float | int]]]) -> list[NativePlayerState]:
        if self.done:
            return self.state

        if self._sim is None:
            self._sim = orbit_native.Simulator(self._state)

        # Comets are spawned in Python for exact Kaggle RNG parity, so we only
        # materialize/reload native state on spawn ticks. Other ticks stay in C++.
        if int(self._sim.step_index) + 1 in ow.COMET_SPAWN_STEPS:
            self._state = dict(self._sim.state())
            self._state["seed"] = self.seed
            _spawn_comets(self._state, seed=self.seed, comet_speed=self.comet_speed)
            self._sim.reset(self._state)

        self._sim.advance(actions)
        self._apply_terminal()
        return self.state

    def _apply_terminal(self) -> None:
        if self._sim is None:
            scores = _scores(self._state, self.num_agents)
            alive_count = len(_alive_players(self._state))
            step = int(self._state.get("step", 0))
        else:
            scores = [int(v) for v in self._sim.scores(self.num_agents)]
            alive_count = int(self._sim.alive_count())
            step = int(self._sim.step_index)

        terminated = step >= MAX_EPISODE_STEPS - 1
        if alive_count <= 1:
            terminated = True

        if not terminated:
            self._sync_player_states(status="ACTIVE", rewards=[0.0] * self.num_agents)
            return

        self.done = True
        max_score = max(scores) if scores else 0
        rewards = [1.0 if score == max_score and max_score > 0 else -1.0 for score in scores]
        self._sync_player_states(status="DONE", rewards=rewards)

    def _sync_player_states(self, *, status: str, rewards: list[float | None]) -> None:
        self.state = [
            NativePlayerState(
                env=self if self._sim is not None else None,
                player=player,
                observation=(
                    None if self._sim is not None else _copy_obs_for_player(self._state, player)
                ),
                status=status,
                reward=rewards[player],
            )
            for player in range(self.num_agents)
        ]


def native_state_as_namespace(player_state: NativePlayerState) -> SimpleNamespace:
    return SimpleNamespace(
        observation=SimpleNamespace(**player_state.observation),
        status=player_state.status,
        reward=player_state.reward,
    )
