"""Orbit Wars v1 native agent."""

from agent_common import load_orbit_native, native_obs

_ENGINES = {}


def agent(obs):
    orbit_native = load_orbit_native()
    parsed = native_obs(obs)
    player = parsed["player"]
    engine = _ENGINES.get(player)
    if engine is None or parsed["step"] == 0:
        engine = orbit_native.Engine()
        _ENGINES[player] = engine
    return engine.act(parsed)
