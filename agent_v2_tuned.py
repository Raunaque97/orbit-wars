"""Orbit Wars v2 tuned native agent.

This keeps agent_v2 as the frozen baseline while allowing MDB/risk ramp
experiments through ORBIT_WARS_V2_TUNED_RISK_* environment variables.
"""

import os

from agent_common import load_orbit_native, native_obs

_ENGINES = {}
_DEFAULT_RISK_START = 45
_DEFAULT_RISK_SLOPE_END = 60
_DEFAULT_RISK_END_VALUE = 1.0


def _risk_ramp():
    return (
        int(os.environ.get("ORBIT_WARS_V2_TUNED_RISK_START", _DEFAULT_RISK_START)),
        int(
            os.environ.get(
                "ORBIT_WARS_V2_TUNED_RISK_SLOPE_END",
                _DEFAULT_RISK_SLOPE_END,
            )
        ),
        float(
            os.environ.get(
                "ORBIT_WARS_V2_TUNED_RISK_END_VALUE",
                _DEFAULT_RISK_END_VALUE,
            )
        ),
    )


def agent(obs):
    orbit_native = load_orbit_native()
    parsed = native_obs(obs)
    player = parsed["player"]
    engine = _ENGINES.get(player)
    if engine is None or parsed["step"] == 0:
        engine = orbit_native.Engine()
        engine.set_v2_risk_ramp(*_risk_ramp())
        _ENGINES[player] = engine
    return engine.act_v2(parsed)
