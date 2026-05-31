"""Kaggle entrypoint for the Orbit Wars native C++ agent."""

import importlib
import importlib.machinery
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

_ENGINES = {}
_LAST_STEP_BY_PLAYER = {}
_ORBIT_NATIVE = None


def _submission_root():
    if "__file__" in globals():
        root = Path(__file__).resolve().parent
        if (root / "src" / "orbit_native").exists():
            return root

    root = Path.cwd()
    if (root / "src" / "orbit_native").exists():
        return root

    kaggle_root = Path("/kaggle_simulations/agent")
    if (kaggle_root / "src" / "orbit_native").exists():
        return kaggle_root

    return root


def _load_orbit_native():
    global _ORBIT_NATIVE
    if _ORBIT_NATIVE is not None:
        return _ORBIT_NATIVE

    try:
        _ORBIT_NATIVE = importlib.import_module("orbit_native")
        return _ORBIT_NATIVE
    except ImportError:
        pass

    root = _submission_root()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    build_dir = Path("/tmp/orbit_wars_native")
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / f"orbit_native{suffix}"

    sources = [
        root / "src" / "orbit_native" / "bindings.cpp",
        root / "src" / "orbit_native" / "orbit_core.cpp",
    ]
    include_dirs = [
        root / "src" / "orbit_native",
        root / "third_party" / "pybind11_include",
        Path(sysconfig.get_paths()["include"]),
    ]

    if not output.exists():
        compiler = os.environ.get("CXX", "c++")
        cmd = [
            compiler,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            *[f"-I{path}" for path in include_dirs],
            *[str(source) for source in sources],
            "-o",
            str(output),
        ]
        subprocess.check_call(cmd)

    sys.path.insert(0, str(build_dir))
    _ORBIT_NATIVE = importlib.import_module("orbit_native")
    return _ORBIT_NATIVE


def _get(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    if hasattr(obs, "get"):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _native_obs(obs):
    player = int(_get(obs, "player", 0))
    raw_step = _get(obs, "step", None)
    if raw_step is None:
        step = _LAST_STEP_BY_PLAYER.get(player, -1) + 1
    else:
        step = int(raw_step)
    _LAST_STEP_BY_PLAYER[player] = step

    remaining = _get(obs, "remainingOverageTime", None)
    budget_ms = 950
    if remaining is not None:
        budget_ms = max(25, min(950, int(float(remaining) * 1000) - 25))

    return {
        "player": player,
        "step": step,
        "time_budget_ms": budget_ms,
        "angular_velocity": _get(obs, "angular_velocity", 0.0),
        "planets": _get(obs, "planets", []),
        "initial_planets": _get(obs, "initial_planets", []),
        "fleets": _get(obs, "fleets", []),
        "comets": _get(obs, "comets", []),
        "comet_planet_ids": _get(obs, "comet_planet_ids", []),
    }


def agent(obs):
    orbit_native = _load_orbit_native()
    native_obs = _native_obs(obs)
    player = native_obs["player"]
    engine = _ENGINES.get(player)
    if engine is None or native_obs["step"] == 0:
        engine = orbit_native.Engine()
        _ENGINES[player] = engine
    return engine.act(native_obs)
