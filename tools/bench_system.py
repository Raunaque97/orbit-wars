from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.model import FeatureSpec, make_model
from rl.native_env import NativeOrbitEnv
from rl.native_rollout import (
    collect_native_rollout,
    compute_native_gae,
    export_torchscript_model,
    ppo_update_native,
)

try:
    import orbit_native
except ImportError as exc:  # pragma: no cover - exercised before extension install.
    raise SystemExit(
        "Native extensions are not importable. Build/install the repo first, for example:\n"
        "  python -m pip install -e ."
    ) from exc


DEFAULT_WORK_DIR = Path(tempfile.gettempdir()) / "orbit_wars_system_bench"
DEFAULT_BATCH_SIZES = "1,8,32,128"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _rss_mb() -> float | None:
    try:
        import psutil

        proc = psutil.Process()
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                pass
        return total / (1024.0 * 1024.0)
    except Exception:
        pass

    try:
        import resource

        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0
    except Exception:
        return None


class MemoryMonitor:
    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = float(interval_sec)
        self.peak_mb = _rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "MemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()
        latest = _rss_mb()
        if latest is not None:
            self.peak_mb = latest if self.peak_mb is None else max(self.peak_mb, latest)

    def _run(self) -> None:
        while not self._stop.is_set():
            latest = _rss_mb()
            if latest is not None:
                self.peak_mb = latest if self.peak_mb is None else max(self.peak_mb, latest)
            self._stop.wait(self.interval_sec)


class JsonlReporter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        record = {"event": event, "time_ms": _now_ms(), **payload}
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def _parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one integer value is required")
    return values


def _device_from_arg(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _system_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "selected_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "rss_mb": _rss_mb(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "memory_gb": round(
                    torch.cuda.get_device_properties(idx).total_memory
                    / (1024.0**3),
                    2,
                ),
                "capability": ".".join(
                    str(v)
                    for v in torch.cuda.get_device_capability(idx)
                ),
            }
            for idx in range(torch.cuda.device_count())
        ]
    return info


def _load_or_init_model(
    checkpoint: Path | None, device: torch.device
) -> tuple[torch.nn.Module, FeatureSpec, str]:
    spec = FeatureSpec()
    model = make_model(spec).to(device)
    source = "random_init"
    if checkpoint is not None and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        spec = payload.get("spec", spec)
        model = make_model(spec).to(device)
        model.load_state_dict(payload["model"], strict=False)
        source = str(checkpoint)
    model.eval()
    return model, spec, source


def _random_actions(
    obs: dict[str, Any],
    rng: random.Random,
    *,
    launch_prob: float = 0.12,
    min_fleet: int = 5,
    max_fleet: int = 30,
) -> list[list[int | float]]:
    player = int(obs.get("player", 0))
    moves: list[list[int | float]] = []
    for planet in obs.get("planets", []) or []:
        if int(planet[1]) != player:
            continue
        ships = int(planet[5])
        if ships < min_fleet or rng.random() > launch_prob:
            continue
        moves.append(
            [
                int(planet[0]),
                rng.random() * 2.0 * math.pi,
                min(ships, rng.randint(min_fleet, max_fleet)),
            ]
        )
    return moves


def benchmark_native_sim(
    *,
    games: int,
    steps: int,
    seed: int,
    policy: str,
) -> dict[str, Any]:
    total_steps = 0
    completed_games = 0
    started = time.perf_counter()
    for game_idx in range(games):
        env = NativeOrbitEnv(seed=seed + game_idx, num_agents=2)
        sim = orbit_native.Simulator(env._state)
        rng = random.Random(seed * 1000003 + game_idx)
        for _ in range(steps):
            if policy == "noop":
                actions = [[], []]
            else:
                state = dict(sim.state())
                actions = []
                for player in (0, 1):
                    obs = dict(state)
                    obs["player"] = player
                    actions.append(_random_actions(obs, rng))
            sim.step(actions)
            total_steps += 1
        completed_games += 1
    elapsed = time.perf_counter() - started
    equivalent_games = total_steps / max(1, steps)
    return {
        "games": games,
        "completed_games": completed_games,
        "steps_cap": steps,
        "policy": policy,
        "steps": total_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": total_steps / elapsed if elapsed > 0 else 0.0,
        "equivalent_games_per_sec": equivalent_games / elapsed if elapsed > 0 else 0.0,
        "ms_per_equivalent_game": elapsed * 1000.0 / max(1.0, equivalent_games),
    }


def benchmark_model_forward(
    *,
    model: torch.nn.Module,
    spec: FeatureSpec,
    device: torch.device,
    batch_sizes: list[int],
    planets: int,
    iters: int,
    warmup: int,
) -> list[dict[str, Any]]:
    results = []
    model.eval()
    with torch.inference_mode():
        for batch_size in batch_sizes:
            planet_features = torch.randn(
                batch_size,
                planets,
                spec.planet_dim,
                dtype=torch.float32,
                device=device,
            )
            edge_features = torch.randn(
                batch_size,
                planets,
                planets,
                spec.edge_dim,
                dtype=torch.float32,
                device=device,
            )
            planet_mask = torch.ones(batch_size, planets, dtype=torch.bool, device=device)
            times: list[float] = []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for idx in range(iters + warmup):
                _sync_device(device)
                started = time.perf_counter()
                out = model(planet_features, edge_features, planet_mask)
                _sync_device(device)
                elapsed = time.perf_counter() - started
                if idx >= warmup:
                    times.append(elapsed)
            mean_sec = sum(times) / max(1, len(times))
            result: dict[str, Any] = {
                "device": str(device),
                "batch_size": batch_size,
                "planets": planets,
                "iters": iters,
                "warmup": warmup,
                "mean_ms": mean_sec * 1000.0,
                "states_per_sec": batch_size / mean_sec if mean_sec > 0 else 0.0,
                "ms_per_state": mean_sec * 1000.0 / max(1, batch_size),
                "last_value_mean": float(out["value"].mean().detach().cpu()),
            }
            if device.type == "cuda":
                result["cuda_peak_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                )
            results.append(result)
    return results


def benchmark_ppo_stack(
    *,
    model: torch.nn.Module,
    spec: FeatureSpec,
    device: torch.device,
    work_dir: Path,
    episodes: int,
    max_steps: int,
    seed: int,
    worker_threads: int,
    torch_threads: int,
    update_epochs: int,
    minibatch_size: int,
) -> dict[str, Any]:
    learner_path = export_torchscript_model(
        model,
        spec,
        work_dir / "models" / "learner.ts.pt",
        batch_size=min(max(1, episodes), 16),
    )
    seeds = [seed + idx for idx in range(episodes)]
    opponents = [
        {"name": "random_v2", "model_path": None, "population_index": 0}
        for _ in seeds
    ]

    rss_before = _rss_mb()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with MemoryMonitor() as monitor:
        rollout_started = time.perf_counter()
        batch = collect_native_rollout(
            learner_model_path=learner_path,
            seeds=seeds,
            opponents=opponents,
            random_v2_policy_prob=1.0,
            max_steps=max_steps,
            seed=seed,
            early_ship_share=0.8,
            early_production_share=0.7,
            delay_cache_dir=work_dir / "delay_cache",
            torch_threads=torch_threads,
            worker_threads=worker_threads,
        )
        rollout_sec = time.perf_counter() - rollout_started
        transitions = int(batch["reward"].shape[0])
        advantages, returns = compute_native_gae(batch, 0.995, 0.95)
        train_model = model.to(device)
        train_model.train()
        optimizer = torch.optim.AdamW(train_model.parameters(), lr=3e-5)
        config = argparse.Namespace(
            device=str(device),
            seed=seed,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.001,
            max_grad_norm=1.0,
        )
        update_started = time.perf_counter()
        update_stats = ppo_update_native(
            train_model, optimizer, batch, advantages, returns, config
        )
        _sync_device(device)
        update_sec = time.perf_counter() - update_started
        peak_rss_mb = monitor.peak_mb

    lengths = batch["episode_lengths"].to(torch.long).tolist()
    rewards = batch["final_rewards"].to(torch.float32).tolist()
    rollout_stats = dict(batch.get("stats", {}))
    result: dict[str, Any] = {
        "episodes": episodes,
        "max_steps": max_steps,
        "steps": sum(int(v) for v in lengths),
        "transitions": transitions,
        "rollout_sec": rollout_sec,
        "update_sec": update_sec,
        "total_sec": rollout_sec + update_sec,
        "games_per_sec_rollout": episodes / rollout_sec if rollout_sec > 0 else 0.0,
        "games_per_sec_total": episodes / (rollout_sec + update_sec)
        if rollout_sec + update_sec > 0
        else 0.0,
        "steps_per_sec_rollout": sum(int(v) for v in lengths) / rollout_sec
        if rollout_sec > 0
        else 0.0,
        "transitions_per_sec_update": transitions / update_sec
        if update_sec > 0
        else 0.0,
        "mean_episode_length": sum(int(v) for v in lengths) / max(1, len(lengths)),
        "win_rate": sum(1 for reward in rewards if reward > 0) / max(1, len(rewards)),
        "worker_threads": worker_threads,
        "native_torch_threads": torch_threads,
        "update_device": str(device),
        "rss_before_mb": rss_before,
        "peak_rss_mb": peak_rss_mb,
        "delay_cache_hits": int(rollout_stats.get("delay_cache_hits", 0)),
        "feature_calls": int(rollout_stats.get("feature_calls", transitions)),
        "feature_ms": float(rollout_stats.get("feature_ms", 0.0)),
        "collect_total_ms": rollout_stats.get("collect_total_ms"),
        "learner_forward_ms": rollout_stats.get("learner_forward_ms"),
        "opponent_random_ms": rollout_stats.get("opponent_random_ms"),
        "sim_step_ms": rollout_stats.get("sim_step_ms"),
        **{f"update_{key}": value for key, value in update_stats.items()},
    }
    feature_calls = int(result["feature_calls"])
    result["delay_cache_hit_rate"] = int(result["delay_cache_hits"]) / max(1, feature_calls)
    if device.type == "cuda":
        result["cuda_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        )
    return result


def _print_summary(records: dict[str, Any]) -> None:
    print("\nSummary", flush=True)
    system = records["system"]
    print(
        f"device={system['selected_device']} torch={system['torch_version']} "
        f"cpu_count={system['cpu_count']} torch_threads={system['torch_threads']}",
        flush=True,
    )
    native = records.get("native_sim")
    if native:
        print(
            f"native_sim={native['steps_per_sec']:.0f} steps/s "
            f"({native['equivalent_games_per_sec']:.2f}x {native['steps_cap']}-step games/s)",
            flush=True,
        )
    model = records.get("model_forward", [])
    if model:
        best = max(model, key=lambda item: float(item["states_per_sec"]))
        print(
            f"model_forward_best=batch{best['batch_size']} "
            f"{best['states_per_sec']:.0f} states/s on {best['device']}",
            flush=True,
        )
    ppo = records.get("ppo_stack")
    if ppo:
        print(
            f"ppo_stack={ppo['games_per_sec_total']:.2f} games/s total "
            f"rollout={ppo['games_per_sec_rollout']:.2f} games/s "
            f"update={ppo['transitions_per_sec_update']:.0f} transitions/s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Orbit Wars RL training throughput for cloud GPU/Colab machines."
        )
    )
    parser.add_argument("--quick", action="store_true", help="Use short smoke-test sizes.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--worker-threads", type=int, default=4)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--skip-native-sim", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--native-games", type=int, default=50)
    parser.add_argument("--native-steps", type=int, default=250)
    parser.add_argument("--native-policy", choices=["noop", "random"], default="random")
    parser.add_argument("--model-batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--model-planets", type=int, default=32)
    parser.add_argument("--model-iters", type=int, default=100)
    parser.add_argument("--model-warmup", type=int, default=10)
    parser.add_argument("--ppo-episodes", type=int, default=16)
    parser.add_argument("--ppo-max-steps", type=int, default=128)
    parser.add_argument("--ppo-update-epochs", type=int, default=1)
    parser.add_argument("--ppo-minibatch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.quick:
        args.native_games = min(args.native_games, 4)
        args.native_steps = min(args.native_steps, 32)
        args.model_batch_sizes = "1,8"
        args.model_iters = min(args.model_iters, 5)
        args.model_warmup = min(args.model_warmup, 1)
        args.ppo_episodes = min(args.ppo_episodes, 2)
        args.ppo_max_steps = min(args.ppo_max_steps, 8)
        args.ppo_minibatch_size = min(args.ppo_minibatch_size, 16)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.torch_threads))
    device = _device_from_arg(args.device)
    reporter = JsonlReporter(args.jsonl)

    records: dict[str, Any] = {}
    system = _system_info(device)
    records["system"] = system
    reporter.emit("system", system)

    model, spec, model_source = _load_or_init_model(args.checkpoint, device)
    reporter.emit(
        "model_source",
        {
            "source": model_source,
            "planet_dim": spec.planet_dim,
            "edge_dim": spec.edge_dim,
        },
    )

    if not args.skip_native_sim:
        native_result = benchmark_native_sim(
            games=args.native_games,
            steps=args.native_steps,
            seed=args.seed,
            policy=args.native_policy,
        )
        records["native_sim"] = native_result
        reporter.emit("native_sim", native_result)

    if not args.skip_model:
        model_results = benchmark_model_forward(
            model=model,
            spec=spec,
            device=device,
            batch_sizes=_parse_ints(args.model_batch_sizes),
            planets=args.model_planets,
            iters=args.model_iters,
            warmup=args.model_warmup,
        )
        records["model_forward"] = model_results
        for result in model_results:
            reporter.emit("model_forward", result)

    if not args.skip_ppo:
        ppo_result = benchmark_ppo_stack(
            model=model,
            spec=spec,
            device=device,
            work_dir=args.work_dir,
            episodes=args.ppo_episodes,
            max_steps=args.ppo_max_steps,
            seed=args.seed,
            worker_threads=args.worker_threads,
            torch_threads=args.torch_threads,
            update_epochs=args.ppo_update_epochs,
            minibatch_size=args.ppo_minibatch_size,
        )
        records["ppo_stack"] = ppo_result
        reporter.emit("ppo_stack", ppo_result)

    _print_summary(records)


if __name__ == "__main__":
    main()
