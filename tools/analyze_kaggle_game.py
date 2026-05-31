import argparse
import json
import math
import subprocess
from pathlib import Path

try:
    import orbit_native
except Exception:
    orbit_native = None

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    SUN_RADIUS,
    point_to_segment_distance,
    swept_pair_hit,
)


def speed(ships, max_speed=6.0):
    if ships <= 1:
        return 1.0
    return 1.0 + (max_speed - 1.0) * min(
        1.0, (math.log(ships) / math.log(1000.0)) ** 1.5
    )


def angle_delta(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def moving_point_circle_clearance(f0, f1, p0, p1, radius):
    d0x = f0[0] - p0[0]
    d0y = f0[1] - p0[1]
    dvx = (f1[0] - f0[0]) - (p1[0] - p0[0])
    dvy = (f1[1] - f0[1]) - (p1[1] - p0[1])
    a = dvx * dvx + dvy * dvy
    if a <= 1e-12:
        return math.hypot(d0x, d0y) - radius
    t = max(0.0, min(1.0, -(d0x * dvx + d0y * dvy) / a))
    cx = d0x + dvx * t
    cy = d0y + dvy * t
    return math.hypot(cx, cy) - radius


def planets_by_id(obs):
    return {p[0]: p for p in obs.get("planets", [])}


def fleets_by_id(obs):
    return {f[0]: f for f in obs.get("fleets", [])}


def obs_with_step(obs, step, player=None):
    out = dict(obs)
    out["step"] = step
    if player is not None:
        out["player"] = player
    return out


def match_birth_action(step_state, fleet):
    actions = step_state.get("action") or []
    best = None
    for action in actions:
        if len(action) != 3:
            continue
        if int(action[0]) != fleet[5] or int(action[2]) != fleet[6]:
            continue
        delta = angle_delta(float(action[1]), fleet[4])
        item = {"action": action, "angle_delta": delta}
        if best is None or delta < best["angle_delta"]:
            best = item
    if best is not None and best["angle_delta"] <= 1e-9:
        return best
    return best


def download_episode(episode_id, replay_path, logs_dir):
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not replay_path.exists():
        subprocess.run(
            ["kaggle", "competitions", "replay", str(episode_id), "-p", str(replay_path.parent)],
            check=True,
        )
    for player in (0, 1):
        log_path = logs_dir / f"episode-{episode_id}-agent-{player}-logs.json"
        if not log_path.exists():
            subprocess.run(
                ["kaggle", "competitions", "logs", str(episode_id), str(player), "-p", str(logs_dir)],
                check=True,
            )


def summarize_logs(logs_dir, episode_id):
    summary = {}
    for player in (0, 1):
        path = Path(logs_dir) / f"episode-{episode_id}-agent-{player}-logs.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        calls = []
        for row in raw:
            if isinstance(row, list):
                calls.extend(item for item in row if isinstance(item, dict))
            elif isinstance(row, dict):
                calls.append(row)
        durations = sorted(float(call.get("duration", 0.0)) for call in calls)
        stderr = [call.get("stderr", "") for call in calls if call.get("stderr")]
        stdout = [call.get("stdout", "") for call in calls if call.get("stdout")]
        p95 = durations[int(0.95 * (len(durations) - 1))] if durations else 0.0
        summary[player] = {
            "calls": len(calls),
            "max_duration": max(durations) if durations else 0.0,
            "p95_duration": p95,
            "stderr_count": len(stderr),
            "stdout_count": len(stdout),
            "stderr_samples": stderr[:3],
            "stdout_samples": stdout[:3],
        }
    return summary


def infer_intended_target(prev_obs, fleet):
    if orbit_native is None:
        return None
    if fleet[1] != prev_obs.get("player"):
        # The observation belongs to one player. It is still enough for geometry,
        # but skip ownership-dependent search reconstruction for the opponent.
        obs = dict(prev_obs)
        obs["player"] = fleet[1]
    else:
        obs = prev_obs

    engine = orbit_native.Engine()
    engine.initialize(obs)
    best = None
    for planet in obs.get("planets", []):
        if planet[0] == fleet[5] or planet[1] == fleet[1]:
            continue
        route = engine.query_route(fleet[5], planet[0], fleet[6], obs.get("step", 0))
        if not route["reachable"]:
            continue
        da = angle_delta(route["angle"], fleet[4])
        item = {
            "target_id": planet[0],
            "angle_delta": da,
            "route": route,
            "target": planet,
        }
        if best is None or item["angle_delta"] < best["angle_delta"]:
            best = item
    return best


def orbit_alignment(steps, limit=8):
    if not steps:
        return None
    obs0 = steps[0][0]["observation"]
    planets0 = obs0.get("planets", [])
    angular_velocity = obs0.get("angular_velocity", 0.0)
    moving = None
    for planet in planets0:
        orbital_radius = math.hypot(planet[2] - CENTER, planet[3] - CENTER)
        if orbital_radius + planet[4] < 50.0:
            moving = planet
            break
    if moving is None:
        return None

    dx = moving[2] - CENTER
    dy = moving[3] - CENTER
    orbital_radius = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    rows = []
    for index in range(min(limit, len(steps))):
        obs = steps[index][0]["observation"]
        planet = planets_by_id(obs).get(moving[0])
        if planet is None:
            continue
        best = None
        for orbit_step in range(limit + 2):
            x = CENTER + orbital_radius * math.cos(theta + angular_velocity * orbit_step)
            y = CENTER + orbital_radius * math.sin(theta + angular_velocity * orbit_step)
            err = math.hypot(planet[2] - x, planet[3] - y)
            if best is None or err < best["error"]:
                best = {"orbit_step": orbit_step, "error": err}
        rows.append(
            {
                "replay_index": index,
                "obs_step": obs.get("step", index),
                "best_orbit_step": best["orbit_step"],
                "error": best["error"],
            }
        )
    return {"planet_id": moving[0], "rows": rows}


def classify_transition(obs, next_obs, fleet):
    old = (fleet[2], fleet[3])
    spd = speed(fleet[6])
    new = (fleet[2] + math.cos(fleet[4]) * spd, fleet[3] + math.sin(fleet[4]) * spd)
    planets0 = planets_by_id(obs)
    planets1 = planets_by_id(next_obs)

    nearest = None
    for pid, p0 in planets0.items():
        p1 = planets1.get(pid, p0)
        p_old = (p0[2], p0[3])
        p_new = (p1[2], p1[3])
        clearance = moving_point_circle_clearance(old, new, p_old, p_new, p0[4])
        if nearest is None or clearance < nearest["clearance"]:
            nearest = {"planet_id": pid, "clearance": clearance, "planet": p0}
        if swept_pair_hit(old, new, p_old, p_new, p0[4]):
            return {
                "kind": "planet",
                "planet_id": pid,
                "clearance": clearance,
                "new_pos": new,
                "nearest": nearest,
            }

    if not (0 <= new[0] <= 100 and 0 <= new[1] <= 100):
        return {"kind": "bounds", "planet_id": None, "new_pos": new, "nearest": nearest}
    if point_to_segment_distance((CENTER, CENTER), old, new) < SUN_RADIUS:
        return {"kind": "sun", "planet_id": None, "new_pos": new, "nearest": nearest}
    return {"kind": "none", "planet_id": None, "new_pos": new, "nearest": nearest}


def analyze_replay(
    replay_path,
    submission_id=None,
    episode_id=None,
    near_threshold=0.75,
    logs_dir="logs",
):
    replay = json.loads(Path(replay_path).read_text())
    steps = replay["steps"]
    births = {}
    deaths = {}
    near_misses = []
    inferred = {}
    birth_actions = {}
    action_count = 0

    for i in range(1, len(steps)):
        prev_obs_by_player = [
            obs_with_step(steps[i - 1][p]["observation"], i - 1, p)
            for p in range(len(steps[i]))
        ]
        obs = steps[i][0]["observation"]
        prev = steps[i - 1][0]["observation"]
        fleets_prev = fleets_by_id(prev)
        fleets_now = fleets_by_id(obs)

        for state in steps[i]:
            action_count += len(state.get("action") or [])

        for fid, fleet in fleets_now.items():
            if fid not in births:
                player = fleet[1]
                prev_obs = prev_obs_by_player[player] if player < len(prev_obs_by_player) else prev
                births[fid] = {
                    "fleet": fleet,
                    "birth_step": i,
                    "decision_step": i - 1,
                    "owner": player,
                    "source": fleet[5],
                    "ships": fleet[6],
                    "angle": fleet[4],
                }
                if player < len(steps[i]):
                    birth_actions[fid] = match_birth_action(steps[i][player], fleet)
                inferred[fid] = infer_intended_target(prev_obs, fleet)

        for fid, fleet in fleets_prev.items():
            result = classify_transition(prev, obs, fleet)
            inferred_target = inferred.get(fid)
            if inferred_target is not None:
                tid = inferred_target["target_id"]
                planets0 = planets_by_id(prev)
                planets1 = planets_by_id(obs)
                if tid in planets0:
                    p0 = planets0[tid]
                    p1 = planets1.get(tid, p0)
                    old = (fleet[2], fleet[3])
                    spd = speed(fleet[6])
                    new = (
                        fleet[2] + math.cos(fleet[4]) * spd,
                        fleet[3] + math.sin(fleet[4]) * spd,
                    )
                    clearance = moving_point_circle_clearance(
                        old, new, (p0[2], p0[3]), (p1[2], p1[3]), p0[4]
                    )
                    if -1e-9 < clearance <= near_threshold:
                        near_misses.append(
                            {
                                "step": i,
                                "fleet_id": fid,
                                "owner": fleet[1],
                                "source": fleet[5],
                                "ships": fleet[6],
                                "angle": fleet[4],
                                "target_id": tid,
                                "clearance": clearance,
                                "target_radius": p0[4],
                                "inferred_angle_delta": inferred_target["angle_delta"],
                                "route": inferred_target["route"],
                            }
                        )

            if fid not in fleets_now:
                deaths[fid] = {
                    "death_step": i,
                    "classification": result,
                    "birth": births.get(fid),
                    "inferred_target": inferred_target,
                }

    by_kind = {}
    for death in deaths.values():
        kind = death["classification"]["kind"]
        by_kind[kind] = by_kind.get(kind, 0) + 1

    suspicious = []
    for fid, death in deaths.items():
        inferred_target = death.get("inferred_target")
        classification = death["classification"]
        if inferred_target is None:
            continue
        target_id = inferred_target["target_id"]
        if classification["kind"] != "planet" or classification.get("planet_id") != target_id:
            nearest = classification.get("nearest") or {}
            suspicious.append(
                {
                    "fleet_id": fid,
                    "owner": death["birth"]["owner"] if death.get("birth") else None,
                    "birth_step": death["birth"]["birth_step"] if death.get("birth") else None,
                    "decision_step": death["birth"]["decision_step"] if death.get("birth") else None,
                    "death_step": death["death_step"],
                    "source": death["birth"]["source"] if death.get("birth") else None,
                    "ships": death["birth"]["ships"] if death.get("birth") else None,
                    "target_id": target_id,
                    "target_route": inferred_target["route"],
                    "death_kind": classification["kind"],
                    "death_planet": classification.get("planet_id"),
                    "nearest_planet": nearest.get("planet_id"),
                    "nearest_clearance": nearest.get("clearance"),
                    "angle_delta": inferred_target["angle_delta"],
                    "birth_action": birth_actions.get(fid),
                }
            )

    suspicious.sort(key=lambda x: (
        abs(x["nearest_clearance"]) if x["nearest_clearance"] is not None else 999,
        x["birth_step"] if x["birth_step"] is not None else 999,
    ))
    near_misses.sort(key=lambda x: x["clearance"])

    return {
        "submission_id": submission_id,
        "episode_id": episode_id,
        "steps": len(steps),
        "actions": action_count,
        "fleets_seen": len(births),
        "deaths_by_kind": by_kind,
        "orbit_alignment": orbit_alignment(steps),
        "suspicious": suspicious,
        "near_misses": near_misses,
        "logs": summarize_logs(logs_dir, episode_id) if episode_id is not None else {},
        "rewards": replay.get("rewards"),
        "statuses": replay.get("statuses"),
    }


def print_report(report, limit):
    print(f"episode={report.get('episode_id')} submission={report.get('submission_id')}")
    print(f"steps={report['steps']} actions={report['actions']} fleets_seen={report['fleets_seen']}")
    print(f"rewards={report.get('rewards')} statuses={report.get('statuses')}")
    print(f"deaths_by_kind={report['deaths_by_kind']}")
    if report.get("logs"):
        for player, logs in report["logs"].items():
            print(
                f"logs player={player}: calls={logs['calls']} "
                f"max={logs['max_duration']:.3f}s p95={logs['p95_duration']:.3f}s "
                f"stderr={logs['stderr_count']} stdout={logs['stdout_count']}"
            )
    alignment = report.get("orbit_alignment")
    if alignment:
        rows = alignment["rows"][:4]
        offsets = [
            row["obs_step"] - row["best_orbit_step"]
            for row in rows
            if row["error"] < 1e-7
        ]
        print(
            f"orbit_alignment planet={alignment['planet_id']} "
            f"first_offsets(obs_step - orbit_step)={offsets}"
        )
    print()
    print(f"suspicious target mismatches/removals: {len(report['suspicious'])}")
    for row in report["suspicious"][:limit]:
        route = row["target_route"]
        birth_action = row.get("birth_action") or {}
        action_delta = birth_action.get("angle_delta")
        print(
            "  "
            f"fleet={row['fleet_id']} owner={row['owner']} birth={row['birth_step']} "
            f"decision={row.get('decision_step')} death={row['death_step']} "
            f"src={row['source']} target={row['target_id']} "
            f"ships={row['ships']} death={row['death_kind']}:{row['death_planet']} "
            f"nearest={row['nearest_planet']} clearance={row['nearest_clearance']:.6f} "
            f"route_arrival={route['arrival_tick']} route_dt={route['travel_time']} "
            f"angle_delta={row['angle_delta']:.3g} "
            f"action_delta={(action_delta if action_delta is not None else float('nan')):.3g}"
        )
    print()
    print(f"near misses within threshold: {len(report['near_misses'])}")
    for row in report["near_misses"][:limit]:
        print(
            "  "
            f"step={row['step']} fleet={row['fleet_id']} owner={row['owner']} "
            f"src={row['source']} target={row['target_id']} ships={row['ships']} "
            f"clearance={row['clearance']:.6f} radius={row['target_radius']:.6f} "
            f"route_arrival={row['route']['arrival_tick']} route_dt={row['route']['travel_time']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--submission-id")
    parser.add_argument("--replay", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--near-threshold", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--json-out")
    parser.add_argument("--logs-dir", default="logs")
    args = parser.parse_args()

    replay_path = Path(args.replay or f"replays/episode-{args.episode_id}-replay.json")
    if args.download:
        download_episode(args.episode_id, replay_path, Path(args.logs_dir))

    report = analyze_replay(
        replay_path,
        submission_id=args.submission_id,
        episode_id=args.episode_id,
        near_threshold=args.near_threshold,
        logs_dir=args.logs_dir,
    )
    print_report(report, args.limit)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
