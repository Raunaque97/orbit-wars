import argparse
import math

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    SUN_RADIUS,
    point_to_segment_distance,
    swept_pair_hit,
)


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, "get"):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _speed(ships, max_speed=6.0):
    if ships <= 1:
        return 1.0
    return 1.0 + (max_speed - 1.0) * min(
        1.0, (math.log(ships) / math.log(1000.0)) ** 1.5
    )


def first_collision(obs, move, synthetic_step):
    from_id, angle, ships = move
    planets = {p[0]: p for p in _get(obs, "planets", [])}
    if from_id not in planets:
        return ("missing_source", None, None)

    source = planets[from_id]
    x = source[2] + math.cos(angle) * (source[4] + 0.1)
    y = source[3] + math.sin(angle) * (source[4] + 0.1)
    comet_ids = set(_get(obs, "comet_planet_ids", []))
    initial_by_id = {p[0]: p for p in _get(obs, "initial_planets", [])}
    step = _get(obs, "step", synthetic_step)
    angular_velocity = _get(obs, "angular_velocity", 0.0)

    for dt in range(1, 180):
        old = (x, y)
        x += math.cos(angle) * _speed(ships)
        y += math.sin(angle) * _speed(ships)
        new = (x, y)
        tick_old = step + dt - 1
        tick_new = step + dt

        for planet in _get(obs, "planets", []):
            if planet[0] in comet_ids:
                continue
            initial = initial_by_id[planet[0]]

            def position(tick):
                dx = initial[2] - CENTER
                dy = initial[3] - CENTER
                radius = math.hypot(dx, dy)
                if radius + planet[4] < 50.0:
                    theta = math.atan2(dy, dx) + angular_velocity * tick
                    return (
                        CENTER + radius * math.cos(theta),
                        CENTER + radius * math.sin(theta),
                    )
                return (planet[2], planet[3])

            if swept_pair_hit(old, new, position(tick_old), position(tick_new), planet[4]):
                return ("planet", planet[0], dt)

        if not (0 <= x <= 100 and 0 <= y <= 100):
            return ("bounds", None, dt)
        if point_to_segment_distance((CENTER, CENTER), old, new) < SUN_RADIUS:
            return ("sun", None, dt)

    return ("none", None, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--agent-a", default="main.py")
    parser.add_argument("--agent-b", default="main.py")
    args = parser.parse_args()

    checked = 0
    for seed in range(args.seeds):
        env = make("orbit_wars", configuration={"seed": seed}, debug=True)
        env.run([args.agent_a, args.agent_b])
        synthetic_steps = [-1, -1]
        for step_index in range(1, len(env.steps)):
            previous = env.steps[step_index - 1]
            current = env.steps[step_index]
            for player, state in enumerate(current):
                obs = previous[player].observation
                raw_step = _get(obs, "step", None)
                if raw_step is None:
                    synthetic_steps[player] += 1
                else:
                    synthetic_steps[player] = raw_step
                for move in state.action or []:
                    checked += 1
                    collision = first_collision(obs, move, synthetic_steps[player])
                    if collision[0] in {"sun", "bounds", "none", "missing_source"}:
                        raise AssertionError(
                            f"bad action seed={seed} step={step_index} player={player} "
                            f"move={move} first_collision={collision}"
                        )

    print(f"checked {checked} actions across {args.seeds} seeds")


if __name__ == "__main__":
    main()
