from __future__ import annotations

import math
import random
from typing import Any


BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]


def distance(p1: tuple[float, float] | list[float], p2: tuple[float, float] | list[float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_planets(rng: Any | None = None) -> list[list[float | int]]:
    if rng is None:
        rng = random
    planets: list[list[float | int]] = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0

    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, 5)
        radius = 1 + math.log(prod)
        angle = rng.uniform(0, math.pi / 2)
        min_orbital = ROTATION_RADIUS_LIMIT - radius
        max_orbital = (BOARD_SIZE - CENTER - radius) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)

        if x + radius > BOARD_SIZE or x - radius < 0 or y + radius > BOARD_SIZE or y - radius < 0:
            continue
        if (BOARD_SIZE - x) - radius < 0 or (BOARD_SIZE - y) - radius < 0:
            continue
        if (x - CENTER) < radius + 5 or (y - CENTER) < radius + 5:
            continue

        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        temp_planets: list[list[float | int]] = [
            [id_counter, -1, y, x, radius, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, radius, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, radius, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, radius, ships, prod],
        ]

        valid = True
        for tp in temp_planets:
            for planet in planets:
                if distance((planet[2], planet[3]), (tp[2], tp[3])) < planet[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            planets.extend(temp_planets)
            id_counter += 4
            static_groups += 1

    attempts = 0
    max_attempts = 5000
    has_orbiting = False

    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < max_attempts):
        attempts += 1
        if attempts >= max_attempts:
            break
        prod = rng.randint(1, 5)
        radius = 1 + math.log(prod)
        x = rng.uniform(CENTER + 15, BOARD_SIZE - radius - 5)
        y = rng.uniform(CENTER + 15, BOARD_SIZE - radius - 5)

        orbital_radius = distance((x, y), (CENTER, CENTER))
        if orbital_radius < SUN_RADIUS + radius + 10:
            continue

        if orbital_radius + radius >= ROTATION_RADIUS_LIMIT:
            if x + radius > BOARD_SIZE or x - radius < 0 or y + radius > BOARD_SIZE or y - radius < 0:
                continue

        valid = True
        ships = rng.randint(5, 30)
        temp_planets = [
            [id_counter, -1, y, x, radius, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, radius, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, radius, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, radius, ships, prod],
        ]

        for tp in temp_planets:
            tp_orbital = distance((tp[2], tp[3]), (CENTER, CENTER))
            tp_is_rotating = tp_orbital + tp[4] < ROTATION_RADIUS_LIMIT

            for planet in planets:
                p_orbital = distance((planet[2], planet[3]), (CENTER, CENTER))
                p_is_rotating = p_orbital + planet[4] < ROTATION_RADIUS_LIMIT

                if distance((planet[2], planet[3]), (tp[2], tp[3])) < planet[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
                if tp_is_rotating != p_is_rotating:
                    if abs(tp_orbital - p_orbital) < tp[4] + planet[4] + PLANET_CLEARANCE:
                        valid = False
                        break

            if not valid:
                break

        if valid:
            if orbital_radius + radius < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(temp_planets)
            id_counter += 4

    return planets


def generate_comet_paths(
    initial_planets: list[list[float | int]],
    angular_velocity: float,
    spawn_step: int,
    comet_planet_ids: list[int] | set[int] | None = None,
    comet_speed: float = 4.0,
    rng: Any | None = None,
) -> list[list[list[float]]] | None:
    if rng is None:
        rng = random
    comet_ids = set(comet_planet_ids or [])
    for _ in range(300):
        eccentricity = rng.uniform(0.75, 0.93)
        semi_major = rng.uniform(60, 150)
        perihelion = semi_major * (1 - eccentricity)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        semi_minor = semi_major * math.sqrt(1 - eccentricity**2)
        focus_offset = semi_major * eccentricity
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        dense = []
        num = 5000
        for i in range(num):
            theta = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = focus_offset + semi_major * math.cos(theta)
            ey = semi_minor * math.sin(theta)
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        path = [dense[0]]
        cumulative = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cumulative += distance(dense[i], dense[i - 1])
            if cumulative >= target:
                path.append(dense[i])
                target += comet_speed

        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in comet_ids:
                continue
            orbital_radius = distance((planet[2], planet[3]), (CENTER, CENTER))
            if orbital_radius + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orbital_radius = math.sqrt(dx**2 + dy**2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orbital_radius * math.cos(cur_angle)
                py = CENTER + orbital_radius * math.sin(cur_angle)
                for sp in sym_pts:
                    if distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None
