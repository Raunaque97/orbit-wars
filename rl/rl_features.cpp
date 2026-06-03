#include "rl_features.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace orbit_rl {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kEps = 1e-9;

Vec2 planet_pos(const Planet& p) { return Vec2{p.x, p.y}; }
Vec2 add(Vec2 a, Vec2 b) { return Vec2{a.x + b.x, a.y + b.y}; }
Vec2 sub(Vec2 a, Vec2 b) { return Vec2{a.x - b.x, a.y - b.y}; }
Vec2 mul(Vec2 a, double s) { return Vec2{a.x * s, a.y * s}; }
double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }

bool in_bounds(Vec2 p) {
  return p.x >= 0.0 && p.x <= kBoardSize && p.y >= 0.0 && p.y <= kBoardSize;
}

void resolve_combat(Planet& planet, const std::vector<Fleet>& arrivals) {
  if (arrivals.empty()) {
    return;
  }

  std::map<int, int> by_owner;
  for (const Fleet& fleet : arrivals) {
    by_owner[fleet.owner] += fleet.ships;
  }

  std::vector<std::pair<int, int>> forces;
  forces.reserve(by_owner.size());
  for (const auto& item : by_owner) {
    forces.push_back(item);
  }
  std::sort(forces.begin(), forces.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) {
      return a.second > b.second;
    }
    return a.first < b.first;
  });

  if (forces.size() >= 2 && forces[0].second == forces[1].second) {
    return;
  }

  int survivor_owner = forces[0].first;
  int survivor_ships = forces[0].second;
  if (forces.size() >= 2) {
    survivor_ships -= forces[1].second;
  }
  if (survivor_ships <= 0) {
    return;
  }

  if (survivor_owner == planet.owner) {
    planet.ships += survivor_ships;
  } else if (survivor_ships > planet.ships) {
    planet.owner = survivor_owner;
    planet.ships = survivor_ships - planet.ships;
  } else {
    planet.ships -= survivor_ships;
  }
}

}  // namespace

double dist2(Vec2 a, Vec2 b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  return dx * dx + dy * dy;
}

double dist(Vec2 a, Vec2 b) { return std::sqrt(dist2(a, b)); }

double normalize_angle(double angle) {
  double out = std::fmod(angle, kTwoPi);
  if (out <= -kPi) {
    out += kTwoPi;
  } else if (out > kPi) {
    out -= kTwoPi;
  }
  return out;
}

double fleet_speed(int ships) {
  if (ships <= 1) {
    return 1.0;
  }
  const double ratio = std::log(static_cast<double>(ships)) / std::log(1000.0);
  const double curved = std::pow(std::max(0.0, ratio), 1.5);
  return 1.0 + (kMaxFleetSpeed - 1.0) * std::min(1.0, curved);
}

bool segment_circle_intersects(Vec2 a, Vec2 b, Vec2 center, double radius) {
  const Vec2 ab = sub(b, a);
  const double len2 = dot(ab, ab);
  if (len2 <= kEps) {
    return dist2(a, center) <= radius * radius + kEps;
  }

  const double t = std::clamp(dot(sub(center, a), ab) / len2, 0.0, 1.0);
  const Vec2 closest = add(a, mul(ab, t));
  return dist2(closest, center) <= radius * radius + kEps;
}

bool swept_pair_hit(Vec2 fleet_old, Vec2 fleet_new, Vec2 planet_old, Vec2 planet_new,
                    double radius) {
  const double d0x = fleet_old.x - planet_old.x;
  const double d0y = fleet_old.y - planet_old.y;
  const double dvx = (fleet_new.x - fleet_old.x) - (planet_new.x - planet_old.x);
  const double dvy = (fleet_new.y - fleet_old.y) - (planet_new.y - planet_old.y);
  const double a = dvx * dvx + dvy * dvy;
  const double b = 2.0 * (d0x * dvx + d0y * dvy);
  const double c = d0x * d0x + d0y * d0y - radius * radius;
  if (a < 1e-12) {
    return c <= 0.0;
  }
  const double disc = b * b - 4.0 * a * c;
  if (disc < 0.0) {
    return false;
  }
  const double root = std::sqrt(disc);
  const double t1 = (-b - root) / (2.0 * a);
  const double t2 = (-b + root) / (2.0 * a);
  return t2 >= 0.0 && t1 <= 1.0;
}

bool is_orbiting_planet(const Planet& p) {
  return dist(planet_pos(p), Vec2{kCenterX, kCenterY}) + p.radius < 50.0;
}

Vec2 rotated_position(const Planet& initial, double angular_velocity, int step) {
  if (!is_orbiting_planet(initial)) {
    return planet_pos(initial);
  }

  const double dx = initial.x - kCenterX;
  const double dy = initial.y - kCenterY;
  const double radius = std::sqrt(dx * dx + dy * dy);
  const double theta = std::atan2(dy, dx) + angular_velocity * step;
  return Vec2{kCenterX + radius * std::cos(theta), kCenterY + radius * std::sin(theta)};
}

int observed_orbit_step(int observation_step) {
  return std::clamp(observation_step - 1, 0, kMaxSteps);
}

void FeatureEngine::initialize(const Observation& obs) {
  current_ = obs;
  if (current_.initial_planets.empty()) {
    current_.initial_planets = current_.planets;
  }
  initialized_ = true;
  rebuild_initial_cache();
  refresh_current(obs);
}

void FeatureEngine::rebuild_initial_cache() {
  initial_index_.clear();
  position_cache_.assign(current_.initial_planets.size(), std::vector<Vec2>(kMaxSteps + 1));
  for (std::size_t i = 0; i < current_.initial_planets.size(); ++i) {
    initial_index_[current_.initial_planets[i].id] = i;
    for (int step = 0; step <= kMaxSteps; ++step) {
      position_cache_[i][step] =
          rotated_position(current_.initial_planets[i], current_.angular_velocity, step);
    }
  }
}

bool FeatureEngine::cache_matches(const Observation& obs) const {
  const std::vector<Planet>& next_initial =
      obs.initial_planets.empty() ? obs.planets : obs.initial_planets;
  if (next_initial.size() != current_.initial_planets.size() ||
      std::abs(obs.angular_velocity - current_.angular_velocity) > 1e-12) {
    return false;
  }
  for (std::size_t i = 0; i < next_initial.size(); ++i) {
    const Planet& a = next_initial[i];
    const Planet& b = current_.initial_planets[i];
    if (a.id != b.id || std::abs(a.x - b.x) > 1e-9 ||
        std::abs(a.y - b.y) > 1e-9 || std::abs(a.radius - b.radius) > 1e-9) {
      return false;
    }
  }
  return true;
}

void FeatureEngine::refresh_current(const Observation& obs) {
  current_.player = obs.player;
  current_.step = obs.step;
  current_.angular_velocity = obs.angular_velocity;
  current_.planets = obs.planets;
  current_.fleets = obs.fleets;
  current_.comet_planet_ids = obs.comet_planet_ids;
  current_.comets = obs.comets;
  if (!obs.initial_planets.empty()) {
    current_.initial_planets = obs.initial_planets;
  }

  current_index_.clear();
  for (std::size_t i = 0; i < current_.planets.size(); ++i) {
    current_index_[current_.planets[i].id] = i;
  }

  comet_path_index_.clear();
  for (std::size_t group_i = 0; group_i < current_.comets.size(); ++group_i) {
    const CometGroup& group = current_.comets[group_i];
    const std::size_t limit = std::min(group.planet_ids.size(), group.paths.size());
    for (std::size_t path_i = 0; path_i < limit; ++path_i) {
      comet_path_index_[group.planet_ids[path_i]] =
          std::make_pair(static_cast<int>(group_i), static_cast<int>(path_i));
    }
  }
}

bool FeatureEngine::is_comet_id(int planet_id) const {
  return std::find(current_.comet_planet_ids.begin(), current_.comet_planet_ids.end(),
                   planet_id) != current_.comet_planet_ids.end();
}

bool FeatureEngine::planet_present_at(const Planet& planet, int absolute_step) const {
  auto comet_it = comet_path_index_.find(planet.id);
  if (comet_it == comet_path_index_.end()) {
    return true;
  }

  const CometGroup& group = current_.comets[comet_it->second.first];
  const std::vector<Vec2>& path = group.paths[comet_it->second.second];
  if (path.empty()) {
    return false;
  }

  const int offset = std::max(0, absolute_step - current_.step);
  const int path_index = group.path_index + offset;
  return path_index >= 0 && path_index < static_cast<int>(path.size());
}

Vec2 FeatureEngine::planet_position_at(const Planet& planet, int absolute_step) const {
  auto comet_it = comet_path_index_.find(planet.id);
  if (comet_it != comet_path_index_.end()) {
    const CometGroup& group = current_.comets[comet_it->second.first];
    const std::vector<Vec2>& path = group.paths[comet_it->second.second];
    if (!path.empty()) {
      const int offset = std::max(0, absolute_step - current_.step);
      const int path_index = group.path_index + offset;
      if (path_index >= 0 && path_index < static_cast<int>(path.size())) {
        return path[path_index];
      }
    }
    return Vec2{std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
  }

  auto initial_it = initial_index_.find(planet.id);
  if (initial_it != initial_index_.end() && !is_comet_id(planet.id)) {
    const int orbit_step = observed_orbit_step(absolute_step);
    return position_cache_[initial_it->second][std::clamp(orbit_step, 0, kMaxSteps)];
  }

  return planet_pos(planet);
}

void FeatureEngine::fill_comet_stats(FeatureBatch& batch, int horizon) const {
  batch.comet_spawn_steps.assign(
      kCometSpawnSteps,
      kCometSpawnSteps + sizeof(kCometSpawnSteps) / sizeof(kCometSpawnSteps[0]));

  for (int spawn_step : batch.comet_spawn_steps) {
    if (spawn_step >= current_.step) {
      batch.stats.next_comet_spawn_step = spawn_step;
      batch.stats.turns_until_next_comet_spawn = spawn_step - current_.step;
      break;
    }
  }

  std::set<int> expiring_ids;
  for (const Planet& planet : current_.planets) {
    if (!is_comet_id(planet.id) || !planet_present_at(planet, current_.step)) {
      continue;
    }
    ++batch.stats.active_comets;
    for (int dt = 0; dt < horizon; ++dt) {
      if (!planet_present_at(planet, current_.step + dt)) {
        expiring_ids.insert(planet.id);
        break;
      }
    }
  }
  batch.stats.expiring_comets_within_horizon = static_cast<int>(expiring_ids.size());
}

std::vector<FeatureEngine::Arrival> FeatureEngine::predict_arrivals(
    int horizon, FeatureStats& stats) const {
  std::vector<Arrival> arrivals;
  arrivals.reserve(current_.fleets.size());

  for (const Fleet& fleet : current_.fleets) {
    const double speed = fleet_speed(fleet.ships);
    Vec2 prev{fleet.x, fleet.y};

    for (int dt = 1; dt < horizon; ++dt) {
      const int tick = current_.step + dt;
      const Vec2 next{prev.x + std::cos(fleet.angle) * speed,
                      prev.y + std::sin(fleet.angle) * speed};

      for (const Planet& planet : current_.planets) {
        if (!planet_present_at(planet, tick - 1) && !planet_present_at(planet, tick)) {
          continue;
        }
        const Vec2 old_pos = planet_position_at(planet, tick - 1);
        const Vec2 new_pos = planet_position_at(planet, tick);
        if (!std::isfinite(old_pos.x) || !std::isfinite(new_pos.x)) {
          continue;
        }
        if (swept_pair_hit(prev, next, old_pos, new_pos, planet.radius)) {
          arrivals.push_back(Arrival{planet.id, fleet.owner, fleet.ships, dt});
          ++stats.predicted_arrivals;
          dt = horizon;
          break;
        }
      }

      if (dt >= horizon) {
        break;
      }
      if (!in_bounds(next) ||
          segment_circle_intersects(prev, next, Vec2{kCenterX, kCenterY}, kSunRadius)) {
        break;
      }

      prev = next;
    }
  }

  std::sort(arrivals.begin(), arrivals.end(), [](const Arrival& a, const Arrival& b) {
    if (a.dt != b.dt) {
      return a.dt < b.dt;
    }
    return a.planet_id < b.planet_id;
  });
  return arrivals;
}

FeatureEngine::RouteEval FeatureEngine::estimate_route_without_proxy(
    const Planet& src, const Planet& target, int ships, int max_route_delay) const {
  if (src.id == target.id || ships <= 0) {
    return RouteEval{0, 0.0, false};
  }

  const Vec2 src_pos = planet_position_at(src, current_.step);
  if (!std::isfinite(src_pos.x) || !planet_present_at(src, current_.step)) {
    return RouteEval{};
  }
  const double speed = fleet_speed(ships);
  const bool target_moves =
      (is_orbiting_planet(target) && !is_comet_id(target.id)) ||
      comet_path_index_.find(target.id) != comet_path_index_.end();

  int estimate = -1;
  Vec2 aim_pos = planet_position_at(target, current_.step);
  if (!std::isfinite(aim_pos.x) || !planet_present_at(target, current_.step)) {
    return RouteEval{};
  }
  if (target_moves) {
    for (int dt = 1; dt <= max_route_delay; ++dt) {
      if (!planet_present_at(target, current_.step + dt)) {
        break;
      }
      const Vec2 target_pos = planet_position_at(target, current_.step + dt);
      if (!std::isfinite(target_pos.x)) {
        break;
      }
      const double center_distance = dist(src_pos, target_pos);
      if (speed * static_cast<double>(dt) + target.radius + 0.1 >= center_distance) {
        estimate = dt;
        aim_pos = target_pos;
        break;
      }
    }
  } else {
    const double center_distance = dist(src_pos, aim_pos);
    estimate = std::max(
        1, static_cast<int>(
               std::ceil(std::max(0.0, center_distance - target.radius - 0.1) / speed)));
  }

  if (estimate < 0 || estimate > max_route_delay) {
    return RouteEval{};
  }

  return RouteEval{estimate,
                   normalize_angle(std::atan2(aim_pos.y - src_pos.y, aim_pos.x - src_pos.x)),
                   false};
}

void FeatureEngine::build_delay_matrix_batched(FeatureBatch& batch,
                                               int max_route_delay) const {
  struct VirtualFleet {
    std::size_t offset = 0;
    int target_id = -1;
    double angle = 0.0;
    double speed = 0.0;
    Vec2 launch;
    Vec2 prev;
    bool active = false;
  };

  const std::size_t n = current_.planets.size();
  const std::size_t bucket_count = batch.ship_buckets.size();
  batch.delay_flat.assign(bucket_count * n * n, kBlockedDelay);
  batch.angle_flat.assign(bucket_count * n * n, 0.0);

  std::vector<VirtualFleet> fleets;
  fleets.reserve(bucket_count * n * (n > 0 ? n - 1 : 0));

  for (std::size_t b = 0; b < bucket_count; ++b) {
    for (std::size_t i = 0; i < n; ++i) {
      const Planet& src = current_.planets[i];
      if (!planet_present_at(src, current_.step)) {
        continue;
      }
      const Vec2 src_pos = planet_position_at(src, current_.step);
      if (!std::isfinite(src_pos.x)) {
        continue;
      }
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t offset = (b * n + i) * n + j;
        if (i == j) {
          batch.delay_flat[offset] = 0;
          continue;
        }

        ++batch.stats.route_queries;
        const Planet& target = current_.planets[j];
        const RouteEval estimate =
            estimate_route_without_proxy(src, target, batch.ship_buckets[b], max_route_delay);
        if (estimate.blocked) {
          ++batch.stats.blocked_routes;
          continue;
        }

        const double c = std::cos(estimate.angle);
        const double s = std::sin(estimate.angle);
        const Vec2 launch{src_pos.x + c * (src.radius + 0.1),
                          src_pos.y + s * (src.radius + 0.1)};
        fleets.push_back(VirtualFleet{offset,
                                      target.id,
                                      estimate.angle,
                                      fleet_speed(batch.ship_buckets[b]),
                                      launch,
                                      launch,
                                      true});
        batch.angle_flat[offset] = estimate.angle;
      }
    }
  }

  if (fleets.empty()) {
    return;
  }

  ++batch.stats.route_proxy_simulations;
  std::vector<std::size_t> active;
  active.reserve(fleets.size());
  for (std::size_t idx = 0; idx < fleets.size(); ++idx) {
    active.push_back(idx);
  }

  for (int dt = 1; dt <= max_route_delay && !active.empty(); ++dt) {
    const int tick = current_.step + dt;
    std::vector<std::size_t> still_active;
    still_active.reserve(active.size());

    for (std::size_t fleet_idx : active) {
      VirtualFleet& fleet = fleets[fleet_idx];
      ++batch.stats.route_sim_ticks;

      const double c = std::cos(fleet.angle);
      const double s = std::sin(fleet.angle);
      const Vec2 next{fleet.launch.x + c * fleet.speed * dt,
                      fleet.launch.y + s * fleet.speed * dt};

      bool finished = false;
      for (const Planet& planet : current_.planets) {
        if (!planet_present_at(planet, tick - 1) && !planet_present_at(planet, tick)) {
          continue;
        }
        const Vec2 old_pos = planet_position_at(planet, tick - 1);
        const Vec2 new_pos = planet_position_at(planet, tick);
        if (!std::isfinite(old_pos.x) || !std::isfinite(new_pos.x)) {
          continue;
        }
        if (!swept_pair_hit(fleet.prev, next, old_pos, new_pos, planet.radius)) {
          continue;
        }

        if (planet.id == fleet.target_id) {
          batch.delay_flat[fleet.offset] = dt;
        } else {
          ++batch.stats.blocked_routes;
        }
        finished = true;
        break;
      }

      if (finished) {
        continue;
      }
      if (!in_bounds(next) ||
          segment_circle_intersects(fleet.prev, next, Vec2{kCenterX, kCenterY}, kSunRadius)) {
        ++batch.stats.blocked_routes;
        continue;
      }

      fleet.prev = next;
      still_active.push_back(fleet_idx);
    }

    active.swap(still_active);
  }

  batch.stats.blocked_routes += static_cast<int>(active.size());
}

FeatureBatch FeatureEngine::compute(const Observation& obs, int horizon, int max_route_delay) {
  const auto started = std::chrono::steady_clock::now();
  if (!initialized_ || !cache_matches(obs)) {
    initialize(obs);
  } else {
    refresh_current(obs);
  }

  horizon = std::clamp(horizon, 1, kMaxSteps + 1);
  max_route_delay = std::clamp(max_route_delay, 1, kMaxSteps);

  FeatureBatch batch;
  batch.ship_buckets = {5, 10, 20, 40, 80, 160};
  batch.stats.planets = static_cast<int>(current_.planets.size());
  batch.stats.fleets = static_cast<int>(current_.fleets.size());
  batch.stats.horizon = horizon;
  fill_comet_stats(batch, horizon);

  const std::size_t n = current_.planets.size();
  batch.planet_ids.reserve(n);
  for (const Planet& planet : current_.planets) {
    batch.planet_ids.push_back(planet.id);
  }

  const std::vector<Arrival> arrivals = predict_arrivals(horizon, batch.stats);
  batch.garrison_flat.assign(n * static_cast<std::size_t>(horizon) * 2, 0);
  for (std::size_t i = 0; i < n; ++i) {
    Planet forecast = current_.planets[i];
    for (int dt = 0; dt < horizon; ++dt) {
      if (!planet_present_at(forecast, current_.step + dt)) {
        const std::size_t offset = (i * static_cast<std::size_t>(horizon) + dt) * 2;
        batch.garrison_flat[offset] = 0;
        batch.garrison_flat[offset + 1] = kMissingOwner;
        continue;
      }

      if (dt > 0) {
        if (forecast.owner != kNeutralOwner) {
          forecast.ships += forecast.production;
        }

        std::vector<Fleet> planet_arrivals;
        for (const Arrival& arrival : arrivals) {
          if (arrival.dt != dt || arrival.planet_id != forecast.id) {
            continue;
          }
          planet_arrivals.push_back(
              Fleet{-1, arrival.owner, 0.0, 0.0, 0.0, -1, arrival.ships});
        }
        resolve_combat(forecast, planet_arrivals);
      }

      const std::size_t offset = (i * static_cast<std::size_t>(horizon) + dt) * 2;
      batch.garrison_flat[offset] = forecast.ships;
      batch.garrison_flat[offset + 1] = forecast.owner;
    }
  }

  build_delay_matrix_batched(batch, max_route_delay);

  const auto finished = std::chrono::steady_clock::now();
  batch.stats.elapsed_ms =
      std::chrono::duration<double, std::milli>(finished - started).count();
  return batch;
}

}  // namespace orbit_rl
