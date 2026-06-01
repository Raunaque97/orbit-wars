#include "orbit_core.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <set>

namespace orbit {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kEps = 1e-9;

Vec2 planet_pos(const Planet& p) { return Vec2{p.x, p.y}; }

Vec2 add(Vec2 a, Vec2 b) { return Vec2{a.x + b.x, a.y + b.y}; }
Vec2 sub(Vec2 a, Vec2 b) { return Vec2{a.x - b.x, a.y - b.y}; }
Vec2 mul(Vec2 a, double s) { return Vec2{a.x * s, a.y * s}; }

double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }

bool same_point(Vec2 a, Vec2 b) {
  return std::abs(a.x - b.x) < 1e-8 && std::abs(a.y - b.y) < 1e-8;
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
    return;
  }

  if (survivor_ships > planet.ships) {
    planet.owner = survivor_owner;
    planet.ships = survivor_ships - planet.ships;
  } else {
    planet.ships -= survivor_ships;
  }
}

int ship_bucket(int ships) {
  // Exact values are best for parity. Bucket only extreme values to keep cache bounded.
  return std::min(ships, 5000);
}

std::int64_t transfer_key(int source_id, int target_id, int ships) {
  return (static_cast<std::int64_t>(source_id) << 42) ^
         (static_cast<std::int64_t>(target_id) << 21) ^
         static_cast<std::int64_t>(std::max(0, ships));
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

bool in_bounds(Vec2 p) {
  return p.x >= 0.0 && p.x <= kBoardSize && p.y >= 0.0 && p.y <= kBoardSize;
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
  const double orbital_radius = dist(planet_pos(p), Vec2{kCenterX, kCenterY});
  return orbital_radius + p.radius < 50.0;
}

int observed_orbit_step(int observation_step) {
  // Kaggle observations are post-action states. After the first transition the
  // observation step increments, but planets have not advanced from their
  // initial orbit position yet, so observed planet positions are one orbit tick
  // behind the observation step number.
  return std::clamp(observation_step - 1, 0, kMaxSteps);
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

std::size_t Engine::RouteKeyHash::operator()(const RouteKey& key) const {
  std::size_t h = static_cast<std::size_t>(key.step);
  h = h * 1315423911u + static_cast<std::size_t>(key.src + 17);
  h = h * 1315423911u + static_cast<std::size_t>(key.target + 31);
  h = h * 1315423911u + static_cast<std::size_t>(key.ships + 47);
  return h;
}

void Engine::initialize(const Observation& obs) {
  base_ = obs;
  if (base_.initial_planets.empty()) {
    base_.initial_planets = base_.planets;
  }
  initialized_ = true;
  route_cache_.clear();
  transfer_hints_.clear();
  last_best_moves_.clear();
  v2_best_line_.clear();
  route_warm_until_step_ = base_.step;
  rebuild_base_indexes();
  build_position_cache();
}

void Engine::rebuild_base_indexes() {
  base_planet_index_.clear();
  for (std::size_t i = 0; i < base_.initial_planets.size(); ++i) {
    base_planet_index_[base_.initial_planets[i].id] = i;
  }
}

void Engine::build_position_cache() {
  planet_position_cache_.assign(base_.initial_planets.size(),
                                std::vector<Vec2>(kMaxSteps + 1));
  cached_planet_ids_.clear();
  cached_planet_radii_.clear();
  cached_planet_ids_.reserve(base_.initial_planets.size());
  cached_planet_radii_.reserve(base_.initial_planets.size());
  for (std::size_t i = 0; i < base_.initial_planets.size(); ++i) {
    cached_planet_ids_.push_back(base_.initial_planets[i].id);
    cached_planet_radii_.push_back(base_.initial_planets[i].radius);
    for (int step = 0; step <= kMaxSteps; ++step) {
      planet_position_cache_[i][step] =
          rotated_position(base_.initial_planets[i], base_.angular_velocity, step);
    }
  }
}

void Engine::refresh_dynamic_observation(const Observation& obs) {
  bool initial_planets_changed =
      !obs.initial_planets.empty() &&
      (obs.initial_planets.size() != base_.initial_planets.size());
  if (!initial_planets_changed && !obs.initial_planets.empty()) {
    for (std::size_t i = 0; i < obs.initial_planets.size(); ++i) {
      const Planet& next = obs.initial_planets[i];
      const Planet& prev = base_.initial_planets[i];
      if (next.id != prev.id || std::abs(next.x - prev.x) > 1e-9 ||
          std::abs(next.y - prev.y) > 1e-9 ||
          std::abs(next.radius - prev.radius) > 1e-9) {
        initial_planets_changed = true;
        break;
      }
    }
  }
  const bool angular_velocity_changed =
      std::abs(obs.angular_velocity - base_.angular_velocity) > 1e-12;

  base_.player = obs.player;
  base_.step = obs.step;
  base_.angular_velocity = obs.angular_velocity;
  base_.planets = obs.planets;
  base_.fleets = obs.fleets;
  base_.comets = obs.comets;
  base_.comet_planet_ids = obs.comet_planet_ids;

  if (initial_planets_changed || angular_velocity_changed) {
    base_.initial_planets = obs.initial_planets.empty() ? obs.planets : obs.initial_planets;
    route_cache_.clear();
    transfer_hints_.clear();
    last_best_moves_.clear();
    v2_best_line_.clear();
    route_warm_until_step_ = base_.step;
    rebuild_base_indexes();
    build_position_cache();
  }
}

Vec2 Engine::cached_planet_position(int planet_id, int step) const {
  auto it = base_planet_index_.find(planet_id);
  if (it == base_planet_index_.end()) {
    return Vec2{std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
  }
  const int clamped = std::clamp(step, 0, kMaxSteps);
  return planet_position_cache_[it->second][clamped];
}

const Planet* Engine::find_planet(const std::vector<Planet>& planets, int id) const {
  for (const Planet& planet : planets) {
    if (planet.id == id) {
      return &planet;
    }
  }
  return nullptr;
}

Planet* Engine::find_planet_mut(std::vector<Planet>& planets, int id) const {
  for (Planet& planet : planets) {
    if (planet.id == id) {
      return &planet;
    }
  }
  return nullptr;
}

bool Engine::is_comet_id(int planet_id) const {
  return std::find(base_.comet_planet_ids.begin(), base_.comet_planet_ids.end(),
                   planet_id) != base_.comet_planet_ids.end();
}

FleetArrival Engine::predict_fleet_arrival(const Fleet& fleet, int step) const {
  FleetArrival arrival;
  arrival.owner = fleet.owner;
  arrival.ships = fleet.ships;

  const double speed = fleet_speed(fleet.ships);
  Vec2 prev{fleet.x, fleet.y};

  for (int tick = step + 1; tick <= kMaxSteps; ++tick) {
    const Vec2 next{prev.x + std::cos(fleet.angle) * speed,
                    prev.y + std::sin(fleet.angle) * speed};

    const int old_step = observed_orbit_step(tick - 1);
    const int new_step = observed_orbit_step(tick);
    for (std::size_t i = 0; i < planet_position_cache_.size(); ++i) {
      const Vec2 planet_old = planet_position_cache_[i][old_step];
      const Vec2 planet_new = planet_position_cache_[i][new_step];
      if (swept_pair_hit(prev, next, planet_old, planet_new, cached_planet_radii_[i])) {
        arrival.planet_id = cached_planet_ids_[i];
        arrival.arrival_tick = tick;
        return arrival;
      }
    }

    if (!in_bounds(next) ||
        segment_circle_intersects(prev, next, Vec2{kCenterX, kCenterY}, kSunRadius)) {
      return arrival;
    }

    prev = next;
  }

  return arrival;
}

void Engine::build_fleet_arrival_forecast() {
  predicted_fleet_arrivals_.clear();
  predicted_fleet_arrivals_.reserve(base_.fleets.size());
  for (const Fleet& fleet : base_.fleets) {
    FleetArrival arrival = predict_fleet_arrival(fleet, base_.step);
    if (arrival.planet_id >= 0) {
      predicted_fleet_arrivals_.push_back(arrival);
    }
  }

  std::sort(predicted_fleet_arrivals_.begin(), predicted_fleet_arrivals_.end(),
            [](const FleetArrival& a, const FleetArrival& b) {
              if (a.arrival_tick != b.arrival_tick) {
                return a.arrival_tick < b.arrival_tick;
              }
              return a.planet_id < b.planet_id;
            });
}

Planet Engine::forecast_planet_at(const Planet& planet, int tick) const {
  Planet forecast = planet;
  const int final_tick = std::clamp(tick, base_.step, kMaxSteps);

  for (int t = base_.step + 1; t <= final_tick; ++t) {
    if (forecast.owner != kNeutralOwner) {
      forecast.ships += forecast.production;
    }

    std::vector<Fleet> arrivals;
    for (const FleetArrival& arrival : predicted_fleet_arrivals_) {
      if (arrival.arrival_tick != t || arrival.planet_id != planet.id) {
        continue;
      }
      arrivals.push_back(Fleet{-1, arrival.owner, 0.0, 0.0, 0.0, -1, arrival.ships});
    }

    if (!arrivals.empty()) {
      resolve_combat(forecast, arrivals);
    }
  }

  return forecast;
}

int Engine::ships_needed_to_capture(const Planet& planet, int arrival_tick, int player) const {
  Planet forecast = forecast_planet_at(planet, arrival_tick);
  if (forecast.owner == player) {
    return 0;
  }
  return forecast.ships + 1;
}

RouteResult Engine::validate_route_toward(const Planet& src, const Planet& target, int ships,
                                          int step, Vec2 aim_pos) {
  RouteResult result;
  const Vec2 src_pos = planet_pos(src);
  const double angle = std::atan2(aim_pos.y - src_pos.y, aim_pos.x - src_pos.x);
  const Vec2 launch{src_pos.x + std::cos(angle) * (src.radius + 0.1),
                    src_pos.y + std::sin(angle) * (src.radius + 0.1)};
  const double speed = fleet_speed(ships);
  Vec2 prev = launch;

  for (int dt = 1; step + dt <= kMaxSteps; ++dt) {
    const Vec2 next{launch.x + std::cos(angle) * speed * dt,
                    launch.y + std::sin(angle) * speed * dt};

    // Match the official interpreter: planet swept-pair hits are resolved before
    // bounds and sun checks, so a fleet that crosses a planet and the sun in the
    // same tick belongs to the planet combat.
    const int old_step = observed_orbit_step(step + dt - 1);
    const int new_step = observed_orbit_step(step + dt);
    for (std::size_t i = 0; i < planet_position_cache_.size(); ++i) {
      const Vec2 planet_old = planet_position_cache_[i][old_step];
      const Vec2 planet_new = planet_position_cache_[i][new_step];
      if (swept_pair_hit(prev, next, planet_old, planet_new, cached_planet_radii_[i])) {
        if (cached_planet_ids_[i] == target.id) {
          result.reachable = true;
          result.angle = normalize_angle(angle);
          result.arrival_tick = step + dt;
          result.travel_time = dt;
          result.blocked_by = "none";
          result.hit_x = next.x;
          result.hit_y = next.y;
        } else {
          result.blocked_by = "planet:" + std::to_string(cached_planet_ids_[i]);
        }
        return result;
      }
    }

    if (!in_bounds(next)) {
      result.blocked_by = "bounds";
      return result;
    }

    if (segment_circle_intersects(prev, next, Vec2{kCenterX, kCenterY}, kSunRadius)) {
      result.blocked_by = "sun";
      return result;
    }

    prev = next;
  }

  return result;
}

RouteResult Engine::route_to_target_at_tick(const Planet& src, const Planet& target, int ships,
                                            int step, int arrival_tick) {
  const int dt = arrival_tick - step;
  if (dt <= 0) {
    return RouteResult{};
  }

  const Vec2 target_pos =
      cached_planet_position(target.id, observed_orbit_step(arrival_tick));
  if (!std::isfinite(target_pos.x)) {
    return RouteResult{};
  }

  const Vec2 src_pos = planet_pos(src);
  const double distance_to_center = dist(src_pos, target_pos);
  const double speed = fleet_speed(ships);
  const double travel_distance = speed * static_cast<double>(dt);
  if (travel_distance + target.radius + 1e-6 < distance_to_center) {
    return RouteResult{};
  }

  return validate_route_toward(src, target, ships, step, target_pos);
}

RouteResult Engine::query_route(int src_id, int target_id, int ships, int step) {
  if (!initialized_) {
    initialize(base_);
  }

  const RouteKey key{std::clamp(step, 0, kMaxSteps), src_id, target_id, ship_bucket(ships)};
  auto cached = route_cache_.find(key);
  if (cached != route_cache_.end()) {
    return cached->second;
  }

  RouteResult best;
  const Planet* src = find_planet(base_.planets, src_id);
  const Planet* target = find_planet(base_.planets, target_id);
  if (src == nullptr) {
    src = find_planet(base_.initial_planets, src_id);
  }
  if (target == nullptr) {
    target = find_planet(base_.initial_planets, target_id);
  }
  if (src == nullptr || target == nullptr || src_id == target_id || ships <= 0) {
    route_cache_[key] = best;
    return best;
  }

  Planet route_src = *src;
  Planet route_target = *target;
  if (!is_comet_id(src->id)) {
    const Vec2 src_pos = cached_planet_position(src->id, observed_orbit_step(step));
    if (std::isfinite(src_pos.x)) {
      route_src.x = src_pos.x;
      route_src.y = src_pos.y;
    }
  }
  if (!is_comet_id(target->id)) {
    const Vec2 target_pos = cached_planet_position(target->id, observed_orbit_step(step));
    if (std::isfinite(target_pos.x)) {
      route_target.x = target_pos.x;
      route_target.y = target_pos.y;
    }
  }

  if (!is_orbiting_planet(route_target) || is_comet_id(route_target.id)) {
    best = validate_route_toward(route_src, route_target, ships, step, planet_pos(route_target));
  } else {
    for (int arrival = step + 1; arrival <= kMaxSteps; ++arrival) {
      RouteResult candidate =
          route_to_target_at_tick(route_src, route_target, ships, step, arrival);
      if (candidate.reachable) {
        best = candidate;
        break;
      }
      if (best.blocked_by == "unreachable" && candidate.blocked_by != "unreachable") {
        best.blocked_by = candidate.blocked_by;
      }
    }
  }

  route_cache_[key] = best;
  return best;
}

std::vector<RouteResult> Engine::batch_query_routes(
    const std::vector<std::tuple<int, int, int, int>>& requests) {
  std::vector<RouteResult> results;
  results.reserve(requests.size());
  for (const auto& request : requests) {
    results.push_back(query_route(std::get<0>(request), std::get<1>(request),
                                  std::get<2>(request), std::get<3>(request)));
  }
  return results;
}

std::vector<std::vector<Planet>> Engine::forecast_planets(const Observation& obs,
                                                          int horizon) {
  if (!initialized_ || obs.step <= 0 || base_.initial_planets.empty()) {
    initialize(obs);
  } else {
    refresh_dynamic_observation(obs);
  }
  build_fleet_arrival_forecast();

  const int clamped_horizon = std::clamp(horizon, 0, kMaxSteps);
  std::vector<std::vector<Planet>> table(
      base_.planets.size(), std::vector<Planet>(clamped_horizon + 1));
  for (std::size_t i = 0; i < base_.planets.size(); ++i) {
    for (int dt = 0; dt <= clamped_horizon; ++dt) {
      table[i][dt] = forecast_planet_at(base_.planets[i], base_.step + dt);
    }
  }
  return table;
}

std::vector<Move> Engine::act(const Observation& obs) {
  return search(obs, obs.time_budget_ms).moves;
}

std::vector<Move> Engine::act_v2(const Observation& obs) {
  return search_v2(obs, obs.time_budget_ms).moves;
}

SearchResult Engine::search(const Observation& obs, int budget_ms) {
  const auto started = std::chrono::steady_clock::now();
  const int clamped_budget_ms = std::clamp(budget_ms, 1, 1000);
  const int work_budget_ms = std::max(1, clamped_budget_ms - 5);
  const auto deadline = started + std::chrono::milliseconds(work_budget_ms);

  if (!initialized_ || obs.step <= 0 || base_.initial_planets.empty()) {
    initialize(obs);
  } else {
    refresh_dynamic_observation(obs);
  }
  build_fleet_arrival_forecast();

  SearchStats stats;

  std::vector<Move> moves;
  std::vector<const Planet*> mine;
  std::vector<const Planet*> targets;
  for (const Planet& planet : base_.planets) {
    if (planet.owner == obs.player) {
      mine.push_back(&planet);
    } else {
      targets.push_back(&planet);
    }
  }

  struct Candidate {
    double score = -1e100;
    Move move;
    int source_id = -1;
    int target_id = -1;
    int arrival_tick = -1;
    int travel_time = -1;
  };

  std::vector<Candidate> candidates;
  bool timed_out = false;
  for (const Planet* src : mine) {
    const int reserve = std::max(3, src->production * 3);
    const int spendable = src->ships - reserve;
    if (spendable <= 1) {
      continue;
    }

    for (const Planet* target : targets) {
      const int baseline_need = std::max(1, target->ships + 1);

      std::set<int> ship_options;
      const auto add_ship_option = [&](int ships) {
        if (ships > 0 && ships <= spendable) {
          ship_options.insert(ships);
        }
      };
      add_ship_option(baseline_need);
      add_ship_option(baseline_need + std::max(1, target->production * 2));
      add_ship_option(baseline_need + std::max(1, target->production * 4));
      add_ship_option(baseline_need + std::max(1, target->production * 8));
      add_ship_option(std::max(1, spendable / 4));
      add_ship_option(std::max(1, spendable / 3));
      add_ship_option(std::max(1, spendable / 2));
      add_ship_option(std::max(1, (spendable * 2) / 3));
      add_ship_option(std::max(1, (spendable * 3) / 4));
      add_ship_option(spendable);
      for (const TransferHint& hint : transfer_hints_) {
        if (hint.source_id != src->id || hint.target_id != target->id) {
          continue;
        }
        add_ship_option(hint.ships);
        add_ship_option(hint.ships - std::max(1, hint.ships / 10));
        add_ship_option(hint.ships + std::max(1, hint.ships / 10));
      }
      for (const Move& move : last_best_moves_) {
        if (move.from_planet_id != src->id) {
          continue;
        }
        add_ship_option(move.ships);
        add_ship_option(move.ships - std::max(1, move.ships / 8));
        add_ship_option(move.ships + std::max(1, move.ships / 8));
      }

      for (int ships : ship_options) {
        if (std::chrono::steady_clock::now() >= deadline) {
          timed_out = true;
          break;
        }

        ++stats.states_considered;
        ++stats.route_queries;
        RouteResult route = query_route(src->id, target->id, ships, obs.step);
        if (!route.reachable) {
          continue;
        }
        const Planet forecast = forecast_planet_at(*target, route.arrival_tick);
        if (forecast.owner == obs.player) {
          continue;
        }
        const int need = forecast.ships + 1;
        if (ships < need) {
          continue;
        }
        const int remaining_after_arrival = std::max(0, kMaxSteps - route.arrival_tick);
        const double production_value = target->production * remaining_after_arrival;
        const double ownership_bonus = forecast.owner == kNeutralOwner ? 20.0 : 45.0;
        const double overkill_penalty = static_cast<double>(ships - need) * 0.55;
        const double score = production_value + ownership_bonus -
                             static_cast<double>(ships) * 1.25 - overkill_penalty -
                             static_cast<double>(route.travel_time) * 2.0;
        candidates.push_back(
            Candidate{score, Move{src->id, route.angle, ships}, src->id, target->id,
                      route.arrival_tick, route.travel_time});
        ++stats.candidates_generated;
      }
      if (timed_out) {
        break;
      }
    }
    if (timed_out) {
      break;
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

  std::unordered_map<std::int64_t, TransferHint> hint_by_key;
  for (const TransferHint& hint : transfer_hints_) {
    if (obs.step - hint.last_step > 20) {
      continue;
    }
    TransferHint kept = hint;
    kept.score *= 0.92;
    hint_by_key[transfer_key(kept.source_id, kept.target_id, kept.ships)] = kept;
  }
  const std::size_t hint_limit = std::min<std::size_t>(candidates.size(), 512);
  for (std::size_t i = 0; i < hint_limit; ++i) {
    const Candidate& candidate = candidates[i];
    if (candidate.score <= 0.0) {
      break;
    }
    const std::int64_t key =
        transfer_key(candidate.source_id, candidate.target_id, candidate.move.ships);
    auto it = hint_by_key.find(key);
    if (it == hint_by_key.end() || candidate.score >= it->second.score) {
      hint_by_key[key] = TransferHint{candidate.source_id, candidate.target_id,
                                      candidate.move.ships, candidate.score, obs.step};
    }
  }
  transfer_hints_.clear();
  transfer_hints_.reserve(hint_by_key.size());
  for (const auto& item : hint_by_key) {
    transfer_hints_.push_back(item.second);
  }
  std::sort(transfer_hints_.begin(), transfer_hints_.end(),
            [](const TransferHint& a, const TransferHint& b) {
              if (a.score != b.score) {
                return a.score > b.score;
              }
              return a.last_step > b.last_step;
            });
  if (transfer_hints_.size() > 512) {
    transfer_hints_.resize(512);
  }

  auto warm_future_route_cache = [&]() {
    const int first_warm_step = std::max(obs.step + 1, route_warm_until_step_ + 1);
    for (int future_step = first_warm_step; future_step <= kMaxSteps; ++future_step) {
      for (const Planet* src : mine) {
        const int reserve = std::max(3, src->production * 3);
        const int spendable = src->ships - reserve;
        if (spendable <= 1) {
          continue;
        }
        for (const Planet* target : targets) {
          std::set<int> warm_ship_options;
          const auto add_warm_ship_option = [&](int ships) {
            if (ships > 0 && ships <= spendable) {
              warm_ship_options.insert(ships);
            }
          };
          const int baseline_need = std::max(1, target->ships + 1);
          add_warm_ship_option(baseline_need);
          add_warm_ship_option(baseline_need + std::max(1, target->production * 4));
          add_warm_ship_option(std::max(1, spendable / 2));
          add_warm_ship_option(spendable);
          for (const TransferHint& hint : transfer_hints_) {
            if (hint.source_id == src->id && hint.target_id == target->id) {
              add_warm_ship_option(hint.ships);
            }
          }
          for (int ships : warm_ship_options) {
            if (std::chrono::steady_clock::now() >= deadline) {
              timed_out = true;
              return;
            }
            ++stats.states_considered;
            ++stats.route_queries;
            (void)query_route(src->id, target->id, ships, future_step);
          }
        }
      }
      route_warm_until_step_ = future_step;
    }
  };

  struct Plan {
    double score = 0.0;
    std::vector<int> picks;
    std::unordered_map<int, int> spent_by_source;
    std::set<int> claimed_targets;
  };

  auto can_add = [&](const Plan& plan, const Candidate& candidate) {
    if (candidate.score <= 0.0 || plan.picks.size() >= 6 ||
        plan.claimed_targets.find(candidate.target_id) != plan.claimed_targets.end()) {
      return false;
    }
    const Planet* src = find_planet(base_.planets, candidate.source_id);
    if (src == nullptr) {
      return false;
    }
    const int reserve = std::max(3, src->production * 3);
    const int already_spent =
        plan.spent_by_source.count(candidate.source_id) > 0
            ? plan.spent_by_source.at(candidate.source_id)
            : 0;
    return already_spent + candidate.move.ships <= src->ships - reserve;
  };

  auto add_to_plan = [&](const Plan& plan, int candidate_index) {
    Plan next = plan;
    const Candidate& candidate = candidates[candidate_index];
    next.score += candidate.score;
    next.picks.push_back(candidate_index);
    next.spent_by_source[candidate.source_id] += candidate.move.ships;
    next.claimed_targets.insert(candidate.target_id);
    return next;
  };

  Plan best_plan;
  const auto remember_best = [&](const Plan& plan, Plan& best) {
    if (plan.score > best.score ||
        (std::abs(plan.score - best.score) <= 1e-9 &&
         plan.picks.size() > best.picks.size())) {
      best = plan;
    }
  };

  Plan greedy;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    ++stats.states_considered;
    if (can_add(greedy, candidates[i])) {
      greedy = add_to_plan(greedy, static_cast<int>(i));
    }
  }
  remember_best(greedy, best_plan);

  auto run_beam = [&](std::size_t candidate_limit, std::size_t beam_width) {
    std::vector<Plan> beam(1);
    const std::size_t limit = std::min(candidate_limit, candidates.size());
    for (std::size_t i = 0; i < limit; ++i) {
      if ((i & 15u) == 0u && std::chrono::steady_clock::now() >= deadline) {
        timed_out = true;
        return;
      }
      std::vector<Plan> next = beam;
      next.reserve(beam.size() * 2);
      for (const Plan& plan : beam) {
        ++stats.states_considered;
        if (can_add(plan, candidates[i])) {
          Plan expanded = add_to_plan(plan, static_cast<int>(i));
          remember_best(expanded, best_plan);
          next.push_back(std::move(expanded));
        }
      }
      std::sort(next.begin(), next.end(), [](const Plan& a, const Plan& b) {
        if (a.score != b.score) {
          return a.score > b.score;
        }
        return a.picks.size() > b.picks.size();
      });
      if (next.size() > beam_width) {
        next.resize(beam_width);
      }
      beam = std::move(next);
    }
  };

  if (!candidates.empty()) {
    const std::array<std::size_t, 4> limits{64, 128, 256, 512};
    const std::array<std::size_t, 4> widths{32, 64, 128, 256};
    for (std::size_t pass = 0; pass < limits.size(); ++pass) {
      if (std::chrono::steady_clock::now() >= deadline) {
        timed_out = true;
        break;
      }
      run_beam(limits[pass], widths[pass]);
    }

    std::mt19937 rng(static_cast<std::uint32_t>(
        1469598103u ^ static_cast<std::uint32_t>(obs.step * 16777619u) ^
        static_cast<std::uint32_t>(obs.player * 2166136261u)));
    const std::size_t positive_count = static_cast<std::size_t>(
        std::distance(candidates.begin(),
                      std::find_if(candidates.begin(), candidates.end(),
                                   [](const Candidate& candidate) {
                                     return candidate.score <= 0.0;
                                   })));
    const std::size_t rollout_pool =
        std::min<std::size_t>(positive_count == 0 ? candidates.size() : positive_count, 768);
    if (rollout_pool <= 4) {
      warm_future_route_cache();
    } else {
      std::int64_t rollout = 0;
      while (rollout_pool > 0 && std::chrono::steady_clock::now() < deadline) {
        Plan plan;
        const int tries = 16 + static_cast<int>(rollout % 64);
        for (int attempt = 0; attempt < tries && plan.picks.size() < 6; ++attempt) {
          ++stats.states_considered;
          std::size_t index = 0;
          if ((attempt & 3) == 0) {
            const std::size_t elite = std::max<std::size_t>(1, rollout_pool / 8);
            index = static_cast<std::size_t>(rng()) % elite;
          } else {
            index = static_cast<std::size_t>(rng()) % rollout_pool;
          }
          if (can_add(plan, candidates[index])) {
            plan = add_to_plan(plan, static_cast<int>(index));
          }
        }
        remember_best(plan, best_plan);
        ++rollout;
        if ((rollout & 255) == 0 && std::chrono::steady_clock::now() >= deadline) {
          timed_out = true;
          break;
        }
      }
    }
    timed_out = timed_out || std::chrono::steady_clock::now() >= deadline;
  }

  for (int index : best_plan.picks) {
    moves.push_back(candidates[index].move);
  }
  last_best_moves_ = moves;

  const auto finished = std::chrono::steady_clock::now();
  stats.elapsed_ms =
      std::chrono::duration<double, std::milli>(finished - started).count();
  stats.states_per_second =
      stats.elapsed_ms > 0.0
          ? static_cast<double>(stats.states_considered) * 1000.0 / stats.elapsed_ms
          : 0.0;
  stats.timed_out = timed_out;
  last_search_stats_ = stats;

  return SearchResult{moves, stats};
}

SearchResult Engine::search_v2(const Observation& obs, int budget_ms) {
  const auto started = std::chrono::steady_clock::now();
  const int clamped_budget_ms = std::clamp(budget_ms, 1, 1000);
  const int work_budget_ms = std::max(1, clamped_budget_ms - 5);
  const auto deadline = started + std::chrono::milliseconds(work_budget_ms);

  if (!initialized_ || obs.step <= 0 || base_.initial_planets.empty()) {
    initialize(obs);
  } else {
    refresh_dynamic_observation(obs);
  }
  build_fleet_arrival_forecast();

  SearchStats stats;
  bool timed_out = false;
  constexpr int kForecastHorizon = 50;
  constexpr int kMaxCommitmentDepth = 16;
  constexpr std::size_t kBranchLimit = 24;
  const int end_tick = obs.step + kForecastHorizon;

  std::unordered_map<int, std::size_t> planet_index;
  for (std::size_t i = 0; i < base_.planets.size(); ++i) {
    planet_index[base_.planets[i].id] = i;
  }

  struct SearchState {
    int tick = 0;
    std::vector<Planet> planets;
    std::map<int, std::vector<Fleet>> planned_arrivals;
    std::vector<PlannedTransfer> line;
    int total_ships_sent = 0;
  };

  struct Candidate {
    PlannedTransfer transfer;
    double roi_time = 1e100;
    double production_delta = 0.0;
  };

  const auto planet_index_in = [](const std::vector<Planet>& planets, int id) {
    for (std::size_t i = 0; i < planets.size(); ++i) {
      if (planets[i].id == id) {
        return static_cast<int>(i);
      }
    }
    return -1;
  };

  const auto risk_weight = [&](int tick) {
    if (v2_risk_start_tick_ >= kMaxSteps || v2_risk_end_value_ <= 0.0 ||
        tick <= v2_risk_start_tick_) {
      return 0.0;
    }
    if (tick >= v2_risk_end_tick_) {
      return v2_risk_end_value_;
    }
    const int ramp_ticks = std::max(1, v2_risk_end_tick_ - v2_risk_start_tick_);
    const double fraction =
        static_cast<double>(tick - v2_risk_start_tick_) /
        static_cast<double>(ramp_ticks);
    return v2_risk_end_value_ * std::clamp(fraction, 0.0, 1.0);
  };

  std::map<std::tuple<int, int, int, int>, double> mdb_cache;
  const auto compute_mdb = [&](const std::vector<Planet>& planets, const Planet& mine,
                               int tick) {
    if (risk_weight(tick) <= 0.0) {
      return 0.0;
    }
    double best = 0.0;
    for (const Planet& enemy : planets) {
      if (enemy.owner == kNeutralOwner || enemy.owner == obs.player || enemy.ships <= 0 ||
          enemy.id == mine.id || is_comet_id(enemy.id)) {
        continue;
      }
      const int ships = std::max(1, enemy.ships);
      const auto key =
          std::make_tuple(tick, enemy.id, mine.id, ship_bucket(ships));
      auto cached = mdb_cache.find(key);
      double threat = 0.0;
      if (cached != mdb_cache.end()) {
        threat = cached->second;
      } else {
        ++stats.route_queries;
        RouteResult route = query_route(enemy.id, mine.id, ships, tick);
        if (route.reachable && route.arrival_tick <= end_tick) {
          threat = static_cast<double>(ships) -
                   static_cast<double>(mine.production * route.travel_time);
        }
        threat = std::max(0.0, threat);
        mdb_cache[key] = threat;
      }
      best = std::max(best, threat);
    }
    return best;
  };

  const auto planet_value = [&](const std::vector<Planet>& planets, const Planet& planet,
                                int tick) {
    if (planet.owner == obs.player) {
      const double risk = risk_weight(tick);
      if (risk <= 0.0) {
        return static_cast<double>(planet.production);
      }
      const double mdb = compute_mdb(planets, planet, tick);
      if (mdb <= 0.0) {
        return static_cast<double>(planet.production);
      }
      const double multiplier =
          1.0 - risk * std::exp(-static_cast<double>(std::max(0, planet.ships)) / mdb);
      return static_cast<double>(planet.production) * std::clamp(multiplier, 0.0, 1.0);
    }
    if (planet.owner == kNeutralOwner) {
      return 0.0;
    }
    return -static_cast<double>(planet.production);
  };

  const auto advance_to = [&](SearchState& state, int target_tick) {
    const int clamped_target = std::clamp(target_tick, state.tick, end_tick);
    for (int tick = state.tick + 1; tick <= clamped_target; ++tick) {
      for (Planet& planet : state.planets) {
        if (planet.owner != kNeutralOwner) {
          planet.ships += planet.production;
        }
      }

      std::map<int, std::vector<Fleet>> arrivals_by_planet;
      for (const FleetArrival& arrival : predicted_fleet_arrivals_) {
        if (arrival.arrival_tick == tick) {
          arrivals_by_planet[arrival.planet_id].push_back(
              Fleet{-1, arrival.owner, 0.0, 0.0, 0.0, -1, arrival.ships});
        }
      }
      auto planned_it = state.planned_arrivals.find(tick);
      if (planned_it != state.planned_arrivals.end()) {
        for (const Fleet& fleet : planned_it->second) {
          arrivals_by_planet[fleet.from_planet_id].push_back(fleet);
        }
      }
      for (auto& item : arrivals_by_planet) {
        const int idx = planet_index_in(state.planets, item.first);
        if (idx >= 0) {
          resolve_combat(state.planets[static_cast<std::size_t>(idx)], item.second);
        }
      }
      state.tick = tick;
    }
  };

  const auto apply_transfer = [&](SearchState& state, PlannedTransfer transfer,
                                  bool recompute_route) {
    if (transfer.launch_tick < state.tick || transfer.launch_tick > end_tick) {
      return false;
    }
    advance_to(state, transfer.launch_tick);
    const int src_idx = planet_index_in(state.planets, transfer.source_id);
    const int target_idx = planet_index_in(state.planets, transfer.target_id);
    if (src_idx < 0 || target_idx < 0) {
      return false;
    }
    Planet& src = state.planets[static_cast<std::size_t>(src_idx)];
    const Planet& target = state.planets[static_cast<std::size_t>(target_idx)];
    if (src.owner != obs.player || transfer.ships <= 0 || src.ships < transfer.ships) {
      return false;
    }
    if (!transfer.reinforcement && target.owner == obs.player) {
      return false;
    }
    if (recompute_route) {
      ++stats.route_queries;
      RouteResult route = query_route(src.id, target.id, transfer.ships,
                                      transfer.launch_tick);
      if (!route.reachable || route.arrival_tick > end_tick) {
        return false;
      }
      transfer.angle = route.angle;
      transfer.arrival_tick = route.arrival_tick;
    }
    src.ships -= transfer.ships;
    Fleet fleet;
    fleet.id = -1;
    fleet.owner = obs.player;
    fleet.angle = transfer.angle;
    fleet.from_planet_id = transfer.target_id;
    fleet.ships = transfer.ships;
    state.planned_arrivals[transfer.arrival_tick].push_back(fleet);
    state.line.push_back(transfer);
    state.total_ships_sent += transfer.ships;
    return true;
  };

  const auto apply_launches_at = [&](SearchState& state,
                                     const std::vector<PlannedTransfer>& line,
                                     std::size_t& next_index, int tick,
                                     bool recompute_route) {
    while (next_index < line.size() && line[next_index].launch_tick == tick) {
      if (!apply_transfer(state, line[next_index], recompute_route)) {
        return false;
      }
      ++next_index;
    }
    return true;
  };

  const auto transfer_delivers_control = [&](const SearchState& after_launch,
                                             const PlannedTransfer& transfer) {
    SearchState landed = after_launch;
    advance_to(landed, transfer.arrival_tick);
    const int target_idx = planet_index_in(landed.planets, transfer.target_id);
    if (target_idx < 0) {
      return false;
    }
    const Planet& landed_target =
        landed.planets[static_cast<std::size_t>(target_idx)];
    return landed_target.owner == obs.player;
  };

  const auto score_line = [&](std::vector<PlannedTransfer> line) {
    std::sort(line.begin(), line.end(), [](const PlannedTransfer& a,
                                           const PlannedTransfer& b) {
      if (a.launch_tick != b.launch_tick) {
        return a.launch_tick < b.launch_tick;
      }
      if (a.arrival_tick != b.arrival_tick) {
        return a.arrival_tick < b.arrival_tick;
      }
      return a.source_id < b.source_id;
    });

    SearchState state;
    state.tick = obs.step;
    state.planets = base_.planets;
    std::size_t next_transfer = 0;
    if (!apply_launches_at(state, line, next_transfer, obs.step, false)) {
      return -1e100;
    }

    double value = 0.0;
    for (int tick = obs.step + 1; tick <= end_tick; ++tick) {
      advance_to(state, tick);
      for (const Planet& planet : state.planets) {
        value += planet_value(state.planets, planet, tick);
      }
      if (!apply_launches_at(state, line, next_transfer, tick, false)) {
        return -1e100;
      }
    }
    return value - static_cast<double>(state.total_ships_sent);
  };

  const double baseline_score = score_line({});

  const auto sanitize_cached_line = [&]() {
    std::vector<PlannedTransfer> sorted = v2_best_line_;
    std::sort(sorted.begin(), sorted.end(), [](const PlannedTransfer& a,
                                               const PlannedTransfer& b) {
      if (a.launch_tick != b.launch_tick) {
        return a.launch_tick < b.launch_tick;
      }
      return a.arrival_tick < b.arrival_tick;
    });

    SearchState state;
    state.tick = obs.step;
    state.planets = base_.planets;
    std::vector<PlannedTransfer> kept;
    for (PlannedTransfer transfer : sorted) {
      if (transfer.launch_tick < obs.step || transfer.launch_tick > end_tick) {
        continue;
      }
      SearchState trial = state;
      if (!apply_transfer(trial, transfer, true)) {
        continue;
      }
      if (!transfer_delivers_control(trial, trial.line.back())) {
        continue;
      }
      state = std::move(trial);
      kept.push_back(state.line.back());
    }
    return kept;
  };

  std::vector<PlannedTransfer> best_line = sanitize_cached_line();
  double best_score = score_line(best_line);
  if (best_score <= -1e90) {
    best_line.clear();
    best_score = 0.0;
  } else {
    best_score -= baseline_score;
  }
  int best_ship_count = 0;
  for (const PlannedTransfer& transfer : best_line) {
    best_ship_count += transfer.ships;
  }

  const auto remember_best = [&](const std::vector<PlannedTransfer>& line,
                                 double score) {
    int ships = 0;
    for (const PlannedTransfer& transfer : line) {
      ships += transfer.ships;
    }
    if (score > best_score + 1e-9 ||
        (std::abs(score - best_score) <= 1e-9 &&
         (line.size() > best_line.size() ||
          (line.size() == best_line.size() &&
           (best_ship_count == 0 || ships < best_ship_count))))) {
      best_line = line;
      best_score = score;
      best_ship_count = ships;
    }
  };

  const auto reserved_for_capture = [](const SearchState& state, int target_id) {
    for (const PlannedTransfer& transfer : state.line) {
      if (!transfer.reinforcement && transfer.target_id == target_id &&
          transfer.arrival_tick > state.tick) {
        return true;
      }
    }
    return false;
  };

  const auto earliest_launch_tick = [&](const Planet& src, int ships, int tick) {
    if (ships <= src.ships) {
      return tick;
    }
    if (src.owner != obs.player || src.production <= 0) {
      return end_tick + 1;
    }
    const int deficit = ships - src.ships;
    return tick + (deficit + src.production - 1) / src.production;
  };

  const auto add_candidate = [&](std::vector<Candidate>& candidates,
                                 const SearchState& state, const Planet& src,
                                 const Planet& target, int ships,
                                 bool reinforcement, double production_delta) {
    if (ships <= 0 || src.id == target.id) {
      return;
    }
    const int launch_tick = earliest_launch_tick(src, ships, state.tick);
    if (launch_tick > end_tick) {
      return;
    }
    ++stats.states_considered;
    ++stats.route_queries;
    RouteResult route = query_route(src.id, target.id, ships, launch_tick);
    if (!route.reachable || route.arrival_tick > end_tick) {
      return;
    }
    const double wait_time = static_cast<double>(launch_tick - state.tick);
    const double roi_time = wait_time + static_cast<double>(route.travel_time) +
                            static_cast<double>(ships) /
                                std::max(0.25, production_delta);
    candidates.push_back(Candidate{
        PlannedTransfer{launch_tick, src.id, target.id, ships, route.angle,
                        route.arrival_tick, reinforcement},
        roi_time,
        production_delta});
    ++stats.candidates_generated;
  };

  struct LossForecast {
    bool lost = false;
    int tick = -1;
    int owner = kNeutralOwner;
    int ships = 0;
  };

  const auto first_visible_loss = [&](const SearchState& state, int target_id) {
    SearchState forecast = state;
    for (int tick = state.tick + 1; tick <= end_tick; ++tick) {
      advance_to(forecast, tick);
      const int target_idx = planet_index_in(forecast.planets, target_id);
      if (target_idx < 0) {
        break;
      }
      const Planet& future_target =
          forecast.planets[static_cast<std::size_t>(target_idx)];
      if (future_target.owner != obs.player) {
        return LossForecast{true, tick, future_target.owner, future_target.ships};
      }
    }
    return LossForecast{};
  };

  const auto generate_candidates = [&](const SearchState& state) {
    std::vector<Candidate> candidates;
    std::set<std::tuple<int, int, int, bool>> seen;

    for (const Planet& src : state.planets) {
      if (std::chrono::steady_clock::now() >= deadline) {
        timed_out = true;
        return candidates;
      }
      if (src.owner != obs.player || src.production < 0 || is_comet_id(src.id)) {
        continue;
      }
      for (const Planet& target : state.planets) {
        if (std::chrono::steady_clock::now() >= deadline) {
          timed_out = true;
          return candidates;
        }
        if (src.id == target.id || is_comet_id(target.id)) {
          continue;
        }

        if (target.owner != obs.player) {
          if (reserved_for_capture(state, target.id)) {
            continue;
          }
          int exact_ships = std::max(1, target.ships + 1);
          int capture_ships = -1;
          for (int iter = 0; iter < 5; ++iter) {
            if (std::chrono::steady_clock::now() >= deadline) {
              timed_out = true;
              return candidates;
            }
            const int launch_tick = earliest_launch_tick(src, exact_ships, state.tick);
            if (launch_tick > end_tick) {
              break;
            }
            ++stats.states_considered;
            ++stats.route_queries;
            RouteResult route = query_route(src.id, target.id, exact_ships, launch_tick);
            if (!route.reachable || route.arrival_tick > end_tick) {
              break;
            }
            SearchState forecast = state;
            advance_to(forecast, route.arrival_tick);
            const int target_idx = planet_index_in(forecast.planets, target.id);
            if (target_idx < 0) {
              break;
            }
            const Planet& future_target =
                forecast.planets[static_cast<std::size_t>(target_idx)];
            if (future_target.owner == obs.player) {
              break;
            }
            const int needed = future_target.ships + 1;
            if (needed == exact_ships) {
              const double production_delta =
                  future_target.owner == kNeutralOwner
                      ? static_cast<double>(target.production)
                      : static_cast<double>(target.production) * 2.0;
              const auto key =
                  std::make_tuple(src.id, target.id, exact_ships, false);
              if (seen.insert(key).second) {
                add_candidate(candidates, state, src, target, exact_ships, false,
                              production_delta);
              }
              capture_ships = exact_ships;
              break;
            }
            exact_ships = needed;
          }

          if (capture_ships > 0 && src.ships > capture_ships) {
            const double production_delta =
                target.owner == kNeutralOwner
                    ? static_cast<double>(target.production)
                    : static_cast<double>(target.production) * 2.0;
            const auto key = std::make_tuple(src.id, target.id, src.ships, false);
            if (seen.insert(key).second) {
              add_candidate(candidates, state, src, target, src.ships, false,
                            production_delta);
            }
          }
          continue;
        }

        const LossForecast visible_loss = first_visible_loss(state, target.id);
        if (state.tick >= 50 && visible_loss.lost && visible_loss.tick - state.tick <= 20) {
          int defense_ships = std::max(1, visible_loss.ships + 1);
          for (int iter = 0; iter < 5; ++iter) {
            if (std::chrono::steady_clock::now() >= deadline) {
              timed_out = true;
              return candidates;
            }
            const int launch_tick = earliest_launch_tick(src, defense_ships, state.tick);
            if (launch_tick > end_tick) {
              break;
            }
            ++stats.states_considered;
            ++stats.route_queries;
            RouteResult route = query_route(src.id, target.id, defense_ships, launch_tick);
            if (!route.reachable || route.arrival_tick > end_tick) {
              break;
            }
            SearchState forecast = state;
            advance_to(forecast, route.arrival_tick);
            const int target_idx = planet_index_in(forecast.planets, target.id);
            if (target_idx < 0) {
              break;
            }
            const Planet& future_target =
                forecast.planets[static_cast<std::size_t>(target_idx)];
            int needed = defense_ships;
            if (future_target.owner != obs.player) {
              needed = future_target.ships + 1;
            }
            if (needed == defense_ships) {
              const double production_delta =
                  std::max(1.0, static_cast<double>(target.production) * 2.0);
              const auto key = std::make_tuple(src.id, target.id, defense_ships, true);
              if (seen.insert(key).second) {
                add_candidate(candidates, state, src, target, defense_ships, true,
                              production_delta);
              }
              break;
            }
            defense_ships = needed;
          }

        }

        if (risk_weight(state.tick) <= 0.0 || src.ships <= 0) {
          continue;
        }
        const double mdb = compute_mdb(state.planets, target, state.tick);
        if (mdb <= 0.0 || static_cast<double>(target.ships) >= mdb) {
          continue;
        }
        const int wanted = std::max(1, static_cast<int>(std::ceil(mdb - target.ships)));
        const int ships = std::min(src.ships, wanted);
        Planet reinforced = target;
        reinforced.ships += ships;
        const double gain =
            planet_value(state.planets, reinforced, state.tick) -
            planet_value(state.planets, target, state.tick);
        if (gain <= 1e-6) {
          continue;
        }
        const auto key = std::make_tuple(src.id, target.id, ships, true);
        if (seen.insert(key).second) {
          add_candidate(candidates, state, src, target, ships, true, gain);
        }
      }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                if (a.roi_time != b.roi_time) {
                  return a.roi_time < b.roi_time;
                }
                if (a.transfer.launch_tick != b.transfer.launch_tick) {
                  return a.transfer.launch_tick < b.transfer.launch_tick;
                }
                return a.transfer.ships < b.transfer.ships;
              });
    if (candidates.size() > kBranchLimit) {
      candidates.resize(kBranchLimit);
    }
    return candidates;
  };

  const auto maybe_remember_state = [&](const SearchState& state) {
    const double score = score_line(state.line) - baseline_score;
    remember_best(state.line, score);
  };

  const auto next_arrival_tick = [&](const SearchState& state) {
    int next_arrival = end_tick + 1;
    for (const auto& item : state.planned_arrivals) {
      if (item.first > state.tick) {
        next_arrival = std::min(next_arrival, item.first);
      }
    }
    return next_arrival;
  };

  const auto state_score = [&](const SearchState& state) {
    return score_line(state.line) - baseline_score;
  };

  struct BeamItem {
    SearchState state;
    double score = 0.0;
  };

  constexpr std::size_t kBeamWidth = 160;
  std::vector<BeamItem> frontier;
  SearchState root;
  root.tick = obs.step;
  root.planets = base_.planets;
  frontier.push_back(BeamItem{root, 0.0});

  if (!best_line.empty()) {
    SearchState cached;
    cached.tick = obs.step;
    cached.planets = base_.planets;
    bool ok = true;
    for (const PlannedTransfer& transfer : best_line) {
      if (!apply_transfer(cached, transfer, true)) {
        ok = false;
        break;
      }
    }
    if (ok) {
      frontier.push_back(BeamItem{cached, state_score(cached)});
    }
  }

  for (int depth = 0; depth < kMaxCommitmentDepth && !frontier.empty(); ++depth) {
    std::vector<BeamItem> next_frontier;
    for (const BeamItem& item : frontier) {
      if (std::chrono::steady_clock::now() >= deadline) {
        timed_out = true;
        break;
      }
      ++stats.states_considered;
      maybe_remember_state(item.state);
      if (item.state.tick >= end_tick) {
        continue;
      }

      std::vector<Candidate> candidates = generate_candidates(item.state);
      for (const Candidate& candidate : candidates) {
        if (std::chrono::steady_clock::now() >= deadline) {
          timed_out = true;
          break;
        }
        SearchState next = item.state;
        if (!apply_transfer(next, candidate.transfer, false)) {
          continue;
        }
        const double next_score = state_score(next);
        maybe_remember_state(next);
        next_frontier.push_back(BeamItem{next, next_score});

        const int arrival_tick = next_arrival_tick(next);
        if (arrival_tick <= end_tick && arrival_tick > next.tick) {
          SearchState waited = next;
          advance_to(waited, arrival_tick);
          const double waited_score = state_score(waited);
          maybe_remember_state(waited);
          next_frontier.push_back(BeamItem{std::move(waited), waited_score});
        }
      }
      if (timed_out) {
        break;
      }
    }

    std::sort(next_frontier.begin(), next_frontier.end(),
              [](const BeamItem& a, const BeamItem& b) {
                if (a.score != b.score) {
                  return a.score > b.score;
                }
                if (a.state.line.size() != b.state.line.size()) {
                  return a.state.line.size() > b.state.line.size();
                }
                return a.state.total_ships_sent < b.state.total_ships_sent;
              });
    if (next_frontier.size() > kBeamWidth) {
      next_frontier.resize(kBeamWidth);
    }
    frontier = std::move(next_frontier);
    if (timed_out) {
      break;
    }
  }

  std::sort(best_line.begin(), best_line.end(), [](const PlannedTransfer& a,
                                                   const PlannedTransfer& b) {
    if (a.launch_tick != b.launch_tick) {
      return a.launch_tick < b.launch_tick;
    }
    return a.arrival_tick < b.arrival_tick;
  });
  v2_best_line_.clear();
  for (const PlannedTransfer& transfer : best_line) {
    if (transfer.launch_tick >= obs.step && transfer.launch_tick <= end_tick) {
      v2_best_line_.push_back(transfer);
    }
  }

  std::vector<Move> moves;
  for (const PlannedTransfer& transfer : v2_best_line_) {
    if (transfer.launch_tick == obs.step) {
      RouteResult route =
          query_route(transfer.source_id, transfer.target_id, transfer.ships, obs.step);
      if (route.reachable && route.arrival_tick <= end_tick) {
        moves.push_back(Move{transfer.source_id, route.angle, transfer.ships});
      }
    }
  }

  const auto finished = std::chrono::steady_clock::now();
  stats.elapsed_ms =
      std::chrono::duration<double, std::milli>(finished - started).count();
  stats.states_per_second =
      stats.elapsed_ms > 0.0
          ? static_cast<double>(stats.states_considered) * 1000.0 / stats.elapsed_ms
          : 0.0;
  stats.timed_out = timed_out;
  last_search_stats_ = stats;

  return SearchResult{moves, stats};
}

SimState Engine::simulate_step(const SimState& state,
                               const std::vector<std::vector<Move>>& actions_by_player) {
  SimState next = state;
  std::map<int, std::vector<Fleet>> arrivals;

  for (std::size_t owner = 0; owner < actions_by_player.size(); ++owner) {
    for (const Move& move : actions_by_player[owner]) {
      Planet* src = find_planet_mut(next.planets, move.from_planet_id);
      if (src == nullptr || src->owner != static_cast<int>(owner) || move.ships <= 0 ||
          src->ships < move.ships) {
        continue;
      }
      src->ships -= move.ships;
      const Vec2 src_pos = planet_pos(*src);
      Fleet fleet;
      fleet.id = next.next_fleet_id++;
      fleet.owner = static_cast<int>(owner);
      fleet.angle = move.angle;
      fleet.from_planet_id = src->id;
      fleet.ships = move.ships;
      fleet.x = src_pos.x + std::cos(move.angle) * (src->radius + 0.1);
      fleet.y = src_pos.y + std::sin(move.angle) * (src->radius + 0.1);
      next.fleets.push_back(fleet);
    }
  }

  for (Planet& planet : next.planets) {
    if (planet.owner != kNeutralOwner) {
      planet.ships += planet.production;
    }
  }

  struct PlanetPath {
    int id = -1;
    Vec2 old_pos;
    Vec2 new_pos;
    double radius = 0.0;
    bool check_collision = true;
  };

  std::vector<PlanetPath> planet_paths;
  planet_paths.reserve(next.planets.size());
  const int next_step = state.step + 1;
  for (const Planet& planet : next.planets) {
    Vec2 old_pos = planet_pos(planet);
    Vec2 new_pos = old_pos;
    const Planet* initial = find_planet(next.initial_planets, planet.id);
    if (initial != nullptr && is_orbiting_planet(*initial)) {
      new_pos =
          rotated_position(*initial, next.angular_velocity, observed_orbit_step(next_step));
    }
    planet_paths.push_back(PlanetPath{planet.id, old_pos, new_pos, planet.radius, true});
  }

  std::vector<Fleet> surviving_fleets;
  for (Fleet& fleet : next.fleets) {
    const Vec2 before{fleet.x, fleet.y};
    const double speed = fleet_speed(fleet.ships);
    const Vec2 after{fleet.x + std::cos(fleet.angle) * speed,
                     fleet.y + std::sin(fleet.angle) * speed};

    int hit_planet_id = -1;
    for (const PlanetPath& path : planet_paths) {
      if (!path.check_collision) {
        continue;
      }
      if (swept_pair_hit(before, after, path.old_pos, path.new_pos, path.radius)) {
        hit_planet_id = path.id;
        break;
      }
    }

    if (hit_planet_id >= 0) {
      fleet.x = after.x;
      fleet.y = after.y;
      arrivals[hit_planet_id].push_back(fleet);
      continue;
    }

    if (!in_bounds(after) ||
        segment_circle_intersects(before, after, Vec2{kCenterX, kCenterY}, kSunRadius)) {
      continue;
    }

    fleet.x = after.x;
    fleet.y = after.y;
    surviving_fleets.push_back(fleet);
  }
  next.fleets = surviving_fleets;

  for (Planet& planet : next.planets) {
    for (const PlanetPath& path : planet_paths) {
      if (path.id == planet.id) {
        planet.x = path.new_pos.x;
        planet.y = path.new_pos.y;
        break;
      }
    }
  }

  for (auto& item : arrivals) {
    Planet* planet = find_planet_mut(next.planets, item.first);
    if (planet != nullptr) {
      resolve_combat(*planet, item.second);
    }
  }

  next.step = next_step;
  return next;
}

Observation observation_from_python_like(int player, int step, double angular_velocity,
                                        const std::vector<Planet>& planets,
                                        const std::vector<Planet>& initial_planets,
                                        const std::vector<Fleet>& fleets) {
  Observation obs;
  obs.player = player;
  obs.step = step;
  obs.angular_velocity = angular_velocity;
  obs.planets = planets;
  obs.initial_planets = initial_planets.empty() ? planets : initial_planets;
  obs.fleets = fleets;
  return obs;
}

}  // namespace orbit
