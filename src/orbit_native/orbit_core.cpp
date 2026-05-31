#include "orbit_core.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
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

    const int old_step = std::clamp(tick - 1, 0, kMaxSteps);
    const int new_step = std::clamp(tick, 0, kMaxSteps);
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
    const int old_step = std::clamp(step + dt - 1, 0, kMaxSteps);
    const int new_step = std::clamp(step + dt, 0, kMaxSteps);
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

  const Vec2 target_pos = cached_planet_position(target.id, arrival_tick);
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

  if (!is_orbiting_planet(*target) || is_comet_id(target->id)) {
    best = validate_route_toward(*src, *target, ships, step, planet_pos(*target));
  } else {
    for (int arrival = step + 1; arrival <= kMaxSteps; ++arrival) {
      RouteResult candidate = route_to_target_at_tick(*src, *target, ships, step, arrival);
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

std::vector<Move> Engine::act(const Observation& obs) {
  return search(obs, obs.time_budget_ms).moves;
}

SearchResult Engine::search(const Observation& obs, int budget_ms) {
  const auto started = std::chrono::steady_clock::now();
  const int clamped_budget_ms = std::clamp(budget_ms, 1, 1000);
  const auto deadline = started + std::chrono::milliseconds(clamped_budget_ms);

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
      add_ship_option(baseline_need + std::max(1, target->production * 4));
      add_ship_option(std::max(1, spendable / 2));
      add_ship_option(spendable);

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
            Candidate{score, Move{src->id, route.angle, ships}, src->id, target->id});
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

  std::unordered_map<int, int> spent_by_source;
  std::set<int> claimed_targets;
  for (const Candidate& candidate : candidates) {
    if (moves.size() >= 6 || candidate.score <= 0.0) {
      break;
    }
    if (claimed_targets.find(candidate.target_id) != claimed_targets.end()) {
      continue;
    }
    const Planet* src = find_planet(base_.planets, candidate.source_id);
    if (src == nullptr) {
      continue;
    }
    const int reserve = std::max(3, src->production * 3);
    const int already_spent = spent_by_source[candidate.source_id];
    if (already_spent + candidate.move.ships <= src->ships - reserve) {
      moves.push_back(candidate.move);
      spent_by_source[candidate.source_id] += candidate.move.ships;
      claimed_targets.insert(candidate.target_id);
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
      fleet.x = src_pos.x + std::cos(move.angle) * (src->radius + 1e-6);
      fleet.y = src_pos.y + std::sin(move.angle) * (src->radius + 1e-6);
      next.fleets.push_back(fleet);
    }
  }

  for (Planet& planet : next.planets) {
    if (planet.owner != kNeutralOwner) {
      planet.ships += planet.production;
    }
  }

  std::vector<Fleet> surviving_fleets;
  for (Fleet& fleet : next.fleets) {
    const Vec2 before{fleet.x, fleet.y};
    const double speed = fleet_speed(fleet.ships);
    const Vec2 after{fleet.x + std::cos(fleet.angle) * speed,
                     fleet.y + std::sin(fleet.angle) * speed};

    if (!in_bounds(after) ||
        segment_circle_intersects(before, after, Vec2{kCenterX, kCenterY}, kSunRadius)) {
      continue;
    }

    int hit_planet_id = -1;
    for (const Planet& planet : next.planets) {
      if (planet.id == fleet.from_planet_id && same_point(before, planet_pos(planet))) {
        continue;
      }
      if (swept_pair_hit(before, after, planet_pos(planet), planet_pos(planet),
                         planet.radius)) {
        hit_planet_id = planet.id;
        break;
      }
    }

    if (hit_planet_id >= 0) {
      fleet.x = after.x;
      fleet.y = after.y;
      arrivals[hit_planet_id].push_back(fleet);
    } else {
      fleet.x = after.x;
      fleet.y = after.y;
      surviving_fleets.push_back(fleet);
    }
  }
  next.fleets = surviving_fleets;

  const int next_step = state.step + 1;
  for (Planet& planet : next.planets) {
    const Planet* initial = find_planet(next.initial_planets, planet.id);
    if (initial != nullptr && is_orbiting_planet(*initial)) {
      const Vec2 pos = rotated_position(*initial, next.angular_velocity, next_step);
      planet.x = pos.x;
      planet.y = pos.y;
    }
  }

  surviving_fleets.clear();
  for (Fleet& fleet : next.fleets) {
    bool swept = false;
    const Vec2 fleet_pos{fleet.x, fleet.y};
    for (const Planet& planet : next.planets) {
      if (dist2(fleet_pos, planet_pos(planet)) <= planet.radius * planet.radius + kEps) {
        arrivals[planet.id].push_back(fleet);
        swept = true;
        break;
      }
    }
    if (!swept) {
      surviving_fleets.push_back(fleet);
    }
  }
  next.fleets = surviving_fleets;

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
