#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace orbit {

constexpr double kBoardSize = 100.0;
constexpr double kCenterX = 50.0;
constexpr double kCenterY = 50.0;
constexpr double kSunRadius = 10.0;
constexpr double kMaxFleetSpeed = 6.0;
constexpr int kMaxSteps = 500;
constexpr int kNeutralOwner = -1;

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

struct Planet {
  int id = 0;
  int owner = kNeutralOwner;
  double x = 0.0;
  double y = 0.0;
  double radius = 0.0;
  int ships = 0;
  int production = 0;
};

struct Fleet {
  int id = 0;
  int owner = kNeutralOwner;
  double x = 0.0;
  double y = 0.0;
  double angle = 0.0;
  int from_planet_id = -1;
  int ships = 0;
};

struct Move {
  int from_planet_id = -1;
  double angle = 0.0;
  int ships = 0;
};

struct CometGroup {
  std::vector<int> planet_ids;
  std::vector<std::vector<Vec2>> paths;
  int path_index = 0;
};

struct Observation {
  int player = 0;
  int step = 0;
  int time_budget_ms = 950;
  double angular_velocity = 0.0;
  std::vector<Planet> planets;
  std::vector<Planet> initial_planets;
  std::vector<Fleet> fleets;
  std::vector<int> comet_planet_ids;
  std::vector<CometGroup> comets;
};

struct RouteResult {
  bool reachable = false;
  double angle = 0.0;
  int arrival_tick = -1;
  int travel_time = -1;
  std::string blocked_by = "unreachable";
  double hit_x = std::numeric_limits<double>::quiet_NaN();
  double hit_y = std::numeric_limits<double>::quiet_NaN();
};

struct SimState {
  int step = 0;
  double angular_velocity = 0.0;
  std::vector<Planet> planets;
  std::vector<Planet> initial_planets;
  std::vector<Fleet> fleets;
  std::vector<int> comet_planet_ids;
  std::vector<CometGroup> comets;
  int next_fleet_id = 1;
};

struct SearchStats {
  std::int64_t states_considered = 0;
  std::int64_t route_queries = 0;
  std::int64_t candidates_generated = 0;
  double elapsed_ms = 0.0;
  double states_per_second = 0.0;
  bool timed_out = false;
};

struct SearchResult {
  std::vector<Move> moves;
  SearchStats stats;
};

struct FleetArrival {
  int planet_id = -1;
  int owner = kNeutralOwner;
  int ships = 0;
  int arrival_tick = -1;
};

double dist2(Vec2 a, Vec2 b);
double dist(Vec2 a, Vec2 b);
double normalize_angle(double angle);
double fleet_speed(int ships);
bool in_bounds(Vec2 p);
bool segment_circle_intersects(Vec2 a, Vec2 b, Vec2 center, double radius);
bool swept_pair_hit(Vec2 fleet_old, Vec2 fleet_new, Vec2 planet_old, Vec2 planet_new,
                    double radius);
bool is_orbiting_planet(const Planet& p);
Vec2 rotated_position(const Planet& initial, double angular_velocity, int step);

class Engine {
 public:
  Engine() = default;

  void initialize(const Observation& obs);
  std::vector<Move> act(const Observation& obs);
  std::vector<Move> act_v2(const Observation& obs);
  SearchResult search(const Observation& obs, int budget_ms);
  SearchResult search_v2(const Observation& obs, int budget_ms);
  SearchStats last_search_stats() const { return last_search_stats_; }
  RouteResult query_route(int src_id, int target_id, int ships, int step);
  std::vector<RouteResult> batch_query_routes(
      const std::vector<std::tuple<int, int, int, int>>& requests);
  std::vector<std::vector<Planet>> forecast_planets(const Observation& obs, int horizon);
  SimState simulate_step(const SimState& state,
                         const std::vector<std::vector<Move>>& actions_by_player);

  bool initialized() const { return initialized_; }

 private:
  struct RouteKey {
    int step = 0;
    int src = 0;
    int target = 0;
    int ships = 0;

    bool operator==(const RouteKey& other) const {
      return step == other.step && src == other.src && target == other.target &&
             ships == other.ships;
    }
  };

  struct RouteKeyHash {
    std::size_t operator()(const RouteKey& key) const;
  };

  struct TransferHint {
    int source_id = -1;
    int target_id = -1;
    int ships = 0;
    double score = 0.0;
    int last_step = -1;
  };

  Observation base_;
  bool initialized_ = false;
  std::unordered_map<int, std::size_t> base_planet_index_;
  std::vector<std::vector<Vec2>> planet_position_cache_;
  std::vector<int> cached_planet_ids_;
  std::vector<double> cached_planet_radii_;
  std::unordered_map<RouteKey, RouteResult, RouteKeyHash> route_cache_;
  std::vector<FleetArrival> predicted_fleet_arrivals_;
  std::vector<TransferHint> transfer_hints_;
  std::vector<Move> last_best_moves_;
  int route_warm_until_step_ = 0;
  SearchStats last_search_stats_;

  void rebuild_base_indexes();
  void build_position_cache();
  void refresh_dynamic_observation(const Observation& obs);
  Vec2 cached_planet_position(int planet_id, int step) const;
  const Planet* find_planet(const std::vector<Planet>& planets, int id) const;
  Planet* find_planet_mut(std::vector<Planet>& planets, int id) const;
  bool is_comet_id(int planet_id) const;
  void build_fleet_arrival_forecast();
  FleetArrival predict_fleet_arrival(const Fleet& fleet, int step) const;
  Planet forecast_planet_at(const Planet& planet, int tick) const;
  int ships_needed_to_capture(const Planet& planet, int arrival_tick, int player) const;
  RouteResult validate_route_toward(const Planet& src, const Planet& target, int ships,
                                    int step, Vec2 aim_pos);
  RouteResult route_to_target_at_tick(const Planet& src, const Planet& target, int ships,
                                      int step, int arrival_tick);
};

Observation observation_from_python_like(int player, int step, double angular_velocity,
                                        const std::vector<Planet>& planets,
                                        const std::vector<Planet>& initial_planets,
                                        const std::vector<Fleet>& fleets);

}  // namespace orbit
