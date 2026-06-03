#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace orbit_rl {

constexpr double kBoardSize = 100.0;
constexpr double kCenterX = 50.0;
constexpr double kCenterY = 50.0;
constexpr double kSunRadius = 10.0;
constexpr double kMaxFleetSpeed = 6.0;
constexpr int kMaxSteps = 500;
constexpr int kMaxRouteDelay = 141;
constexpr int kNeutralOwner = -1;
constexpr int kMissingOwner = -2;
constexpr int kBlockedDelay = 200;
constexpr int kCometSpawnSteps[] = {50, 150, 250, 350, 450};

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

struct CometGroup {
  std::vector<int> planet_ids;
  std::vector<std::vector<Vec2>> paths;
  int path_index = 0;
};

struct Observation {
  int player = 0;
  int step = 0;
  double angular_velocity = 0.0;
  std::vector<Planet> planets;
  std::vector<Planet> initial_planets;
  std::vector<Fleet> fleets;
  std::vector<int> comet_planet_ids;
  std::vector<CometGroup> comets;
};

struct FeatureStats {
  double elapsed_ms = 0.0;
  int planets = 0;
  int fleets = 0;
  int horizon = 0;
  int route_queries = 0;
  int route_proxy_simulations = 0;
  int route_sim_ticks = 0;
  int predicted_arrivals = 0;
  int blocked_routes = 0;
  int active_comets = 0;
  int expiring_comets_within_horizon = 0;
  int next_comet_spawn_step = -1;
  int turns_until_next_comet_spawn = -1;
};

struct FeatureBatch {
  std::vector<int> planet_ids;
  std::vector<int> ship_buckets;
  std::vector<int> comet_spawn_steps;
  std::vector<int> garrison_flat;
  std::vector<int> delay_flat;
  std::vector<double> angle_flat;
  FeatureStats stats;
};

struct ExactRoute {
  bool reachable = false;
  int delay = kBlockedDelay;
  double angle = 0.0;
  std::string blocked_by = "unreachable";
};

double dist2(Vec2 a, Vec2 b);
double dist(Vec2 a, Vec2 b);
double normalize_angle(double angle);
double fleet_speed(int ships);
bool segment_circle_intersects(Vec2 a, Vec2 b, Vec2 center, double radius);
bool swept_pair_hit(Vec2 fleet_old, Vec2 fleet_new, Vec2 planet_old, Vec2 planet_new,
                    double radius);
bool is_orbiting_planet(const Planet& p);
Vec2 rotated_position(const Planet& initial, double angular_velocity, int step);
int observed_orbit_step(int observation_step);

class FeatureEngine {
 public:
  FeatureEngine() = default;

  void initialize(const Observation& obs);
  FeatureBatch compute(const Observation& obs, int horizon = 50,
                       int max_route_delay = kMaxRouteDelay);
  ExactRoute query_route(const Observation& obs, int src_id, int target_id, int ships,
                         int max_route_delay = kMaxRouteDelay);
  bool initialized() const { return initialized_; }

 private:
  struct Arrival {
    int planet_id = -1;
    int owner = kNeutralOwner;
    int ships = 0;
    int dt = -1;
  };

  struct RouteEval {
    int delay = kBlockedDelay;
    double angle = 0.0;
    bool blocked = true;
  };

  Observation current_;
  bool initialized_ = false;
  std::unordered_map<int, std::size_t> initial_index_;
  std::unordered_map<int, std::size_t> current_index_;
  std::unordered_map<int, std::pair<int, int>> comet_path_index_;
  std::vector<std::vector<Vec2>> position_cache_;

  void rebuild_initial_cache();
  void refresh_current(const Observation& obs);
  bool cache_matches(const Observation& obs) const;
  bool is_comet_id(int planet_id) const;
  bool planet_present_at(const Planet& planet, int absolute_step) const;
  Vec2 planet_position_at(const Planet& planet, int absolute_step) const;
  void fill_comet_stats(FeatureBatch& batch, int horizon) const;
  std::vector<Arrival> predict_arrivals(int horizon, FeatureStats& stats) const;
  RouteEval estimate_route_without_proxy(const Planet& src, const Planet& target, int ships,
                                         int max_route_delay) const;
  ExactRoute validate_exact_route(const Planet& src, const Planet& target, int ships,
                                  double angle, int max_route_delay) const;
  void build_delay_matrix_batched(FeatureBatch& batch, int max_route_delay) const;
};

}  // namespace orbit_rl
