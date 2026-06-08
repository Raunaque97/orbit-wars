#include "rl_features.hpp"
#include "orbit_core.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using torch::indexing::Slice;
using torch::indexing::TensorIndex;

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template <typename Fn>
void parallel_for(std::size_t count, int worker_threads, Fn fn) {
  if (count == 0) {
    return;
  }
  const int workers = std::max(1, std::min(worker_threads, static_cast<int>(count)));
  if (workers <= 1) {
    for (std::size_t i = 0; i < count; ++i) {
      fn(i);
    }
    return;
  }

  std::atomic<std::size_t> next{0};
  std::mutex error_mutex;
  std::exception_ptr error;
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(workers));
  for (int worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&]() {
      while (true) {
        const std::size_t idx = next.fetch_add(1);
        if (idx >= count) {
          break;
        }
        try {
          fn(idx);
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!error) {
            error = std::current_exception();
          }
          break;
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

constexpr int kHorizon = 50;
constexpr int kNearestAllies = 3;
constexpr int kPlanetDim = 12 + kHorizon * 4 + kNearestAllies * 2;
constexpr int kEdgeDim = 11;
constexpr int kAmountBins = 6;
constexpr int kInferenceBatchSize = 16;
constexpr int kAllyDelayBucket = 20;
constexpr int kRouteTimeout = 141;
constexpr int kBlockedDelay = 200;
constexpr int kMaxEpisodeSteps = 500;
constexpr double kStepReward = -0.002;
constexpr double kTerminalWinReward = 10.0;
constexpr double kTerminalLossReward = -10.0;
constexpr double kTerminalDrawReward = 0.0;
constexpr double kQuickWinRewardScale = 2.0;
constexpr double kTimeoutReward = -2.0;
constexpr double kCaptureNeutralReward = 0.20;
constexpr double kCaptureEnemyReward = 0.40;
constexpr double kCaptureProductionReward = 0.05;
constexpr double kLosePlanetReward = -0.40;
constexpr double kLoseProductionReward = -0.08;

constexpr int kRandomMinFleetSize = 5;
constexpr double kRandomMaxLaunchProb = 0.10;
constexpr int kRandomLaunchProbMaxShips = 50;
constexpr double kRandomLaunchProbDecayShips = 15.0;
constexpr int kRandomRouteAttemptsPerSource = 32;

const std::array<int, 6> kEdgeDelayBuckets{5, 10, 20, 40, 80, 160};
const std::array<int, 5> kCometSpawnSteps{50, 150, 250, 350, 450};

struct SpawnEvent {
  int step = 0;
  std::vector<orbit::Planet> planets;
  std::vector<orbit::Planet> initial_planets;
  std::vector<int> comet_planet_ids;
  std::vector<orbit::CometGroup> comets;
};

struct GraphData {
  torch::Tensor planet_features;
  torch::Tensor edge_features;
  torch::Tensor planet_mask;
  std::vector<int> planet_ids;
  orbit_rl::FeatureBatch feature_batch;
  orbit_rl::Observation obs;
  int delay_cache_hit = 0;
};

struct DecisionData {
  int source_idx = -1;
  int stop_action = 1;
  int target_idx = -1;
  int amount_idx = -1;
  std::array<bool, kAmountBins> amount_mask{false, false, false, false, false, false};
};

struct TransitionData {
  GraphData graph;
  std::vector<DecisionData> decisions;
  double old_logprob = 0.0;
  double old_entropy = 0.0;
  double value = 0.0;
  double reward = 0.0;
  bool done = false;
  int action_terms = 0;
  std::map<std::string, int> invalid_counts;
};

struct EpisodeState {
  int seed = 0;
  orbit::SimState state;
  orbit::Engine sim_engine;
  orbit::Engine random_route_engine;
  orbit_rl::FeatureEngine feature_engine;
  std::vector<SpawnEvent> spawns;
  std::mt19937 rng;
  bool done = false;
  double outcome = 0.0;
  int length = 0;
  std::vector<TransitionData> transitions;
};

struct OpponentSpec {
  std::string name;
  std::string model_path;
  int population_index = -1;
};

struct ModelHandle {
  std::string path;
  torch::jit::script::Module module;
};

double sigmoid(double x) {
  if (x >= 0.0) {
    const double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(x);
  return z / (1.0 + z);
}

double logsumexp(const std::vector<double>& logits, const std::vector<bool>& mask) {
  double max_value = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (mask[i]) {
      max_value = std::max(max_value, logits[i]);
    }
  }
  if (!std::isfinite(max_value)) {
    return max_value;
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (mask[i]) {
      sum += std::exp(logits[i] - max_value);
    }
  }
  return max_value + std::log(std::max(sum, 1e-300));
}

int sample_categorical(const std::vector<double>& logits, const std::vector<bool>& mask,
                       std::mt19937& rng) {
  const double lse = logsumexp(logits, mask);
  if (!std::isfinite(lse)) {
    return -1;
  }
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const double draw = unit(rng);
  double cumulative = 0.0;
  int last = -1;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (!mask[i]) {
      continue;
    }
    last = static_cast<int>(i);
    cumulative += std::exp(logits[i] - lse);
    if (draw <= cumulative) {
      return static_cast<int>(i);
    }
  }
  return last;
}

double categorical_entropy(const std::vector<double>& logits, const std::vector<bool>& mask) {
  const double lse = logsumexp(logits, mask);
  if (!std::isfinite(lse)) {
    return 0.0;
  }
  double entropy = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (!mask[i]) {
      continue;
    }
    const double logp = logits[i] - lse;
    const double p = std::exp(logp);
    entropy -= p * logp;
  }
  return entropy;
}

orbit::Planet orbit_planet_from_sequence(const py::handle& item) {
  py::sequence seq = py::reinterpret_borrow<py::sequence>(item);
  orbit::Planet planet;
  planet.id = seq[0].cast<int>();
  planet.owner = seq[1].cast<int>();
  planet.x = seq[2].cast<double>();
  planet.y = seq[3].cast<double>();
  planet.radius = seq[4].cast<double>();
  planet.ships = seq[5].cast<int>();
  planet.production = seq[6].cast<int>();
  return planet;
}

orbit::Fleet orbit_fleet_from_sequence(const py::handle& item) {
  py::sequence seq = py::reinterpret_borrow<py::sequence>(item);
  orbit::Fleet fleet;
  fleet.id = seq[0].cast<int>();
  fleet.owner = seq[1].cast<int>();
  fleet.x = seq[2].cast<double>();
  fleet.y = seq[3].cast<double>();
  fleet.angle = seq[4].cast<double>();
  fleet.from_planet_id = seq[5].cast<int>();
  fleet.ships = seq[6].cast<int>();
  return fleet;
}

std::vector<orbit::Planet> orbit_planets_from_object(const py::object& obj) {
  std::vector<orbit::Planet> planets;
  if (obj.is_none()) {
    return planets;
  }
  for (const py::handle& item : obj) {
    planets.push_back(orbit_planet_from_sequence(item));
  }
  return planets;
}

std::vector<orbit::Fleet> orbit_fleets_from_object(const py::object& obj) {
  std::vector<orbit::Fleet> fleets;
  if (obj.is_none()) {
    return fleets;
  }
  for (const py::handle& item : obj) {
    fleets.push_back(orbit_fleet_from_sequence(item));
  }
  return fleets;
}

template <typename T>
T get_attr_or_item(const py::object& obj, const char* name, T fallback) {
  if (py::isinstance<py::dict>(obj)) {
    py::dict dict = obj.cast<py::dict>();
    py::object value = dict.attr("get")(name, py::cast(fallback));
    return value.is_none() ? fallback : value.cast<T>();
  }
  if (py::hasattr(obj, name)) {
    py::object value = obj.attr(name);
    return value.is_none() ? fallback : value.cast<T>();
  }
  return fallback;
}

py::object get_object_attr_or_item(const py::object& obj, const char* name) {
  if (py::isinstance<py::dict>(obj)) {
    py::dict dict = obj.cast<py::dict>();
    return dict.attr("get")(name, py::none());
  }
  if (py::hasattr(obj, name)) {
    return obj.attr(name);
  }
  return py::none();
}

std::vector<orbit::CometGroup> orbit_comets_from_object(const py::object& obj) {
  std::vector<orbit::CometGroup> comets;
  if (obj.is_none()) {
    return comets;
  }
  for (const py::handle& comet_item : obj) {
    py::object comet = py::reinterpret_borrow<py::object>(comet_item);
    orbit::CometGroup group;
    py::object planet_ids = get_object_attr_or_item(comet, "planet_ids");
    if (!planet_ids.is_none()) {
      for (const py::handle& id : planet_ids) {
        group.planet_ids.push_back(id.cast<int>());
      }
    }
    group.path_index = get_attr_or_item<int>(comet, "path_index", 0);
    py::object paths = get_object_attr_or_item(comet, "paths");
    if (!paths.is_none()) {
      for (const py::handle& path_item : paths) {
        std::vector<orbit::Vec2> path;
        for (const py::handle& point_item : path_item) {
          py::sequence point = py::reinterpret_borrow<py::sequence>(point_item);
          path.push_back(orbit::Vec2{point[0].cast<double>(), point[1].cast<double>()});
        }
        group.paths.push_back(std::move(path));
      }
    }
    comets.push_back(std::move(group));
  }
  return comets;
}

orbit::SimState state_from_py(const py::object& obj) {
  orbit::SimState state;
  state.step = get_attr_or_item<int>(obj, "step", 0);
  state.angular_velocity = get_attr_or_item<double>(obj, "angular_velocity", 0.0);
  state.planets = orbit_planets_from_object(get_object_attr_or_item(obj, "planets"));
  state.initial_planets =
      orbit_planets_from_object(get_object_attr_or_item(obj, "initial_planets"));
  if (state.initial_planets.empty()) {
    state.initial_planets = state.planets;
  }
  state.fleets = orbit_fleets_from_object(get_object_attr_or_item(obj, "fleets"));
  py::object comet_ids = get_object_attr_or_item(obj, "comet_planet_ids");
  if (!comet_ids.is_none()) {
    for (const py::handle& item : comet_ids) {
      state.comet_planet_ids.push_back(item.cast<int>());
    }
  }
  state.comets = orbit_comets_from_object(get_object_attr_or_item(obj, "comets"));
  state.next_fleet_id = get_attr_or_item<int>(obj, "next_fleet_id", 0);
  for (const orbit::Fleet& fleet : state.fleets) {
    state.next_fleet_id = std::max(state.next_fleet_id, fleet.id + 1);
  }
  return state;
}

SpawnEvent spawn_event_from_py(const py::object& obj) {
  SpawnEvent event;
  event.step = get_attr_or_item<int>(obj, "step", 0);
  event.planets = orbit_planets_from_object(get_object_attr_or_item(obj, "planets"));
  event.initial_planets =
      orbit_planets_from_object(get_object_attr_or_item(obj, "initial_planets"));
  py::object comet_ids = get_object_attr_or_item(obj, "comet_planet_ids");
  if (!comet_ids.is_none()) {
    for (const py::handle& item : comet_ids) {
      event.comet_planet_ids.push_back(item.cast<int>());
    }
  }
  event.comets = orbit_comets_from_object(get_object_attr_or_item(obj, "comets"));
  return event;
}

orbit::Observation orbit_obs_from_state(const orbit::SimState& state, int player) {
  orbit::Observation obs;
  obs.player = player;
  obs.step = state.step;
  obs.angular_velocity = state.angular_velocity;
  obs.planets = state.planets;
  obs.initial_planets = state.initial_planets;
  obs.fleets = state.fleets;
  obs.comet_planet_ids = state.comet_planet_ids;
  obs.comets = state.comets;
  return obs;
}

orbit_rl::Observation rl_obs_from_state(const orbit::SimState& state, int player) {
  orbit_rl::Observation obs;
  obs.player = player;
  obs.step = state.step;
  obs.angular_velocity = state.angular_velocity;
  for (const orbit::Planet& p : state.planets) {
    obs.planets.push_back(
        orbit_rl::Planet{p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production});
  }
  for (const orbit::Planet& p : state.initial_planets) {
    obs.initial_planets.push_back(
        orbit_rl::Planet{p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production});
  }
  for (const orbit::Fleet& f : state.fleets) {
    obs.fleets.push_back(
        orbit_rl::Fleet{f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships});
  }
  obs.comet_planet_ids = state.comet_planet_ids;
  for (const orbit::CometGroup& group : state.comets) {
    orbit_rl::CometGroup out;
    out.planet_ids = group.planet_ids;
    out.path_index = group.path_index;
    for (const std::vector<orbit::Vec2>& path : group.paths) {
      std::vector<orbit_rl::Vec2> converted;
      for (const orbit::Vec2& point : path) {
        converted.push_back(orbit_rl::Vec2{point.x, point.y});
      }
      out.paths.push_back(std::move(converted));
    }
    obs.comets.push_back(std::move(out));
  }
  if (obs.initial_planets.empty()) {
    obs.initial_planets = obs.planets;
  }
  return obs;
}

void apply_spawn_event(orbit::SimState& state, const SpawnEvent& event) {
  for (const orbit::Planet& planet : event.planets) {
    state.planets.push_back(planet);
  }
  for (const orbit::Planet& planet : event.initial_planets) {
    state.initial_planets.push_back(planet);
  }
  for (int id : event.comet_planet_ids) {
    state.comet_planet_ids.push_back(id);
  }
  for (const orbit::CometGroup& comet : event.comets) {
    state.comets.push_back(comet);
  }
}

bool is_spawn_step(int step) {
  return std::find(kCometSpawnSteps.begin(), kCometSpawnSteps.end(), step) !=
         kCometSpawnSteps.end();
}

std::string cache_path_for(const std::string& root, int seed, int step) {
  char seed_buf[64];
  char step_buf[64];
  std::snprintf(seed_buf, sizeof(seed_buf), "seed_%06d_native", seed);
  std::snprintf(step_buf, sizeof(step_buf), "step_%04d.bin", step);
  return (std::filesystem::path(root) / seed_buf / step_buf).string();
}

std::mutex& delay_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

bool load_delay_cache(const std::string& root, int seed, int step,
                      const std::vector<int>& planet_ids,
                      const std::vector<int>& ship_buckets, orbit_rl::FeatureBatch& batch) {
  if (root.empty()) {
    return false;
  }
  const std::string path = cache_path_for(root, seed, step);
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint32_t n = 0;
  std::uint32_t buckets = 0;
  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char*>(&version), sizeof(version));
  in.read(reinterpret_cast<char*>(&n), sizeof(n));
  in.read(reinterpret_cast<char*>(&buckets), sizeof(buckets));
  if (!in || magic != 0x4f574443u || version != 1u || n != planet_ids.size() ||
      buckets != ship_buckets.size()) {
    return false;
  }
  std::vector<int> cached_planets(n);
  std::vector<int> cached_buckets(buckets);
  in.read(reinterpret_cast<char*>(cached_planets.data()), cached_planets.size() * sizeof(int));
  in.read(reinterpret_cast<char*>(cached_buckets.data()), cached_buckets.size() * sizeof(int));
  if (!in || cached_planets != planet_ids || cached_buckets != ship_buckets) {
    return false;
  }
  const std::size_t total = static_cast<std::size_t>(buckets) * n * n;
  batch.delay_flat.resize(total);
  batch.angle_flat.resize(total);
  in.read(reinterpret_cast<char*>(batch.delay_flat.data()), total * sizeof(int));
  in.read(reinterpret_cast<char*>(batch.angle_flat.data()), total * sizeof(double));
  return static_cast<bool>(in);
}

void save_delay_cache(const std::string& root, int seed, int step,
                      const orbit_rl::FeatureBatch& batch) {
  if (root.empty()) {
    return;
  }
  const std::string path = cache_path_for(root, seed, step);
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  const std::string tmp_path = path + ".tmp";
  std::ofstream out(tmp_path, std::ios::binary);
  if (!out) {
    return;
  }
  const std::uint32_t magic = 0x4f574443u;
  const std::uint32_t version = 1u;
  const std::uint32_t n = static_cast<std::uint32_t>(batch.planet_ids.size());
  const std::uint32_t buckets = static_cast<std::uint32_t>(batch.ship_buckets.size());
  out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char*>(&version), sizeof(version));
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(reinterpret_cast<const char*>(&buckets), sizeof(buckets));
  out.write(reinterpret_cast<const char*>(batch.planet_ids.data()),
            batch.planet_ids.size() * sizeof(int));
  out.write(reinterpret_cast<const char*>(batch.ship_buckets.data()),
            batch.ship_buckets.size() * sizeof(int));
  out.write(reinterpret_cast<const char*>(batch.delay_flat.data()),
            batch.delay_flat.size() * sizeof(int));
  out.write(reinterpret_cast<const char*>(batch.angle_flat.data()),
            batch.angle_flat.size() * sizeof(double));
  out.close();
  if (out) {
    std::filesystem::rename(tmp_path, path);
  }
}

std::array<float, 3> owner_vec(int owner, int player) {
  if (owner == orbit_rl::kMissingOwner || owner == orbit_rl::kNeutralOwner) {
    return {0.0f, 0.0f, 0.0f};
  }
  if (owner == player) {
    return {-1.0f, -1.0f, -1.0f};
  }
  std::array<float, 3> out{0.0f, 0.0f, 0.0f};
  int cursor = 0;
  for (int pid = 0; pid < 4; ++pid) {
    if (pid == player) {
      continue;
    }
    if (pid == owner && cursor < 3) {
      out[cursor] = 1.0f;
    }
    ++cursor;
  }
  return out;
}

bool segment_sun_intersects(double ax, double ay, double bx, double by) {
  return orbit_rl::segment_circle_intersects(
      orbit_rl::Vec2{ax, ay}, orbit_rl::Vec2{bx, by},
      orbit_rl::Vec2{orbit_rl::kCenterX, orbit_rl::kCenterY}, orbit_rl::kSunRadius);
}

int ship_bucket_index(const std::vector<int>& buckets, int value) {
  auto it = std::find(buckets.begin(), buckets.end(), value);
  if (it == buckets.end()) {
    return -1;
  }
  return static_cast<int>(std::distance(buckets.begin(), it));
}

int delay_at(const orbit_rl::FeatureBatch& batch, int bucket_idx, int i, int j) {
  const int n = static_cast<int>(batch.planet_ids.size());
  return batch.delay_flat[(bucket_idx * n + i) * n + j];
}

int garrison_ships_at(const orbit_rl::FeatureBatch& batch, int row, int dt) {
  return batch.garrison_flat[(row * kHorizon + dt) * 2];
}

int garrison_owner_at(const orbit_rl::FeatureBatch& batch, int row, int dt) {
  return batch.garrison_flat[(row * kHorizon + dt) * 2 + 1];
}

int planet_index_by_id(const std::vector<int>& planet_ids, int id) {
  auto it = std::find(planet_ids.begin(), planet_ids.end(), id);
  if (it == planet_ids.end()) {
    return -1;
  }
  return static_cast<int>(std::distance(planet_ids.begin(), it));
}

std::unordered_map<int, int> comet_remaining_by_id(const orbit_rl::Observation& obs) {
  std::unordered_map<int, int> remaining;
  for (const orbit_rl::CometGroup& group : obs.comets) {
    const std::size_t limit = std::min(group.planet_ids.size(), group.paths.size());
    for (std::size_t i = 0; i < limit; ++i) {
      remaining[group.planet_ids[i]] =
          std::max(0, static_cast<int>(group.paths[i].size()) - group.path_index);
    }
  }
  return remaining;
}

int forecast_surplus_for_planet(const orbit_rl::FeatureBatch& batch, int planet_id, int owner) {
  const int row = planet_index_by_id(batch.planet_ids, planet_id);
  if (row < 0) {
    return 0;
  }
  int min_ships = std::numeric_limits<int>::max();
  for (int dt = 0; dt < kHorizon; ++dt) {
    const int future_owner = garrison_owner_at(batch, row, dt);
    if (future_owner != owner) {
      return 0;
    }
    min_ships = std::min(min_ships, garrison_ships_at(batch, row, dt));
  }
  return std::max(0, min_ships == std::numeric_limits<int>::max() ? 0 : min_ships);
}

int minimum_to_capture_at_arrival(const orbit_rl::FeatureBatch& batch, int target_id,
                                  int player, int delay) {
  const int row = planet_index_by_id(batch.planet_ids, target_id);
  if (row < 0) {
    return 0;
  }
  const int dt = std::clamp(delay, 0, kHorizon - 1);
  const int owner = garrison_owner_at(batch, row, dt);
  if (owner == player) {
    return 0;
  }
  return std::max(1, garrison_ships_at(batch, row, dt) + 1);
}

std::array<int, kAmountBins> amount_bin_ship_counts(int source_ships, int surplus,
                                                    int minimum_to_capture) {
  source_ships = std::max(0, source_ships);
  surplus = std::max(0, surplus);
  minimum_to_capture = std::max(0, minimum_to_capture);
  std::array<int, kAmountBins> raw{
      minimum_to_capture > 0 ? minimum_to_capture + 1 : 0,
      static_cast<int>(std::llround(0.20 * surplus)),
      static_cast<int>(std::llround(0.50 * surplus)),
      static_cast<int>(std::llround(0.80 * surplus)),
      surplus,
      source_ships,
  };
  for (int& value : raw) {
    value = std::max(0, std::min(source_ships, value));
  }
  return raw;
}

GraphData build_graph(EpisodeState& episode, int player, const std::string& delay_cache_dir) {
  GraphData graph;
  graph.obs = rl_obs_from_state(episode.state, player);
  if (!episode.feature_engine.initialized() || episode.state.step == 0) {
    episode.feature_engine.initialize(graph.obs);
  }

  orbit_rl::FeatureBatch batch =
      episode.feature_engine.compute(graph.obs, kHorizon, orbit_rl::kMaxRouteDelay, false);
  {
    std::lock_guard<std::mutex> lock(delay_cache_mutex());
    graph.delay_cache_hit =
        load_delay_cache(delay_cache_dir, episode.seed, episode.state.step, batch.planet_ids,
                         batch.ship_buckets, batch)
            ? 1
            : 0;
  }
  if (!graph.delay_cache_hit) {
    batch = episode.feature_engine.compute(graph.obs, kHorizon, orbit_rl::kMaxRouteDelay, true);
    std::lock_guard<std::mutex> lock(delay_cache_mutex());
    save_delay_cache(delay_cache_dir, episode.seed, episode.state.step, batch);
  }
  graph.feature_batch = std::move(batch);
  graph.planet_ids = graph.feature_batch.planet_ids;

  const int n = static_cast<int>(graph.planet_ids.size());
  std::unordered_map<int, const orbit_rl::Planet*> by_id;
  for (const orbit_rl::Planet& planet : graph.obs.planets) {
    by_id[planet.id] = &planet;
  }

  std::set<int> comet_ids(graph.obs.comet_planet_ids.begin(), graph.obs.comet_planet_ids.end());
  std::unordered_map<int, int> comet_remaining = comet_remaining_by_id(graph.obs);
  std::unordered_map<int, double> owner_total_prod;
  std::unordered_map<int, double> owner_total_ships;
  for (int planet_id : graph.planet_ids) {
    const orbit_rl::Planet* planet = by_id.at(planet_id);
    owner_total_prod[planet->owner] += planet->production;
    owner_total_ships[planet->owner] += planet->ships;
  }

  std::vector<float> planet_features(static_cast<std::size_t>(n) * kPlanetDim, 0.0f);
  std::vector<float> edge_features(static_cast<std::size_t>(n) * n * kEdgeDim, 0.0f);
  const int ally_bucket_idx = ship_bucket_index(graph.feature_batch.ship_buckets, kAllyDelayBucket);

  for (int i = 0; i < n; ++i) {
    const orbit_rl::Planet* planet = by_id.at(graph.planet_ids[i]);
    const bool is_comet = comet_ids.count(planet->id) > 0;
    const double total_ships = owner_total_ships[planet->owner];
    const double dx = planet->x - orbit_rl::kCenterX;
    const double dy = planet->y - orbit_rl::kCenterY;
    std::vector<std::pair<float, float>> same_owner;
    for (int j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      const orbit_rl::Planet* other = by_id.at(graph.planet_ids[j]);
      if (other->owner != planet->owner) {
        continue;
      }
      same_owner.push_back(
          {static_cast<float>(delay_at(graph.feature_batch, ally_bucket_idx, i, j)),
           static_cast<float>(other->ships)});
    }
    std::sort(same_owner.begin(), same_owner.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::size_t offset = static_cast<std::size_t>(i) * kPlanetDim;
    const auto ov = owner_vec(planet->owner, player);
    planet_features[offset++] = ov[0];
    planet_features[offset++] = ov[1];
    planet_features[offset++] = ov[2];
    planet_features[offset++] = static_cast<float>(owner_total_prod[planet->owner]);
    planet_features[offset++] = static_cast<float>(total_ships);
    planet_features[offset++] = static_cast<float>(planet->radius);
    planet_features[offset++] = static_cast<float>(std::hypot(dx, dy));
    planet_features[offset++] = is_comet ? 1.0f : 0.0f;
    planet_features[offset++] =
        static_cast<float>(is_comet ? comet_remaining[planet->id]
                                    : orbit_rl::kMaxSteps - graph.obs.step);
    planet_features[offset++] =
        (orbit_rl::is_orbiting_planet(*planet) && !is_comet) ? 1.0f : 0.0f;
    planet_features[offset++] = static_cast<float>(planet->production);
    planet_features[offset++] =
        static_cast<float>(planet->ships / std::max(1.0, total_ships));

    for (int dt = 0; dt < kHorizon; ++dt) {
      const auto future_owner =
          owner_vec(garrison_owner_at(graph.feature_batch, i, dt), player);
      planet_features[offset++] = future_owner[0];
      planet_features[offset++] = future_owner[1];
      planet_features[offset++] = future_owner[2];
      planet_features[offset++] = static_cast<float>(garrison_ships_at(graph.feature_batch, i, dt));
    }

    for (int k = 0; k < kNearestAllies; ++k) {
      planet_features[offset++] =
          k < static_cast<int>(same_owner.size())
              ? std::min(same_owner[k].first, static_cast<float>(kRouteTimeout))
              : static_cast<float>(kRouteTimeout);
    }
    for (int k = 0; k < kNearestAllies; ++k) {
      planet_features[offset++] =
          k < static_cast<int>(same_owner.size()) ? same_owner[k].second : 0.0f;
    }
  }

  std::array<int, 6> delay_indices;
  for (std::size_t k = 0; k < kEdgeDelayBuckets.size(); ++k) {
    delay_indices[k] = ship_bucket_index(graph.feature_batch.ship_buckets, kEdgeDelayBuckets[k]);
  }
  for (int i = 0; i < n; ++i) {
    const orbit_rl::Planet* src = by_id.at(graph.planet_ids[i]);
    for (int j = 0; j < n; ++j) {
      const orbit_rl::Planet* dst = by_id.at(graph.planet_ids[j]);
      const std::size_t base = (static_cast<std::size_t>(i) * n + j) * kEdgeDim;
      for (int k = 0; k < 6; ++k) {
        edge_features[base + k] =
            static_cast<float>(delay_at(graph.feature_batch, delay_indices[k], i, j));
      }
      edge_features[base + 6] = static_cast<float>(dst->production);
      edge_features[base + 7] = static_cast<float>(dst->ships);
      edge_features[base + 8] = static_cast<float>(src->production);
      edge_features[base + 9] = static_cast<float>(src->ships);
      edge_features[base + 10] =
          segment_sun_intersects(src->x, src->y, dst->x, dst->y) ? 1.0f : 0.0f;
    }
  }

  graph.planet_features =
      torch::from_blob(planet_features.data(), {n, kPlanetDim}, torch::kFloat32).clone();
  graph.edge_features =
      torch::from_blob(edge_features.data(), {n, n, kEdgeDim}, torch::kFloat32).clone();
  graph.planet_mask = torch::ones({n}, torch::kBool);
  return graph;
}

torch::Tensor pad_planets(const std::vector<GraphData*>& graphs, int max_n) {
  torch::Tensor out = torch::zeros(
      {static_cast<long>(graphs.size()), max_n, kPlanetDim}, torch::kFloat32);
  for (std::size_t b = 0; b < graphs.size(); ++b) {
    const int n = static_cast<int>(graphs[b]->planet_ids.size());
    std::vector<TensorIndex> indices{
        static_cast<int64_t>(b), Slice(0, n), Slice()};
    out.index(indices).copy_(graphs[b]->planet_features);
  }
  return out;
}

torch::Tensor pad_edges(const std::vector<GraphData*>& graphs, int max_n) {
  torch::Tensor out = torch::zeros(
      {static_cast<long>(graphs.size()), max_n, max_n, kEdgeDim}, torch::kFloat32);
  for (std::size_t b = 0; b < graphs.size(); ++b) {
    const int n = static_cast<int>(graphs[b]->planet_ids.size());
    std::vector<TensorIndex> indices{
        static_cast<int64_t>(b), Slice(0, n), Slice(0, n), Slice()};
    out.index(indices).copy_(graphs[b]->edge_features);
  }
  return out;
}

torch::Tensor pad_masks(const std::vector<GraphData*>& graphs, int max_n) {
  torch::Tensor out =
      torch::zeros({static_cast<long>(graphs.size()), max_n}, torch::kBool);
  for (std::size_t b = 0; b < graphs.size(); ++b) {
    const int n = static_cast<int>(graphs[b]->planet_ids.size());
    std::vector<TensorIndex> indices{static_cast<int64_t>(b), Slice(0, n)};
    out.index(indices).copy_(graphs[b]->planet_mask);
  }
  return out;
}

std::unordered_map<std::string, torch::Tensor> forward_model_padded(
    torch::jit::script::Module& module, const std::vector<GraphData*>& graphs, int max_n) {
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(pad_planets(graphs, max_n));
  inputs.emplace_back(pad_edges(graphs, max_n));
  inputs.emplace_back(pad_masks(graphs, max_n));
  auto out = module.forward(inputs).toGenericDict();
  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors["edge_logits"] = out.at("edge_logits").toTensor().cpu();
  tensors["amount_logits"] = out.at("amount_logits").toTensor().cpu();
  tensors["stop_logits"] = out.at("stop_logits").toTensor().cpu();
  tensors["value"] = out.at("value").toTensor().cpu();
  return tensors;
}

std::unordered_map<std::string, torch::Tensor> forward_model(
    torch::jit::script::Module& module, const std::vector<GraphData*>& graphs) {
  int max_n = 1;
  for (const GraphData* graph : graphs) {
    max_n = std::max(max_n, static_cast<int>(graph->planet_ids.size()));
  }
  return forward_model_padded(module, graphs, max_n);
}

std::unordered_map<std::string, torch::Tensor> forward_model_chunked(
    torch::jit::script::Module& module, const std::vector<GraphData*>& graphs) {
  if (graphs.size() <= static_cast<std::size_t>(kInferenceBatchSize)) {
    return forward_model(module, graphs);
  }

  int max_n = 1;
  for (const GraphData* graph : graphs) {
    max_n = std::max(max_n, static_cast<int>(graph->planet_ids.size()));
  }
  std::map<std::string, std::vector<torch::Tensor>> chunks;
  for (std::size_t start = 0; start < graphs.size();
       start += static_cast<std::size_t>(kInferenceBatchSize)) {
    const std::size_t end =
        std::min(graphs.size(), start + static_cast<std::size_t>(kInferenceBatchSize));
    std::vector<GraphData*> slice(graphs.begin() + static_cast<std::ptrdiff_t>(start),
                                  graphs.begin() + static_cast<std::ptrdiff_t>(end));
    auto out = forward_model_padded(module, slice, max_n);
    for (auto& item : out) {
      chunks[item.first].push_back(item.second);
    }
  }

  std::unordered_map<std::string, torch::Tensor> merged;
  for (auto& item : chunks) {
    merged[item.first] = torch::cat(item.second, 0);
  }
  return merged;
}

void add_invalid(TransitionData& transition, const std::string& reason) {
  transition.invalid_counts[reason] += 1;
  if (reason == "route_sun") {
    transition.reward -= 0.15;
  } else if (reason == "bad_amount" || reason == "mincapture_unaffordable") {
    transition.reward -= 0.05;
  } else {
    transition.reward -= 0.10;
  }
}

std::string route_invalid_reason(const std::string& blocked_by) {
  if (blocked_by == "sun" || blocked_by == "planet" || blocked_by == "bounds" ||
      blocked_by == "timeout" || blocked_by == "wrong_planet") {
    return "route_" + blocked_by;
  }
  return "route_blocked";
}

std::vector<orbit::Move> sample_from_output(EpisodeState& episode, TransitionData& transition,
                                            const std::unordered_map<std::string, torch::Tensor>& out,
                                            int batch_idx) {
  const int player = transition.graph.obs.player;
  const int n = static_cast<int>(transition.graph.planet_ids.size());
  std::unordered_map<int, const orbit_rl::Planet*> planets_by_id;
  for (const orbit_rl::Planet& planet : transition.graph.obs.planets) {
    planets_by_id[planet.id] = &planet;
  }
  const int ally_bucket_idx =
      ship_bucket_index(transition.graph.feature_batch.ship_buckets, kAllyDelayBucket);

  std::vector<orbit::Move> moves;
  auto stop_logits = out.at("stop_logits");
  auto edge_logits = out.at("edge_logits");
  auto amount_logits = out.at("amount_logits");
  auto values = out.at("value");
  transition.value = values.index(std::vector<TensorIndex>{batch_idx}).item<double>();

  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int src_idx = 0; src_idx < n; ++src_idx) {
    const int source_id = transition.graph.planet_ids[src_idx];
    const orbit_rl::Planet* source = planets_by_id[source_id];
    if (source == nullptr || source->owner != player || source->ships <= 0) {
      continue;
    }

    const double stop_logit =
        stop_logits.index(std::vector<TensorIndex>{batch_idx, src_idx}).item<double>();
    const double stop_p = sigmoid(stop_logit);
    const int stop_action = unit(episode.rng) < stop_p ? 1 : 0;
    transition.old_logprob += stop_action ? std::log(std::max(stop_p, 1e-30))
                                          : std::log(std::max(1.0 - stop_p, 1e-30));
    transition.old_entropy +=
        -(stop_p * std::log(std::max(stop_p, 1e-30)) +
          (1.0 - stop_p) * std::log(std::max(1.0 - stop_p, 1e-30)));
    transition.action_terms += 1;
    DecisionData decision;
    decision.source_idx = src_idx;
    decision.stop_action = stop_action;
    transition.decisions.push_back(decision);
    DecisionData& stored_decision = transition.decisions.back();
    if (stop_action == 1) {
      continue;
    }

    std::vector<double> target_logits(n, 0.0);
    std::vector<bool> target_mask(n, true);
    for (int j = 0; j < n; ++j) {
      target_logits[j] =
          edge_logits.index(std::vector<TensorIndex>{batch_idx, src_idx, j}).item<double>();
      target_mask[j] = std::isfinite(target_logits[j]);
    }
    target_mask[src_idx] = false;
    const int target_idx = sample_categorical(target_logits, target_mask, episode.rng);
    if (target_idx < 0) {
      add_invalid(transition, "route_blocked");
      continue;
    }
    const double target_lse = logsumexp(target_logits, target_mask);
    transition.old_logprob += target_logits[target_idx] - target_lse;
    transition.old_entropy += categorical_entropy(target_logits, target_mask);
    transition.action_terms += 1;
    stored_decision.target_idx = target_idx;

    const int target_id = transition.graph.planet_ids[target_idx];
    const int available = source->ships;
    const int delay = delay_at(transition.graph.feature_batch, ally_bucket_idx, src_idx, target_idx);
    const int coarse_delay = delay < kBlockedDelay ? delay : kRouteTimeout;
    const int surplus = forecast_surplus_for_planet(transition.graph.feature_batch, source_id, player);
    const int minimum_to_capture =
        minimum_to_capture_at_arrival(transition.graph.feature_batch, target_id, player, coarse_delay);
    const auto candidates = amount_bin_ship_counts(available, surplus, minimum_to_capture);

    std::vector<double> amount_logit_values(kAmountBins, 0.0);
    std::vector<bool> amount_mask(kAmountBins, false);
    bool any_amount = false;
    for (int k = 0; k < kAmountBins; ++k) {
      amount_logit_values[k] =
          amount_logits
              .index(std::vector<TensorIndex>{batch_idx, src_idx, target_idx, k})
              .item<double>();
      amount_mask[k] =
          candidates[k] > 0 && (k != 0 || minimum_to_capture <= available) &&
          std::isfinite(amount_logit_values[k]);
      any_amount = any_amount || amount_mask[k];
    }
    if (!any_amount) {
      amount_mask[kAmountBins - 1] = true;
    }
    const int amount_idx = sample_categorical(amount_logit_values, amount_mask, episode.rng);
    const double amount_lse = logsumexp(amount_logit_values, amount_mask);
    transition.old_logprob += amount_logit_values[amount_idx] - amount_lse;
    transition.old_entropy += categorical_entropy(amount_logit_values, amount_mask);
    transition.action_terms += 1;
    stored_decision.amount_idx = amount_idx;
    for (int k = 0; k < kAmountBins; ++k) {
      stored_decision.amount_mask[k] = amount_mask[k];
    }

    const int ships = candidates[amount_idx];
    if (ships <= 0 || ships > available) {
      add_invalid(transition, "bad_amount");
      continue;
    }

    orbit_rl::ExactRoute route = episode.feature_engine.query_route(
        transition.graph.obs, source_id, target_id, ships, orbit_rl::kMaxRouteDelay);
    if (!route.reachable) {
      add_invalid(transition, route_invalid_reason(route.blocked_by));
      continue;
    }
    moves.push_back(orbit::Move{source_id, route.angle, ships});
  }
  return moves;
}

std::vector<orbit::Move> valid_random_moves(EpisodeState& episode, int player) {
  orbit::Observation obs = orbit_obs_from_state(episode.state, player);
  std::vector<const orbit::Planet*> my_planets;
  std::vector<const orbit::Planet*> targets;
  for (const orbit::Planet& planet : episode.state.planets) {
    if (planet.owner == player && planet.ships >= kRandomMinFleetSize) {
      my_planets.push_back(&planet);
    } else if (planet.owner != player) {
      targets.push_back(&planet);
    }
  }
  if (my_planets.empty() || targets.empty()) {
    return {};
  }
  episode.random_route_engine.initialize(obs);
  std::shuffle(my_planets.begin(), my_planets.end(), episode.rng);
  std::vector<orbit::Move> moves;
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (const orbit::Planet* src : my_planets) {
    const double usable = std::max(0.0, static_cast<double>(src->ships - kRandomMinFleetSize));
    const double max_usable =
        std::max(1.0, static_cast<double>(kRandomLaunchProbMaxShips - kRandomMinFleetSize));
    const double raw = 1.0 - std::exp(-usable / kRandomLaunchProbDecayShips);
    const double normalizer = 1.0 - std::exp(-max_usable / kRandomLaunchProbDecayShips);
    const double launch_prob =
        std::clamp(kRandomMaxLaunchProb * raw / std::max(1e-9, normalizer), 0.0,
                   kRandomMaxLaunchProb);
    if (unit(episode.rng) >= launch_prob) {
      continue;
    }
    for (int attempt = 0; attempt < kRandomRouteAttemptsPerSource; ++attempt) {
      std::uniform_int_distribution<int> target_dist(0, static_cast<int>(targets.size()) - 1);
      std::uniform_int_distribution<int> ship_dist(kRandomMinFleetSize, src->ships);
      const orbit::Planet* target = targets[target_dist(episode.rng)];
      const int ships = ship_dist(episode.rng);
      orbit::RouteResult route =
          episode.random_route_engine.query_route(src->id, target->id, ships, episode.state.step);
      if (route.reachable) {
        moves.push_back(orbit::Move{src->id, route.angle, ships});
        break;
      }
    }
  }
  return moves;
}

std::vector<orbit::Move> opponent_moves(EpisodeState& episode, const OpponentSpec& spec,
                                        ModelHandle* model,
                                        const std::string& delay_cache_dir,
                                        double random_v2_prob) {
  if (spec.name == "random_v2") {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (unit(episode.rng) < random_v2_prob) {
      return valid_random_moves(episode, 1);
    }
    orbit::Observation obs = orbit_obs_from_state(episode.state, 1);
    return episode.random_route_engine.search_v2(obs, 100).moves;
  }
  if (model == nullptr) {
    return {};
  }
  TransitionData transition;
  transition.graph = build_graph(episode, 1, delay_cache_dir);
  std::vector<GraphData*> graph_ptrs{&transition.graph};
  auto out = forward_model(model->module, graph_ptrs);
  return sample_from_output(episode, transition, out, 0);
}

double event_reward(const orbit::SimState& before, const orbit::SimState& after, int player) {
  std::unordered_map<int, std::pair<int, int>> before_owner;
  for (const orbit::Planet& planet : before.planets) {
    before_owner[planet.id] = {planet.owner, planet.production};
  }
  double reward = 0.0;
  for (const orbit::Planet& planet : after.planets) {
    const int prev_owner =
        before_owner.count(planet.id) > 0 ? before_owner[planet.id].first : -999;
    if (prev_owner != player && planet.owner == player) {
      reward += prev_owner == orbit::kNeutralOwner ? kCaptureNeutralReward : kCaptureEnemyReward;
      reward += kCaptureProductionReward * planet.production;
    } else if (prev_owner == player && planet.owner != player) {
      reward += kLosePlanetReward + kLoseProductionReward * planet.production;
    }
  }
  return reward;
}

std::vector<int> scores(const orbit::SimState& state, int players) {
  std::vector<int> out(players, 0);
  for (const orbit::Planet& planet : state.planets) {
    if (planet.owner >= 0 && planet.owner < players) {
      out[planet.owner] += planet.ships;
    }
  }
  for (const orbit::Fleet& fleet : state.fleets) {
    if (fleet.owner >= 0 && fleet.owner < players) {
      out[fleet.owner] += fleet.ships;
    }
  }
  return out;
}

int alive_count(const orbit::SimState& state) {
  std::set<int> alive;
  for (const orbit::Planet& planet : state.planets) {
    if (planet.owner >= 0) {
      alive.insert(planet.owner);
    }
  }
  for (const orbit::Fleet& fleet : state.fleets) {
    if (fleet.owner >= 0) {
      alive.insert(fleet.owner);
    }
  }
  return static_cast<int>(alive.size());
}

int dominant_player(const orbit::SimState& state, double ship_share, double production_share) {
  std::unordered_map<int, int> ships;
  std::unordered_map<int, int> production;
  for (const orbit::Planet& planet : state.planets) {
    if (planet.owner >= 0) {
      ships[planet.owner] += planet.ships;
      production[planet.owner] += planet.production;
    }
  }
  for (const orbit::Fleet& fleet : state.fleets) {
    if (fleet.owner >= 0) {
      ships[fleet.owner] += fleet.ships;
    }
  }
  int total_ships = 0;
  int total_production = 0;
  for (const auto& item : ships) {
    total_ships += item.second;
  }
  for (const auto& item : production) {
    total_production += item.second;
  }
  if (total_ships <= 0 || total_production <= 0) {
    return -1;
  }
  for (const auto& item : ships) {
    if (static_cast<double>(item.second) / total_ships < ship_share) {
      continue;
    }
    if (static_cast<double>(production[item.first]) / total_production >= production_share) {
      return item.first;
    }
  }
  return -1;
}

double terminal_reward(double outcome) {
  if (outcome > 0.0) {
    return kTerminalWinReward;
  }
  if (outcome < 0.0) {
    return kTerminalLossReward;
  }
  return kTerminalDrawReward;
}

double quick_win_reward(int steps_played, int max_steps) {
  if (max_steps <= 0) {
    return 0.0;
  }
  const double remaining =
      std::clamp(static_cast<double>(max_steps - steps_played) / max_steps, 0.0, 1.0);
  return kQuickWinRewardScale * remaining;
}

py::dict pack_rollout(const std::vector<EpisodeState>& episodes, int feature_calls,
                      int delay_cache_hits, double feature_ms) {
  std::vector<const TransitionData*> flat;
  std::vector<int64_t> episode_offsets;
  std::vector<int64_t> episode_lengths;
  std::vector<float> final_rewards;
  episode_offsets.push_back(0);
  int max_n = 1;
  std::map<std::string, int> invalid_counts;
  for (const EpisodeState& episode : episodes) {
    for (const TransitionData& transition : episode.transitions) {
      flat.push_back(&transition);
      max_n = std::max(max_n, static_cast<int>(transition.graph.planet_ids.size()));
      for (const auto& item : transition.invalid_counts) {
        invalid_counts[item.first] += item.second;
      }
    }
    episode_offsets.push_back(static_cast<int64_t>(flat.size()));
    episode_lengths.push_back(episode.length);
    final_rewards.push_back(static_cast<float>(episode.outcome));
  }

  const int64_t t = static_cast<int64_t>(flat.size());
  torch::Tensor planet_features = torch::zeros({t, max_n, kPlanetDim}, torch::kFloat32);
  torch::Tensor edge_features = torch::zeros({t, max_n, max_n, kEdgeDim}, torch::kFloat32);
  torch::Tensor planet_mask = torch::zeros({t, max_n}, torch::kBool);
  torch::Tensor old_logprob = torch::zeros({t}, torch::kFloat32);
  torch::Tensor old_entropy = torch::zeros({t}, torch::kFloat32);
  torch::Tensor value = torch::zeros({t}, torch::kFloat32);
  torch::Tensor reward = torch::zeros({t}, torch::kFloat32);
  torch::Tensor done = torch::zeros({t}, torch::kBool);
  torch::Tensor action_terms = torch::zeros({t}, torch::kInt32);

  std::vector<int64_t> decision_offsets;
  std::vector<int32_t> source_idx;
  std::vector<int32_t> stop_action;
  std::vector<int32_t> target_idx;
  std::vector<int32_t> amount_idx;
  std::vector<uint8_t> amount_masks;
  decision_offsets.push_back(0);

  for (int64_t idx = 0; idx < t; ++idx) {
    const TransitionData& transition = *flat[static_cast<std::size_t>(idx)];
    const int n = static_cast<int>(transition.graph.planet_ids.size());
    std::vector<TensorIndex> planet_indices{idx, Slice(0, n), Slice()};
    planet_features.index(planet_indices).copy_(transition.graph.planet_features);
    std::vector<TensorIndex> edge_indices{idx, Slice(0, n), Slice(0, n), Slice()};
    edge_features.index(edge_indices).copy_(transition.graph.edge_features);
    std::vector<TensorIndex> mask_indices{idx, Slice(0, n)};
    planet_mask.index(mask_indices).copy_(transition.graph.planet_mask);
    std::vector<TensorIndex> scalar_index{idx};
    old_logprob.index(scalar_index).fill_(static_cast<float>(transition.old_logprob));
    old_entropy.index(scalar_index).fill_(static_cast<float>(transition.old_entropy));
    value.index(scalar_index).fill_(static_cast<float>(transition.value));
    reward.index(scalar_index).fill_(static_cast<float>(transition.reward));
    done.index(scalar_index).fill_(transition.done);
    action_terms.index(scalar_index).fill_(transition.action_terms);
    for (const DecisionData& decision : transition.decisions) {
      source_idx.push_back(decision.source_idx);
      stop_action.push_back(decision.stop_action);
      target_idx.push_back(decision.target_idx);
      amount_idx.push_back(decision.amount_idx);
      for (bool v : decision.amount_mask) {
        amount_masks.push_back(v ? 1 : 0);
      }
    }
    decision_offsets.push_back(static_cast<int64_t>(source_idx.size()));
  }

  const int64_t d = static_cast<int64_t>(source_idx.size());
  py::dict out;
  out["planet_features"] = planet_features;
  out["edge_features"] = edge_features;
  out["planet_mask"] = planet_mask;
  out["old_logprob"] = old_logprob;
  out["old_entropy"] = old_entropy;
  out["value"] = value;
  out["reward"] = reward;
  out["done"] = done;
  out["action_terms"] = action_terms;
  out["decision_offsets"] =
      torch::from_blob(decision_offsets.data(), {static_cast<int64_t>(decision_offsets.size())},
                       torch::kInt64)
          .clone();
  out["source_idx"] = torch::from_blob(source_idx.data(), {d}, torch::kInt32).clone();
  out["stop_action"] = torch::from_blob(stop_action.data(), {d}, torch::kInt32).clone();
  out["target_idx"] = torch::from_blob(target_idx.data(), {d}, torch::kInt32).clone();
  out["amount_idx"] = torch::from_blob(amount_idx.data(), {d}, torch::kInt32).clone();
  out["amount_mask"] =
      torch::from_blob(amount_masks.data(), {d, kAmountBins}, torch::kUInt8).clone().to(torch::kBool);
  out["episode_offsets"] =
      torch::from_blob(episode_offsets.data(), {static_cast<int64_t>(episode_offsets.size())},
                       torch::kInt64)
          .clone();
  out["episode_lengths"] =
      torch::from_blob(episode_lengths.data(), {static_cast<int64_t>(episode_lengths.size())},
                       torch::kInt64)
          .clone();
  out["final_rewards"] =
      torch::from_blob(final_rewards.data(), {static_cast<int64_t>(final_rewards.size())},
                       torch::kFloat32)
          .clone();
  py::dict invalid;
  for (const auto& item : invalid_counts) {
    invalid[py::str(item.first)] = item.second;
  }
  py::dict stats;
  stats["feature_calls"] = feature_calls;
  stats["delay_cache_hits"] = delay_cache_hits;
  stats["feature_ms"] = feature_calls > 0 ? feature_ms / feature_calls : 0.0;
  out["invalid_counts"] = invalid;
  out["stats"] = stats;
  return out;
}

py::dict collect_native_rollout(const py::list& initial_states, const py::list& spawn_events,
                                const py::list& opponents, const std::string& learner_model_path,
                                double random_v2_prob, int max_steps, int seed,
                                double early_ship_share, double early_production_share,
                                const std::string& delay_cache_dir, int torch_threads,
                                int worker_threads) {
  const auto collect_start = Clock::now();
  torch::NoGradGuard no_grad;
  torch::set_num_threads(std::max(1, torch_threads));
  worker_threads = std::max(1, worker_threads);

  double model_load_ms = 0.0;
  double init_ms = 0.0;
  double graph_wall_ms = 0.0;
  double learner_forward_ms = 0.0;
  double learner_sample_ms = 0.0;
  double opponent_ms = 0.0;
  double opponent_graph_ms = 0.0;
  double opponent_forward_ms = 0.0;
  double opponent_sample_ms = 0.0;
  double opponent_random_ms = 0.0;
  double sim_step_ms = 0.0;
  double pack_ms = 0.0;

  auto phase_start = Clock::now();
  ModelHandle learner;
  learner.path = learner_model_path;
  learner.module = torch::jit::load(learner_model_path, torch::kCPU);
  learner.module.eval();

  std::vector<OpponentSpec> opponent_specs;
  std::unordered_map<std::string, ModelHandle> model_handles;
  for (const py::handle& item : opponents) {
    py::dict raw = py::reinterpret_borrow<py::dict>(item);
    OpponentSpec spec;
    spec.name = raw["name"].cast<std::string>();
    spec.population_index = raw.contains("population_index") && !raw["population_index"].is_none()
                                ? raw["population_index"].cast<int>()
                                : -1;
    if (raw.contains("model_path") && !raw["model_path"].is_none()) {
      spec.model_path = raw["model_path"].cast<std::string>();
      if (!spec.model_path.empty() && model_handles.count(spec.model_path) == 0) {
        ModelHandle handle;
        handle.path = spec.model_path;
        handle.module = torch::jit::load(spec.model_path, torch::kCPU);
        handle.module.eval();
        model_handles.emplace(spec.model_path, std::move(handle));
      }
    }
    opponent_specs.push_back(std::move(spec));
  }
  model_load_ms += elapsed_ms(phase_start);

  phase_start = Clock::now();
  std::vector<EpisodeState> episodes;
  episodes.reserve(initial_states.size());
  for (std::size_t idx = 0; idx < initial_states.size(); ++idx) {
    py::object initial = py::reinterpret_borrow<py::object>(initial_states[idx]);
    EpisodeState episode;
    episode.seed = get_attr_or_item<int>(initial, "seed", seed + static_cast<int>(idx));
    episode.state = state_from_py(initial);
    episode.sim_engine.initialize(orbit_obs_from_state(episode.state, 0));
    episode.random_route_engine.initialize(orbit_obs_from_state(episode.state, 1));
    episode.feature_engine.initialize(rl_obs_from_state(episode.state, 0));
    episode.rng.seed(static_cast<std::uint32_t>(seed * 1000003 + idx * 9176 + episode.seed));
    if (idx < spawn_events.size()) {
      py::object raw_events = py::reinterpret_borrow<py::object>(spawn_events[idx]);
      for (const py::handle& raw_event : raw_events) {
        episode.spawns.push_back(spawn_event_from_py(py::reinterpret_borrow<py::object>(raw_event)));
      }
      std::sort(episode.spawns.begin(), episode.spawns.end(),
                [](const SpawnEvent& a, const SpawnEvent& b) { return a.step < b.step; });
    }
    episodes.push_back(std::move(episode));
  }
  init_ms += elapsed_ms(phase_start);

  int feature_calls = 0;
  int delay_cache_hits = 0;
  double feature_ms = 0.0;

  for (int step = 0; step < max_steps; ++step) {
    std::vector<int> active;
    for (std::size_t idx = 0; idx < episodes.size(); ++idx) {
      if (!episodes[idx].done) {
        active.push_back(static_cast<int>(idx));
      }
    }
    if (active.empty()) {
      break;
    }

    std::vector<TransitionData> pending(active.size());
    std::vector<GraphData*> learner_graphs;
    learner_graphs.reserve(active.size());
    phase_start = Clock::now();
    parallel_for(active.size(), worker_threads, [&](std::size_t local) {
      EpisodeState& episode = episodes[active[local]];
      pending[local].graph = build_graph(episode, 0, delay_cache_dir);
      pending[local].reward += kStepReward;
    });
    for (std::size_t local = 0; local < active.size(); ++local) {
      feature_calls += 1;
      delay_cache_hits += pending[local].graph.delay_cache_hit;
      feature_ms += pending[local].graph.feature_batch.stats.elapsed_ms;
      learner_graphs.push_back(&pending[local].graph);
    }
    graph_wall_ms += elapsed_ms(phase_start);

    phase_start = Clock::now();
    auto learner_out = forward_model_chunked(learner.module, learner_graphs);
    learner_forward_ms += elapsed_ms(phase_start);
    std::vector<std::vector<orbit::Move>> learner_moves(active.size());
    phase_start = Clock::now();
    parallel_for(active.size(), worker_threads, [&](std::size_t local) {
      learner_moves[local] =
          sample_from_output(episodes[active[local]], pending[local], learner_out,
                             static_cast<int>(local));
    });
    learner_sample_ms += elapsed_ms(phase_start);

    std::vector<std::vector<orbit::Move>> opp_moves(active.size());
    std::vector<TransitionData> opponent_pending(active.size());
    std::vector<int> model_opponent_locals;
    std::map<std::string, std::vector<int>> model_locals_by_path;
    for (std::size_t local = 0; local < active.size(); ++local) {
      const OpponentSpec& spec = opponent_specs[active[local]];
      if (spec.model_path.empty()) {
        continue;
      }
      model_opponent_locals.push_back(static_cast<int>(local));
      model_locals_by_path[spec.model_path].push_back(static_cast<int>(local));
    }

    phase_start = Clock::now();
    parallel_for(active.size(), worker_threads, [&](std::size_t local) {
      EpisodeState& episode = episodes[active[local]];
      const OpponentSpec& spec = opponent_specs[active[local]];
      if (spec.name != "random_v2") {
        return;
      }
      opp_moves[local] = opponent_moves(episode, spec, nullptr, delay_cache_dir, random_v2_prob);
    });
    opponent_random_ms += elapsed_ms(phase_start);

    phase_start = Clock::now();
    parallel_for(model_opponent_locals.size(), worker_threads, [&](std::size_t idx) {
      const int local = model_opponent_locals[idx];
      EpisodeState& episode = episodes[active[local]];
      opponent_pending[local].graph = build_graph(episode, 1, delay_cache_dir);
    });
    opponent_graph_ms += elapsed_ms(phase_start);

    for (const auto& group : model_locals_by_path) {
      phase_start = Clock::now();
      std::vector<GraphData*> opponent_graphs;
      opponent_graphs.reserve(group.second.size());
      for (int local : group.second) {
        opponent_graphs.push_back(&opponent_pending[local].graph);
      }
      auto opponent_out =
          forward_model_chunked(model_handles.at(group.first).module, opponent_graphs);
      opponent_forward_ms += elapsed_ms(phase_start);

      phase_start = Clock::now();
      parallel_for(group.second.size(), worker_threads, [&](std::size_t idx) {
        const int local = group.second[idx];
        opp_moves[local] =
            sample_from_output(episodes[active[local]], opponent_pending[local],
                               opponent_out, static_cast<int>(idx));
      });
      opponent_sample_ms += elapsed_ms(phase_start);
    }

    phase_start = Clock::now();
    parallel_for(active.size(), worker_threads, [&](std::size_t local) {
      EpisodeState& episode = episodes[active[local]];
      for (const SpawnEvent& spawn : episode.spawns) {
        if (episode.state.step + 1 == spawn.step) {
          apply_spawn_event(episode.state, spawn);
          episode.sim_engine.initialize(orbit_obs_from_state(episode.state, 0));
          episode.feature_engine.initialize(rl_obs_from_state(episode.state, 0));
          break;
        }
      }

      orbit::SimState before = episode.state;
      std::vector<std::vector<orbit::Move>> actions{learner_moves[local], opp_moves[local]};
      episode.state = episode.sim_engine.simulate_step(episode.state, actions);
      episode.length = step + 1;
      pending[local].reward += event_reward(before, episode.state, 0);

      bool is_done = episode.state.step >= kMaxEpisodeSteps - 1 || alive_count(episode.state) <= 1;
      int early_winner = -1;
      if (!is_done) {
        early_winner =
            dominant_player(episode.state, early_ship_share, early_production_share);
        is_done = early_winner >= 0;
      }
      if (is_done) {
        if (early_winner >= 0) {
          episode.outcome = early_winner == 0 ? 1.0 : -1.0;
          pending[local].reward +=
              early_winner == 0 ? kTerminalWinReward : kTerminalLossReward;
          if (early_winner == 0) {
            pending[local].reward += quick_win_reward(step + 1, max_steps);
          }
        } else {
          std::vector<int> score_values = scores(episode.state, 2);
          const int max_score = *std::max_element(score_values.begin(), score_values.end());
          episode.outcome = max_score > 0 && score_values[0] == max_score ? 1.0 : -1.0;
          pending[local].reward += terminal_reward(episode.outcome);
          if (episode.outcome > 0.0) {
            pending[local].reward += quick_win_reward(step + 1, max_steps);
          }
        }
      }
      pending[local].done = is_done;
      episode.done = is_done;
      episode.transitions.push_back(std::move(pending[local]));
    });
    sim_step_ms += elapsed_ms(phase_start);
  }

  for (EpisodeState& episode : episodes) {
    if (!episode.transitions.empty() && !episode.transitions.back().done) {
      episode.transitions.back().reward += kTimeoutReward;
      episode.outcome = 0.0;
    }
  }

  phase_start = Clock::now();
  py::dict out = pack_rollout(episodes, feature_calls, delay_cache_hits, feature_ms);
  pack_ms = elapsed_ms(phase_start);
  opponent_ms =
      opponent_random_ms + opponent_graph_ms + opponent_forward_ms + opponent_sample_ms;
  py::dict stats = py::reinterpret_borrow<py::dict>(out["stats"]);
  stats["model_load_ms"] = model_load_ms;
  stats["init_ms"] = init_ms;
  stats["graph_wall_ms"] = graph_wall_ms;
  stats["learner_forward_ms"] = learner_forward_ms;
  stats["learner_sample_ms"] = learner_sample_ms;
  stats["opponent_ms"] = opponent_ms;
  stats["opponent_graph_ms"] = opponent_graph_ms;
  stats["opponent_forward_ms"] = opponent_forward_ms;
  stats["opponent_sample_ms"] = opponent_sample_ms;
  stats["opponent_random_ms"] = opponent_random_ms;
  stats["sim_step_ms"] = sim_step_ms;
  stats["pack_ms"] = pack_ms;
  stats["collect_total_ms"] = elapsed_ms(collect_start);
  stats["torch_threads"] = std::max(1, torch_threads);
  stats["worker_threads"] = worker_threads;
  return out;
}

}  // namespace

PYBIND11_MODULE(orbit_rollout_native, m) {
  m.doc() = "Native C++ TorchScript rollout collector for Orbit Wars PPO";
  m.def("collect_rollout", &collect_native_rollout, py::arg("initial_states"),
        py::arg("spawn_events"), py::arg("opponents"), py::arg("learner_model_path"),
        py::arg("random_v2_prob"), py::arg("max_steps"), py::arg("seed"),
        py::arg("early_ship_share"), py::arg("early_production_share"),
        py::arg("delay_cache_dir"), py::arg("torch_threads") = 4,
        py::arg("worker_threads") = 1);
}
