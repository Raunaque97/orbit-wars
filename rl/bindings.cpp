#include "rl_features.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>

namespace py = pybind11;

namespace {

orbit_rl::Planet planet_from_sequence(const py::handle& item) {
  py::sequence seq = py::reinterpret_borrow<py::sequence>(item);
  orbit_rl::Planet planet;
  planet.id = seq[0].cast<int>();
  planet.owner = seq[1].cast<int>();
  planet.x = seq[2].cast<double>();
  planet.y = seq[3].cast<double>();
  planet.radius = seq[4].cast<double>();
  planet.ships = seq[5].cast<int>();
  planet.production = seq[6].cast<int>();
  return planet;
}

orbit_rl::Fleet fleet_from_sequence(const py::handle& item) {
  py::sequence seq = py::reinterpret_borrow<py::sequence>(item);
  orbit_rl::Fleet fleet;
  fleet.id = seq[0].cast<int>();
  fleet.owner = seq[1].cast<int>();
  fleet.x = seq[2].cast<double>();
  fleet.y = seq[3].cast<double>();
  fleet.angle = seq[4].cast<double>();
  fleet.from_planet_id = seq[5].cast<int>();
  fleet.ships = seq[6].cast<int>();
  return fleet;
}

std::vector<orbit_rl::Planet> planets_from_object(const py::object& obj) {
  std::vector<orbit_rl::Planet> planets;
  if (obj.is_none()) {
    return planets;
  }
  for (const py::handle& item : obj) {
    planets.push_back(planet_from_sequence(item));
  }
  return planets;
}

std::vector<orbit_rl::Fleet> fleets_from_object(const py::object& obj) {
  std::vector<orbit_rl::Fleet> fleets;
  if (obj.is_none()) {
    return fleets;
  }
  for (const py::handle& item : obj) {
    fleets.push_back(fleet_from_sequence(item));
  }
  return fleets;
}

template <typename T>
T get_attr_or_item(const py::object& obs, const char* name, T fallback) {
  if (py::isinstance<py::dict>(obs)) {
    py::dict dict = obs.cast<py::dict>();
    py::object value = dict.attr("get")(name, py::cast(fallback));
    if (value.is_none()) {
      return fallback;
    }
    return value.cast<T>();
  }
  if (py::hasattr(obs, "get")) {
    py::object value = obs.attr("get")(name, py::cast(fallback));
    if (value.is_none()) {
      return fallback;
    }
    return value.cast<T>();
  }
  if (py::hasattr(obs, name)) {
    py::object value = obs.attr(name);
    if (value.is_none()) {
      return fallback;
    }
    return value.cast<T>();
  }
  return fallback;
}

py::object get_object_attr_or_item(const py::object& obs, const char* name) {
  if (py::isinstance<py::dict>(obs)) {
    py::dict dict = obs.cast<py::dict>();
    return dict.attr("get")(name, py::none());
  }
  if (py::hasattr(obs, "get")) {
    return obs.attr("get")(name, py::none());
  }
  if (py::hasattr(obs, name)) {
    return obs.attr(name);
  }
  return py::none();
}

orbit_rl::Observation observation_from_py(const py::object& obs) {
  orbit_rl::Observation out;
  out.player = get_attr_or_item<int>(obs, "player", 0);
  out.step = get_attr_or_item<int>(obs, "step", 0);
  out.angular_velocity = get_attr_or_item<double>(obs, "angular_velocity", 0.0);
  out.planets = planets_from_object(get_object_attr_or_item(obs, "planets"));
  out.initial_planets = planets_from_object(get_object_attr_or_item(obs, "initial_planets"));
  out.fleets = fleets_from_object(get_object_attr_or_item(obs, "fleets"));

  py::object comet_ids = get_object_attr_or_item(obs, "comet_planet_ids");
  if (!comet_ids.is_none()) {
    for (const py::handle& item : comet_ids) {
      out.comet_planet_ids.push_back(item.cast<int>());
    }
  }

  py::object comets = get_object_attr_or_item(obs, "comets");
  if (!comets.is_none()) {
    for (const py::handle& comet_item : comets) {
      py::object comet = py::reinterpret_borrow<py::object>(comet_item);
      orbit_rl::CometGroup group;
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
          std::vector<orbit_rl::Vec2> path;
          for (const py::handle& point_item : path_item) {
            py::sequence point = py::reinterpret_borrow<py::sequence>(point_item);
            path.push_back(
                orbit_rl::Vec2{point[0].cast<double>(), point[1].cast<double>()});
          }
          group.paths.push_back(path);
        }
      }
      out.comets.push_back(group);
    }
  }

  if (out.initial_planets.empty()) {
    out.initial_planets = out.planets;
  }
  return out;
}

py::dict stats_to_dict(const orbit_rl::FeatureStats& stats) {
  py::dict out;
  out["elapsed_ms"] = stats.elapsed_ms;
  out["planets"] = stats.planets;
  out["fleets"] = stats.fleets;
  out["horizon"] = stats.horizon;
  out["route_queries"] = stats.route_queries;
  out["route_proxy_simulations"] = stats.route_proxy_simulations;
  out["route_sim_ticks"] = stats.route_sim_ticks;
  out["predicted_arrivals"] = stats.predicted_arrivals;
  out["blocked_routes"] = stats.blocked_routes;
  return out;
}

py::dict batch_to_dict(const orbit_rl::FeatureBatch& batch) {
  const py::ssize_t planets = static_cast<py::ssize_t>(batch.planet_ids.size());
  const py::ssize_t horizon = static_cast<py::ssize_t>(batch.stats.horizon);
  const py::ssize_t buckets = static_cast<py::ssize_t>(batch.ship_buckets.size());

  py::array_t<int> garrisons({planets, horizon, py::ssize_t{2}});
  std::memcpy(garrisons.mutable_data(), batch.garrison_flat.data(),
              batch.garrison_flat.size() * sizeof(int));

  py::array_t<int> delays({buckets, planets, planets});
  std::memcpy(delays.mutable_data(), batch.delay_flat.data(),
              batch.delay_flat.size() * sizeof(int));

  py::array_t<double> angles({buckets, planets, planets});
  std::memcpy(angles.mutable_data(), batch.angle_flat.data(),
              batch.angle_flat.size() * sizeof(double));

  py::dict out;
  out["planet_ids"] = batch.planet_ids;
  out["ship_buckets"] = batch.ship_buckets;
  out["garrisons"] = garrisons;
  out["delays"] = delays;
  out["angles"] = angles;
  out["stats"] = stats_to_dict(batch.stats);
  return out;
}

std::unique_ptr<orbit_rl::FeatureEngine> global_engine =
    std::make_unique<orbit_rl::FeatureEngine>();

}  // namespace

PYBIND11_MODULE(orbit_rl_native, m) {
  m.doc() = "Native C++ RL feature builder for Orbit Wars";

  py::class_<orbit_rl::FeatureEngine>(m, "FeatureEngine")
      .def(py::init<>())
      .def("initialize", [](orbit_rl::FeatureEngine& engine, const py::object& obs) {
        engine.initialize(observation_from_py(obs));
      })
      .def("compute",
           [](orbit_rl::FeatureEngine& engine, const py::object& obs, int horizon,
              int max_route_delay) {
             return batch_to_dict(
                 engine.compute(observation_from_py(obs), horizon, max_route_delay));
           },
           py::arg("obs"), py::arg("horizon") = 50,
           py::arg("max_route_delay") = orbit_rl::kMaxRouteDelay)
      .def("initialized", &orbit_rl::FeatureEngine::initialized);

  m.def("initialize", [](const py::object& obs) {
    global_engine->initialize(observation_from_py(obs));
  });
  m.def("compute",
        [](const py::object& obs, int horizon, int max_route_delay) {
          return batch_to_dict(
              global_engine->compute(observation_from_py(obs), horizon, max_route_delay));
        },
        py::arg("obs"), py::arg("horizon") = 50,
        py::arg("max_route_delay") = orbit_rl::kMaxRouteDelay);
  m.def("fleet_speed", &orbit_rl::fleet_speed);
}
