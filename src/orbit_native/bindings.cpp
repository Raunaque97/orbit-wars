#include "orbit_core.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

namespace py = pybind11;

namespace {

orbit::Planet planet_from_sequence(const py::handle& item) {
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

orbit::Fleet fleet_from_sequence(const py::handle& item) {
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

std::vector<orbit::Planet> planets_from_object(const py::object& obj) {
  std::vector<orbit::Planet> planets;
  if (obj.is_none()) {
    return planets;
  }
  for (const py::handle& item : obj) {
    planets.push_back(planet_from_sequence(item));
  }
  return planets;
}

std::vector<orbit::Fleet> fleets_from_object(const py::object& obj) {
  std::vector<orbit::Fleet> fleets;
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

orbit::Observation observation_from_py(const py::object& obs) {
  orbit::Observation out;
  out.player = get_attr_or_item<int>(obs, "player", 0);
  out.step = get_attr_or_item<int>(obs, "step", 0);
  out.time_budget_ms = get_attr_or_item<int>(obs, "time_budget_ms", 950);
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

py::dict route_to_dict(const orbit::RouteResult& route) {
  py::dict d;
  d["reachable"] = route.reachable;
  d["angle"] = route.angle;
  d["arrival_tick"] = route.arrival_tick;
  d["travel_time"] = route.travel_time;
  d["blocked_by"] = route.blocked_by;
  d["hit_x"] = route.hit_x;
  d["hit_y"] = route.hit_y;
  return d;
}

py::list moves_to_py(const std::vector<orbit::Move>& moves) {
  py::list out;
  for (const orbit::Move& move : moves) {
    py::list item;
    item.append(move.from_planet_id);
    item.append(move.angle);
    item.append(move.ships);
    out.append(item);
  }
  return out;
}

orbit::Move move_from_sequence(const py::handle& item) {
  py::sequence seq = py::reinterpret_borrow<py::sequence>(item);
  orbit::Move move;
  move.from_planet_id = seq[0].cast<int>();
  move.angle = seq[1].cast<double>();
  move.ships = seq[2].cast<int>();
  return move;
}

std::vector<std::vector<orbit::Move>> actions_from_py(const py::object& obj) {
  std::vector<std::vector<orbit::Move>> actions;
  if (obj.is_none()) {
    return actions;
  }
  for (const py::handle& player_actions : obj) {
    std::vector<orbit::Move> moves;
    for (const py::handle& move : player_actions) {
      moves.push_back(move_from_sequence(move));
    }
    actions.push_back(moves);
  }
  return actions;
}

orbit::SimState state_from_observation(const orbit::Observation& obs) {
  orbit::SimState state;
  state.step = obs.step;
  state.angular_velocity = obs.angular_velocity;
  state.planets = obs.planets;
  state.initial_planets = obs.initial_planets.empty() ? obs.planets : obs.initial_planets;
  state.fleets = obs.fleets;
  state.comet_planet_ids = obs.comet_planet_ids;
  state.comets = obs.comets;
  for (const orbit::Fleet& fleet : obs.fleets) {
    state.next_fleet_id = std::max(state.next_fleet_id, fleet.id + 1);
  }
  return state;
}

py::list planets_to_py(const std::vector<orbit::Planet>& planets) {
  py::list out;
  for (const orbit::Planet& planet : planets) {
    py::list item;
    item.append(planet.id);
    item.append(planet.owner);
    item.append(planet.x);
    item.append(planet.y);
    item.append(planet.radius);
    item.append(planet.ships);
    item.append(planet.production);
    out.append(item);
  }
  return out;
}

py::list fleets_to_py(const std::vector<orbit::Fleet>& fleets) {
  py::list out;
  for (const orbit::Fleet& fleet : fleets) {
    py::list item;
    item.append(fleet.id);
    item.append(fleet.owner);
    item.append(fleet.x);
    item.append(fleet.y);
    item.append(fleet.angle);
    item.append(fleet.from_planet_id);
    item.append(fleet.ships);
    out.append(item);
  }
  return out;
}

py::list forecast_table_to_py(const std::vector<std::vector<orbit::Planet>>& table) {
  py::list out;
  for (const std::vector<orbit::Planet>& row : table) {
    out.append(planets_to_py(row));
  }
  return out;
}

py::dict state_to_dict(const orbit::SimState& state) {
  py::dict out;
  out["step"] = state.step;
  out["angular_velocity"] = state.angular_velocity;
  out["planets"] = planets_to_py(state.planets);
  out["initial_planets"] = planets_to_py(state.initial_planets);
  out["fleets"] = fleets_to_py(state.fleets);
  return out;
}

py::dict stats_to_dict(const orbit::SearchStats& stats) {
  py::dict out;
  out["states_considered"] = stats.states_considered;
  out["route_queries"] = stats.route_queries;
  out["candidates_generated"] = stats.candidates_generated;
  out["elapsed_ms"] = stats.elapsed_ms;
  out["states_per_second"] = stats.states_per_second;
  out["timed_out"] = stats.timed_out;
  return out;
}

py::dict search_result_to_dict(const orbit::SearchResult& result) {
  py::dict out;
  out["moves"] = moves_to_py(result.moves);
  out["stats"] = stats_to_dict(result.stats);
  return out;
}

std::unique_ptr<orbit::Engine> global_engine = std::make_unique<orbit::Engine>();

}  // namespace

PYBIND11_MODULE(orbit_native, m) {
  m.doc() = "Native C++ Orbit Wars simulator, route query engine, and agent";

  py::class_<orbit::Vec2>(m, "Vec2")
      .def(py::init<>())
      .def_readwrite("x", &orbit::Vec2::x)
      .def_readwrite("y", &orbit::Vec2::y);

  py::class_<orbit::Planet>(m, "Planet")
      .def(py::init<>())
      .def_readwrite("id", &orbit::Planet::id)
      .def_readwrite("owner", &orbit::Planet::owner)
      .def_readwrite("x", &orbit::Planet::x)
      .def_readwrite("y", &orbit::Planet::y)
      .def_readwrite("radius", &orbit::Planet::radius)
      .def_readwrite("ships", &orbit::Planet::ships)
      .def_readwrite("production", &orbit::Planet::production);

  py::class_<orbit::Fleet>(m, "Fleet")
      .def(py::init<>())
      .def_readwrite("id", &orbit::Fleet::id)
      .def_readwrite("owner", &orbit::Fleet::owner)
      .def_readwrite("x", &orbit::Fleet::x)
      .def_readwrite("y", &orbit::Fleet::y)
      .def_readwrite("angle", &orbit::Fleet::angle)
      .def_readwrite("from_planet_id", &orbit::Fleet::from_planet_id)
      .def_readwrite("ships", &orbit::Fleet::ships);

  py::class_<orbit::Move>(m, "Move")
      .def(py::init<>())
      .def_readwrite("from_planet_id", &orbit::Move::from_planet_id)
      .def_readwrite("angle", &orbit::Move::angle)
      .def_readwrite("ships", &orbit::Move::ships);

  py::class_<orbit::Engine>(m, "Engine")
      .def(py::init<>())
      .def("initialize", [](orbit::Engine& engine, const py::object& obs) {
        engine.initialize(observation_from_py(obs));
      })
      .def("set_v2_risk_start_tick", &orbit::Engine::set_v2_risk_start_tick)
      .def("set_v2_risk_ramp", &orbit::Engine::set_v2_risk_ramp)
      .def("act", [](orbit::Engine& engine, const py::object& obs) {
        return moves_to_py(engine.act(observation_from_py(obs)));
      })
      .def("act_v2", [](orbit::Engine& engine, const py::object& obs) {
        return moves_to_py(engine.act_v2(observation_from_py(obs)));
      })
      .def("search",
           [](orbit::Engine& engine, const py::object& obs, int budget_ms) {
             return search_result_to_dict(engine.search(observation_from_py(obs), budget_ms));
           },
           py::arg("obs"), py::arg("budget_ms") = 950)
      .def("search_v2",
           [](orbit::Engine& engine, const py::object& obs, int budget_ms) {
             return search_result_to_dict(engine.search_v2(observation_from_py(obs), budget_ms));
           },
           py::arg("obs"), py::arg("budget_ms") = 950)
      .def("last_search_stats",
           [](orbit::Engine& engine) { return stats_to_dict(engine.last_search_stats()); })
      .def("query_route",
           [](orbit::Engine& engine, int src_id, int target_id, int ships, int step) {
             return route_to_dict(engine.query_route(src_id, target_id, ships, step));
           })
      .def("batch_query_routes",
           [](orbit::Engine& engine,
              const std::vector<std::tuple<int, int, int, int>>& requests) {
             py::list out;
             for (const orbit::RouteResult& route : engine.batch_query_routes(requests)) {
               out.append(route_to_dict(route));
             }
             return out;
           })
      .def("forecast_planets",
           [](orbit::Engine& engine, const py::object& obs, int horizon) {
             return forecast_table_to_py(
                 engine.forecast_planets(observation_from_py(obs), horizon));
           })
      .def("simulate_step",
           [](orbit::Engine& engine, const py::object& obs, const py::object& actions) {
             orbit::Observation parsed = observation_from_py(obs);
             orbit::SimState state = state_from_observation(parsed);
             return state_to_dict(engine.simulate_step(state, actions_from_py(actions)));
           });

  m.def("initialize", [](const py::object& obs) {
    global_engine->initialize(observation_from_py(obs));
  });
  m.def("act", [](const py::object& obs) {
    return moves_to_py(global_engine->act(observation_from_py(obs)));
  });
  m.def("act_v2", [](const py::object& obs) {
    return moves_to_py(global_engine->act_v2(observation_from_py(obs)));
  });
  m.def("search",
        [](const py::object& obs, int budget_ms) {
          return search_result_to_dict(global_engine->search(observation_from_py(obs), budget_ms));
        },
        py::arg("obs"), py::arg("budget_ms") = 950);
  m.def("search_v2",
        [](const py::object& obs, int budget_ms) {
          return search_result_to_dict(
              global_engine->search_v2(observation_from_py(obs), budget_ms));
        },
        py::arg("obs"), py::arg("budget_ms") = 950);
  m.def("last_search_stats",
        []() { return stats_to_dict(global_engine->last_search_stats()); });
  m.def("query_route", [](int src_id, int target_id, int ships, int step) {
    return route_to_dict(global_engine->query_route(src_id, target_id, ships, step));
  });
  m.def("batch_query_routes",
        [](const std::vector<std::tuple<int, int, int, int>>& requests) {
          py::list out;
          for (const orbit::RouteResult& route : global_engine->batch_query_routes(requests)) {
            out.append(route_to_dict(route));
          }
          return out;
        });
  m.def("forecast_planets", [](const py::object& obs, int horizon) {
    return forecast_table_to_py(
        global_engine->forecast_planets(observation_from_py(obs), horizon));
  });
  m.def("simulate_step", [](const py::object& obs, const py::object& actions) {
    orbit::Observation parsed = observation_from_py(obs);
    orbit::SimState state = state_from_observation(parsed);
    return state_to_dict(global_engine->simulate_step(state, actions_from_py(actions)));
  });

  m.def("fleet_speed", &orbit::fleet_speed);
  m.def("segment_circle_intersects", &orbit::segment_circle_intersects);
  m.def("swept_pair_hit", &orbit::swept_pair_hit);
}
