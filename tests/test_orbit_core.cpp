#include "orbit_core.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool close(double a, double b, double eps = 1e-6) {
  return std::abs(a - b) <= eps;
}

void test_speed() {
  assert(close(orbit::fleet_speed(1), 1.0));
  assert(orbit::fleet_speed(500) > 4.0);
  assert(close(orbit::fleet_speed(1000), 6.0));
  assert(close(orbit::fleet_speed(5000), 6.0));
}

void test_geometry() {
  assert(orbit::segment_circle_intersects({0, 0}, {10, 0}, {5, 0}, 1));
  assert(orbit::segment_circle_intersects({0, 0}, {10, 0}, {5, 1}, 1));
  assert(!orbit::segment_circle_intersects({0, 0}, {10, 0}, {5, 2}, 1));
  assert(orbit::swept_pair_hit({0, 0}, {10, 0}, {5, 2}, {5, 0}, 1));
  assert(!orbit::swept_pair_hit({0, 0}, {10, 0}, {5, 3}, {5, 2}, 1));
}

void test_rotation() {
  orbit::Planet p{1, -1, 70.0, 50.0, 2.0, 10, 1};
  orbit::Vec2 pos = orbit::rotated_position(p, 0.1, 10);
  assert(close(pos.x, 50.0 + 20.0 * std::cos(1.0)));
  assert(close(pos.y, 50.0 + 20.0 * std::sin(1.0)));
}

void test_static_route() {
  orbit::Observation obs;
  obs.player = 0;
  obs.angular_velocity = 0.0;
  obs.planets = {
      orbit::Planet{0, 0, 10.0, 10.0, 2.0, 50, 1},
      orbit::Planet{1, -1, 20.0, 10.0, 2.0, 5, 3},
  };
  obs.initial_planets = obs.planets;

  orbit::Engine engine;
  engine.initialize(obs);
  orbit::RouteResult route = engine.query_route(0, 1, 6, 0);
  assert(route.reachable);
  assert(route.travel_time > 0);
  assert(std::abs(route.angle) < 1e-6);
}

void test_sun_blocked_route() {
  orbit::Observation obs;
  obs.player = 0;
  obs.angular_velocity = 0.0;
  obs.planets = {
      orbit::Planet{0, 0, 20.0, 50.0, 2.0, 50, 1},
      orbit::Planet{1, -1, 80.0, 50.0, 2.0, 5, 3},
  };
  obs.initial_planets = obs.planets;

  orbit::Engine engine;
  engine.initialize(obs);
  orbit::RouteResult route = engine.query_route(0, 1, 10, 0);
  assert(!route.reachable);
  assert(route.blocked_by == "sun");
}

void test_agent_prefers_production() {
  orbit::Observation obs;
  obs.player = 0;
  obs.angular_velocity = 0.0;
  obs.planets = {
      orbit::Planet{0, 0, 10.0, 10.0, 2.0, 80, 1},
      orbit::Planet{1, -1, 20.0, 10.0, 2.0, 5, 1},
      orbit::Planet{2, -1, 22.0, 15.0, 2.0, 6, 5},
  };
  obs.initial_planets = obs.planets;

  orbit::Engine engine;
  std::vector<orbit::Move> moves = engine.act(obs);
  assert(!moves.empty());
}

}  // namespace

int main() {
  test_speed();
  test_geometry();
  test_rotation();
  test_static_route();
  test_sun_blocked_route();
  test_agent_prefers_production();
  std::cout << "orbit_core tests passed\n";
  return 0;
}
