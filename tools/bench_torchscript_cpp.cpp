#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int arg_int(char** argv, int idx, int fallback) {
  if (argv[idx] == nullptr) {
    return fallback;
  }
  return std::atoi(argv[idx]);
}

double now_seconds() {
  using clock = std::chrono::steady_clock;
  static const auto start = clock::now();
  const auto elapsed = clock::now() - start;
  return std::chrono::duration<double>(elapsed).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr
        << "usage: bench_torchscript_cpp MODEL [batch=16] [planets=32] [iters=300] "
           "[warmup=30] [threads=4] [planet_dim=218] [edge_dim=11]\n";
    return 2;
  }

  const std::string model_path = argv[1];
  const int batch = argc > 2 ? arg_int(argv, 2, 16) : 16;
  const int planets = argc > 3 ? arg_int(argv, 3, 32) : 32;
  const int iters = argc > 4 ? arg_int(argv, 4, 300) : 300;
  const int warmup = argc > 5 ? arg_int(argv, 5, 30) : 30;
  const int threads = argc > 6 ? arg_int(argv, 6, 4) : 4;
  const int planet_dim = argc > 7 ? arg_int(argv, 7, 218) : 218;
  const int edge_dim = argc > 8 ? arg_int(argv, 8, 11) : 11;

  torch::set_num_threads(std::max(1, threads));
  torch::NoGradGuard no_grad;

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path, torch::kCPU);
    module.eval();
  } catch (const c10::Error& error) {
    std::cerr << "failed to load TorchScript model: " << error.what() << "\n";
    return 1;
  }

  auto planet_features = torch::randn({batch, planets, planet_dim}, torch::kFloat32);
  auto edge_features = torch::randn({batch, planets, planets, edge_dim}, torch::kFloat32);
  auto planet_mask = torch::ones({batch, planets}, torch::kBool);
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(planet_features);
  inputs.emplace_back(edge_features);
  inputs.emplace_back(planet_mask);

  double total = 0.0;
  double value_sum = 0.0;
  for (int idx = 0; idx < iters + warmup; ++idx) {
    const double started = now_seconds();
    auto out = module.forward(inputs);
    const double elapsed = now_seconds() - started;
    if (idx >= warmup) {
      total += elapsed;
    }

    if (idx == iters + warmup - 1) {
      auto dict = out.toGenericDict();
      auto value = dict.at("value").toTensor();
      value_sum = value.mean().item<double>();
    }
  }

  const double mean_sec = total / std::max(1, iters);
  std::cout << "{"
            << "\"mode\":\"cpp_torchscript\","
            << "\"batch_size\":" << batch << ","
            << "\"planets\":" << planets << ","
            << "\"iters\":" << iters << ","
            << "\"threads\":" << torch::get_num_threads() << ","
            << "\"mean_ms\":" << mean_sec * 1000.0 << ","
            << "\"states_per_sec\":" << (mean_sec > 0.0 ? batch / mean_sec : 0.0)
            << ","
            << "\"ms_per_state\":" << (mean_sec * 1000.0 / std::max(1, batch))
            << ","
            << "\"last_value_mean\":" << value_sum << "}\n";
  return 0;
}
