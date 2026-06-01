#include "inference.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace munet;

namespace {

struct BenchmarkConfig {
  Device device{DeviceType::VULKAN, 0};
  DataType dtype{DataType::Float32};
  int batch = 32;
  int input_dim = 256;
  int hidden_dim = 512;
  int output_dim = 128;
  int warmup_runs = 5;
  int single_run_iters = 50;
  int batch_run_inputs = 4;
  int batch_run_iters = 20;
};

class BenchmarkMLP : public inference::Module {
public:
  BenchmarkMLP(int input_dim, int hidden_dim, int output_dim,
               const TensorOptions &options)
      : w1_({input_dim, hidden_dim}, options.device, options.dtype, true),
        b1_({hidden_dim}, options.device, options.dtype, true),
        w2_({hidden_dim, hidden_dim}, options.device, options.dtype, true),
        b2_({hidden_dim}, options.device, options.dtype, true),
        w3_({hidden_dim, output_dim}, options.device, options.dtype, true),
        b3_({output_dim}, options.device, options.dtype, true) {
    initialize_parameter(w1_, input_dim);
    initialize_parameter(b1_, hidden_dim);
    initialize_parameter(w2_, hidden_dim);
    initialize_parameter(b2_, hidden_dim);
    initialize_parameter(w3_, hidden_dim);
    initialize_parameter(b3_, output_dim);

    register_parameter("w1", w1_);
    register_parameter("b1", b1_);
    register_parameter("w2", w2_);
    register_parameter("b2", b2_);
    register_parameter("w3", w3_);
    register_parameter("b3", b3_);
  }

  Tensor forward_impl(Tensor x) override {
    x = x.matmul(w1_) + b1_;
    x = x.relu();
    x = x.matmul(w2_) + b2_;
    x = x.relu();
    x = x.matmul(w3_) + b3_;
    return x;
  }

private:
  static void initialize_parameter(Tensor &tensor, int fan_in) {
    const float limit = 1.0f / std::sqrt(static_cast<float>(fan_in));
    tensor.uniform_(-limit, limit);
  }

  Tensor w1_;
  Tensor b1_;
  Tensor w2_;
  Tensor b2_;
  Tensor w3_;
  Tensor b3_;
};

std::string quote(const std::string &value) {
  std::ostringstream oss;
  oss << '"';
  for (const char c : value) {
    switch (c) {
    case '\\':
      oss << "\\\\";
      break;
    case '"':
      oss << "\\\"";
      break;
    case '\n':
      oss << "\\n";
      break;
    default:
      oss << c;
      break;
    }
  }
  oss << '"';
  return oss.str();
}

std::string to_json_array(const std::vector<int> &values) {
  std::ostringstream oss;
  oss << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0)
      oss << ',';
    oss << values[i];
  }
  oss << ']';
  return oss.str();
}

Device parse_device(const std::string &value) {
  if (value == "vulkan")
    return Device{DeviceType::VULKAN, 0};
  throw std::runtime_error("Vulkan device '" + value +
                           "'. Expected: vulkan");
}

DataType parse_dtype(const std::string &value) {
  if (value == "float32")
    return DataType::Float32;
  if (value == "float16")
    return DataType::Float16;
  throw std::runtime_error("Vulkan dtype '" + value +
                           "'. Expected one of: float32, float16");
}

int parse_positive_int(const std::string &name, const std::string &value) {
  const int parsed = std::stoi(value);
  if (parsed <= 0)
    throw std::runtime_error(name + " must be > 0");
  return parsed;
}

int parse_nonnegative_int(const std::string &name, const std::string &value) {
  const int parsed = std::stoi(value);
  if (parsed < 0)
    throw std::runtime_error(name + " must be >= 0");
  return parsed;
}

BenchmarkConfig parse_args(int argc, char **argv) {
  BenchmarkConfig cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string &flag) -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error("Missing value for " + flag);
      return argv[++i];
    };

    if (arg == "--device") {
      cfg.device = parse_device(require_value(arg));
    } else if (arg == "--dtype") {
      cfg.dtype = parse_dtype(require_value(arg));
    } else if (arg == "--batch") {
      cfg.batch = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--input-dim") {
      cfg.input_dim = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--hidden-dim") {
      cfg.hidden_dim = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--output-dim") {
      cfg.output_dim = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--warmup-runs") {
      cfg.warmup_runs = parse_nonnegative_int(arg, require_value(arg));
    } else if (arg == "--single-run-iters") {
      cfg.single_run_iters = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--batch-run-inputs") {
      cfg.batch_run_inputs = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--batch-run-iters") {
      cfg.batch_run_iters = parse_positive_int(arg, require_value(arg));
    } else if (arg == "--help") {
      std::cout << "MuNet inference baseline benchmark\n"
                << "Usage: munet_inference_baseline [options]\n"
                << "  --device <vulkan>\n"
                << "  --dtype <float32|float16>\n"
                << "  --batch <int>\n"
                << "  --input-dim <int>\n"
                << "  --hidden-dim <int>\n"
                << "  --output-dim <int>\n"
                << "  --warmup-runs <int>\n"
                << "  --single-run-iters <int>\n"
                << "  --batch-run-inputs <int>\n"
                << "  --batch-run-iters <int>\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Vulkan argument: " + arg);
    }
  }
  return cfg;
}

Tensor make_input(const BenchmarkConfig &cfg) {
  Tensor input({cfg.batch, cfg.input_dim}, Device{DeviceType::VULKAN, 0},
               cfg.dtype, false);
  input.uniform_(-1.0f, 1.0f);
  return input;
}

void synchronize_tensor(const Tensor &tensor) {
  if (tensor.impl_) {
    tensor.impl_->backend().synchronize();
  }
}

double
elapsed_ms(const std::chrono::high_resolution_clock::time_point &start_time) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - start_time)
      .count();
}

std::string build_profile_hint(const Device &) { return "vulkan_runtime"; }

} // namespace

int main(int argc, char **argv) {
  try {
    const BenchmarkConfig cfg = parse_args(argc, argv);
    TensorOptions options;
    options.device = Device{DeviceType::VULKAN, 0};
    options.dtype = cfg.dtype;
    options.requires_grad = true;

    auto module = std::make_shared<BenchmarkMLP>(cfg.input_dim, cfg.hidden_dim,
                                                 cfg.output_dim, options);

    inference::EngineConfig engine_cfg;
    engine_cfg.device = cfg.device;
    engine_cfg.strict_shape_check = true;
    engine_cfg.allow_autograd_inputs = false;
    inference::Engine engine(engine_cfg);

    const Tensor input = make_input(cfg);
    std::vector<Tensor> batch_inputs;
    batch_inputs.reserve(static_cast<size_t>(cfg.batch_run_inputs));
    for (int i = 0; i < cfg.batch_run_inputs; ++i) {
      batch_inputs.push_back(make_input(cfg));
    }

    const auto load_start = std::chrono::high_resolution_clock::now();
    engine.load(module);
    const double load_wall_ms = elapsed_ms(load_start);
    engine.compile(input, {-1, cfg.input_dim}, {-1, cfg.output_dim});
    const auto compile_stats = engine.stats();

    for (int i = 0; i < cfg.warmup_runs; ++i) {
      Tensor warm = engine.run(input);
      synchronize_tensor(warm);
    }

    double single_wall_ms_total = 0.0;
    double single_engine_run_ms_total = 0.0;

    for (int i = 0; i < cfg.single_run_iters; ++i) {
      const auto iter_start = std::chrono::high_resolution_clock::now();
      Tensor out = engine.run(input);
      synchronize_tensor(out);
      single_wall_ms_total += elapsed_ms(iter_start);

      const auto stats = engine.stats();
      single_engine_run_ms_total += stats.last_run_ms;
    }

    double batch_wall_ms_total = 0.0;
    for (int i = 0; i < cfg.batch_run_iters; ++i) {
      const auto iter_start = std::chrono::high_resolution_clock::now();
      std::vector<Tensor> batch_outputs = engine.run_batch(batch_inputs);
      if (!batch_outputs.empty()) {
        synchronize_tensor(batch_outputs.back());
      }
      batch_wall_ms_total += elapsed_ms(iter_start);
    }

    const auto final_stats = engine.stats();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "{\n";
    std::cout << "  \"device\": " << quote(cfg.device.to_string()) << ",\n";
    std::cout << "  \"dtype\": " << quote(dtype_name(cfg.dtype)) << ",\n";
    std::cout << "  \"build_profile_hint\": "
              << quote(build_profile_hint(cfg.device)) << ",\n";
    std::cout << "  \"shape_contract\": {\n";
    std::cout << "    \"compiled_input_shape\": "
              << to_json_array(compile_stats.compiled_input_shape) << ",\n";
    std::cout << "    \"compiled_output_shape\": "
              << to_json_array(compile_stats.compiled_output_shape) << "\n";
    std::cout << "  },\n";
    std::cout << "  \"cold_load\": {\n";
    std::cout << "    \"wall_ms\": " << load_wall_ms << "\n";
    std::cout << "  },\n";
    std::cout << "  \"compile\": {\n";
    std::cout << "    \"compile_ms\": " << compile_stats.compile_ms << "\n";
    std::cout << "  },\n";
    std::cout << "  \"steady_single_run\": {\n";
    std::cout << "    \"iters\": " << cfg.single_run_iters << ",\n";
    std::cout << "    \"avg_wall_ms\": "
              << (single_wall_ms_total / cfg.single_run_iters) << ",\n";
    std::cout << "    \"avg_engine_run_ms\": "
              << (single_engine_run_ms_total / cfg.single_run_iters) << "\n";
    std::cout << "  },\n";
    std::cout << "  \"steady_batch_run\": {\n";
    std::cout << "    \"iters\": " << cfg.batch_run_iters << ",\n";
    std::cout << "    \"inputs_per_iter\": " << cfg.batch_run_inputs << ",\n";
    std::cout << "    \"avg_wall_ms\": "
              << (batch_wall_ms_total / cfg.batch_run_iters) << ",\n";
    std::cout << "    \"avg_per_input_wall_ms\": "
              << (batch_wall_ms_total /
                  (cfg.batch_run_iters * cfg.batch_run_inputs))
              << "\n";
    std::cout << "  },\n";
    std::cout << "  \"engine_stats\": {\n";
    std::cout << "    \"runs\": " << final_stats.runs << ",\n";
    std::cout << "    \"batch_runs\": " << final_stats.batch_runs << "\n";
    std::cout << "  }\n";
    std::cout << "}\n";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "munet_inference_baseline failed: " << e.what() << std::endl;
    return 1;
  }
}
