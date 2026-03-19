#include "inference.hpp"
#include "core/util/profiler.hpp"

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
  Device device{DeviceType::CPU, 0};
  DataType dtype{DataType::Float32};
  int batch = 32;
  int input_dim = 256;
  int hidden_dim = 512;
  int output_dim = 128;
  int warmup_runs = 5;
  int single_run_iters = 50;
  int batch_run_inputs = 4;
  int batch_run_iters = 20;
  bool capture_profiler_memory = false;
  bool lean_mode = false;
  int prepared_input_cache_entries = 8;
  size_t prepared_input_cache_max_bytes = 64 * 1024 * 1024;
  bool preallocate_batch_inputs = false;
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
  if (value == "cpu")
    return Device{DeviceType::CPU, 0};
  if (value == "cuda")
    return Device{DeviceType::CUDA, 0};
  if (value == "vulkan")
    return Device{DeviceType::VULKAN, 0};
  throw std::runtime_error("Unknown device '" + value +
                           "'. Expected one of: cpu, cuda, vulkan");
}

DataType parse_dtype(const std::string &value) {
  if (value == "float32")
    return DataType::Float32;
  if (value == "float16")
    return DataType::Float16;
  throw std::runtime_error("Unknown dtype '" + value +
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

bool parse_bool(const std::string &name, const std::string &value) {
  if (value == "1" || value == "true")
    return true;
  if (value == "0" || value == "false")
    return false;
  throw std::runtime_error(name + " must be one of: 0, 1, false, true");
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
    } else if (arg == "--capture-profiler-memory") {
      cfg.capture_profiler_memory =
          parse_bool(arg, require_value(arg));
    } else if (arg == "--lean-mode") {
      cfg.lean_mode = parse_bool(arg, require_value(arg));
    } else if (arg == "--prepared-input-cache-entries") {
      cfg.prepared_input_cache_entries =
          parse_nonnegative_int(arg, require_value(arg));
    } else if (arg == "--prepared-input-cache-max-bytes") {
      cfg.prepared_input_cache_max_bytes =
          static_cast<size_t>(std::stoull(require_value(arg)));
    } else if (arg == "--preallocate-batch-inputs") {
      cfg.preallocate_batch_inputs = parse_bool(arg, require_value(arg));
    } else if (arg == "--help") {
      std::cout
          << "MuNet inference baseline benchmark\n"
          << "Usage: munet_inference_baseline [options]\n"
          << "  --device <cpu|cuda|vulkan>\n"
          << "  --dtype <float32|float16>\n"
          << "  --batch <int>\n"
          << "  --input-dim <int>\n"
          << "  --hidden-dim <int>\n"
          << "  --output-dim <int>\n"
          << "  --warmup-runs <int>\n"
          << "  --single-run-iters <int>\n"
          << "  --batch-run-inputs <int>\n"
          << "  --batch-run-iters <int>\n"
          << "  --capture-profiler-memory <0|1|false|true>\n"
          << "  --lean-mode <0|1|false|true>\n"
          << "  --prepared-input-cache-entries <int>\n"
          << "  --prepared-input-cache-max-bytes <int>\n"
          << "  --preallocate-batch-inputs <0|1|false|true>\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return cfg;
}

Tensor make_input(const BenchmarkConfig &cfg) {
  Tensor input({cfg.batch, cfg.input_dim}, Device{DeviceType::CPU, 0},
               cfg.dtype, false);
  input.uniform_(-1.0f, 1.0f);
  return input;
}

void synchronize_tensor(const Tensor &tensor) {
  if (tensor.impl_) {
    tensor.impl_->backend().synchronize();
  }
}

double elapsed_ms(
    const std::chrono::high_resolution_clock::time_point &start_time) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - start_time)
      .count();
}

std::string build_profile_hint(const Device &device) {
  switch (device.type) {
  case DeviceType::CPU:
    return "minimal_cpu_edge";
  case DeviceType::CUDA:
  case DeviceType::VULKAN:
    return "accelerator_enabled_runtime";
  default:
    return "general_purpose_runtime";
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const BenchmarkConfig cfg = parse_args(argc, argv);
    Profiler::get().reset();

    TensorOptions options;
    options.device = Device{DeviceType::CPU, 0};
    options.dtype = cfg.dtype;
    options.requires_grad = true;

    auto module = std::make_shared<BenchmarkMLP>(cfg.input_dim, cfg.hidden_dim,
                                                 cfg.output_dim, options);

    inference::EngineConfig engine_cfg;
    engine_cfg.device = cfg.device;
    engine_cfg.warmup_runs = cfg.warmup_runs;
    engine_cfg.strict_shape_check = true;
    engine_cfg.allow_autograd_inputs = false;
    engine_cfg.capture_profiler_memory = cfg.capture_profiler_memory;
    engine_cfg.lean_mode = cfg.lean_mode;
    engine_cfg.prepared_input_cache_entries =
        static_cast<size_t>(cfg.prepared_input_cache_entries);
    engine_cfg.prepared_input_cache_max_bytes =
        cfg.prepared_input_cache_max_bytes;
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
    const auto load_stats = engine.stats();

    engine.compile(input, {-1, cfg.input_dim}, {-1, cfg.output_dim});
    const auto compile_stats = engine.stats();

    for (int i = 0; i < cfg.warmup_runs; ++i) {
      Tensor warm = engine.run(input);
      synchronize_tensor(warm);
    }

    double single_wall_ms_total = 0.0;
    double single_engine_run_ms_total = 0.0;
    double single_prepare_ms_total = 0.0;
    double single_forward_ms_total = 0.0;
    double single_validate_ms_total = 0.0;

    for (int i = 0; i < cfg.single_run_iters; ++i) {
      const auto iter_start = std::chrono::high_resolution_clock::now();
      Tensor out = engine.run(input);
      synchronize_tensor(out);
      single_wall_ms_total += elapsed_ms(iter_start);

      const auto stats = engine.stats();
      single_engine_run_ms_total += stats.last_run_ms;
      single_prepare_ms_total += stats.last_prepare_input_ms;
      single_forward_ms_total += stats.last_forward_ms;
      single_validate_ms_total += stats.last_output_validation_ms;
    }

    double batch_wall_ms_total = 0.0;
    if (cfg.preallocate_batch_inputs) {
      engine.prepare_batch(batch_inputs);
    }
    std::vector<Tensor> batch_outputs;
    batch_outputs.reserve(batch_inputs.size());
    for (int i = 0; i < cfg.batch_run_iters; ++i) {
      const auto iter_start = std::chrono::high_resolution_clock::now();
      engine.run_batch_into(batch_inputs, batch_outputs);
      if (!batch_outputs.empty()) {
        synchronize_tensor(batch_outputs.back());
      }
      batch_wall_ms_total += elapsed_ms(iter_start);
    }

    const auto final_stats = engine.stats();
    const auto profiler_snapshot = Profiler::get().snapshot();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "{\n";
    std::cout << "  \"device\": " << quote(cfg.device.to_string()) << ",\n";
    std::cout << "  \"dtype\": " << quote(dtype_name(cfg.dtype)) << ",\n";
    std::cout << "  \"build_profile_hint\": "
              << quote(build_profile_hint(cfg.device)) << ",\n";
    std::cout << "  \"lean_mode\": "
              << (cfg.lean_mode ? "true" : "false") << ",\n";
    std::cout << "  \"memory_policy\": {\n";
    std::cout << "    \"prepared_input_cache_entries\": "
              << engine.prepared_input_cache_entries_limit() << ",\n";
    std::cout << "    \"prepared_input_cache_max_bytes\": "
              << engine.prepared_input_cache_max_bytes_limit() << ",\n";
    std::cout << "    \"preallocate_batch_inputs\": "
              << (cfg.preallocate_batch_inputs ? "true" : "false") << "\n";
    std::cout << "  },\n";
    std::cout << "  \"shape_contract\": {\n";
    std::cout << "    \"compiled_input_shape\": "
              << to_json_array(compile_stats.compiled_input_shape) << ",\n";
    std::cout << "    \"compiled_output_shape\": "
              << to_json_array(compile_stats.compiled_output_shape) << "\n";
    std::cout << "  },\n";
    std::cout << "  \"cold_load\": {\n";
    std::cout << "    \"wall_ms\": " << load_wall_ms << ",\n";
    std::cout << "    \"to_device_ms\": " << load_stats.load_to_device_ms
              << ",\n";
    std::cout << "    \"eval_ms\": " << load_stats.load_eval_ms << "\n";
    std::cout << "  },\n";
    std::cout << "  \"compile\": {\n";
    std::cout << "    \"compile_ms\": " << compile_stats.compile_ms << ",\n";
    std::cout << "    \"prepare_input_ms\": "
              << compile_stats.compile_prepare_input_ms << ",\n";
    std::cout << "    \"forward_ms\": " << compile_stats.compile_forward_ms
              << ",\n";
    std::cout << "    \"warmup_ms\": " << compile_stats.compile_warmup_ms
              << "\n";
    std::cout << "  },\n";
    std::cout << "  \"steady_single_run\": {\n";
    std::cout << "    \"iters\": " << cfg.single_run_iters << ",\n";
    std::cout << "    \"avg_wall_ms\": "
              << (single_wall_ms_total / cfg.single_run_iters) << ",\n";
    std::cout << "    \"avg_engine_run_ms\": "
              << (single_engine_run_ms_total / cfg.single_run_iters) << ",\n";
    std::cout << "    \"avg_prepare_input_ms\": "
              << (single_prepare_ms_total / cfg.single_run_iters) << ",\n";
    std::cout << "    \"avg_forward_ms\": "
              << (single_forward_ms_total / cfg.single_run_iters) << ",\n";
    std::cout << "    \"avg_output_validation_ms\": "
              << (single_validate_ms_total / cfg.single_run_iters) << "\n";
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
    std::cout << "  \"memory_bytes\": {\n";
    std::cout << "    \"engine_current\": " << final_stats.current_memory_bytes
              << ",\n";
    std::cout << "    \"engine_peak\": " << final_stats.peak_memory_bytes
              << ",\n";
    std::cout << "    \"prepared_input_cache_entries\": "
              << final_stats.prepared_input_cache_entries << ",\n";
    std::cout << "    \"prepared_input_cache_bytes\": "
              << final_stats.prepared_input_cache_bytes << ",\n";
    std::cout << "    \"prepared_input_cache_hits\": "
              << final_stats.prepared_input_cache_hits << ",\n";
    std::cout << "    \"prepared_input_cache_misses\": "
              << final_stats.prepared_input_cache_misses << ",\n";
    std::cout << "    \"prepared_input_cache_evictions\": "
              << final_stats.prepared_input_cache_evictions << ",\n";
    std::cout << "    \"profiler_current\": "
              << profiler_snapshot.current_memory_bytes << ",\n";
    std::cout << "    \"profiler_peak\": "
              << profiler_snapshot.peak_memory_bytes << "\n";
    std::cout << "  }\n";
    std::cout << "}\n";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "munet_inference_baseline failed: " << e.what() << std::endl;
    return 1;
  }
}
