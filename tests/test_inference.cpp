#include "backend/cpu_backend.hpp"
#include "core/util/profiler.hpp"
#include "inference.hpp"
#include "nn.hpp"
#include "test_utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <type_traits>

using namespace munet;

namespace {
class ScopedProfileOverride {
public:
  explicit ScopedProfileOverride(bool enabled) {
    set_profile_enabled_override(enabled);
  }

  ~ScopedProfileOverride() { set_profile_enabled_override(std::nullopt); }
};

class IdentityLayer : public inference::Module {
public:
  Tensor forward_impl(Tensor x) override { return x; }
};

class IdentityLayerWithState : public inference::Module {
public:
  explicit IdentityLayerWithState(const TensorOptions &options) {
    weight = Tensor({1}, options.device, options.dtype, true);
    weight.fill_(1.0f);
    register_parameter("weight", weight);

    running = Tensor(
        {1}, options.device,
        accumulation_type(AccumulationOp::Elementwise, options.dtype), false);
    running.fill_(0.0f);
    register_buffer("running", running);
  }

  Tensor forward_impl(Tensor x) override { return x; }

  Tensor weight;
  Tensor running;
};

class CountingCopyBackend : public CPUBackend {
public:
  explicit CountingCopyBackend(std::shared_ptr<int> copy_count)
      : copy_count_(std::move(copy_count)) {}

  const char *name() const override { return "counting_copy"; }

  void copy(const void *src, void *dst, size_t bytes, Device src_dev,
            Device dst_dev) override {
    ++(*copy_count_);
    CPUBackend::copy(src, dst, bytes, src_dev, dst_dev);
  }

private:
  std::shared_ptr<int> copy_count_;
};
} // namespace

namespace {

void append_u16_le(std::vector<uint8_t> &out, uint16_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xff));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
}

void append_u32_le(std::vector<uint8_t> &out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xff));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
  out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
  out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
}

std::vector<uint8_t> encode_unicode_scalar_npy(const std::string &value) {
  std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
  std::ostringstream header;
  header << "{'descr': '<U" << value.size()
         << "', 'fortran_order': False, 'shape': (), }";
  std::string header_str = header.str();
  size_t header_len = header_str.size() + 1;
  while ((10 + header_len) % 16 != 0) {
    ++header_len;
  }
  header_str.append(header_len - header_str.size() - 1, ' ');
  header_str.push_back('\n');
  append_u16_le(out, static_cast<uint16_t>(header_str.size()));
  out.insert(out.end(), header_str.begin(), header_str.end());
  for (char ch : value) {
    out.push_back(static_cast<uint8_t>(ch));
    out.push_back(0);
    out.push_back(0);
    out.push_back(0);
  }
  return out;
}

std::vector<uint8_t> encode_bool_scalar_npy(bool value) {
  std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
  std::string header_str =
      "{'descr': '|b1', 'fortran_order': False, 'shape': (), }";
  size_t header_len = header_str.size() + 1;
  while ((10 + header_len) % 16 != 0) {
    ++header_len;
  }
  header_str.append(header_len - header_str.size() - 1, ' ');
  header_str.push_back('\n');
  append_u16_le(out, static_cast<uint16_t>(header_str.size()));
  out.insert(out.end(), header_str.begin(), header_str.end());
  out.push_back(value ? 1 : 0);
  return out;
}

std::vector<uint8_t> encode_int64_scalar_npy(int64_t value) {
  std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
  std::string header_str =
      "{'descr': '<i8', 'fortran_order': False, 'shape': (), }";
  size_t header_len = header_str.size() + 1;
  while ((10 + header_len) % 16 != 0) {
    ++header_len;
  }
  header_str.append(header_len - header_str.size() - 1, ' ');
  header_str.push_back('\n');
  append_u16_le(out, static_cast<uint16_t>(header_str.size()));
  out.insert(out.end(), header_str.begin(), header_str.end());
  for (int i = 0; i < 8; ++i) {
    out.push_back(
        static_cast<uint8_t>((static_cast<uint64_t>(value) >> (8 * i)) & 0xff));
  }
  return out;
}

std::string npy_descr_for_tensor(const Tensor &tensor) {
  switch (tensor.dtype()) {
  case DataType::Float32:
    return "<f4";
  case DataType::Float16:
    return "<f2";
  case DataType::Int32:
    return "<i4";
  default:
    throw std::runtime_error(
        "Unsupported tensor dtype in test artifact writer");
  }
}

std::vector<uint8_t> encode_tensor_npy(const Tensor &tensor) {
  Tensor cpu = tensor.device().type == DeviceType::CPU
                   ? tensor
                   : tensor.to(Device{DeviceType::CPU, 0});
  std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
  std::ostringstream header;
  header << "{'descr': '" << npy_descr_for_tensor(cpu)
         << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < cpu.shape().size(); ++i) {
    if (i > 0) {
      header << ", ";
    }
    header << cpu.shape()[i];
  }
  if (cpu.shape().size() == 1) {
    header << ",";
  }
  header << "), }";
  std::string header_str = header.str();
  size_t header_len = header_str.size() + 1;
  while ((10 + header_len) % 16 != 0) {
    ++header_len;
  }
  header_str.append(header_len - header_str.size() - 1, ' ');
  header_str.push_back('\n');
  append_u16_le(out, static_cast<uint16_t>(header_str.size()));
  out.insert(out.end(), header_str.begin(), header_str.end());
  const auto *bytes = static_cast<const uint8_t *>(cpu.data());
  out.insert(out.end(), bytes, bytes + cpu.bytes());
  return out;
}

std::filesystem::path
write_npz_artifact(const std::string &base_name, const std::string &config_json,
                   const std::map<std::string, Tensor> &tensors) {
  std::vector<std::string> tensor_names;
  tensor_names.reserve(tensors.size());
  for (const auto &entry : tensors) {
    tensor_names.push_back(entry.first);
  }

  std::ostringstream manifest;
  manifest << '[';
  for (size_t i = 0; i < tensor_names.size(); ++i) {
    if (i > 0) {
      manifest << ',';
    }
    manifest << '"' << tensor_names[i] << '"';
  }
  manifest << ']';

  std::vector<std::pair<std::string, std::vector<uint8_t>>> entries;
  entries.push_back({"__config__.npy", encode_unicode_scalar_npy(config_json)});
  entries.push_back(
      {"__format_name__.npy", encode_unicode_scalar_npy("munet_model")});
  entries.push_back({"__format_revision__.npy", encode_int64_scalar_npy(1)});
  entries.push_back(
      {"__format_version__.npy", encode_unicode_scalar_npy("munet_model_v1")});
  entries.push_back({"__producer__.npy", encode_unicode_scalar_npy("munet")});
  entries.push_back(
      {"__artifact_kind__.npy", encode_unicode_scalar_npy("deploy_model")});
  entries.push_back(
      {"__artifact_scope__.npy", encode_unicode_scalar_npy("runtime_only")});
  entries.push_back(
      {"__default_load_mode__.npy", encode_unicode_scalar_npy("eval")});
  entries.push_back(
      {"__contains_training_state__.npy", encode_bool_scalar_npy(false)});
  entries.push_back({"__recommended_loader__.npy",
                     encode_unicode_scalar_npy("load_for_inference")});
  entries.push_back({"__compile_contract_policy__.npy",
                     encode_unicode_scalar_npy("external")});
  entries.push_back(
      {"__tensor_names__.npy", encode_unicode_scalar_npy(manifest.str())});
  for (const auto &entry : tensors) {
    entries.push_back({entry.first + ".npy", encode_tensor_npy(entry.second)});
  }

  const auto dir = std::filesystem::temp_directory_path();
  const auto path = dir / (base_name + ".npz");
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Unable to create test artifact: " +
                             path.string());
  }

  struct CentralDirectoryEntry {
    std::string name;
    uint32_t local_offset = 0;
    uint32_t size = 0;
  };
  std::vector<CentralDirectoryEntry> directory;
  uint32_t offset = 0;

  for (const auto &entry : entries) {
    directory.push_back(
        {entry.first, offset, static_cast<uint32_t>(entry.second.size())});
    std::vector<uint8_t> header;
    append_u32_le(header, 0x04034b50U);
    append_u16_le(header, 20);
    append_u16_le(header, 0);
    append_u16_le(header, 0);
    append_u16_le(header, 0);
    append_u16_le(header, 0);
    append_u32_le(header, 0);
    append_u32_le(header, static_cast<uint32_t>(entry.second.size()));
    append_u32_le(header, static_cast<uint32_t>(entry.second.size()));
    append_u16_le(header, static_cast<uint16_t>(entry.first.size()));
    append_u16_le(header, 0);
    out.write(reinterpret_cast<const char *>(header.data()),
              static_cast<std::streamsize>(header.size()));
    out.write(entry.first.data(),
              static_cast<std::streamsize>(entry.first.size()));
    out.write(reinterpret_cast<const char *>(entry.second.data()),
              static_cast<std::streamsize>(entry.second.size()));
    offset += static_cast<uint32_t>(header.size() + entry.first.size() +
                                    entry.second.size());
  }

  const uint32_t central_directory_offset = offset;
  for (const auto &entry : directory) {
    std::vector<uint8_t> header;
    append_u32_le(header, 0x02014b50U);
    append_u16_le(header, 20);
    append_u16_le(header, 20);
    append_u16_le(header, 0);
    append_u16_le(header, 0);
    append_u16_le(header, 0);
    append_u16_le(header, 0);
    append_u32_le(header, 0);
    append_u32_le(header, entry.size);
    append_u32_le(header, entry.size);
    append_u16_le(header, static_cast<uint16_t>(entry.name.size()));
    append_u16_le(header, 0);
    append_u16_le(header, 0);
    append_u16_le(header, 0);
    append_u16_le(header, 0);
    append_u32_le(header, 0);
    append_u32_le(header, entry.local_offset);
    out.write(reinterpret_cast<const char *>(header.data()),
              static_cast<std::streamsize>(header.size()));
    out.write(entry.name.data(),
              static_cast<std::streamsize>(entry.name.size()));
    offset += static_cast<uint32_t>(header.size() + entry.name.size());
  }

  const uint32_t central_directory_size = offset - central_directory_offset;
  std::vector<uint8_t> eocd;
  append_u32_le(eocd, 0x06054b50U);
  append_u16_le(eocd, 0);
  append_u16_le(eocd, 0);
  append_u16_le(eocd, static_cast<uint16_t>(directory.size()));
  append_u16_le(eocd, static_cast<uint16_t>(directory.size()));
  append_u32_le(eocd, central_directory_size);
  append_u32_le(eocd, central_directory_offset);
  append_u16_le(eocd, 0);
  out.write(reinterpret_cast<const char *>(eocd.data()),
            static_cast<std::streamsize>(eocd.size()));
  out.close();
  return path;
}

} // namespace

TEST(InferenceTest, ModuleInheritsCoreModule) {
  EXPECT_TRUE((std::is_base_of_v<core::Module, inference::Module>));
}

TEST(InferenceTest, TrainCallKeepsEvalMode) {
  auto m = std::make_shared<IdentityLayer>();
  m->train(true);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({1, 1}, cpu);
  x.uniform_(1.0f, 1.0f);

  auto y = m->forward(x);
  EXPECT_EQ(y.size(), x.size());
}

TEST(InferenceTest, EngineLoadPrepareRunAndStats) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.set_warmup_runs(2);
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 2}, cpu);
  x.uniform_(0.5f, 0.5f);

  engine.prepare(x);
  EXPECT_TRUE(engine.is_loaded());
  EXPECT_TRUE(engine.is_prepared());

  Tensor y = engine.run(x).to(cpu);
  EXPECT_EQ(y.shape(), x.shape());

  auto stats = engine.stats();
  EXPECT_EQ(stats.runs, 1u);
  EXPECT_GE(stats.last_run_ms, 0.0);
}

TEST(InferenceTest, EngineThrowsWithoutLoad) {
  inference::Engine engine;
  Device cpu{DeviceType::CPU, 0};
  Tensor x({1, 1}, cpu);

  EXPECT_THROW(engine.run(x), std::runtime_error);
  EXPECT_THROW(engine.prepare(x), std::runtime_error);
}

TEST(InferenceTest, EngineRunBatch) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor a({1, 2}, cpu);
  Tensor b({1, 2}, cpu);
  a.uniform_(1.0f, 1.0f);
  b.uniform_(2.0f, 2.0f);

  auto outs = engine.run_batch({a, b});
  EXPECT_EQ(outs.size(), 2u);
  EXPECT_EQ(outs[0].shape(), a.shape());
  EXPECT_EQ(outs[1].shape(), b.shape());
  EXPECT_EQ(engine.stats().runs, 2u);
}

TEST(InferenceTest, EngineCompileCapturesShapeAndStats) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.set_warmup_runs(1);
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 3}, cpu);
  x.uniform_(0.2f, 0.2f);

  engine.compile(x);
  EXPECT_TRUE(engine.is_compiled());
  EXPECT_TRUE(engine.is_prepared());
  EXPECT_EQ(engine.compiled_input_shape(), x.shape());
  EXPECT_GE(engine.stats().compile_ms, 0.0);
  EXPECT_GE(engine.stats().compile_prepare_input_ms, 0.0);
  EXPECT_GE(engine.stats().compile_forward_ms, 0.0);
}

TEST(InferenceTest, EngineStrictShapeCheckAfterCompile) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 3}, cpu);
  Tensor bad({2, 4}, cpu);
  x.uniform_(0.1f, 0.1f);
  bad.uniform_(0.1f, 0.1f);

  engine.compile(x);
  EXPECT_THROW(engine.run(bad), std::runtime_error);

  engine.set_strict_shape_check(false);
  EXPECT_NO_THROW((void)engine.run(bad));
}

TEST(InferenceTest, EngineCompileWithDynamicInputShape) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x_compile({1, 3, 64, 64}, cpu);
  x_compile.uniform_(0.1f, 0.1f);

  engine.compile(x_compile, {-1, 3, -1, -1}, {-1, 3, -1, -1});

  Tensor x_ok({4, 3, 128, 256}, cpu);
  x_ok.uniform_(0.2f, 0.2f);
  EXPECT_NO_THROW((void)engine.run(x_ok));

  Tensor x_bad({4, 1, 128, 256}, cpu);
  x_bad.uniform_(0.2f, 0.2f);
  EXPECT_THROW((void)engine.run(x_bad), std::runtime_error);
}

TEST(InferenceTest, EngineCompileWithInvalidExpectedOutputShapeThrows) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 3}, cpu);
  x.uniform_(0.3f, 0.3f);

  EXPECT_THROW(engine.compile(x, {-1, 3}, {-1, 4}), std::runtime_error);
}

TEST(InferenceTest, EngineLoadPreservesModelWideDTypeConversion) {
  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float16;

  auto m = std::make_shared<IdentityLayerWithState>(options);
  inference::Engine engine;
  engine.load(m);

  EXPECT_EQ(m->weight.dtype(), DataType::Float16);
  EXPECT_EQ(m->running.dtype(), DataType::Float32);

  Tensor x({2, 2}, Device{DeviceType::CPU, 0}, DataType::Float16, false);
  x.fill_(0.5f);
  Tensor y = engine.run(x);
  EXPECT_EQ(y.dtype(), DataType::Float16);
  EXPECT_EQ(y.shape(), x.shape());
}

TEST(InferenceTest, EnginePreservesFloat16LinearOutputsAcrossAvailableDevices) {
  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float16;

  auto linear = std::make_shared<nn::Linear>(2, 2, true, options);
  Tensor weight_cpu({2, 2}, Device{DeviceType::CPU, 0}, DataType::Float32);
  Tensor bias_cpu({2}, Device{DeviceType::CPU, 0}, DataType::Float32);
  float *w = static_cast<float *>(weight_cpu.data());
  float *b = static_cast<float *>(bias_cpu.data());
  w[0] = 1.0f;
  w[1] = 2.0f;
  w[2] = -1.0f;
  w[3] = 0.5f;
  b[0] = 0.25f;
  b[1] = -0.75f;
  linear->weight = weight_cpu.to(DataType::Float16);
  linear->bias = bias_cpu.to(DataType::Float16);

  Tensor x32({1, 2}, Device{DeviceType::CPU, 0}, DataType::Float32);
  float *x_ptr = static_cast<float *>(x32.data());
  x_ptr[0] = 3.0f;
  x_ptr[1] = -2.0f;
  Tensor x = x32.to(DataType::Float16);

  Tensor expected =
      linear->forward(x).to(Device{DeviceType::CPU, 0}).to(DataType::Float32);

  for (const Device &device : test::get_available_devices()) {
    inference::Engine engine;
    engine.set_device(device);
    engine.load(linear);

    Tensor y = engine.run(x.to(device))
                   .to(Device{DeviceType::CPU, 0})
                   .to(DataType::Float32);
    EXPECT_EQ(y.dtype(), DataType::Float32);
    EXPECT_TRUE(test::all_close(y, expected, 2e-1f)) << device.to_string();
  }
}

TEST(InferenceTest, EngineRunDisablesAutogradGraphConstruction) {
  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float32;

  auto linear = std::make_shared<nn::Linear>(2, 2, true, options);
  inference::Engine engine;
  engine.load(linear);

  EXPECT_TRUE(linear->weight.requires_grad());
  GradMode::set_enabled(true);

  Tensor x({1, 2}, options.device, options.dtype, false);
  x.fill_(1.0f);

  Tensor y = engine.run(x);
  EXPECT_FALSE(y.requires_grad());
  EXPECT_FALSE(y.impl_ && y.impl_->grad_fn);
  EXPECT_TRUE(GradMode::is_enabled());
}

TEST(InferenceTest, EngineRejectsAutogradInputsByDefault) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 2}, cpu, DataType::Float32, true);
  x.fill_(0.25f);

  EXPECT_THROW(engine.run(x), std::runtime_error);
}

TEST(InferenceTest, EngineObserverReceivesLifecycleAndErrorEvents) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  std::vector<inference::EngineEvent> events;
  engine.set_observer([&events](const inference::EngineEvent &event) {
    events.push_back(event);
  });
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 3}, cpu);
  Tensor bad({2, 4}, cpu);
  x.uniform_(0.1f, 0.1f);
  bad.uniform_(0.1f, 0.1f);

  engine.compile(x);
  (void)engine.run(x);
  EXPECT_THROW((void)engine.run(bad), std::runtime_error);

  ASSERT_GE(events.size(), 6u);
  EXPECT_EQ(events.front().type, inference::EngineEventType::LoadStarted);
  EXPECT_EQ(events[1].type, inference::EngineEventType::LoadCompleted);
  EXPECT_EQ(events[2].type, inference::EngineEventType::CompileStarted);
  EXPECT_EQ(events[3].type, inference::EngineEventType::CompileCompleted);
  EXPECT_EQ(events[4].type, inference::EngineEventType::RunStarted);
  EXPECT_EQ(events[5].type, inference::EngineEventType::RunCompleted);
  EXPECT_GT(events[2].trace_id, 0u);
  EXPECT_EQ(events[2].trace_id, events[3].trace_id);
  EXPECT_EQ(events[2].span, "compile");
  EXPECT_EQ(events[3].span, "compile");
  EXPECT_GT(events[4].trace_id, 0u);
  EXPECT_EQ(events[4].trace_id, events[5].trace_id);
  EXPECT_NE(events[2].trace_id, events[4].trace_id);
  EXPECT_EQ(events[4].span, "run");
  EXPECT_EQ(events[5].span, "run");
  EXPECT_EQ(events[3].output_shape, x.shape());
  EXPECT_EQ(events[5].input_shape, x.shape());
  EXPECT_GE(events[5].peak_memory_bytes, events[5].current_memory_bytes);
  EXPECT_EQ(events.back().type, inference::EngineEventType::Error);
  EXPECT_GT(events.back().trace_id, 0u);
  EXPECT_EQ(events.back().span, "run");
  EXPECT_NE(events.back().message.find("run failed"), std::string::npos);
}

TEST(InferenceTest, EngineProfilesLifecyclePhasesIntoStatsAndProfiler) {
  ScopedProfileOverride profile(true);
  Profiler::get().reset();

  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.set_warmup_runs(2);
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 2}, cpu);
  x.fill_(1.0f);

  engine.compile(x);
  (void)engine.run(x);

  const auto stats = engine.stats();
  EXPECT_GT(stats.last_compile_trace_id, 0u);
  EXPECT_GT(stats.last_run_trace_id, 0u);
  EXPECT_NE(stats.last_compile_trace_id, stats.last_run_trace_id);
  EXPECT_GE(stats.load_to_device_ms, 0.0);
  EXPECT_GE(stats.load_eval_ms, 0.0);
  EXPECT_GE(stats.compile_prepare_input_ms, 0.0);
  EXPECT_GE(stats.compile_forward_ms, 0.0);
  EXPECT_GE(stats.compile_warmup_ms, 0.0);
  EXPECT_GE(stats.last_prepare_input_ms, 0.0);
  EXPECT_GE(stats.last_forward_ms, 0.0);
  EXPECT_GE(stats.last_output_validation_ms, 0.0);

  const auto snapshot = Profiler::get().snapshot();
  EXPECT_NE(snapshot.stats.find("inference.load.to_device"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.load.eval"), snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.compile.prepare_input"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.compile.forward"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.compile.warmup"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.run.prepare_input"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.run.forward"), snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.run.validate_output"),
            snapshot.stats.end());
  const auto compile_forward = snapshot.stats.find("inference.compile.forward");
  ASSERT_NE(compile_forward, snapshot.stats.end());
  EXPECT_NE(compile_forward->second.last_shape.find(
                "trace_id=" + std::to_string(stats.last_compile_trace_id)),
            std::string::npos);
  EXPECT_NE(compile_forward->second.last_shape.find("span=compile.forward"),
            std::string::npos);
  const auto run_forward = snapshot.stats.find("inference.run.forward");
  ASSERT_NE(run_forward, snapshot.stats.end());
  EXPECT_NE(run_forward->second.last_shape.find(
                "trace_id=" + std::to_string(stats.last_run_trace_id)),
            std::string::npos);
  EXPECT_NE(run_forward->second.last_shape.find("span=run.forward"),
            std::string::npos);
  const auto module_forward = snapshot.stats.find("module.root.forward");
  ASSERT_NE(module_forward, snapshot.stats.end());
  EXPECT_NE(module_forward->second.last_shape.find(
                "trace_id=" + std::to_string(stats.last_run_trace_id)),
            std::string::npos);
  EXPECT_NE(module_forward->second.last_shape.find("span=run.forward"),
            std::string::npos);
}

TEST(InferenceTest, EngineDefaultHotPathLeavesTraceIdsZeroWithoutDiagnostics) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 2}, cpu);
  x.fill_(1.0f);

  engine.compile(x);
  (void)engine.run(x);

  const auto stats = engine.stats();
  EXPECT_EQ(stats.last_compile_trace_id, 0u);
  EXPECT_EQ(stats.last_run_trace_id, 0u);
  EXPECT_EQ(stats.current_memory_bytes, 0u);
  EXPECT_EQ(stats.peak_memory_bytes, 0u);
}

TEST(InferenceTest, EngineLeanModeDisablesProfilerMemoryByDefault) {
  auto m = std::make_shared<IdentityLayer>();
  inference::EngineConfig cfg;
  cfg.lean_mode = true;
  inference::Engine engine(cfg);
  engine.load(m);

  EXPECT_TRUE(engine.lean_mode());
  EXPECT_FALSE(engine.capture_profiler_memory());

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 2}, cpu);
  x.fill_(1.0f);

  engine.compile(x);
  (void)engine.run(x);

  const auto stats = engine.stats();
  EXPECT_EQ(stats.current_memory_bytes, 0u);
  EXPECT_EQ(stats.peak_memory_bytes, 0u);
}

TEST(InferenceTest, EngineCachesTransferredInputAcrossCompileAndRun) {
  auto copy_count = std::make_shared<int>(0);
  BackendManager::register_backend(DeviceType::UNKNOWN, [copy_count](Device) {
    return std::make_shared<CountingCopyBackend>(copy_count);
  });

  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.set_device(Device{DeviceType::CPU, 0});
  engine.load(m);

  Tensor x({2, 2}, Device{DeviceType::UNKNOWN, 0});
  x.fill_(1.0f);
  *copy_count = 0;

  engine.compile(x);
  EXPECT_EQ(*copy_count, 1);

  (void)engine.run(x);
  (void)engine.run(x);
  EXPECT_EQ(*copy_count, 1);

  const int copies_before_mutation = *copy_count;
  x.fill_(2.0f);
  const int copies_after_mutation = *copy_count;
  (void)engine.run(x);
  EXPECT_EQ(copies_after_mutation, copies_before_mutation + 1);
  EXPECT_EQ(*copy_count, copies_after_mutation + 1);

  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device) {
    return std::make_shared<CPUBackend>();
  });
}

TEST(InferenceTest, EngineRejectsAutogradInputBeforeTransfer) {
  auto copy_count = std::make_shared<int>(0);
  BackendManager::register_backend(DeviceType::UNKNOWN, [copy_count](Device) {
    return std::make_shared<CountingCopyBackend>(copy_count);
  });

  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.set_device(Device{DeviceType::CPU, 0});
  engine.load(m);

  Tensor x({2, 2}, Device{DeviceType::UNKNOWN, 0}, DataType::Float32, true);
  x.fill_(1.0f);
  *copy_count = 0;

  EXPECT_THROW((void)engine.run(x), std::runtime_error);
  EXPECT_EQ(*copy_count, 0);

  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device) {
    return std::make_shared<CPUBackend>();
  });
}

TEST(InferenceTest, EngineCachesPreparedBatchInputsAcrossBatchRuns) {
  auto copy_count = std::make_shared<int>(0);
  BackendManager::register_backend(DeviceType::UNKNOWN, [copy_count](Device) {
    return std::make_shared<CountingCopyBackend>(copy_count);
  });

  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.set_device(Device{DeviceType::CPU, 0});
  engine.load(m);

  Tensor a({2, 2}, Device{DeviceType::UNKNOWN, 0});
  Tensor b({2, 2}, Device{DeviceType::UNKNOWN, 0});
  a.fill_(1.0f);
  b.fill_(2.0f);
  *copy_count = 0;

  auto first = engine.run_batch({a, b});
  ASSERT_EQ(first.size(), 2u);
  EXPECT_EQ(*copy_count, 2);

  auto second = engine.run_batch({a, b});
  ASSERT_EQ(second.size(), 2u);
  EXPECT_EQ(*copy_count, 2);

  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device) {
    return std::make_shared<CPUBackend>();
  });
}

TEST(InferenceTest, EngineLoadSkipsTransfersForPrepositionedModule) {
  auto copy_count = std::make_shared<int>(0);
  BackendManager::register_backend(DeviceType::UNKNOWN, [copy_count](Device) {
    return std::make_shared<CountingCopyBackend>(copy_count);
  });

  TensorOptions options;
  options.device = Device{DeviceType::UNKNOWN, 0};
  options.dtype = DataType::Float32;

  auto m = std::make_shared<IdentityLayerWithState>(options);
  *copy_count = 0;

  inference::Engine engine;
  engine.set_device(Device{DeviceType::UNKNOWN, 0});
  engine.load(m);

  EXPECT_EQ(*copy_count, 0);

  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device) {
    return std::make_shared<CPUBackend>();
  });
}

TEST(InferenceTest, EnginePreparedInputCacheCanBeBoundedToSingleEntry) {
  auto copy_count = std::make_shared<int>(0);
  BackendManager::register_backend(DeviceType::UNKNOWN, [copy_count](Device) {
    return std::make_shared<CountingCopyBackend>(copy_count);
  });

  inference::EngineConfig cfg;
  cfg.device = Device{DeviceType::CPU, 0};
  cfg.prepared_input_cache_entries = 1;
  cfg.prepared_input_cache_max_bytes = 1024;
  inference::Engine engine(cfg);
  engine.load(std::make_shared<IdentityLayer>());

  Tensor a({2, 2}, Device{DeviceType::UNKNOWN, 0});
  Tensor b({2, 2}, Device{DeviceType::UNKNOWN, 0});
  a.fill_(1.0f);
  b.fill_(2.0f);
  *copy_count = 0;

  (void)engine.run_batch({a, b});
  EXPECT_EQ(*copy_count, 2);
  const auto stats_after_first = engine.stats();
  EXPECT_EQ(stats_after_first.prepared_input_cache_entries, 1u);
  EXPECT_LE(stats_after_first.prepared_input_cache_bytes, 1024u);

  (void)engine.run_batch({a, b});
  EXPECT_EQ(*copy_count, 4);
  const auto stats_after_second = engine.stats();
  EXPECT_EQ(stats_after_second.prepared_input_cache_entries, 1u);
  EXPECT_GE(stats_after_second.prepared_input_cache_evictions, 3u);

  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device) {
    return std::make_shared<CPUBackend>();
  });
}

TEST(InferenceTest, EnginePrepareBatchPrepopulatesPreparedInputCache) {
  auto copy_count = std::make_shared<int>(0);
  BackendManager::register_backend(DeviceType::UNKNOWN, [copy_count](Device) {
    return std::make_shared<CountingCopyBackend>(copy_count);
  });

  inference::EngineConfig cfg;
  cfg.device = Device{DeviceType::CPU, 0};
  cfg.prepared_input_cache_entries = 2;
  inference::Engine engine(cfg);
  engine.load(std::make_shared<IdentityLayer>());

  Tensor a({2, 2}, Device{DeviceType::UNKNOWN, 0});
  Tensor b({2, 2}, Device{DeviceType::UNKNOWN, 0});
  a.fill_(1.0f);
  b.fill_(2.0f);
  *copy_count = 0;

  engine.prepare_batch({a, b});
  EXPECT_EQ(*copy_count, 2);
  const auto stats_after_prepare = engine.stats();
  EXPECT_EQ(stats_after_prepare.prepared_input_cache_entries, 2u);
  EXPECT_EQ(stats_after_prepare.prepared_input_cache_misses, 2u);

  (void)engine.run_batch({a, b});
  EXPECT_EQ(*copy_count, 2);
  const auto stats_after_run = engine.stats();
  EXPECT_EQ(stats_after_run.prepared_input_cache_hits, 2u);

  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device) {
    return std::make_shared<CPUBackend>();
  });
}

TEST(InferenceTest, LoadSerializedReconstructsDeployModuleInCpp) {
  Tensor weight({2, 2}, Device{DeviceType::CPU, 0}, DataType::Float32, false);
  Tensor bias({2}, Device{DeviceType::CPU, 0}, DataType::Float32, false);
  auto *w = static_cast<float *>(weight.data());
  auto *b = static_cast<float *>(bias.data());
  w[0] = 1.0f;
  w[1] = -1.0f;
  w[2] = 0.5f;
  w[3] = 2.0f;
  b[0] = 0.25f;
  b[1] = -0.75f;

  const std::string config =
      R"({"type":"Sequential","layers":[{"type":"Linear","in_features":2,"out_features":2,"bias":true,"dtype":"float32"},{"type":"ReLU"}]})";
  const auto path =
      write_npz_artifact("munet_cpp_load_serialized_roundtrip", config,
                         {{"0.bias", bias}, {"0.weight", weight}});

  auto module = inference::load_serialized(path.string());
  ASSERT_TRUE(module);

  Tensor x({1, 2}, Device{DeviceType::CPU, 0}, DataType::Float32, false);
  auto *x_ptr = static_cast<float *>(x.data());
  x_ptr[0] = 2.0f;
  x_ptr[1] = 3.0f;

  Tensor y = module->forward(x).to(Device{DeviceType::CPU, 0});
  const auto *y_ptr = static_cast<const float *>(y.data());
  EXPECT_FLOAT_EQ(y_ptr[0], 3.75f);
  EXPECT_FLOAT_EQ(y_ptr[1], 3.25f);

  std::remove(path.string().c_str());
}

TEST(InferenceTest, LoadSerializedNormalizesModuleForInferenceEval) {
  const std::string config =
      R"({"type":"Sequential","layers":[{"type":"Dropout","p":0.9}]})";
  const auto path =
      write_npz_artifact("munet_cpp_load_serialized_eval_mode", config, {});

  auto module = inference::load_serialized(path.string());
  ASSERT_TRUE(module);

  Tensor x({4, 4}, Device{DeviceType::CPU, 0}, DataType::Float32, false);
  x.fill_(1.0f);
  Tensor y = module->forward(x).to(Device{DeviceType::CPU, 0});
  const auto *y_ptr = static_cast<const float *>(y.data());
  for (size_t i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(y_ptr[i], 1.0f);
  }

  std::remove(path.string().c_str());
}

TEST(InferenceTest, EngineLoadCanAcceptSerializedArtifactPath) {
  Tensor weight({2, 2}, Device{DeviceType::CPU, 0}, DataType::Float32, false);
  Tensor bias({2}, Device{DeviceType::CPU, 0}, DataType::Float32, false);
  auto *w = static_cast<float *>(weight.data());
  auto *b = static_cast<float *>(bias.data());
  w[0] = 2.0f;
  w[1] = 0.0f;
  w[2] = 0.0f;
  w[3] = 3.0f;
  b[0] = 1.0f;
  b[1] = -2.0f;

  const std::string config =
      R"({"type":"Linear","in_features":2,"out_features":2,"bias":true,"dtype":"float32"})";
  const auto path =
      write_npz_artifact("munet_cpp_engine_load_serialized", config,
                         {{"bias", bias}, {"weight", weight}});

  inference::Engine engine;
  engine.load(path.string());

  Tensor x({1, 2}, Device{DeviceType::CPU, 0}, DataType::Float32, false);
  auto *x_ptr = static_cast<float *>(x.data());
  x_ptr[0] = 4.0f;
  x_ptr[1] = 5.0f;

  Tensor y = engine.run(x).to(Device{DeviceType::CPU, 0});
  const auto *y_ptr = static_cast<const float *>(y.data());
  EXPECT_FLOAT_EQ(y_ptr[0], 9.0f);
  EXPECT_FLOAT_EQ(y_ptr[1], 13.0f);

  std::remove(path.string().c_str());
}

TEST(InferenceTest,
     LoadWeightsSerializedRestoresExistingModuleAndSetsEvalMode) {
  Tensor weight({2, 2}, Device{DeviceType::CPU, 0}, DataType::Float32, false);
  Tensor bias({2}, Device{DeviceType::CPU, 0}, DataType::Float32, false);
  auto *w = static_cast<float *>(weight.data());
  auto *b = static_cast<float *>(bias.data());
  w[0] = 1.0f;
  w[1] = 0.0f;
  w[2] = 0.0f;
  w[3] = 1.0f;
  b[0] = 0.0f;
  b[1] = 0.0f;

  const std::string config =
      R"({"type":"Sequential","layers":[{"type":"Dropout","p":0.75},{"type":"Linear","in_features":2,"out_features":2,"bias":true,"dtype":"float32"}]})";
  const auto path =
      write_npz_artifact("munet_cpp_load_weights_serialized", config,
                         {{"1.bias", bias}, {"1.weight", weight}});

  auto module = std::make_shared<nn::Sequential>();
  module->add(std::make_shared<nn::Dropout>(0.75f));
  module->add(std::make_shared<nn::Linear>(2, 2));
  module->train(true);

  inference::load_weights_serialized(module, path.string());

  Tensor x({1, 2}, Device{DeviceType::CPU, 0}, DataType::Float32, false);
  auto *x_ptr = static_cast<float *>(x.data());
  x_ptr[0] = 7.0f;
  x_ptr[1] = 11.0f;
  Tensor y = module->forward(x).to(Device{DeviceType::CPU, 0});
  const auto *y_ptr = static_cast<const float *>(y.data());
  EXPECT_FLOAT_EQ(y_ptr[0], 7.0f);
  EXPECT_FLOAT_EQ(y_ptr[1], 11.0f);

  std::remove(path.string().c_str());
}
