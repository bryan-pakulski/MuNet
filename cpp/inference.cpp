#include "inference.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace munet {
namespace {
using Json = nlohmann::json;
using Bytes = std::vector<uint8_t>;
constexpr size_t max_nodes = 100000;
constexpr uint64_t format_bytes = uint64_t{2} * 1024 * 1024 * 1024;
void require(bool valid, const std::string& message) {
  if (!valid) throw std::runtime_error(message);
}
void bounds(const Bytes& bytes, size_t offset, size_t count) {
  require(offset <= bytes.size() && count <= bytes.size() - offset, "truncated model archive");
}
uint64_t little(const Bytes& bytes, size_t offset, size_t count) {
  bounds(bytes, offset, count);
  uint64_t value = 0;
  for (size_t i = 0; i < count; ++i) value |= uint64_t(bytes[offset + i]) << (i * 8);
  return value;
}
std::string string_at(const Bytes& bytes, size_t offset, size_t count) {
  bounds(bytes, offset, count);
  return std::string(bytes.begin() + offset, bytes.begin() + offset + count);
}
uint32_t crc32(const Bytes& bytes, size_t offset, size_t count) {
  static const auto table = [] {
    std::array<uint32_t, 256> values{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int bit = 0; bit < 8; ++bit) c = (c >> 1) ^ ((c & 1) ? 0xedb88320u : 0u);
      values[i] = c;
    }
    return values;
  }();
  uint32_t c = 0xffffffffu;
  for (size_t i = 0; i < count; ++i) c = table[(c ^ bytes[offset + i]) & 255] ^ (c >> 8);
  return c ^ 0xffffffffu;
}

// Only the uncompressed ZIP subset emitted by munet.save/export is accepted.
// Entries are read in memory, never extracted. ZIP64 entry counts are supported.
class Archive {
 public:
  Bytes bytes;
  std::map<std::string, std::pair<size_t, size_t>> entries;
  std::set<std::string> used;
  explicit Archive(const std::string& path, uint64_t limit) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file), "cannot open model file");
    const auto length = file.tellg();
    require(length >= 22 && uint64_t(length) <= limit, "invalid or oversized model archive");
    bytes.resize(static_cast<size_t>(length));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(bool(file), "could not read model archive");
    size_t end = bytes.size() - 22;
    const size_t first = bytes.size() > 65557 ? bytes.size() - 65557 : 0;
    for (;;) {
      if (little(bytes, end, 4) == 0x06054b50 && end + 22 + little(bytes, end + 20, 2) == bytes.size()) break;
      require(end > first, "missing ZIP directory");
      --end;
    }
    require(little(bytes, end + 4, 2) == 0 && little(bytes, end + 6, 2) == 0, "multi-disk ZIP is unsupported");
    uint64_t count = little(bytes, end + 10, 2), size = little(bytes, end + 12, 4), offset = little(bytes, end + 16, 4);
    require(little(bytes, end + 8, 2) == count, "inconsistent ZIP entry count");
    size_t directory_end = end;
    if (count == 65535 || size == 0xffffffffu || offset == 0xffffffffu) {
      require(end >= 20 && little(bytes, end - 20, 4) == 0x07064b50, "missing ZIP64 locator");
      require(little(bytes, end - 16, 4) == 0 && little(bytes, end - 4, 4) == 1, "multi-disk ZIP64 is unsupported");
      auto at = little(bytes, end - 12, 8);
      require(at <= end - 20 && end - 20 - at >= 56, "invalid ZIP64 directory");
      require(little(bytes, at, 4) == 0x06064b50 && little(bytes, at + 4, 8) == 44, "invalid ZIP64 record");
      require(little(bytes, at + 16, 4) == 0 && little(bytes, at + 20, 4) == 0, "multi-disk ZIP64 is unsupported");
      count = little(bytes, at + 32, 8);
      require(little(bytes, at + 24, 8) == count, "inconsistent ZIP64 entry count");
      size = little(bytes, at + 40, 8); offset = little(bytes, at + 48, 8);
      directory_end = static_cast<size_t>(at);
    }
    require(count <= 3 * max_nodes + 1 && offset <= directory_end && size == directory_end - offset, "invalid ZIP directory bounds");
    size_t cursor = static_cast<size_t>(offset);
    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t i = 0; i < count; ++i) {
      bounds(bytes, cursor, 46);
      require(little(bytes, cursor, 4) == 0x02014b50, "invalid ZIP directory entry");
      auto flags = little(bytes, cursor + 8, 2), method = little(bytes, cursor + 10, 2);
      require(method == 0 && (flags & ~uint64_t{0x808}) == 0, "only uncompressed, unencrypted MuNet ZIP entries are supported");
      auto compressed = little(bytes, cursor + 20, 4), unpacked = little(bytes, cursor + 24, 4);
      auto name_size = little(bytes, cursor + 28, 2), extra_size = little(bytes, cursor + 30, 2), comment_size = little(bytes, cursor + 32, 2);
      require(little(bytes, cursor + 34, 2) == 0, "multi-disk ZIP entry is unsupported");
      uint64_t local = little(bytes, cursor + 42, 4);
      size_t next = cursor + 46 + name_size + extra_size + comment_size;
      require(next <= directory_end, "invalid ZIP entry length");
      auto name = string_at(bytes, cursor + 46, name_size);
      require(!name.empty() && name.size() <= 512 && name.find('\0') == std::string::npos, "invalid ZIP entry name");
      if (compressed == 0xffffffffu || unpacked == 0xffffffffu || local == 0xffffffffu) {
        size_t extra = cursor + 46 + name_size, extra_end = extra + extra_size;
        bool found = false;
        while (extra + 4 <= extra_end) {
          auto kind = little(bytes, extra, 2), len = little(bytes, extra + 2, 2);
          extra += 4;
          require(len <= extra_end - extra, "invalid ZIP extra field");
          if (kind == 1) {
            auto field = [&](uint64_t& value) {
              if (value != 0xffffffffu) return;
              require(len >= 8, "truncated ZIP64 extra field");
              value = little(bytes, extra, 8); extra += 8; len -= 8;
            };
            field(unpacked); field(compressed); field(local); found = true; break;
          }
          extra += len;
        }
        require(found, "missing ZIP64 entry sizes");
      }
      require(compressed == unpacked && unpacked <= limit && local < offset, "invalid ZIP entry size/offset");
      require(little(bytes, local, 4) == 0x04034b50 && little(bytes, local + 6, 2) == flags && little(bytes, local + 8, 2) == method, "invalid ZIP local header");
      auto local_name = little(bytes, local + 26, 2), local_extra = little(bytes, local + 28, 2);
      require(string_at(bytes, local + 30, local_name) == name, "ZIP local/directory names differ");
      uint64_t start = local + 30 + local_name + local_extra;
      require(start <= offset && unpacked <= offset - start, "ZIP entry overlaps directory");
      require(crc32(bytes, start, unpacked) == little(bytes, cursor + 16, 4), "ZIP entry checksum mismatch");
      require(entries.emplace(name, std::make_pair(start, unpacked)).second, "duplicate archive entry");
      ranges.emplace_back(local, start + unpacked);
      cursor = next;
    }
    require(cursor == directory_end, "unexpected ZIP directory bytes");
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) require(ranges[i - 1].second <= ranges[i].first, "overlapping ZIP entries");
  }
  std::pair<size_t, size_t> get(const std::string& name) {
    auto entry = entries.find(name);
    require(entry != entries.end(), "missing archive entry: " + name);
    used.insert(name);
    return entry->second;
  }
  std::string text(const std::string& name) {
    auto entry = get(name);
    return string_at(bytes, entry.first, entry.second);
  }
};

int64_t integer(const Json& value) {
  require(value.is_number_integer() && (!value.is_number_unsigned() || value.get<uint64_t>() <= uint64_t(INT64_MAX)), "expected bounded integer metadata");
  return value.get<int64_t>();
}
Shape integers(const Json& array) {
  require(array.is_array(), "expected integer array");
  Shape result;
  for (const auto& item : array) result.push_back(integer(item));
  return result;
}
std::vector<Id> ids(const Json& array, size_t upper) {
  std::vector<Id> result;
  for (auto id : integers(array)) {
    require(id >= 0 && uint64_t(id) < upper, "graph is not topologically ordered or references an invalid value");
    result.push_back(static_cast<Id>(id));
  }
  return result;
}
std::vector<float> tensor(Archive& archive, const std::string& name, const Shape& shape) {
  auto [at, size] = archive.get(name);
  require(size >= 10 && string_at(archive.bytes, at, 6) == std::string("\x93NUMPY", 6), "invalid NPY tensor");
  auto version = little(archive.bytes, at + 6, 2);
  require(version == 1 || version == 2, "unsupported NPY tensor version");
  size_t base = version == 1 ? 10 : 12;
  auto header_size = little(archive.bytes, at + 8, base - 8);
  require(base <= size && header_size <= 4096 && header_size <= size - base, "invalid NPY header size");
  auto header = string_at(archive.bytes, at + base, header_size);
  static const std::regex format(R"(^\{\s*'descr'\s*:\s*'<f4'\s*,\s*'fortran_order'\s*:\s*False\s*,\s*'shape'\s*:\s*\(([0-9, ]*)\)\s*,?\s*\}\s*$)");
  std::smatch match;
  require(std::regex_match(header, match, format), "tensor must use the MuNet little-endian FP32 row-major NPY encoding");
  Shape declared;
  std::istringstream dimensions(match[1].str());
  std::string part;
  while (std::getline(dimensions, part, ',')) {
    if (part.find_first_not_of(' ') == std::string::npos) continue;
    declared.push_back(std::stoll(part));
  }
  const auto count = numel(shape);
  require(declared == shape && count * 4 == size - base - header_size, "tensor shape/payload length mismatch");
  std::vector<float> result(count);
  at += base + header_size;
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "MuNet requires IEEE-754 FP32");
  const uint16_t endian = 1;
  if (*reinterpret_cast<const uint8_t*>(&endian) == 1) {
    std::memcpy(result.data(), archive.bytes.data() + at, count * 4);
    return result;
  }
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits = static_cast<uint32_t>(little(archive.bytes, at + i * 4, 4));
    std::memcpy(&result[i], &bits, 4);
  }
  return result;
}
std::vector<std::string> names(const Json& manifest, const std::string& prefix, size_t count) {
  auto field = manifest.find(prefix + "_names");
  std::vector<std::string> result;
  if (field == manifest.end()) {
    for (size_t i = 0; i < count; ++i) result.push_back(prefix + "_" + std::to_string(i));
  } else {
    require(field->is_array() && field->size() == count, "invalid " + prefix + " names");
    for (const auto& name : *field) {
      require(name.is_string(), "invalid tensor name");
      auto text = name.get<std::string>();
      require(!text.empty() && text.size() <= 1024 && text.find('\0') == std::string::npos, "invalid tensor name");
      result.push_back(text);
    }
  }
  require(std::set<std::string>(result.begin(), result.end()).size() == count, "duplicate tensor name");
  return result;
}
void validate_tree(const Json& tree, size_t count, int depth = 0) {
  require(depth <= 64 && tree.is_array() && tree.size() == 2 && tree[0].is_string(), "invalid output tree");
  const auto kind = tree[0].get<std::string>();
  const auto& value = tree[1];
  if (kind == "tensor") {
    auto index = integer(value);
    require(index >= 0 && uint64_t(index) < count, "invalid output tensor index");
  } else if (kind == "constant") {
    require(value.is_primitive(), "invalid output constant");
  } else if (kind == "tuple" || kind == "list") {
    require(value.is_array(), "invalid output sequence");
    for (const auto& child : value) validate_tree(child, count, depth + 1);
  } else if (kind == "dict") {
    require(value.is_array(), "invalid output dictionary");
    std::set<std::string> keys;
    for (const auto& pair : value) {
      require(pair.is_array() && pair.size() == 2 && pair[0].is_string(), "invalid output dictionary entry");
      require(keys.insert(pair[0].get<std::string>()).second, "duplicate output dictionary key");
      validate_tree(pair[1], count, depth + 1);
    }
  } else throw std::runtime_error("unknown output container");
}
std::string shape_string(const Shape& shape) {
  std::string result = "(";
  for (size_t i = 0; i < shape.size(); ++i) result += (i ? ", " : "") + std::to_string(shape[i]);
  return result + ")";
}
}  // namespace

struct Model::Impl {
  std::unique_ptr<Plan> plan;
  std::vector<TensorInfo> inputs, outputs;
  std::vector<size_t> feed_indices;
  mutable std::mutex mutex;
};

Model::Model(const std::string& path, const ModelOptions& options) : impl_(std::make_unique<Impl>()) {
  try {
    require(options.max_memory_bytes > 0, "max_memory_bytes must be positive");
    bool gpu = options.device != "cpu";
    unsigned device_index = 0;
    if (gpu && options.device != "vulkan") {
      require(options.device.rfind("vulkan:", 0) == 0, "device must be cpu, vulkan or vulkan:N");
      auto index = options.device.substr(7);
      require(!index.empty() && index.find_first_not_of("0123456789") == std::string::npos, "invalid Vulkan device index");
      auto parsed = std::stoull(index);
      require(parsed <= std::numeric_limits<unsigned>::max(), "Vulkan device index out of range");
      device_index = static_cast<unsigned>(parsed);
    }
    Archive archive(path, std::min(options.max_memory_bytes, format_bytes));
    require(archive.get("manifest.json").second <= 32 * 1024 * 1024, "oversized manifest");
    auto manifest = Json::parse(archive.text("manifest.json"), [](int depth, Json::parse_event_t, Json&) {
      require(depth <= 128, "model metadata is too deeply nested"); return true;
    });
    require(manifest.at("format") == "munet-program" && integer(manifest.at("version")) == 1, "expected an inference program; training checkpoints are not executable models");
    require(manifest.at("dtype") == "float32", "unsupported model dtype");
    require(manifest.at("updates").is_array() && manifest.at("updates").empty(), "training/state-update programs cannot be loaded for inference; export the eval-mode model");
    const auto& nodes = manifest.at("nodes");
    require(nodes.is_array() && nodes.size() <= max_nodes, "invalid node count");
    Graph graph;
    std::vector<size_t> depths;
    for (size_t i = 0; i < nodes.size(); ++i) {
      const auto& node = nodes[i];
      auto op = node.at("op").get<std::string>();
      auto shape = integers(node.at("shape")), attrs = integers(node.at("attrs"));
      numel(shape);
      auto arguments = ids(node.at("inputs"), i);
      size_t depth = 0;
      for (auto id : arguments) depth = std::max(depth, depths[id] + 1);
      require(depth <= 1024, "graph dependency depth exceeds the inference limit (1024)");
      depths.push_back(depth);
      if (op == "input" || op == "parameter" || op == "constant") {
        require(arguments.empty() && attrs.empty(), "leaf contains inputs/attributes");
        std::vector<float> data;
        if (op != "input") {
          auto entry = "tensors/" + std::to_string(i) + ".npy";
          require(node.at("tensor") == entry, "invalid tensor entry name");
          data = tensor(archive, entry, shape);
        }
        graph.leaf(op, node.at("name").get<std::string>(), shape, data);
      } else {
        auto id = graph.op(op, arguments, attrs);
        require(graph.at(id).shape == shape, "declared shape differs from inferred shape");
      }
    }
    auto output_ids = ids(manifest.at("outputs"), nodes.size());
    require(manifest.at("single_output").is_boolean() && (!manifest.at("single_output").get<bool>() || output_ids.size() == 1), "invalid single-output contract");
    if (manifest.contains("output_tree") && !manifest["output_tree"].is_null()) validate_tree(manifest["output_tree"], output_ids.size());
    bool fuse = true;
    if (manifest.contains("fuse")) {
      require(manifest["fuse"].is_boolean(), "invalid fusion setting"); fuse = manifest["fuse"].get<bool>();
    }
    impl_->plan = std::make_unique<Plan>(graph, output_ids, std::vector<std::pair<Id, Id>>{}, fuse);
    require(impl_->plan->stats().at("arena_bytes") <= options.max_memory_bytes, "planned arena exceeds max_memory_bytes");
    const auto& specs = manifest.at("input_specs");
    require(specs.is_array() && specs.size() <= max_nodes, "invalid input specifications");
    auto input_names = names(manifest, "input", specs.size()), output_names = names(manifest, "output", output_ids.size());
    for (size_t i = 0; i < specs.size(); ++i) {
      auto shape = integers(specs[i]); numel(shape); impl_->inputs.push_back({input_names[i], shape});
    }
    for (size_t i = 0; i < output_ids.size(); ++i) impl_->outputs.push_back({output_names[i], graph.at(output_ids[i]).shape});
    auto mapping = integers(manifest.at("feed_indices"));
    require(mapping.size() == impl_->plan->inputs.size() && std::set<int64_t>(mapping.begin(), mapping.end()).size() == mapping.size(), "invalid runtime input mapping");
    for (size_t i = 0; i < mapping.size(); ++i) {
      require(mapping[i] >= 0 && uint64_t(mapping[i]) < specs.size(), "runtime input index out of range");
      require(impl_->inputs[mapping[i]].shape == graph.at(impl_->plan->inputs[i]).shape, "runtime input mapping shape mismatch");
      impl_->feed_indices.push_back(static_cast<size_t>(mapping[i]));
    }
    std::vector<std::vector<uint32_t>> binaries;
    if (manifest.contains("vulkan")) {
      const auto& shaders = manifest["vulkan"];
      require(shaders.is_array() && shaders.size() == impl_->plan->kernels.size(), "Vulkan kernel count mismatch; re-export with this runtime");
      for (size_t i = 0; i < shaders.size(); ++i) {
        auto source_name = "shaders/" + std::to_string(i) + ".comp", binary_name = "shaders/" + std::to_string(i) + ".spv";
        require(shaders[i] == Json{{"source", source_name}, {"spirv", binary_name}}, "invalid shader entry name");
        require(archive.text(source_name) == impl_->plan->kernels[i].source, "Vulkan source does not match this runtime; re-export the model");
        auto [offset, count] = archive.get(binary_name);
        require(count >= 20 && count % 4 == 0 && little(archive.bytes, offset, 4) == 0x07230203, "invalid embedded SPIR-V");
        std::vector<uint32_t> binary(count / 4);
        for (size_t j = 0; j < binary.size(); ++j) binary[j] = static_cast<uint32_t>(little(archive.bytes, offset + j * 4, 4));
        binaries.push_back(std::move(binary));
      }
    }
    require(archive.used.size() == archive.entries.size(), "unexpected entries in model archive");
    if (gpu) {
      require(manifest.contains("vulkan"), "Vulkan inference needs shaders; export with include_vulkan=True in Python");
      require(vulkan_built(), "this SDK was built without Vulkan support");
      impl_->plan->enable_vulkan(binaries, device_index);
    }
  } catch (const std::exception& error) {
    throw std::runtime_error("MuNet model '" + path + "': " + error.what());
  }
}
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
Model::Impl& Model::impl() const {
  if (!impl_) throw std::logic_error("cannot use a moved-from MuNet Model");
  return *impl_;
}
const std::vector<TensorInfo>& Model::inputs() const { return impl().inputs; }
const std::vector<TensorInfo>& Model::outputs() const { return impl().outputs; }
std::vector<Tensor> Model::run(const std::vector<Tensor>& values) {
  auto& self = impl();
  std::lock_guard<std::mutex> lock(self.mutex);
  require(values.size() == self.inputs.size(), "expected " + std::to_string(self.inputs.size()) + " input tensors, received " + std::to_string(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    require(values[i].shape == self.inputs[i].shape, "input '" + self.inputs[i].name + "' expected shape " + shape_string(self.inputs[i].shape) + ", received " + shape_string(values[i].shape));
    require(values[i].data.size() == numel(values[i].shape), "input '" + self.inputs[i].name + "' data length does not match its shape");
  }
  std::vector<std::vector<float>> feeds;
  for (auto index : self.feed_indices) feeds.push_back(values[index].data);
  self.plan->run(feeds);
  std::vector<Tensor> result;
  for (auto id : self.plan->outputs) result.push_back({self.plan->graph.at(id).shape, self.plan->read(id)});
  return result;
}
std::map<std::string, Tensor> Model::run_named(const std::map<std::string, Tensor>& values) {
  auto& self = impl();
  require(values.size() == self.inputs.size(), "named inputs must match the model signature exactly");
  std::vector<Tensor> ordered;
  for (const auto& input : self.inputs) {
    auto found = values.find(input.name);
    require(found != values.end(), "missing input: " + input.name);
    ordered.push_back(found->second);
  }
  auto outputs = run(ordered);
  std::map<std::string, Tensor> result;
  for (size_t i = 0; i < outputs.size(); ++i) result.emplace(self.outputs[i].name, std::move(outputs[i]));
  return result;
}
std::map<std::string, uint64_t> Model::stats() const {
  auto& self = impl(); std::lock_guard<std::mutex> lock(self.mutex); return self.plan->stats();
}
std::string Model::device_name() const {
  auto& self = impl(); std::lock_guard<std::mutex> lock(self.mutex); return self.plan->device_name();
}
void Model::synchronize() {
  auto& self = impl(); std::lock_guard<std::mutex> lock(self.mutex); self.plan->synchronize();
}
}  // namespace munet
