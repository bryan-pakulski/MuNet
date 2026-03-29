#include "inference.hpp"

#include "core/backend.hpp"
#include "nn.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace munet {
namespace inference {
namespace detail {
namespace {

constexpr const char *kSerializationFormatName = "munet_model";
constexpr int64_t kSerializationFormatRevision = 1;
constexpr const char *kSerializationLegacyTag = "munet_model_v1";
constexpr const char *kSerializationArtifactKind = "deploy_model";
constexpr const char *kSerializationArtifactScope = "runtime_only";
constexpr const char *kSerializationDefaultLoadMode = "eval";
constexpr const char *kSerializationRecommendedLoader = "load_for_inference";
constexpr const char *kSerializationCompileContractPolicy = "external";
constexpr const char *kSerializationProducer = "munet";
const std::vector<std::string> kForbiddenTrainingKeyTokens = {
    "optim",      "optimizer", "scheduler", "scaler", "master_weight",
    "checkpoint", "epoch",     "step",      "grad",
};

std::vector<uint8_t> read_file_bytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Unable to open serialized artifact: " + path);
  }

  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (size < 0) {
    throw std::runtime_error("Unable to determine serialized artifact size: " +
                             path);
  }

  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (!bytes.empty()) {
    in.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  }
  if (!in) {
    throw std::runtime_error("Failed to read serialized artifact: " + path);
  }
  return bytes;
}

uint16_t read_u16_le(const std::vector<uint8_t> &bytes, size_t offset) {
  if (offset + 2 > bytes.size()) {
    throw std::runtime_error("Unexpected end of file while reading uint16");
  }
  return static_cast<uint16_t>(bytes[offset]) |
         (static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

uint32_t read_u32_le(const std::vector<uint8_t> &bytes, size_t offset) {
  if (offset + 4 > bytes.size()) {
    throw std::runtime_error("Unexpected end of file while reading uint32");
  }
  return static_cast<uint32_t>(bytes[offset]) |
         (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

struct NpyArray {
  std::string descr;
  bool fortran_order = false;
  std::vector<size_t> shape;
  std::vector<uint8_t> data;
};

std::string trim(std::string value) {
  const auto begin =
      std::find_if_not(value.begin(), value.end(),
                       [](unsigned char ch) { return std::isspace(ch) != 0; });
  const auto end =
      std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
      }).base();
  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

std::string extract_npy_header_string(const std::string &header,
                                      const std::string &key) {
  const std::string needle = "'" + key + "':";
  const size_t key_pos = header.find(needle);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("NPY header missing key '" + key + "'");
  }

  size_t value_pos = header.find_first_not_of(" ", key_pos + needle.size());
  if (value_pos == std::string::npos || header[value_pos] != '\'') {
    throw std::runtime_error("NPY header key '" + key + "' is not a string");
  }
  ++value_pos;
  const size_t value_end = header.find('\'', value_pos);
  if (value_end == std::string::npos) {
    throw std::runtime_error("NPY header key '" + key +
                             "' has unterminated string");
  }
  return header.substr(value_pos, value_end - value_pos);
}

bool extract_npy_header_bool(const std::string &header,
                             const std::string &key) {
  const std::string needle = "'" + key + "':";
  const size_t key_pos = header.find(needle);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("NPY header missing key '" + key + "'");
  }

  size_t value_pos = header.find_first_not_of(" ", key_pos + needle.size());
  if (value_pos == std::string::npos) {
    throw std::runtime_error("NPY header key '" + key + "' has no value");
  }
  if (header.compare(value_pos, 4, "True") == 0) {
    return true;
  }
  if (header.compare(value_pos, 5, "False") == 0) {
    return false;
  }
  throw std::runtime_error("NPY header key '" + key + "' is not a boolean");
}

std::vector<size_t> extract_npy_header_shape(const std::string &header) {
  const std::string needle = "'shape':";
  const size_t key_pos = header.find(needle);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("NPY header missing key 'shape'");
  }

  size_t value_pos = header.find('(', key_pos + needle.size());
  if (value_pos == std::string::npos) {
    throw std::runtime_error("NPY header shape is missing '('");
  }
  const size_t value_end = header.find(')', value_pos);
  if (value_end == std::string::npos) {
    throw std::runtime_error("NPY header shape is missing ')'");
  }

  const std::string inside =
      trim(header.substr(value_pos + 1, value_end - value_pos - 1));
  if (inside.empty()) {
    return {};
  }

  std::vector<size_t> shape;
  std::stringstream ss(inside);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim(token);
    if (token.empty()) {
      continue;
    }
    const long long dim = std::stoll(token);
    if (dim < 0) {
      throw std::runtime_error("NPY shape contains a negative dimension");
    }
    shape.push_back(static_cast<size_t>(dim));
  }
  return shape;
}

NpyArray parse_npy_array(const std::vector<uint8_t> &bytes) {
  static constexpr uint8_t kMagic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  if (bytes.size() < 10 ||
      !std::equal(std::begin(kMagic), std::end(kMagic), bytes.begin())) {
    throw std::runtime_error("Serialized payload is not a valid .npy array");
  }

  const uint8_t major = bytes[6];
  const uint8_t minor = bytes[7];
  (void)minor;

  size_t header_len = 0;
  size_t header_offset = 0;
  if (major == 1) {
    header_len = read_u16_le(bytes, 8);
    header_offset = 10;
  } else if (major == 2) {
    header_len = read_u32_le(bytes, 8);
    header_offset = 12;
  } else {
    throw std::runtime_error("Unsupported .npy version: " +
                             std::to_string(major));
  }

  if (header_offset + header_len > bytes.size()) {
    throw std::runtime_error("Invalid .npy header length");
  }

  const std::string header(
      reinterpret_cast<const char *>(bytes.data() + header_offset), header_len);

  NpyArray array;
  array.descr = extract_npy_header_string(header, "descr");
  array.fortran_order = extract_npy_header_bool(header, "fortran_order");
  array.shape = extract_npy_header_shape(header);
  if (array.fortran_order) {
    throw std::runtime_error("Fortran-ordered .npy arrays are not supported");
  }

  const size_t data_offset = header_offset + header_len;
  array.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_offset),
                    bytes.end());
  return array;
}

std::unordered_map<std::string, std::vector<uint8_t>>
parse_npz_archive(const std::string &path) {
  const std::vector<uint8_t> bytes = read_file_bytes(path);
  if (bytes.size() < 22) {
    throw std::runtime_error(
        "Serialized artifact is too small to be a valid .npz archive: " + path);
  }

  size_t eocd_offset = std::string::npos;
  const size_t min_search =
      bytes.size() > (22 + 65535) ? bytes.size() - (22 + 65535) : 0;
  for (size_t pos = bytes.size() - 22 + 1; pos-- > min_search;) {
    if (read_u32_le(bytes, pos) == 0x06054b50U) {
      eocd_offset = pos;
      break;
    }
    if (pos == 0) {
      break;
    }
  }
  if (eocd_offset == std::string::npos) {
    throw std::runtime_error(
        "Unable to locate .npz central directory in artifact: " + path);
  }

  const uint16_t entry_count = read_u16_le(bytes, eocd_offset + 10);
  const uint32_t central_dir_size = read_u32_le(bytes, eocd_offset + 12);
  const uint32_t central_dir_offset = read_u32_le(bytes, eocd_offset + 16);
  if (static_cast<size_t>(central_dir_offset) + central_dir_size >
      bytes.size()) {
    throw std::runtime_error(
        "Invalid .npz central directory bounds in artifact: " + path);
  }

  std::unordered_map<std::string, std::vector<uint8_t>> entries;
  size_t pos = central_dir_offset;
  for (uint16_t i = 0; i < entry_count; ++i) {
    if (read_u32_le(bytes, pos) != 0x02014b50U) {
      throw std::runtime_error(
          "Invalid .npz central directory entry in artifact: " + path);
    }

    const uint16_t compression_method = read_u16_le(bytes, pos + 10);
    const uint32_t compressed_size = read_u32_le(bytes, pos + 20);
    const uint32_t uncompressed_size = read_u32_le(bytes, pos + 24);
    const uint16_t file_name_len = read_u16_le(bytes, pos + 28);
    const uint16_t extra_len = read_u16_le(bytes, pos + 30);
    const uint16_t comment_len = read_u16_le(bytes, pos + 32);
    const uint32_t local_header_offset = read_u32_le(bytes, pos + 42);

    const size_t file_name_offset = pos + 46;
    const size_t header_end =
        file_name_offset + file_name_len + extra_len + comment_len;
    if (header_end > bytes.size()) {
      throw std::runtime_error(
          "Invalid .npz central directory filename bounds in artifact: " +
          path);
    }

    const std::string filename(
        reinterpret_cast<const char *>(
            bytes.data() + static_cast<std::ptrdiff_t>(file_name_offset)),
        file_name_len);

    if (compression_method != 0) {
      throw std::runtime_error("Compressed .npz entries are not supported by "
                               "the C++ deploy loader: " +
                               filename);
    }
    if (compressed_size != uncompressed_size) {
      throw std::runtime_error(
          "Unexpected compressed size mismatch in serialized artifact: " +
          filename);
    }

    if (static_cast<size_t>(local_header_offset) + 30 > bytes.size() ||
        read_u32_le(bytes, local_header_offset) != 0x04034b50U) {
      throw std::runtime_error("Invalid .npz local header in artifact: " +
                               filename);
    }

    const uint16_t local_name_len =
        read_u16_le(bytes, local_header_offset + 26);
    const uint16_t local_extra_len =
        read_u16_le(bytes, local_header_offset + 28);
    const size_t data_offset = static_cast<size_t>(local_header_offset) + 30 +
                               local_name_len + local_extra_len;
    const size_t data_end = data_offset + uncompressed_size;
    if (data_end > bytes.size()) {
      throw std::runtime_error("Invalid .npz entry bounds in artifact: " +
                               filename);
    }

    entries.emplace(
        filename, std::vector<uint8_t>(
                      bytes.begin() + static_cast<std::ptrdiff_t>(data_offset),
                      bytes.begin() + static_cast<std::ptrdiff_t>(data_end)));

    pos = header_end;
  }

  return entries;
}

size_t element_count(const std::vector<size_t> &shape) {
  if (shape.empty()) {
    return 1;
  }
  size_t count = 1;
  for (size_t dim : shape) {
    count *= dim;
  }
  return count;
}

std::string decode_unicode_scalar(const NpyArray &array) {
  if (array.shape.size() > 1 || (!array.shape.empty() && array.shape[0] != 1)) {
    throw std::runtime_error("Expected scalar or size-1 unicode array");
  }
  if (array.descr.size() < 3 || array.descr[0] != '<' ||
      array.descr[1] != 'U') {
    throw std::runtime_error("Expected little-endian unicode scalar array");
  }

  const size_t char_count =
      static_cast<size_t>(std::stoull(array.descr.substr(2)));
  const size_t expected_bytes = char_count * 4;
  if (array.data.size() < expected_bytes) {
    throw std::runtime_error("Unicode scalar payload is truncated");
  }

  std::string value;
  value.reserve(char_count);
  for (size_t i = 0; i < char_count; ++i) {
    const uint32_t codepoint =
        static_cast<uint32_t>(array.data[i * 4]) |
        (static_cast<uint32_t>(array.data[i * 4 + 1]) << 8) |
        (static_cast<uint32_t>(array.data[i * 4 + 2]) << 16) |
        (static_cast<uint32_t>(array.data[i * 4 + 3]) << 24);
    if (codepoint == 0) {
      break;
    }
    if (codepoint > 0x7f) {
      throw std::runtime_error(
          "Only ASCII unicode scalar metadata is supported");
    }
    value.push_back(static_cast<char>(codepoint));
  }
  return value;
}

std::string decode_bytes_scalar(const NpyArray &array) {
  if (array.descr.size() < 3 ||
      (array.descr[0] != '|' && array.descr[0] != '<') ||
      array.descr[1] != 'S') {
    throw std::runtime_error("Expected bytes scalar array");
  }
  const size_t byte_count =
      static_cast<size_t>(std::stoull(array.descr.substr(2)));
  if (array.data.size() < byte_count) {
    throw std::runtime_error("Bytes scalar payload is truncated");
  }
  std::string value(reinterpret_cast<const char *>(array.data.data()),
                    byte_count);
  const size_t nul = value.find('\0');
  if (nul != std::string::npos) {
    value.resize(nul);
  }
  return value;
}

std::string decode_string_scalar(const NpyArray &array) {
  if (array.descr.size() >= 2 && array.descr[1] == 'U') {
    return decode_unicode_scalar(array);
  }
  if (array.descr.size() >= 2 && array.descr[1] == 'S') {
    return decode_bytes_scalar(array);
  }
  throw std::runtime_error("Unsupported string scalar descriptor: " +
                           array.descr);
}

int64_t decode_int_scalar(const NpyArray &array) {
  if (!(array.shape.empty() ||
        (array.shape.size() == 1 && array.shape[0] == 1))) {
    throw std::runtime_error("Expected scalar integer metadata");
  }
  if (array.descr == "<i8") {
    if (array.data.size() < 8) {
      throw std::runtime_error("Integer scalar payload is truncated");
    }
    int64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      value |= static_cast<int64_t>(array.data[static_cast<size_t>(i)])
               << (8 * i);
    }
    return value;
  }
  if (array.descr == "<i4") {
    if (array.data.size() < 4) {
      throw std::runtime_error("Integer scalar payload is truncated");
    }
    int32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      value |= static_cast<int32_t>(array.data[static_cast<size_t>(i)])
               << (8 * i);
    }
    return value;
  }
  throw std::runtime_error("Unsupported integer scalar descriptor: " +
                           array.descr);
}

bool decode_bool_scalar(const NpyArray &array) {
  if (!(array.shape.empty() ||
        (array.shape.size() == 1 && array.shape[0] == 1))) {
    throw std::runtime_error("Expected scalar boolean metadata");
  }
  if (array.data.empty()) {
    throw std::runtime_error("Boolean scalar payload is truncated");
  }
  if (array.descr == "|b1" || array.descr == "|?" || array.descr == "?") {
    return array.data[0] != 0;
  }
  throw std::runtime_error("Unsupported boolean scalar descriptor: " +
                           array.descr);
}

DataType data_type_from_npy_descr(const std::string &descr) {
  if (descr == "<f4") {
    return DataType::Float32;
  }
  if (descr == "<f2") {
    return DataType::Float16;
  }
  if (descr == "<i4") {
    return DataType::Int32;
  }
  throw std::runtime_error("Unsupported tensor dtype in serialized artifact: " +
                           descr);
}

Tensor tensor_from_npy_array(const NpyArray &array) {
  const DataType dtype = data_type_from_npy_descr(array.descr);
  Shape shape;
  shape.reserve(array.shape.size());
  for (size_t dim : array.shape) {
    if (dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "Serialized tensor dimension exceeds MuNet Shape range");
    }
    shape.push_back(static_cast<int>(dim));
  }

  Tensor tensor(shape, Device{DeviceType::CPU, 0}, dtype, false);
  const size_t expected_bytes = element_count(array.shape) * dtype_size(dtype);
  if (array.data.size() != expected_bytes) {
    throw std::runtime_error("Serialized tensor payload size mismatch");
  }
  if (expected_bytes > 0) {
    std::memcpy(tensor.data(), array.data.data(), expected_bytes);
  }
  return tensor;
}

struct JsonValue {
  using Object = std::map<std::string, JsonValue>;
  using Array = std::vector<JsonValue>;
  using Value =
      std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

  Value value;

  bool is_object() const { return std::holds_alternative<Object>(value); }
  bool is_array() const { return std::holds_alternative<Array>(value); }
  bool is_string() const { return std::holds_alternative<std::string>(value); }
  bool is_bool() const { return std::holds_alternative<bool>(value); }
  bool is_number() const { return std::holds_alternative<double>(value); }

  const Object &as_object() const { return std::get<Object>(value); }
  const Array &as_array() const { return std::get<Array>(value); }
  const std::string &as_string() const { return std::get<std::string>(value); }
  bool as_bool() const { return std::get<bool>(value); }
  double as_number() const { return std::get<double>(value); }
  int as_int() const { return static_cast<int>(std::llround(as_number())); }
};

class JsonParser {
public:
  explicit JsonParser(std::string text) : text_(std::move(text)) {}

  JsonValue parse() {
    skip_ws();
    JsonValue value = parse_value();
    skip_ws();
    if (pos_ != text_.size()) {
      throw std::runtime_error("Unexpected trailing JSON content");
    }
    return value;
  }

private:
  JsonValue parse_value() {
    skip_ws();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("Unexpected end of JSON input");
    }

    const char ch = text_[pos_];
    if (ch == '{') {
      return JsonValue{parse_object()};
    }
    if (ch == '[') {
      return JsonValue{parse_array()};
    }
    if (ch == '"') {
      return JsonValue{parse_string()};
    }
    if (ch == 't' || ch == 'f') {
      return JsonValue{parse_bool()};
    }
    if (ch == 'n') {
      parse_null();
      return JsonValue{nullptr};
    }
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch)) != 0) {
      return JsonValue{parse_number()};
    }
    throw std::runtime_error(
        std::string("Unsupported JSON token starting with '") + ch + "'");
  }

  JsonValue::Object parse_object() {
    expect('{');
    JsonValue::Object object;
    skip_ws();
    if (try_consume('}')) {
      return object;
    }

    while (true) {
      const std::string key = parse_string();
      skip_ws();
      expect(':');
      object.emplace(key, parse_value());
      skip_ws();
      if (try_consume('}')) {
        break;
      }
      expect(',');
    }
    return object;
  }

  JsonValue::Array parse_array() {
    expect('[');
    JsonValue::Array array;
    skip_ws();
    if (try_consume(']')) {
      return array;
    }

    while (true) {
      array.push_back(parse_value());
      skip_ws();
      if (try_consume(']')) {
        break;
      }
      expect(',');
    }
    return array;
  }

  std::string parse_string() {
    expect('"');
    std::string value;
    while (pos_ < text_.size()) {
      char ch = text_[pos_++];
      if (ch == '"') {
        return value;
      }
      if (ch == '\\') {
        if (pos_ >= text_.size()) {
          throw std::runtime_error("Invalid JSON escape sequence");
        }
        const char esc = text_[pos_++];
        switch (esc) {
        case '"':
          value.push_back('"');
          break;
        case '\\':
          value.push_back('\\');
          break;
        case '/':
          value.push_back('/');
          break;
        case 'b':
          value.push_back('\b');
          break;
        case 'f':
          value.push_back('\f');
          break;
        case 'n':
          value.push_back('\n');
          break;
        case 'r':
          value.push_back('\r');
          break;
        case 't':
          value.push_back('\t');
          break;
        default:
          throw std::runtime_error("Unsupported JSON escape sequence");
        }
        continue;
      }
      value.push_back(ch);
    }
    throw std::runtime_error("Unterminated JSON string");
  }

  bool parse_bool() {
    if (text_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      return true;
    }
    if (text_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      return false;
    }
    throw std::runtime_error("Invalid JSON boolean literal");
  }

  void parse_null() {
    if (text_.compare(pos_, 4, "null") != 0) {
      throw std::runtime_error("Invalid JSON null literal");
    }
    pos_ += 4;
  }

  double parse_number() {
    const size_t start = pos_;
    if (text_[pos_] == '-') {
      ++pos_;
    }
    while (pos_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0) {
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0) {
        ++pos_;
      }
    }
    return std::stod(text_.substr(start, pos_ - start));
  }

  void skip_ws() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_])) != 0) {
      ++pos_;
    }
  }

  void expect(char ch) {
    skip_ws();
    if (pos_ >= text_.size() || text_[pos_] != ch) {
      throw std::runtime_error(std::string("Expected JSON character '") + ch +
                               "'");
    }
    ++pos_;
  }

  bool try_consume(char ch) {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == ch) {
      ++pos_;
      return true;
    }
    return false;
  }

  std::string text_;
  size_t pos_ = 0;
};

const JsonValue &required_field(const JsonValue::Object &object,
                                const std::string &key) {
  const auto it = object.find(key);
  if (it == object.end()) {
    throw std::runtime_error(
        "Serialized module config is missing required field '" + key + "'");
  }
  return it->second;
}

DataType dtype_from_name(const std::string &name) {
  if (name == "float32") {
    return DataType::Float32;
  }
  if (name == "float16") {
    return DataType::Float16;
  }
  if (name == "int32") {
    return DataType::Int32;
  }
  throw std::runtime_error("Unsupported serialized dtype name: " + name);
}

TensorOptions options_from_config(const JsonValue::Object &config) {
  TensorOptions options;
  const auto dtype_it = config.find("dtype");
  if (dtype_it != config.end()) {
    options.dtype = dtype_from_name(dtype_it->second.as_string());
  }
  return options;
}

std::shared_ptr<nn::Module>
build_module_from_config(const JsonValue &config_value) {
  if (!config_value.is_object()) {
    throw std::runtime_error(
        "Serialized module config root must be a JSON object");
  }

  const auto &config = config_value.as_object();
  const std::string type = required_field(config, "type").as_string();
  const TensorOptions options = options_from_config(config);

  if (type == "Sequential") {
    auto sequence = std::make_shared<nn::Sequential>(options);
    const auto &layers = required_field(config, "layers").as_array();
    for (const auto &layer : layers) {
      sequence->add(build_module_from_config(layer));
    }
    return sequence;
  }
  if (type == "Linear") {
    return std::make_shared<nn::Linear>(
        required_field(config, "in_features").as_int(),
        required_field(config, "out_features").as_int(),
        required_field(config, "bias").as_bool(), options);
  }
  if (type == "Conv2d") {
    return std::make_shared<nn::Conv2d>(
        required_field(config, "in_channels").as_int(),
        required_field(config, "out_channels").as_int(),
        required_field(config, "kernel_size").as_int(),
        required_field(config, "stride").as_int(),
        required_field(config, "padding").as_int(), options);
  }
  if (type == "MaxPool2d") {
    return std::make_shared<nn::MaxPool2d>(
        required_field(config, "kernel_size").as_int(),
        required_field(config, "stride").as_int(),
        required_field(config, "padding").as_int());
  }
  if (type == "BatchNorm2d") {
    return std::make_shared<nn::BatchNorm2d>(
        required_field(config, "num_features").as_int(),
        static_cast<float>(required_field(config, "eps").as_number()),
        static_cast<float>(required_field(config, "momentum").as_number()),
        options);
  }
  if (type == "Upsample") {
    return std::make_shared<nn::Upsample>(
        required_field(config, "scale_factor").as_int());
  }
  if (type == "GlobalAvgPool2d") {
    return std::make_shared<nn::GlobalAvgPool2d>();
  }
  if (type == "ReLU") {
    return std::make_shared<nn::ReLU>();
  }
  if (type == "Sigmoid") {
    return std::make_shared<nn::Sigmoid>();
  }
  if (type == "Tanh") {
    return std::make_shared<nn::Tanh>();
  }
  if (type == "GELU") {
    return std::make_shared<nn::GELU>();
  }
  if (type == "LeakyReLU") {
    return std::make_shared<nn::LeakyReLU>(static_cast<float>(
        required_field(config, "negative_slope").as_number()));
  }
  if (type == "Dropout") {
    return std::make_shared<nn::Dropout>(
        static_cast<float>(required_field(config, "p").as_number()));
  }
  if (type == "Embedding") {
    return std::make_shared<nn::Embedding>(
        required_field(config, "num_embeddings").as_int(),
        required_field(config, "embedding_dim").as_int(), options);
  }
  if (type == "LayerNorm") {
    return std::make_shared<nn::LayerNorm>(
        required_field(config, "normalized_shape").as_int(),
        static_cast<float>(required_field(config, "eps").as_number()), options);
  }
  if (type == "RMSNorm") {
    return std::make_shared<nn::RMSNorm>(
        required_field(config, "normalized_shape").as_int(),
        static_cast<float>(required_field(config, "eps").as_number()), options);
  }
  if (type == "MultiHeadAttention") {
    return std::make_shared<nn::MultiHeadAttention>(
        required_field(config, "embed_dim").as_int(),
        required_field(config, "num_heads").as_int(),
        required_field(config, "causal").as_bool(), options);
  }
  if (type == "Flatten") {
    return std::make_shared<nn::Flatten>();
  }

  throw std::runtime_error(
      "Unsupported serialized module type for C++ reconstruction: " + type);
}

void copy_tensor_into(Tensor &target, const Tensor &source) {
  if (target.shape() != source.shape()) {
    throw std::runtime_error("Serialized tensor shape mismatch for '" +
                             target.name() + "': expected " +
                             to_string(target.shape()) + ", got " +
                             to_string(source.shape()));
  }

  Tensor prepared = source;
  if (prepared.dtype() != target.dtype()) {
    prepared = prepared.to(target.dtype());
  }
  if (prepared.device() != target.device()) {
    prepared = prepared.to(target.device());
  }
  if (prepared.bytes() != target.bytes()) {
    throw std::runtime_error("Serialized tensor byte-size mismatch for '" +
                             target.name() + "'");
  }

  target.impl_->backend().copy(prepared.data(), target.data(), target.bytes(),
                               prepared.device(), target.device());
  target.bump_version();
}

std::vector<std::string>
parse_manifest_tensor_names(const std::string &json_text) {
  const JsonValue parsed = JsonParser(json_text).parse();
  if (!parsed.is_array()) {
    throw std::runtime_error("Serialized tensor manifest must be a JSON array");
  }

  std::vector<std::string> names;
  for (const auto &entry : parsed.as_array()) {
    names.push_back(entry.as_string());
  }
  std::sort(names.begin(), names.end());
  return names;
}

struct ParsedArtifact {
  std::string format_name;
  int64_t format_revision = 0;
  std::string legacy_tag;
  std::string producer;
  std::string artifact_kind;
  std::string artifact_scope;
  std::string default_load_mode;
  bool contains_training_state = false;
  std::string recommended_loader;
  std::string compile_contract_policy;
  std::string config_json;
  std::vector<std::string> tensor_manifest;
  std::unordered_map<std::string, Tensor> tensors;
};

ParsedArtifact parse_and_validate_artifact(const std::string &path) {
  const auto entries = parse_npz_archive(path);
  ParsedArtifact artifact;
  std::vector<std::string> payload_names;

  for (const auto &entry : entries) {
    std::string name = entry.first;
    if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") {
      name.resize(name.size() - 4);
    }

    const NpyArray array = parse_npy_array(entry.second);
    if (name.rfind("__", 0) == 0) {
      if (name == "__format_name__") {
        artifact.format_name = decode_string_scalar(array);
      } else if (name == "__format_revision__") {
        artifact.format_revision = decode_int_scalar(array);
      } else if (name == "__format_version__") {
        artifact.legacy_tag = decode_string_scalar(array);
      } else if (name == "__producer__") {
        artifact.producer = decode_string_scalar(array);
      } else if (name == "__artifact_kind__") {
        artifact.artifact_kind = decode_string_scalar(array);
      } else if (name == "__artifact_scope__") {
        artifact.artifact_scope = decode_string_scalar(array);
      } else if (name == "__default_load_mode__") {
        artifact.default_load_mode = decode_string_scalar(array);
      } else if (name == "__contains_training_state__") {
        artifact.contains_training_state = decode_bool_scalar(array);
      } else if (name == "__recommended_loader__") {
        artifact.recommended_loader = decode_string_scalar(array);
      } else if (name == "__compile_contract_policy__") {
        artifact.compile_contract_policy = decode_string_scalar(array);
      } else if (name == "__config__") {
        artifact.config_json = decode_string_scalar(array);
      } else if (name == "__tensor_names__") {
        artifact.tensor_manifest =
            parse_manifest_tensor_names(decode_string_scalar(array));
      }
      continue;
    }

    std::string lowered = name;
    std::transform(
        lowered.begin(), lowered.end(), lowered.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    for (const auto &token : kForbiddenTrainingKeyTokens) {
      if (lowered.find(token) != std::string::npos) {
        throw std::runtime_error(
            "Unsupported training/checkpoint payload key in deploy artifact: " +
            name);
      }
    }

    payload_names.push_back(name);
    artifact.tensors.emplace(name, tensor_from_npy_array(array));
  }

  std::sort(payload_names.begin(), payload_names.end());
  if (!artifact.tensor_manifest.empty() &&
      artifact.tensor_manifest != payload_names) {
    throw std::runtime_error(
        "Serialized tensor manifest does not match payload keys");
  }

  if (artifact.format_name != kSerializationFormatName) {
    throw std::runtime_error("Unsupported serialization format name: " +
                             artifact.format_name);
  }
  if (artifact.format_revision != kSerializationFormatRevision) {
    throw std::runtime_error("Unsupported serialization format revision: " +
                             std::to_string(artifact.format_revision));
  }
  if (!artifact.legacy_tag.empty() &&
      artifact.legacy_tag != kSerializationLegacyTag) {
    throw std::runtime_error("Unsupported legacy serialization tag: " +
                             artifact.legacy_tag);
  }
  if (!artifact.producer.empty() &&
      artifact.producer != kSerializationProducer) {
    throw std::runtime_error("Unsupported serialization producer: " +
                             artifact.producer);
  }
  if (artifact.artifact_kind != kSerializationArtifactKind) {
    throw std::runtime_error("Unsupported serialization artifact kind: " +
                             artifact.artifact_kind);
  }
  if (artifact.artifact_scope != kSerializationArtifactScope) {
    throw std::runtime_error("Unsupported serialization artifact scope: " +
                             artifact.artifact_scope);
  }
  if (artifact.default_load_mode != kSerializationDefaultLoadMode) {
    throw std::runtime_error("Unsupported deploy load mode: " +
                             artifact.default_load_mode);
  }
  if (artifact.contains_training_state) {
    throw std::runtime_error("Unsupported serialization payload: deploy "
                             "artifacts must not contain training-only state");
  }
  if (artifact.recommended_loader != kSerializationRecommendedLoader) {
    throw std::runtime_error("Unsupported recommended loader: " +
                             artifact.recommended_loader);
  }
  if (artifact.compile_contract_policy != kSerializationCompileContractPolicy) {
    throw std::runtime_error("Unsupported compile contract policy: " +
                             artifact.compile_contract_policy);
  }
  if (artifact.config_json.empty()) {
    throw std::runtime_error(
        "Serialized deploy artifact does not contain architecture config "
        "required for C++ reconstruction");
  }

  return artifact;
}

void apply_state(
    const std::shared_ptr<core::Module> &module,
    const std::unordered_map<std::string, Tensor> &serialized_tensors) {
  if (!module) {
    throw std::runtime_error(
        "Cannot load serialized weights into a null module");
  }

  auto named = module->named_parameters();
  for (const auto &entry : serialized_tensors) {
    auto it = named.find(entry.first);
    if (it == named.end()) {
      throw std::runtime_error("Serialized tensor '" + entry.first +
                               "' does not match any named parameter/buffer in "
                               "the reconstructed module");
    }
    copy_tensor_into(it->second, entry.second);
  }
}

void normalize_loaded_module(const std::shared_ptr<core::Module> &module,
                             const std::optional<Device> &device) {
  if (device.has_value() && !module->is_on(device.value())) {
    module->to(device.value());
  }
  module->eval();
}

} // namespace

std::shared_ptr<core::Module>
load_serialized_module(const std::string &path,
                       const std::optional<Device> &device) {
  const ParsedArtifact artifact = parse_and_validate_artifact(path);
  const JsonValue config = JsonParser(artifact.config_json).parse();
  std::shared_ptr<core::Module> module = build_module_from_config(config);
  apply_state(module, artifact.tensors);
  normalize_loaded_module(module, device);
  return module;
}

void load_serialized_weights_into(const std::shared_ptr<core::Module> &module,
                                  const std::string &path,
                                  const std::optional<Device> &device) {
  const ParsedArtifact artifact = parse_and_validate_artifact(path);
  apply_state(module, artifact.tensors);
  normalize_loaded_module(module, device);
}

} // namespace detail

std::shared_ptr<core::Module> load_serialized(const std::string &path,
                                              std::optional<Device> device) {
  return detail::load_serialized_module(path, device);
}

void load_weights_serialized(const std::shared_ptr<core::Module> &module,
                             const std::string &path,
                             std::optional<Device> device) {
  detail::load_serialized_weights_into(module, path, device);
}

} // namespace inference
} // namespace munet
