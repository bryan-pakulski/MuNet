#pragma once

#include "layer.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace munet {

class Model {
public:
  inline void add(std::shared_ptr<Layer> layer);
  inline Tensor forward(const Tensor &input);
  inline void backward(const Tensor &grad_output);

  void train() {
    for (auto &l : layers_)
      l->train();
  }
  void eval() {
    for (auto &l : layers_)
      l->eval();
  }

  inline std::vector<Tensor *> parameters() {
    std::vector<Tensor *> params;
    for (const auto &layer : layers_) {
      for (const auto &[name, tensor] : layer->get_parameters()) {
        params.push_back(tensor);
      }
    }
    return params;
  }

  void save_weights(const std::string &path) const;
  void load_weights(const std::string &path);
  inline void export_onnx(const std::string &path) const;

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};

inline void Model::add(std::shared_ptr<Layer> layer) {
  layers_.push_back(std::move(layer));
}

inline Tensor Model::forward(const Tensor &input) {
  if (layers_.empty())
    return input.clone();

  Tensor current = layers_.front()->forward(input);

  for (size_t i = 1; i < layers_.size(); ++i) {
    current = layers_[i]->forward(current);
  }
  return current;
}

inline void Model::backward(const Tensor &grad_output) {
  if (layers_.empty())
    return;

  Tensor current_grad = layers_.back()->backward(grad_output);

  for (auto it = layers_.rbegin() + 1; it != layers_.rend(); ++it) {
    current_grad = (*it)->backward(current_grad);
  }
}

inline void Model::export_onnx(const std::string &path) const {
#ifdef MUNET_USE_ONNX
  // ONNX Protobuf logic here
#else
  throw std::runtime_error(
      "MuNet compiled without ONNX support. Define MUNET_USE_ONNX.");
#endif
}

inline void Model::save_weights(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("Cannot open file for writing: " + path);

  // Write magic number "MUNT"
  uint32_t magic = 0x4D554E54;
  out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));

  uint32_t num_layers = layers_.size();
  out.write(reinterpret_cast<const char *>(&num_layers), sizeof(num_layers));

  for (const auto &layer : layers_) {
    auto params = layer->get_parameters();
    uint32_t num_params = params.size();
    out.write(reinterpret_cast<const char *>(&num_params), sizeof(num_params));

    for (const auto &[name, tensor] : params) {

      if (tensor->device_ != Device::CPU) {
        tensor->to_cpu();
      }

      // Write param name
      uint32_t name_len = name.size();
      out.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
      out.write(name.data(), name_len);

      // Write shape
      const auto &shape = tensor->shape();
      uint32_t ndims = shape.size();
      out.write(reinterpret_cast<const char *>(&ndims), sizeof(ndims));
      out.write(reinterpret_cast<const char *>(shape.data()),
                ndims * sizeof(int));

      // Write raw data
      uint64_t bytes = tensor->bytes();
      out.write(reinterpret_cast<const char *>(&bytes), sizeof(bytes));
      out.write(reinterpret_cast<const char *>(tensor->data()), bytes);
    }
  }
}

inline void Model::load_weights(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open file for reading: " + path);

  uint32_t magic;
  in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  if (magic != 0x4D554E54)
    throw std::runtime_error("Invalid weight file format");

  uint32_t num_layers;
  in.read(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));
  if (num_layers != layers_.size())
    throw std::runtime_error("Model layer count mismatch");

  for (size_t i = 0; i < num_layers; ++i) {
    auto params = layers_[i]->get_parameters();
    uint32_t num_params;
    in.read(reinterpret_cast<char *>(&num_params), sizeof(num_params));
    if (num_params != params.size())
      throw std::runtime_error("Layer parameter count mismatch");

    for (uint32_t p = 0; p < num_params; ++p) {
      uint32_t name_len;
      in.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
      std::string name(name_len, '\0');
      in.read(&name[0], name_len);

      if (params.find(name) == params.end())
        throw std::runtime_error("Param " + name + " not found");

      uint32_t ndims;
      in.read(reinterpret_cast<char *>(&ndims), sizeof(ndims));
      std::vector<int> shape(ndims);
      in.read(reinterpret_cast<char *>(shape.data()), ndims * sizeof(int));
      if (shape != params[name]->shape())
        throw std::runtime_error("Shape mismatch for " + name);

      uint64_t bytes;
      in.read(reinterpret_cast<char *>(&bytes), sizeof(bytes));
      if (bytes != params[name]->bytes())
        throw std::runtime_error("Byte size mismatch for " + name);

      in.read(reinterpret_cast<char *>(params[name]->data()), bytes);
    }
  }
}
} // namespace munet
