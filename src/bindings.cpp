#include "autograd/engine.hpp"
#include "inference.hpp"
#include "nn/module.hpp"
#include "nn.hpp"
#include "ops.hpp"
#include "optim.hpp"
#include "tensor.hpp"
#include <optional>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
using namespace munet;

// Trampoline for Module to allow python inheritance
class PyModule : public nn::Module {
public:
  using nn::Module::Module;
  Tensor forward(Tensor x) override {
    PYBIND11_OVERRIDE_PURE(Tensor, nn::Module, forward, x);
  }
};

PYBIND11_MODULE(munet, m) {
  m.doc() = "MuNet: C++ Machine Learning Framework";

  // ============================================================================
  // Enums and Devices
  // ============================================================================
  py::enum_<DeviceType>(m, "DeviceType", "Types of compute devices available.")
      .value("CPU", DeviceType::CPU)
      .value("CUDA", DeviceType::CUDA)
      .value("VULKAN", DeviceType::VULKAN)
      .export_values();

  py::enum_<DataType>(m, "DataType", "Supported data types for tensors.")
      .value("Float32", DataType::Float32)
      .value("Float16", DataType::Float16)
      .value("Int32", DataType::Int32)
      .export_values();

  py::class_<Device>(
      m, "Device",
      "Represents a compute device (e.g., CPU, CUDA) and its index.")
      .def(py::init<DeviceType, int>(),
           py::arg_v("type", DeviceType::CPU, "munet.DeviceType.CPU"),
           py::arg("index") = 0, "Initializes a new Device.")
      .def("__repr__", &Device::to_string)
      .def_readwrite("type", &Device::type, "The type of the device.")
      .def_readwrite("index", &Device::index, "The index of the device.");

  py::class_<TensorOptions>(m, "TensorOptions",
                            "Tensor construction and conversion options.")
      .def(py::init<>())
      .def_readwrite("device", &TensorOptions::device)
      .def_readwrite("dtype", &TensorOptions::dtype)
      .def_readwrite("requires_grad", &TensorOptions::requires_grad);

  // ============================================================================
  // Tensor Core (Moved above factory functions for correct return type parsing)
  // ============================================================================
  py::class_<Tensor>(m, "Tensor", py::buffer_protocol(),
                     "Multi-dimensional matrix with autograd support.")
      .def(py::init<Shape, Device, DataType, bool>(), py::arg("shape"),
           py::arg_v("device", Device{DeviceType::CPU, 0},
                     "munet.Device(munet.DeviceType.CPU, 0)"),
           py::arg_v("dtype", DataType::Float32, "munet.DataType.Float32"),
           py::arg("requires_grad") = false)
      .def(py::init<Shape, const TensorOptions &>(), py::arg("shape"),
           py::arg("options"))

      // Properties
      .def_property(
          "name",
          [](const Tensor &t) { return t.impl_ ? t.name() : "uninitialized"; },
          [](Tensor &t, const std::string &n) {
            if (t.impl_)
              t.set_name(n);
          },
          "Optional name for debugging purposes.")
      .def_property_readonly(
          "shape",
          [](const Tensor &t) { return t.impl_ ? t.shape() : Shape{}; },
          "The shape (dimensions) of the tensor.")
      .def_property_readonly(
          "device",
          [](const Tensor &t) {
            return t.impl_ ? t.device() : Device{DeviceType::UNKNOWN, 0};
          },
          "The device where this tensor is allocated.")
      .def_property_readonly(
          "dtype",
          [](const Tensor &t) {
            return t.impl_ ? t.dtype() : DataType::Float32;
          },
          "The data type of the tensor elements.")
      .def_property_readonly("strides", &Tensor::strides)
      .def_property_readonly("storage_offset", &Tensor::storage_offset)
      .def_property_readonly("is_contiguous", &Tensor::is_contiguous)
      .def_property(
          "requires_grad",
          [](const Tensor &t) { return t.impl_ ? t.requires_grad() : false; },
          [](Tensor &t, bool req) {
            if (t.impl_)
              t.set_requires_grad(req);
          },
          "Whether this tensor tracks operations for automatic "
          "differentiation.")
      .def_property_readonly("grad", &Tensor::grad,
                             "The gradient of this tensor.")

      // Core Methods
      .def("__len__",
           [](const Tensor &t) {
             return (!t.impl_ || t.shape().empty()) ? 0 : t.shape()[0];
           })
      .def(
          "numel", [](const Tensor &t) { return t.impl_ ? t.size() : 0; },
          "Returns the total number of elements in the tensor.")
      .def("detach", &Tensor::detach,
           "Returns a new Tensor, detached from the current autograd graph.")
      .def("__repr__",
           [](const Tensor &t) {
             if (!t.impl_)
               return std::string("Tensor(uninitialized)");
             return "Tensor(shape=" + to_string(t.shape()) + ", device='" +
                    t.device().to_string() + "'" +
                    (t.requires_grad() ? ", requires_grad=True)" : ")");
           })
      .def("to", py::overload_cast<Device>(&Tensor::to, py::const_), py::arg("device"),
           "Moves the tensor to the specified device.")
      .def("to", py::overload_cast<DataType>(&Tensor::to, py::const_), py::arg("dtype"),
           "Converts the tensor to the specified dtype.")
      .def("to_options", py::overload_cast<const TensorOptions &>(&Tensor::to, py::const_), py::arg("options"),
           "Converts the tensor using explicit tensor options.")
      .def(
          "copy_from_numpy",
          [](Tensor &t, py::array_t<float> input) {
            if (t.device().type != DeviceType::CPU)
              throw std::runtime_error("Target tensor must be on CPU.");
            py::buffer_info buf = input.request();
            size_t bytes = buf.size * sizeof(float);
            if (bytes != t.bytes())
              throw std::runtime_error("Size mismatch.");
            std::memcpy(t.data(), buf.ptr, bytes);
          },
          py::arg("input"), "Copies data from a NumPy array into this tensor.")
      .def(
          "replace_",
          [](Tensor &self, const Tensor &other) { self.impl_ = other.impl_; },
          py::arg("other"),
          "In-place replacement of underlying tensor implementation.")
      .def("transpose", &ops::transpose, py::arg("dim0"), py::arg("dim1"),
           "Returns a tensor that is a transposed version of this tensor.")
      .def("permute", &Tensor::permute, py::arg("dims"),
           "Returns a tensor view with dimensions permuted.")
      .def("contiguous", &Tensor::contiguous,
           "Returns a contiguous tensor containing the same data as this "
           "tensor.")

      // Autograd
      .def("zero_grad", &Tensor::zero_grad,
           "Clears the gradient of the tensor.")
      .def(
          "backward", [](Tensor &t) { t.backward(); },
          "Computes the gradient of current tensor w.r.t. graph leaves.")
      .def(
          "backward", [](Tensor &t, const Tensor &grad) { t.backward(grad); },
          py::arg("grad"),
          "Computes the gradient with a given upstream gradient.")

      // Math & Ops
      .def("__add__", [](const Tensor &a, const Tensor &b) { return a + b; })
      .def("__add__",
           [](const Tensor &a, float b) {
             Tensor bt({1}, a.device(), a.dtype());
             bt.uniform_(b, b);
             return a + bt;
           })
      .def("__sub__", [](const Tensor &a, const Tensor &b) { return a - b; })
      .def("__sub__",
           [](const Tensor &a, float b) {
             Tensor bt({1}, a.device(), a.dtype());
             bt.uniform_(b, b);
             return a - bt;
           })
      .def("__mul__", [](const Tensor &a, const Tensor &b) { return a * b; })
      .def("__mul__",
           [](const Tensor &a, float b) {
             Tensor bt({1}, a.device(), a.dtype());
             bt.uniform_(b, b);
             return a * bt;
           })
      .def("__truediv__", [](const Tensor &a, const Tensor &b) { return a / b; })
      .def("__truediv__",
           [](const Tensor &a, float b) {
             Tensor bt({1}, a.device(), a.dtype());
             bt.uniform_(b, b);
             return a / bt;
           })
      .def("__matmul__",
           [](const Tensor &a, const Tensor &b) { return a.matmul(b); })
      .def("sum", &Tensor::sum,
           "Returns the sum of all elements in the tensor.")
      .def("reshape", &Tensor::reshape, py::arg("shape"),
           "Returns a tensor with the same data and number of elements, but "
           "with the specified shape.")
      .def("masked_fill", &Tensor::masked_fill, py::arg("mask"), py::arg("value"),
           "Fills entries where mask is 1 with the given value.")
      .def_static("cat", &Tensor::cat, py::arg("inputs"), py::arg("dim") = 1,
                  "Concatenates tensors along a given dimension.")
      .def(
          "numpy", [](py::object self) { return py::cast<py::array>(self); },
          "Returns the tensor as a NumPy ndarray. The returned array and the "
          "tensor will share their storage (CPU only).")
      .def("item", &Tensor::item,
           "Returns the value of this tensor as a standard Python number. Only "
           "works for tensors with one element.")
      .def("uniform_", &Tensor::uniform_, py::arg("low") = -1.0f,
           py::arg("high") = 1.0f,
           "Fills the tensor with values from a uniform distribution.")
      .def("step", &Tensor::step, py::arg("lr"),
           "Applies a simple SGD step manually directly to the tensor.")

      // NN & Activations
      .def("relu", &Tensor::relu,
           "Applies the Rectified Linear Unit function element-wise.")
      .def("sigmoid", &Tensor::sigmoid,
           "Applies the Sigmoid function element-wise.")
      .def("softmax", &Tensor::softmax, py::arg("dim") = -1, "Applies softmax along a dimension.")
      .def("log_softmax", &Tensor::log_softmax, py::arg("dim") = -1,
           "Applies log-softmax along a dimension.")
      .def("conv2d", &Tensor::conv2d, py::arg("weight"),
           py::arg_v("bias", Tensor(), "munet.Tensor()"), py::arg("stride") = 1,
           py::arg("padding") = 0,
           "Applies a 2D convolution over an input signal.")
      .def("max_pool2d", &Tensor::max_pool2d, py::arg("kernel_size"),
           py::arg("stride"), py::arg("padding") = 0,
           "Applies a 2D max pooling over an input signal.")
      .def("upsample2d", &Tensor::upsample2d, py::arg("scale_factor"),
           "Upsamples the input by the given scale factor.")
      .def("batch_norm", &Tensor::batch_norm, py::arg("running_mean"),
           py::arg("running_var"), py::arg("weight"), py::arg("bias"),
           py::arg("training"), py::arg("momentum") = 0.1,
           py::arg("eps") = 1e-5, "Applies Batch Normalization.")
      .def("mse_loss", &Tensor::mse_loss, py::arg("target"),
           "Computes Mean Squared Error against the target tensor.")
      .def("cross_entropy", &Tensor::cross_entropy, py::arg("target"),
           "Computes Cross Entropy loss against the target tensor.")

      // Buffer Protocol (Zero-copy to NumPy)
      .def_buffer([](Tensor &t) -> py::buffer_info {
        if (t.device().type != DeviceType::CPU) {
          throw std::runtime_error(
              "Cannot convert GPU tensor to NumPy array directly. Call "
              "`.to(Device(DeviceType.CPU))` first.");
        }

        if (t.requires_grad()) {
          throw std::runtime_error(
              "Can't access buffer on a tensor that requires grad. "
              "Use .detach().numpy() instead.");
        }

        std::vector<py::ssize_t> py_shape;
        py_shape.reserve(t.shape().size());
        for (int dim : t.shape()) {
          py_shape.push_back(static_cast<py::ssize_t>(dim));
        }

        std::vector<py::ssize_t> py_strides(py_shape.size());
        py::ssize_t stride = dtype_size(t.dtype());
        for (int i = (int)py_shape.size() - 1; i >= 0; --i) {
          py_strides[i] = stride;
          stride *= py_shape[i];
        }

        return py::buffer_info(t.data(), dtype_size(t.dtype()),
                               py::format_descriptor<float>::format(),
                               py_shape.size(), py_shape, py_strides);
      });

  // ============================================================================
  // Factory Functions
  // ============================================================================
  m.def("cat", &ops::cat, py::arg("tensors"), py::arg("dim") = 1,
        "Concatenates a sequence of tensors along the specified dimension.");

  m.def(
      "zeros",
      [](Shape shape, std::optional<Device> device, bool requires_grad) {
        Device dev = device.value_or(Device{DeviceType::CPU, 0});
        Tensor t(shape, dev, DataType::Float32, requires_grad);
        t.impl_->backend().memset(t.data(), 0, t.bytes());
        return t;
      },
      py::arg("shape"), py::arg_v("device", py::none(), "None"),
      py::arg("requires_grad") = false,
      "Creates a tensor of the specified shape filled with zeros.");

  m.def(
      "ones",
      [](Shape shape, std::optional<Device> device, bool requires_grad) {
        Device dev = device.value_or(Device{DeviceType::CPU, 0});
        Tensor t(shape, dev, DataType::Float32, requires_grad);
        t.uniform_(1.0f, 1.0f);
        return t;
      },
      py::arg("shape"), py::arg_v("device", py::none(), "None"),
      py::arg("requires_grad") = false,
      "Creates a tensor of the specified shape filled with ones.");

  m.def(
      "rand",
      [](Shape shape, std::optional<Device> device, bool requires_grad) {
        Device dev = device.value_or(Device{DeviceType::CPU, 0});
        Tensor t(shape, dev, DataType::Float32, requires_grad);
        t.uniform_(0.0f, 1.0f);
        return t;
      },
      py::arg("shape"), py::arg_v("device", py::none(), "None"),
      py::arg("requires_grad") = false,
      "Creates a tensor of the specified shape filled with random values from "
      "U[0, 1).");

  m.def(
      "from_numpy",
      [](py::array_t<float> input) {
        py::buffer_info buf = input.request();
        std::vector<int> shape(buf.shape.begin(), buf.shape.end());
        Tensor t(shape, Device{DeviceType::CPU, 0});
        std::memcpy(t.data(), buf.ptr, t.bytes());
        return t;
      },
      py::arg("input"), "Creates a CPU Tensor from a NumPy array.");

  // Alias copy_from_numpy to module level as well
  m.def(
      "copy_from_numpy",
      [](Tensor &t, py::array_t<float> input) {
        if (t.device().type != DeviceType::CPU)
          throw std::runtime_error(
              "copy_from_numpy: Target tensor must be on CPU.");
        py::buffer_info buf = input.request();
        size_t bytes = buf.size * sizeof(float);
        if (bytes != t.bytes())
          throw std::runtime_error("copy_from_numpy: Size mismatch.");
        std::memcpy(t.data(), buf.ptr, bytes);
      },
      py::arg("tensor"), py::arg("input"),
      "Copies data from a NumPy array into the given CPU tensor.");

  // ============================================================================
  // Neural Network Layers (munet.nn)
  // ============================================================================
  auto nn = m.def_submodule("nn", "Neural Network Modules and Layers");

  py::class_<nn::Module, std::shared_ptr<nn::Module>, PyModule>(
      nn, "Module", py::dynamic_attr(),
      "Base class for all neural network modules.")
      .def(py::init<>())
      .def("forward", &nn::Module::forward, py::arg("x"),
           "Defines the computation performed at every call.")
      .def("parameters", &nn::Module::parameters,
           "Returns an iterator over module parameters.")
      .def("named_parameters", &nn::Module::named_parameters,
           py::arg("prefix") = "",
           "Returns an iterator over module parameters, yielding both the name "
           "of the parameter as well as the parameter itself.")
      .def("named_modules",
           [](nn::Module &self, const std::string &prefix) {
             return self.named_modules_typed(prefix);
           },
           py::arg("prefix") = "",
           "Returns an iterator over all nn modules in the network.")
      .def("train", &nn::Module::train, py::arg("mode") = true,
           "Sets the module in training mode.")
      .def("eval", &nn::Module::eval, "Sets the module in evaluation mode.")
      .def("to", &nn::Module::to, py::arg("device"),
           "Moves all parameters and buffers to the specified device.")
      .def("zero_grad", &nn::Module::zero_grad,
           "Clears the gradients of all optimized parameters.")
      .def("__call__", &nn::Module::forward)
      .def("__setattr__", [](py::object self, const std::string &name,
                             py::object value) {
        auto &mod = self.cast<nn::Module &>();
        if (py::isinstance<nn::Module>(value)) {
          mod.register_module(name, value.cast<std::shared_ptr<nn::Module>>());
        } else if (py::isinstance<Tensor>(value)) {
          auto &t = value.cast<Tensor &>();
          if (t.requires_grad())
            mod.register_parameter(name, t);
          else
            mod.register_buffer(name, t);
        }
        py::str py_name(name);
        if (PyObject_GenericSetAttr(self.ptr(), py_name.ptr(), value.ptr()) !=
            0) {
          throw py::error_already_set();
        }
      });

  py::class_<nn::Linear, nn::Module, std::shared_ptr<nn::Linear>>(
      nn, "Linear", "Applies a linear transformation to the incoming data.")
      .def(py::init<int, int, bool>(), py::arg("in_features"),
           py::arg("out_features"), py::arg("bias") = true)
      .def_readonly("weight", &nn::Linear::weight,
                    "The learnable weights of the module.")
      .def_readonly("bias", &nn::Linear::bias,
                    "The learnable bias of the module.");

  py::class_<nn::Conv2d, nn::Module, std::shared_ptr<nn::Conv2d>>(
      nn, "Conv2d", "Applies a 2D convolution over an input signal.")
      .def(py::init<int, int, int, int, int>(), py::arg("in_channels"),
           py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0)
      .def_readonly("stride", &nn::Conv2d::stride_)
      .def_readonly("padding", &nn::Conv2d::padding_)
      .def_readonly("weight", &nn::Conv2d::weight)
      .def_readonly("bias", &nn::Conv2d::bias);

  py::class_<nn::BatchNorm2d, nn::Module, std::shared_ptr<nn::BatchNorm2d>>(
      nn, "BatchNorm2d", "Applies Batch Normalization over a 4D input.")
      .def(py::init<int, float, float>(), py::arg("num_features"),
           py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f)
      .def_readonly("eps", &nn::BatchNorm2d::eps_)
      .def_readonly("momentum", &nn::BatchNorm2d::momentum_)
      .def_readonly("weight", &nn::BatchNorm2d::weight)
      .def_readonly("bias", &nn::BatchNorm2d::bias)
      .def_readonly("running_mean", &nn::BatchNorm2d::running_mean)
      .def_readonly("running_var", &nn::BatchNorm2d::running_var);

  py::class_<nn::Flatten, nn::Module, std::shared_ptr<nn::Flatten>>(
      nn, "Flatten", "Flattens a contiguous range of dims into a tensor.")
      .def(py::init<>());

  py::class_<nn::ReLU, nn::Module, std::shared_ptr<nn::ReLU>>(
      nn, "ReLU", "Applies the rectified linear unit function element-wise.")
      .def(py::init<>());

  py::class_<nn::Sigmoid, nn::Module, std::shared_ptr<nn::Sigmoid>>(
      nn, "Sigmoid", "Applies the element-wise Sigmoid function.")
      .def(py::init<>());

  py::class_<nn::Tanh, nn::Module, std::shared_ptr<nn::Tanh>>(
      nn, "Tanh", "Applies the element-wise Tanh function.")
      .def(py::init<>());

  py::class_<nn::GELU, nn::Module, std::shared_ptr<nn::GELU>>(
      nn, "GELU",
      "Applies a fast GELU approximation: x * sigmoid(1.702*x).")
      .def(py::init<>());

  py::class_<nn::LeakyReLU, nn::Module, std::shared_ptr<nn::LeakyReLU>>(
      nn, "LeakyReLU", "Applies the element-wise LeakyReLU function.")
      .def(py::init<float>(), py::arg("negative_slope") = 0.01f)
      .def_readonly("negative_slope", &nn::LeakyReLU::negative_slope_);

  py::class_<nn::Dropout, nn::Module, std::shared_ptr<nn::Dropout>>(
      nn, "Dropout",
      "Randomly zeroes some of the elements of the input tensor with "
      "probability p during training.")
      .def(py::init<float>(), py::arg("p") = 0.5f)
      .def_readonly("p", &nn::Dropout::p_);

  py::class_<nn::Embedding, nn::Module, std::shared_ptr<nn::Embedding>>(
      nn, "Embedding",
      "Embedding lookup for [B,T] token ids and projection for [B,T,V] one-hot/probability inputs.")
      .def(py::init<int, int>(), py::arg("num_embeddings"),
           py::arg("embedding_dim"))
      .def_readonly("num_embeddings", &nn::Embedding::num_embeddings_)
      .def_readonly("embedding_dim", &nn::Embedding::embedding_dim_)
      .def_readonly("weight", &nn::Embedding::weight);

  py::class_<nn::LayerNorm, nn::Module, std::shared_ptr<nn::LayerNorm>>(
      nn, "LayerNorm",
      "Applies Layer Normalization over the last tensor dimension.")
      .def(py::init<int, float>(), py::arg("normalized_shape"),
           py::arg("eps") = 1e-5f)
      .def_readonly("normalized_shape", &nn::LayerNorm::normalized_shape_)
      .def_readonly("eps", &nn::LayerNorm::eps_)
      .def_readonly("weight", &nn::LayerNorm::weight)
      .def_readonly("bias", &nn::LayerNorm::bias);

  py::class_<nn::MultiHeadAttention, nn::Module, std::shared_ptr<nn::MultiHeadAttention>>(
      nn, "MultiHeadAttention",
      "Applies causal/non-causal multi-head self-attention over [B,T,E].")
      .def(py::init<int, int, bool>(), py::arg("embed_dim"), py::arg("num_heads"),
           py::arg("causal") = true)
      .def_readonly("embed_dim", &nn::MultiHeadAttention::embed_dim_)
      .def_readonly("num_heads", &nn::MultiHeadAttention::num_heads_)
      .def_readonly("causal", &nn::MultiHeadAttention::causal_);

  py::class_<nn::GlobalAvgPool2d, nn::Module, std::shared_ptr<nn::GlobalAvgPool2d>>(
      nn, "GlobalAvgPool2d", "Applies global average pooling over spatial dimensions.")
      .def(py::init<>());

  py::class_<nn::MaxPool2d, nn::Module, std::shared_ptr<nn::MaxPool2d>>(
      nn, "MaxPool2d", "Applies a 2D max pooling over an input signal.")
      .def(py::init<int, int, int>(), py::arg("kernel_size"),
           py::arg("stride") = 2, py::arg("padding") = 0)
      .def_readonly("kernel_size", &nn::MaxPool2d::k_)
      .def_readonly("stride", &nn::MaxPool2d::s_)
      .def_readonly("padding", &nn::MaxPool2d::p_);

  py::class_<nn::Upsample, nn::Module, std::shared_ptr<nn::Upsample>>(
      nn, "Upsample", "Upsamples a given multi-channel 2D spatial data.")
      .def(py::init<int>(), py::arg("scale_factor"))
      .def_readonly("scale_factor", &nn::Upsample::scale_);

  py::class_<nn::Sequential, nn::Module, std::shared_ptr<nn::Sequential>>(
      nn, "Sequential",
      "A sequential container. Modules will be added to it in the order they "
      "are passed in.")
      .def(py::init<>())
      .def("add", &nn::Sequential::add, py::arg("module"),
           "Appends a module to the sequence.")
      .def(py::init([](const std::vector<std::shared_ptr<nn::Module>> &layers) {
             auto seq = std::make_shared<nn::Sequential>();
             for (auto l : layers)
               seq->add(l);
             return seq;
           }),
           py::arg("layers"))
      .def(
          "__iter__",
          [](nn::Sequential &s) {
            return py::make_iterator(s.ordered_modules_.begin(),
                                     s.ordered_modules_.end());
          },
          py::keep_alive<0, 1>());

  // ============================================================================
  // Inference (munet.inference)
  // ============================================================================
  auto inf = m.def_submodule("inference", "Inference runtime APIs");

  py::class_<inference::EngineConfig>(inf, "EngineConfig")
      .def(py::init<>())
      .def_readwrite("device", &inference::EngineConfig::device)
      .def_readwrite("warmup_runs", &inference::EngineConfig::warmup_runs)
      .def_readwrite("strict_shape_check",
                     &inference::EngineConfig::strict_shape_check);

  py::class_<inference::EngineStats>(inf, "EngineStats")
      .def(py::init<>())
      .def_readonly("runs", &inference::EngineStats::runs)
      .def_readonly("last_run_ms", &inference::EngineStats::last_run_ms)
      .def_readonly("compile_ms", &inference::EngineStats::compile_ms)
      .def_readonly("compiled_input_shape",
                    &inference::EngineStats::compiled_input_shape)
      .def_readonly("compiled_output_shape",
                    &inference::EngineStats::compiled_output_shape);

  py::class_<inference::Engine>(inf, "Engine")
      .def(py::init<inference::EngineConfig>(), py::arg("config") = inference::EngineConfig{})
      .def("set_device", &inference::Engine::set_device, py::arg("device"))
      .def("device", &inference::Engine::device)
      .def("set_warmup_runs", &inference::Engine::set_warmup_runs,
           py::arg("warmup_runs"))
      .def("set_strict_shape_check", &inference::Engine::set_strict_shape_check,
           py::arg("enabled"))
      .def(
          "load",
          [](inference::Engine &self, const std::shared_ptr<nn::Module> &module) {
            self.load(std::static_pointer_cast<core::Module>(module));
          },
          py::arg("module"))
      .def("compile",
           [](inference::Engine &self, const Tensor &example_input,
              const std::optional<std::vector<int>> &expected_input_shape,
              const std::optional<std::vector<int>> &expected_output_shape) {
             self.compile(example_input, expected_input_shape.value_or(std::vector<int>{}),
                          expected_output_shape.value_or(std::vector<int>{}));
           },
           py::arg("example_input"),
           py::arg("expected_input_shape") = py::none(),
           py::arg("expected_output_shape") = py::none())
      .def("prepare", &inference::Engine::prepare, py::arg("example_input"))
      .def("run", &inference::Engine::run, py::arg("input"))
      .def("run_batch", &inference::Engine::run_batch, py::arg("inputs"))
      .def("is_loaded", &inference::Engine::is_loaded)
      .def("is_prepared", &inference::Engine::is_prepared)
      .def("is_compiled", &inference::Engine::is_compiled)
      .def("compiled_input_shape", &inference::Engine::compiled_input_shape)
      .def("compiled_output_shape", &inference::Engine::compiled_output_shape)
      .def("stats", &inference::Engine::stats);

  // ============================================================================
  // Optimizers (munet.optim)
  // ============================================================================
  auto optim = m.def_submodule("optim", "Optimization Algorithms");

  py::class_<optim::Optimizer, std::shared_ptr<optim::Optimizer>>(
      optim, "Optimizer", "Base class for all optimizers.")
      .def("step", &optim::Optimizer::step,
           "Performs a single optimization step.")
      .def("zero_grad", &optim::Optimizer::zero_grad,
           "Clears the gradients of all optimized Tensors.");

  py::class_<optim::Adam, optim::Optimizer, std::shared_ptr<optim::Adam>>(
      optim, "Adam", "Adam optimizer.")
      .def(py::init<std::vector<Tensor>, float, float, float, float>(),
           py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
           py::arg("beta2") = 0.999, py::arg("eps") = 1e-8);

  py::class_<optim::SGD, optim::Optimizer, std::shared_ptr<optim::SGD>>(
      optim, "SGD", "Stochastic Gradient Descent optimizer.")
      .def(py::init<std::vector<Tensor>, float>(), py::arg("params"),
           py::arg("lr"));

  // ============================================================================
  // Utilities
  // ============================================================================
  py::class_<GradMode>(m, "GradMode",
                       "Controls whether autograd graph is recorded.")
      .def_static("is_enabled", &GradMode::is_enabled,
                  "Check if grad mode is enabled.")
      .def_static("set_enabled", &GradMode::set_enabled, py::arg("enabled"),
                  "Enable or disable grad mode.");

  m.def(
      "print_profiler_stats", []() { Profiler::get().print_summary(); },
      "Prints current memory and compute profiler stats (stderr).");
  m.def(
      "reset_profiler", []() { Profiler::get().reset(); },
      "Clears all collected performance statistics and resets peak memory "
      "tracking.");

  // ============================================================================
  // Python Injected Helpers
  // ============================================================================
  m.attr("__munet_helper_source_dir__") = py::str(MUNET_PY_HELPER_SOURCE_DIR);
  if (py::hasattr(m, "__file__")) {
    m.attr("__munet_file__") = m.attr("__file__");
  }

  py::exec(
      R"(
 class no_grad:
     """Context-manager that disables gradient calculation."""
     def __enter__(self):
         import munet
         self.prev = munet.GradMode.is_enabled()
         munet.GradMode.set_enabled(False)
     def __exit__(self, exc_type, exc_val, exc_tb):
         import munet
         munet.GradMode.set_enabled(self.prev)

 class enable_grad:
     """Context-manager that enables gradient calculation."""
     def __enter__(self):
         import munet
         self.prev = munet.GradMode.is_enabled()
         munet.GradMode.set_enabled(True)
     def __exit__(self, exc_type, exc_val, exc_tb):
         import munet
         munet.GradMode.set_enabled(self.prev)

 def _tensor_to_numpy(t):
     import numpy as np
     import munet

     cpu = munet.Device(munet.DeviceType.CPU, 0)
     td = t.detach()
     if td.device.type != munet.DeviceType.CPU:
         td = td.to(cpu)
     return np.array(td, copy=False).copy()

 def _copy_numpy_into_tensor(t, arr):
     import numpy as np
     import munet

     req = bool(t.requires_grad)
     target = t.device
     src = munet.from_numpy(np.asarray(arr, dtype=np.float32))
     if target.type != munet.DeviceType.CPU:
         src = src.to(target)
     t.replace_(src)
     t.requires_grad = req

 def save(module, filename):
     """
     Saves a module architecture + parameters/buffers to a compressed .npz file.
     The saved file can be loaded with `load(filename)` (full reconstruction)
     for supported built-in module types, or with `load(module, filename)` for
     weights-only restore into an existing model definition.
     """
     import numpy as np
     import json
     import munet

     def get_config(m):
         name = type(m).__name__
         if name == 'Sequential':
             return {'type': name, 'layers': [get_config(child) for child in m]}
         elif name == 'Linear':
             has_bias = hasattr(m, 'bias') and getattr(m, 'bias') is not None and getattr(m, 'bias').numel() > 0
             return {'type': name, 'in_features': m.weight.shape[0], 'out_features': m.weight.shape[1], 'bias': has_bias}
         elif name == 'Conv2d':
             return {'type': name, 'in_channels': m.weight.shape[1], 'out_channels': m.weight.shape[0], 'kernel_size': m.weight.shape[2], 'stride': m.stride, 'padding': m.padding}
         elif name == 'MaxPool2d':
             return {'type': name, 'kernel_size': m.kernel_size, 'stride': m.stride, 'padding': m.padding}
         elif name == 'BatchNorm2d':
             return {'type': name, 'num_features': m.weight.shape[0], 'eps': m.eps, 'momentum': m.momentum}
         elif name == 'Upsample':
             return {'type': name, 'scale_factor': m.scale_factor}
         elif name == 'GlobalAvgPool2d':
             return {'type': name}
         elif name in ('ReLU', 'Sigmoid', 'Tanh', 'GELU', 'Flatten'):
             return {'type': name}
         elif name == 'LeakyReLU':
             return {'type': name, 'negative_slope': m.negative_slope}
         elif name == 'Dropout':
             return {'type': name, 'p': m.p}
         elif name == 'Embedding':
             return {'type': name, 'num_embeddings': m.num_embeddings, 'embedding_dim': m.embedding_dim}
         elif name == 'LayerNorm':
             return {'type': name, 'normalized_shape': m.normalized_shape, 'eps': m.eps}
         elif name == 'MultiHeadAttention':
             return {'type': name, 'embed_dim': m.embed_dim, 'num_heads': m.num_heads, 'causal': bool(m.causal)}
         else:
             raise ValueError(
                 f"Unsupported module type for full reconstruction: {name}. "
                 "Use `load(existing_model, filename)` for weights-only restore."
             )

     def tensor_to_numpy(t):
         m = __import__("munet")
         cpu = m.Device(m.DeviceType.CPU, 0)
         td = t.detach()
         if td.device.type != m.DeviceType.CPU:
             td = td.to(cpu)
         return np.array(td, copy=False).copy()

     config = get_config(module)
     state = {}
     for name, p in module.named_parameters().items():
         state[name] = tensor_to_numpy(p)

     state['__config__'] = np.array(json.dumps(config))
     state['__format_version__'] = np.array('munet_model_v1')
     np.savez(filename, **state)

 def load(arg, filename=None):
     """
     Loads a previously saved module state.

     Usage:
       - load("model.npz") -> reconstruct full supported model from file.
       - load(module, "model.npz") -> load weights/buffers into existing model.
     """
     import numpy as np
     import json
     import munet

     def build_module(cfg):
         t = cfg['type']
         if t == 'Sequential': return munet.nn.Sequential([build_module(c) for c in cfg['layers']])
         elif t == 'Linear': return munet.nn.Linear(cfg['in_features'], cfg['out_features'], cfg['bias'])
         elif t == 'Conv2d': return munet.nn.Conv2d(cfg['in_channels'], cfg['out_channels'], cfg['kernel_size'], cfg['stride'], cfg['padding'])
         elif t == 'MaxPool2d': return munet.nn.MaxPool2d(cfg['kernel_size'], cfg['stride'], cfg['padding'])
         elif t == 'BatchNorm2d': return munet.nn.BatchNorm2d(cfg['num_features'], cfg['eps'], cfg['momentum'])
         elif t == 'Upsample': return munet.nn.Upsample(cfg['scale_factor'])
         elif t == 'GlobalAvgPool2d': return munet.nn.GlobalAvgPool2d()
         elif t == 'ReLU': return munet.nn.ReLU()
         elif t == 'Sigmoid': return munet.nn.Sigmoid()
         elif t == 'Tanh': return munet.nn.Tanh()
         elif t == 'GELU': return munet.nn.GELU()
         elif t == 'LeakyReLU': return munet.nn.LeakyReLU(cfg.get('negative_slope', 0.01))
         elif t == 'Dropout': return munet.nn.Dropout(cfg.get('p', 0.5))
         elif t == 'Embedding': return munet.nn.Embedding(cfg['num_embeddings'], cfg['embedding_dim'])
         elif t == 'LayerNorm': return munet.nn.LayerNorm(cfg['normalized_shape'], cfg.get('eps', 1e-5))
         elif t == 'MultiHeadAttention': return munet.nn.MultiHeadAttention(cfg['embed_dim'], cfg['num_heads'], cfg.get('causal', True))
         elif t == 'Flatten': return munet.nn.Flatten()
         else:
             raise ValueError(f"Unsupported saved module type: {t}")

     def copy_numpy_into_tensor(t, arr):
         m = __import__("munet")
         req = bool(t.requires_grad)
         target = t.device
         src = m.from_numpy(np.asarray(arr, dtype=np.float32))
         if target.type != m.DeviceType.CPU:
             src = src.to(target)
         t.replace_(src)
         t.requires_grad = req

     def apply_state(module, state):
         for name, p in module.named_parameters().items():
             if name in state:
                 copy_numpy_into_tensor(p, state[name])
         return module

     if filename is None:
         state = np.load(arg, allow_pickle=True)
         if '__config__' not in state:
             raise ValueError("File does not contain architecture config. Use `load(module, filename)` for weights-only restore.")

         config = json.loads(str(state['__config__']))
         module = build_module(config)
         return apply_state(module, state)
     else:
         module = arg
         state = np.load(filename, allow_pickle=True)
         return apply_state(module, state)

 def load_weights(module, filename):
     """Alias for `load(module, filename)` to explicitly do weights-only restore."""
     m = __import__("munet")
     return m.load(module, filename)

 def _load_python_helper(filename):
     import pathlib
     import sys

     # Prefer compile-time source helper dir when available (dev builds).
     helper_dir = globals().get("__munet_helper_source_dir__", None)
     if helper_dir is not None:
         helper_path = pathlib.Path(helper_dir) / filename
         if helper_path.exists():
             src = helper_path.read_text(encoding="utf-8")
             exec(compile(src, str(helper_path), "exec"), globals(), globals())
             return

     # Avoid importing `munet` while module init is still running, which can
     # recursively execute bindings init and trigger pybind duplicate type registration.
     mod_file = globals().get("__munet_file__", None)
     if mod_file is None:
         mod = sys.modules.get(__name__)
         mod_file = getattr(mod, "__file__", None) if mod is not None else None
     if mod_file is None:
         spec = globals().get("__spec__", None)
         mod_file = getattr(spec, "origin", None)
     if mod_file is not None:
         helper_path = pathlib.Path(mod_file).resolve().parent / "python_src" / filename
         if helper_path.exists():
             src = helper_path.read_text(encoding="utf-8")
             exec(compile(src, str(helper_path), "exec"), globals(), globals())
             return

     raise RuntimeError(
         f"Required MuNet python helper '{filename}' could not be located. "
         f"Searched source helper dir={helper_dir!r} and module-adjacent python_src."
     )

 _load_python_helper("onnx_integration.py")
 )",
      m.attr("__dict__"), m.attr("__dict__"));
}
