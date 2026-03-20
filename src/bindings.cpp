#include "autograd/engine.hpp"
#include "inference.hpp"
#include "nn/module.hpp"
#include "nn.hpp"
#include "ops.hpp"
#include "optim.hpp"
#include "tensor.hpp"
#include <optional>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
using namespace munet;

namespace {

DataType numpy_dtype_to_data_type(const py::buffer_info &buf) {
  if (buf.itemsize == static_cast<py::ssize_t>(sizeof(float)) &&
      buf.format == py::format_descriptor<float>::format()) {
    return DataType::Float32;
  }
  if (buf.itemsize == static_cast<py::ssize_t>(sizeof(int32_t)) &&
      buf.format == py::format_descriptor<int32_t>::format()) {
    return DataType::Int32;
  }
  if (buf.itemsize == 2 && buf.format.find('e') != std::string::npos) {
    return DataType::Float16;
  }
  throw std::runtime_error("Unsupported NumPy dtype; expected float32, float16, or int32");
}

std::string numpy_format_for_dtype(DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return py::format_descriptor<float>::format();
  case DataType::Float16:
    return "e";
  case DataType::Int32:
    return py::format_descriptor<int32_t>::format();
  default:
    throw std::runtime_error("Unsupported tensor dtype for NumPy conversion");
  }
}

py::array ensure_c_contiguous(py::array input) {
  py::array contiguous = py::array::ensure(input, py::array::c_style);
  if (!contiguous) {
    throw std::runtime_error("Expected a NumPy array that can be viewed as contiguous");
  }
  return contiguous;
}

Tensor tensor_from_numpy_array(py::array input) {
  py::array contiguous = ensure_c_contiguous(std::move(input));
  py::buffer_info buf = contiguous.request();
  std::vector<int> shape(buf.shape.begin(), buf.shape.end());
  DataType dtype = numpy_dtype_to_data_type(buf);
  Tensor t(shape, Device{DeviceType::CPU, 0}, dtype, false);
  std::memcpy(t.data(), buf.ptr, t.bytes());
  return t;
}

void copy_numpy_array_into_tensor(Tensor &tensor, py::array input) {
  if (tensor.device().type != DeviceType::CPU) {
    throw std::runtime_error("Target tensor must be on CPU.");
  }

  Tensor source = tensor_from_numpy_array(std::move(input));
  if (source.shape() != tensor.shape()) {
    throw std::runtime_error("Size mismatch.");
  }

  Tensor converted = (source.dtype() == tensor.dtype()) ? source
                                                        : source.to(tensor.dtype());
  std::memcpy(tensor.data(), converted.data(), tensor.bytes());
}

Tensor make_scalar_tensor(Device device, DataType dtype, float value) {
  Tensor scalar({1}, device, dtype, false);
  scalar.fill_(value);
  return scalar;
}

Tensor make_constant_tensor(Shape shape, Device device, DataType dtype,
                            bool requires_grad, const ScalarValue &value) {
  Tensor tensor(shape, device, dtype, requires_grad);
  tensor.fill_(value);
  return tensor;
}

} // namespace

// Trampoline for Module to allow python inheritance
class PyModule : public nn::Module {
public:
  using nn::Module::Module;
  Tensor forward_impl(Tensor x) override {
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

  py::enum_<BackendFeature>(m, "BackendFeature",
                            "Coarse-grained backend capability categories.")
      .value("ElementwiseBinary", BackendFeature::ElementwiseBinary)
      .value("BroadcastRow", BackendFeature::BroadcastRow)
      .value("Matmul", BackendFeature::Matmul)
      .value("UnaryActivation", BackendFeature::UnaryActivation)
      .value("Softmax", BackendFeature::Softmax)
      .value("Concat", BackendFeature::Concat)
      .value("Loss", BackendFeature::Loss)
      .value("Convolution", BackendFeature::Convolution)
      .value("Pooling", BackendFeature::Pooling)
      .value("BatchNorm", BackendFeature::BatchNorm)
      .value("OptimizerStep", BackendFeature::OptimizerStep)
      .value("RandomFill", BackendFeature::RandomFill)
      .value("Reduction", BackendFeature::Reduction)
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
          [](Tensor &t, py::array input) { copy_numpy_array_into_tensor(t, input); },
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
          "backward",
          [](Tensor &t, bool retain_graph) { t.backward(retain_graph); },
          py::arg("retain_graph") = false,
          "Computes the gradient of current tensor w.r.t. graph leaves.")
      .def(
          "backward",
          [](Tensor &t, const Tensor &grad, bool retain_graph) {
            t.backward(grad, retain_graph);
          },
          py::arg("grad"), py::arg("retain_graph") = false,
          "Computes the gradient with a given upstream gradient.")

      // Math & Ops
      .def("__add__", [](const Tensor &a, const Tensor &b) { return a + b; })
      .def("__add__",
           [](const Tensor &a, float b) {
             Tensor bt = make_scalar_tensor(a.device(), a.dtype(), b);
             return a + bt;
           })
      .def("__sub__", [](const Tensor &a, const Tensor &b) { return a - b; })
      .def("__sub__",
           [](const Tensor &a, float b) {
             Tensor bt = make_scalar_tensor(a.device(), a.dtype(), b);
             return a - bt;
           })
      .def("__mul__", [](const Tensor &a, const Tensor &b) { return a * b; })
      .def("__mul__",
           [](const Tensor &a, float b) {
             Tensor bt = make_scalar_tensor(a.device(), a.dtype(), b);
             return a * bt;
           })
      .def("__truediv__", [](const Tensor &a, const Tensor &b) { return a / b; })
      .def("__truediv__",
           [](const Tensor &a, float b) {
             Tensor bt = make_scalar_tensor(a.device(), a.dtype(), b);
             return a / bt;
           })
      .def("__matmul__",
           [](const Tensor &a, const Tensor &b) { return a.matmul(b); })
      .def("sum", &Tensor::sum,
           "Returns the sum of all elements in the tensor.")
      .def("reshape", &Tensor::reshape, py::arg("shape"),
           "Returns a tensor with the same data and number of elements, but "
           "with the specified shape.")
      .def("masked_fill",
           py::overload_cast<const Tensor &, float>(&Tensor::masked_fill,
                                                    py::const_),
           py::arg("mask"), py::arg("value"),
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
      .def("exp", &Tensor::exp, "Applies exp element-wise.")
      .def("log", &Tensor::log, "Applies natural log element-wise.")
      .def("sqrt", &Tensor::sqrt, "Applies square root element-wise.")
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
                               numpy_format_for_dtype(t.dtype()),
                               py_shape.size(), py_shape, py_strides);
      });

  // ============================================================================
  // Factory Functions
  // ============================================================================
  m.def("cat", &ops::cat, py::arg("tensors"), py::arg("dim") = 1,
        "Concatenates a sequence of tensors along the specified dimension.");
  m.def(
      "supports",
      [](Device device, BackendFeature feature, DataType dtype) {
        return BackendManager::get(device)->supports(feature, dtype);
      },
      py::arg("device"), py::arg("feature"), py::arg("dtype"),
      "Returns whether the selected backend advertises native support for a "
      "feature/dtype combination.");

  m.def(
      "zeros",
      [](Shape shape, std::optional<Device> device, bool requires_grad,
         DataType dtype) {
        Device dev = device.value_or(Device{DeviceType::CPU, 0});
        return make_constant_tensor(shape, dev, dtype, requires_grad,
                                    make_scalar(0.0, dtype));
      },
      py::arg("shape"), py::arg_v("device", py::none(), "None"),
      py::arg("requires_grad") = false,
      py::arg_v("dtype", DataType::Float32, "munet.DataType.Float32"),
      "Creates a tensor of the specified shape filled with zeros.");

  m.def(
      "ones",
      [](Shape shape, std::optional<Device> device, bool requires_grad,
         DataType dtype) {
        Device dev = device.value_or(Device{DeviceType::CPU, 0});
        return make_constant_tensor(shape, dev, dtype, requires_grad,
                                    make_scalar(1.0, dtype));
      },
      py::arg("shape"), py::arg_v("device", py::none(), "None"),
      py::arg("requires_grad") = false,
      py::arg_v("dtype", DataType::Float32, "munet.DataType.Float32"),
      "Creates a tensor of the specified shape filled with ones.");

  m.def(
      "rand",
      [](Shape shape, std::optional<Device> device, bool requires_grad,
         DataType dtype) {
        Device dev = device.value_or(Device{DeviceType::CPU, 0});
        if (!is_floating(dtype)) {
          throw std::runtime_error("rand only supports floating-point dtypes");
        }
        Tensor t(shape, dev, dtype, requires_grad);
        t.uniform_(0.0f, 1.0f);
        return t;
      },
      py::arg("shape"), py::arg_v("device", py::none(), "None"),
      py::arg("requires_grad") = false,
      py::arg_v("dtype", DataType::Float32, "munet.DataType.Float32"),
      "Creates a tensor of the specified shape filled with random values from "
      "U[0, 1).");

  m.def(
      "from_numpy",
      [](py::array input) { return tensor_from_numpy_array(input); },
      py::arg("input"), "Creates a CPU Tensor from a NumPy array.");

  // Alias copy_from_numpy to module level as well
  m.def(
      "copy_from_numpy",
      [](Tensor &t, py::array input) {
        try {
          copy_numpy_array_into_tensor(t, input);
        } catch (const std::runtime_error &err) {
          throw std::runtime_error(std::string("copy_from_numpy: ") + err.what());
        }
      },
      py::arg("tensor"), py::arg("input"),
      "Copies data from a NumPy array into the given CPU tensor.");

  py::class_<core::Module, std::shared_ptr<core::Module>>(m, "_CoreModule");

  // ============================================================================
  // Neural Network Layers (munet.nn)
  // ============================================================================
  auto nn = m.def_submodule("nn", "Neural Network Modules and Layers");

  py::class_<nn::Module, core::Module, std::shared_ptr<nn::Module>, PyModule>(
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
      .def("to", py::overload_cast<Device>(&nn::Module::to), py::arg("device"),
           "Moves all parameters and buffers to the specified device.")
      .def("to", py::overload_cast<DataType>(&nn::Module::to), py::arg("dtype"),
           "Converts all parameters and buffers to the specified dtype.")
      .def("to_options", py::overload_cast<const TensorOptions &>(&nn::Module::to),
           py::arg("options"),
           "Converts all parameters and buffers using explicit tensor options.")
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
      .def(py::init<int, int, bool, TensorOptions>(), py::arg("in_features"),
           py::arg("out_features"), py::arg("bias") = true,
           py::arg("options") = TensorOptions{})
      .def_readonly("weight", &nn::Linear::weight,
                    "The learnable weights of the module.")
      .def_readonly("bias", &nn::Linear::bias,
                    "The learnable bias of the module.");

  py::class_<nn::Conv2d, nn::Module, std::shared_ptr<nn::Conv2d>>(
      nn, "Conv2d", "Applies a 2D convolution over an input signal.")
      .def(py::init<int, int, int, int, int, TensorOptions>(), py::arg("in_channels"),
           py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0,
           py::arg("options") = TensorOptions{})
      .def_readonly("stride", &nn::Conv2d::stride_)
      .def_readonly("padding", &nn::Conv2d::padding_)
      .def_readonly("weight", &nn::Conv2d::weight)
      .def_readonly("bias", &nn::Conv2d::bias);

  py::class_<nn::BatchNorm2d, nn::Module, std::shared_ptr<nn::BatchNorm2d>>(
      nn, "BatchNorm2d", "Applies Batch Normalization over a 4D input.")
      .def(py::init<int, float, float, TensorOptions>(), py::arg("num_features"),
           py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f,
           py::arg("options") = TensorOptions{})
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
      .def(py::init<int, int, TensorOptions>(), py::arg("num_embeddings"),
           py::arg("embedding_dim"), py::arg("options") = TensorOptions{})
      .def_readonly("num_embeddings", &nn::Embedding::num_embeddings_)
      .def_readonly("embedding_dim", &nn::Embedding::embedding_dim_)
      .def_readonly("weight", &nn::Embedding::weight);

  py::class_<nn::LayerNorm, nn::Module, std::shared_ptr<nn::LayerNorm>>(
      nn, "LayerNorm",
      "Applies Layer Normalization over the last tensor dimension.")
      .def(py::init<int, float, TensorOptions>(), py::arg("normalized_shape"),
           py::arg("eps") = 1e-5f, py::arg("options") = TensorOptions{})
      .def_readonly("normalized_shape", &nn::LayerNorm::normalized_shape_)
      .def_readonly("eps", &nn::LayerNorm::eps_)
      .def_readonly("weight", &nn::LayerNorm::weight)
      .def_readonly("bias", &nn::LayerNorm::bias);

  py::class_<nn::MultiHeadAttention, nn::Module, std::shared_ptr<nn::MultiHeadAttention>>(
      nn, "MultiHeadAttention",
      "Applies causal/non-causal multi-head self-attention over [B,T,E].")
      .def(py::init<int, int, bool, TensorOptions>(), py::arg("embed_dim"),
           py::arg("num_heads"), py::arg("causal") = true,
           py::arg("options") = TensorOptions{})
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

  py::enum_<inference::EngineEventType>(inf, "EngineEventType")
      .value("LoadStarted", inference::EngineEventType::LoadStarted)
      .value("LoadCompleted", inference::EngineEventType::LoadCompleted)
      .value("CompileStarted", inference::EngineEventType::CompileStarted)
      .value("CompileCompleted", inference::EngineEventType::CompileCompleted)
      .value("RunStarted", inference::EngineEventType::RunStarted)
      .value("RunCompleted", inference::EngineEventType::RunCompleted)
      .value("Error", inference::EngineEventType::Error);

  py::class_<inference::EngineEvent>(inf, "EngineEvent")
      .def(py::init<>())
      .def_readonly("type", &inference::EngineEvent::type)
      .def_readonly("device", &inference::EngineEvent::device)
      .def_readonly("trace_id", &inference::EngineEvent::trace_id)
      .def_readonly("run_index", &inference::EngineEvent::run_index)
      .def_readonly("duration_ms", &inference::EngineEvent::duration_ms)
      .def_readonly("input_shape", &inference::EngineEvent::input_shape)
      .def_readonly("output_shape", &inference::EngineEvent::output_shape)
      .def_readonly("current_memory_bytes",
                    &inference::EngineEvent::current_memory_bytes)
      .def_readonly("peak_memory_bytes",
                    &inference::EngineEvent::peak_memory_bytes)
      .def_readonly("span", &inference::EngineEvent::span)
      .def_readonly("message", &inference::EngineEvent::message);

  py::class_<inference::EngineConfig>(inf, "EngineConfig")
      .def(py::init<>())
      .def_readwrite("device", &inference::EngineConfig::device)
      .def_readwrite("warmup_runs", &inference::EngineConfig::warmup_runs)
      .def_readwrite("strict_shape_check",
                     &inference::EngineConfig::strict_shape_check)
      .def_readwrite("allow_autograd_inputs",
                     &inference::EngineConfig::allow_autograd_inputs)
      .def_readwrite("capture_profiler_memory",
                     &inference::EngineConfig::capture_profiler_memory)
      .def_readwrite("lean_mode", &inference::EngineConfig::lean_mode)
      .def_readwrite("prepared_input_cache_entries",
                     &inference::EngineConfig::prepared_input_cache_entries)
      .def_readwrite("prepared_input_cache_max_bytes",
                     &inference::EngineConfig::prepared_input_cache_max_bytes);

  py::class_<inference::EngineStats>(inf, "EngineStats")
      .def(py::init<>())
      .def_readonly("runs", &inference::EngineStats::runs)
      .def_readonly("last_compile_trace_id",
                    &inference::EngineStats::last_compile_trace_id)
      .def_readonly("last_run_trace_id",
                    &inference::EngineStats::last_run_trace_id)
      .def_readonly("load_to_device_ms",
                    &inference::EngineStats::load_to_device_ms)
      .def_readonly("load_eval_ms", &inference::EngineStats::load_eval_ms)
      .def_readonly("last_run_ms", &inference::EngineStats::last_run_ms)
      .def_readonly("last_prepare_input_ms",
                    &inference::EngineStats::last_prepare_input_ms)
      .def_readonly("last_forward_ms", &inference::EngineStats::last_forward_ms)
      .def_readonly("last_output_validation_ms",
                    &inference::EngineStats::last_output_validation_ms)
      .def_readonly("compile_ms", &inference::EngineStats::compile_ms)
      .def_readonly("compile_prepare_input_ms",
                    &inference::EngineStats::compile_prepare_input_ms)
      .def_readonly("compile_forward_ms",
                    &inference::EngineStats::compile_forward_ms)
      .def_readonly("compile_warmup_ms",
                    &inference::EngineStats::compile_warmup_ms)
      .def_readonly("compiled_input_shape",
                    &inference::EngineStats::compiled_input_shape)
      .def_readonly("compiled_output_shape",
                    &inference::EngineStats::compiled_output_shape)
      .def_readonly("current_memory_bytes",
                    &inference::EngineStats::current_memory_bytes)
      .def_readonly("peak_memory_bytes",
                    &inference::EngineStats::peak_memory_bytes)
      .def_readonly("prepared_input_cache_entries",
                    &inference::EngineStats::prepared_input_cache_entries)
      .def_readonly("prepared_input_cache_bytes",
                    &inference::EngineStats::prepared_input_cache_bytes)
      .def_readonly("prepared_input_cache_hits",
                    &inference::EngineStats::prepared_input_cache_hits)
      .def_readonly("prepared_input_cache_misses",
                    &inference::EngineStats::prepared_input_cache_misses)
      .def_readonly("prepared_input_cache_evictions",
                    &inference::EngineStats::prepared_input_cache_evictions);

  py::class_<inference::Engine>(inf, "Engine")
      .def(py::init<inference::EngineConfig>(), py::arg("config") = inference::EngineConfig{})
      .def("set_device", &inference::Engine::set_device, py::arg("device"))
      .def("device", &inference::Engine::device)
      .def("set_warmup_runs", &inference::Engine::set_warmup_runs,
           py::arg("warmup_runs"))
      .def("set_strict_shape_check", &inference::Engine::set_strict_shape_check,
           py::arg("enabled"))
      .def("set_allow_autograd_inputs",
           &inference::Engine::set_allow_autograd_inputs, py::arg("enabled"))
      .def("allow_autograd_inputs", &inference::Engine::allow_autograd_inputs)
      .def("set_capture_profiler_memory",
           &inference::Engine::set_capture_profiler_memory,
           py::arg("enabled"))
      .def("capture_profiler_memory",
           &inference::Engine::capture_profiler_memory)
      .def("set_lean_mode", &inference::Engine::set_lean_mode,
           py::arg("enabled"))
      .def("lean_mode", &inference::Engine::lean_mode)
      .def("set_prepared_input_cache_entries",
           &inference::Engine::set_prepared_input_cache_entries, py::arg("entries"))
      .def("prepared_input_cache_entries_limit",
           &inference::Engine::prepared_input_cache_entries_limit)
      .def("set_prepared_input_cache_max_bytes",
           &inference::Engine::set_prepared_input_cache_max_bytes, py::arg("bytes"))
      .def("prepared_input_cache_max_bytes_limit",
           &inference::Engine::prepared_input_cache_max_bytes_limit)
      .def("clear_prepared_input_cache",
           &inference::Engine::clear_prepared_input_cache)
      .def("set_observer", &inference::Engine::set_observer,
           py::arg("observer"))
      .def("clear_observer", &inference::Engine::clear_observer)
      .def(
          "load",
          [](inference::Engine &self, py::object module) {
            self.load(module.cast<std::shared_ptr<core::Module>>());
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
      .def("prepare_batch", &inference::Engine::prepare_batch, py::arg("inputs"))
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
     src = munet.from_numpy(np.ascontiguousarray(arr))
     if src.dtype != t.dtype:
         src = src.to(t.dtype)
     if target.type != munet.DeviceType.CPU:
         src = src.to(target)
     t.replace_(src)
     t.requires_grad = req

 def _tensor_dtype_name(t):
     m = __import__("munet")
     return {m.DataType.Float32: "float32",
             m.DataType.Float16: "float16",
             m.DataType.Int32: "int32"}[t.dtype]

 def _dtype_from_name(name):
     m = __import__("munet")
     return {"float32": m.DataType.Float32,
             "float16": m.DataType.Float16,
             "int32": m.DataType.Int32}[name]

 def _tensor_options_for_dtype(dtype_name):
     m = __import__("munet")
     opts = m.TensorOptions()
     opts.dtype = _dtype_from_name(dtype_name)
     return opts


 SERIALIZATION_FORMAT_NAME = "munet_model"
 SERIALIZATION_FORMAT_REVISION = 1
 SERIALIZATION_LEGACY_TAG = "munet_model_v1"
 SERIALIZATION_ARTIFACT_KIND = "deploy_model"
 SERIALIZATION_ARTIFACT_SCOPE = "runtime_only"
 SERIALIZATION_DEFAULT_LOAD_MODE = "eval"
 SERIALIZATION_CONTAINS_TRAINING_STATE = False
 SERIALIZATION_DEVICE_POLICY = "caller_specified"
 SERIALIZATION_DTYPE_POLICY = "per_tensor"
 SERIALIZATION_RECOMMENDED_LOADER = "load_for_inference"
 SERIALIZATION_COMPILE_CONTRACT_POLICY = "external"
 SERIALIZATION_FORBIDDEN_TRAINING_KEY_TOKENS = (
     "optim",
     "optimizer",
     "scheduler",
     "scaler",
     "master_weight",
     "checkpoint",
     "epoch",
     "step",
     "grad",
 )

 def serialization_format_info():
     return {
         "format_name": SERIALIZATION_FORMAT_NAME,
         "format_revision": SERIALIZATION_FORMAT_REVISION,
         "legacy_tag": SERIALIZATION_LEGACY_TAG,
         "artifact_kind": SERIALIZATION_ARTIFACT_KIND,
         "artifact_scope": SERIALIZATION_ARTIFACT_SCOPE,
         "default_load_mode": SERIALIZATION_DEFAULT_LOAD_MODE,
         "contains_training_state": SERIALIZATION_CONTAINS_TRAINING_STATE,
         "device_policy": SERIALIZATION_DEVICE_POLICY,
         "dtype_policy": SERIALIZATION_DTYPE_POLICY,
         "recommended_loader": SERIALIZATION_RECOMMENDED_LOADER,
         "compile_contract_policy": SERIALIZATION_COMPILE_CONTRACT_POLICY,
         "load_compatibility": [
             {"format_name": SERIALIZATION_FORMAT_NAME, "format_revision": SERIALIZATION_FORMAT_REVISION},
             {"legacy_tag": SERIALIZATION_LEGACY_TAG},
         ],
         "policy": "Forward-compatible loading is not guaranteed across future major format revisions.",
     }

 def _string_state_value(state, key):
     return str(state[key]) if key in state else None

 def _bool_state_value(state, key):
     return bool(state[key]) if key in state else None

 def _serialization_metadata_from_state(state):
     format_name = _string_state_value(state, '__format_name__')
     format_revision = int(state['__format_revision__']) if '__format_revision__' in state else None
     legacy_tag = _string_state_value(state, '__format_version__')
     producer = _string_state_value(state, '__producer__')
     artifact_kind = _string_state_value(state, '__artifact_kind__')
     artifact_scope = _string_state_value(state, '__artifact_scope__')
     default_load_mode = _string_state_value(state, '__default_load_mode__')
     device_policy = _string_state_value(state, '__device_policy__')
     dtype_policy = _string_state_value(state, '__dtype_policy__')
     recommended_loader = _string_state_value(state, '__recommended_loader__')
     compile_contract_policy = _string_state_value(state, '__compile_contract_policy__')
     contains_training_state = _bool_state_value(state, '__contains_training_state__')
     tensor_names = []
     if '__tensor_names__' in state:
         import json
         tensor_names = list(json.loads(str(state['__tensor_names__'])))
     has_config = '__config__' in state

     if format_name is None and legacy_tag == SERIALIZATION_LEGACY_TAG:
         format_name = SERIALIZATION_FORMAT_NAME
         format_revision = SERIALIZATION_FORMAT_REVISION
     if artifact_kind is None:
         artifact_kind = SERIALIZATION_ARTIFACT_KIND
     if artifact_scope is None:
         artifact_scope = SERIALIZATION_ARTIFACT_SCOPE
     if default_load_mode is None:
         default_load_mode = SERIALIZATION_DEFAULT_LOAD_MODE
     if contains_training_state is None:
         contains_training_state = SERIALIZATION_CONTAINS_TRAINING_STATE
     if device_policy is None:
         device_policy = SERIALIZATION_DEVICE_POLICY
     if dtype_policy is None:
         dtype_policy = SERIALIZATION_DTYPE_POLICY
     if recommended_loader is None:
         recommended_loader = SERIALIZATION_RECOMMENDED_LOADER
     if compile_contract_policy is None:
         compile_contract_policy = SERIALIZATION_COMPILE_CONTRACT_POLICY

     return {
         "format_name": format_name,
         "format_revision": format_revision,
         "legacy_tag": legacy_tag,
         "producer": producer,
         "artifact_kind": artifact_kind,
         "artifact_scope": artifact_scope,
         "default_load_mode": default_load_mode,
         "contains_training_state": contains_training_state,
         "device_policy": device_policy,
         "dtype_policy": dtype_policy,
         "recommended_loader": recommended_loader,
         "compile_contract_policy": compile_contract_policy,
         "tensor_names": tensor_names,
         "tensor_count": len(tensor_names),
         "has_config": has_config,
     }

 def serialization_metadata(filename):
     import numpy as np
     with np.load(filename, allow_pickle=True) as state:
         return _serialization_metadata_from_state(state)

 def _payload_tensor_names(state):
     return sorted([
         key for key in state.files
         if not key.startswith('__')
     ])

 def _validate_serialization_payload_keys(state, metadata):
     payload_tensor_names = _payload_tensor_names(state)

     for key in payload_tensor_names:
         lowered = key.lower()
         if any(token in lowered for token in SERIALIZATION_FORBIDDEN_TRAINING_KEY_TOKENS):
             raise ValueError(
                 f"Unsupported training/checkpoint payload key in deploy artifact: {key!r}."
             )

     manifest_tensor_names = sorted(metadata.get("tensor_names", []))
     if manifest_tensor_names:
         if payload_tensor_names != manifest_tensor_names:
             raise ValueError(
                 "Serialization tensor manifest does not match payload keys. "
                 f"manifest={manifest_tensor_names}, payload={payload_tensor_names}"
             )

     return payload_tensor_names

 def _validate_serialization_metadata(state):
     metadata = _serialization_metadata_from_state(state)
     format_name = metadata["format_name"]
     format_revision = metadata["format_revision"]
     legacy_tag = metadata["legacy_tag"]

     if format_name != SERIALIZATION_FORMAT_NAME:
         raise ValueError(
             f"Unsupported serialization format name: {format_name!r}. "
             f"Expected {SERIALIZATION_FORMAT_NAME!r}."
         )

     if format_revision != SERIALIZATION_FORMAT_REVISION:
         raise ValueError(
             f"Unsupported serialization format revision: {format_revision!r}. "
             f"This build supports revision {SERIALIZATION_FORMAT_REVISION}."
         )

     if legacy_tag not in (None, SERIALIZATION_LEGACY_TAG):
         raise ValueError(
             f"Unsupported legacy serialization tag: {legacy_tag!r}. "
             f"Expected {SERIALIZATION_LEGACY_TAG!r}."
         )

     if metadata["artifact_kind"] != SERIALIZATION_ARTIFACT_KIND:
         raise ValueError(
             f"Unsupported serialization artifact kind: {metadata['artifact_kind']!r}. "
             f"Expected {SERIALIZATION_ARTIFACT_KIND!r}."
         )

     if metadata["artifact_scope"] != SERIALIZATION_ARTIFACT_SCOPE:
         raise ValueError(
             f"Unsupported serialization artifact scope: {metadata['artifact_scope']!r}. "
             f"Expected {SERIALIZATION_ARTIFACT_SCOPE!r}."
         )

     if metadata["contains_training_state"] is not SERIALIZATION_CONTAINS_TRAINING_STATE:
         raise ValueError(
             "Unsupported serialization payload: deploy artifacts must not contain training-only state."
         )

     if metadata["recommended_loader"] != SERIALIZATION_RECOMMENDED_LOADER:
         raise ValueError(
             f"Unsupported recommended loader: {metadata['recommended_loader']!r}. "
             f"Expected {SERIALIZATION_RECOMMENDED_LOADER!r}."
         )

     if metadata["compile_contract_policy"] != SERIALIZATION_COMPILE_CONTRACT_POLICY:
         raise ValueError(
             f"Unsupported compile contract policy: {metadata['compile_contract_policy']!r}. "
             f"Expected {SERIALIZATION_COMPILE_CONTRACT_POLICY!r}."
         )

     _validate_serialization_payload_keys(state, metadata)

     return metadata

 def _direct_named_tensors(module):
     name = type(module).__name__
     items = []
     if name in ('Linear', 'Conv2d', 'Embedding', 'LayerNorm'):
         items.append(('weight', module.weight))
         if hasattr(module, 'bias') and getattr(module, 'bias') is not None and getattr(module, 'bias').numel() > 0:
             items.append(('bias', module.bias))
     elif name == 'BatchNorm2d':
         items.extend([
             ('weight', module.weight),
             ('bias', module.bias),
             ('running_mean', module.running_mean),
             ('running_var', module.running_var),
         ])
     return items

 def _iter_named_tensors(module):
     for name, tensor in _direct_named_tensors(module):
         yield name, tensor
     for prefix, submodule in module.named_modules().items():
         for name, tensor in _direct_named_tensors(submodule):
             yield f"{prefix}.{name}", tensor

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
             return {'type': name, 'in_features': m.weight.shape[0], 'out_features': m.weight.shape[1], 'bias': has_bias, 'dtype': _tensor_dtype_name(m.weight)}
         elif name == 'Conv2d':
             return {'type': name, 'in_channels': m.weight.shape[1], 'out_channels': m.weight.shape[0], 'kernel_size': m.weight.shape[2], 'stride': m.stride, 'padding': m.padding, 'dtype': _tensor_dtype_name(m.weight)}
         elif name == 'MaxPool2d':
             return {'type': name, 'kernel_size': m.kernel_size, 'stride': m.stride, 'padding': m.padding}
         elif name == 'BatchNorm2d':
             return {'type': name, 'num_features': m.weight.shape[0], 'eps': m.eps, 'momentum': m.momentum, 'dtype': _tensor_dtype_name(m.weight)}
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
             return {'type': name, 'num_embeddings': m.num_embeddings, 'embedding_dim': m.embedding_dim, 'dtype': _tensor_dtype_name(m.weight)}
         elif name == 'LayerNorm':
             return {'type': name, 'normalized_shape': m.normalized_shape, 'eps': m.eps, 'dtype': _tensor_dtype_name(m.weight)}
         elif name == 'MultiHeadAttention':
             return {'type': name, 'embed_dim': m.embed_dim, 'num_heads': m.num_heads, 'causal': bool(m.causal), 'dtype': _tensor_dtype_name(m.q_proj.weight)}
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
     for name, tensor in _iter_named_tensors(module):
         state[name] = tensor_to_numpy(tensor)

     tensor_names = sorted(state.keys())
     state['__config__'] = np.array(json.dumps(config))
     state['__format_name__'] = np.array(SERIALIZATION_FORMAT_NAME)
     state['__format_revision__'] = np.array(SERIALIZATION_FORMAT_REVISION)
     state['__format_version__'] = np.array(SERIALIZATION_LEGACY_TAG)
     state['__producer__'] = np.array('munet')
     state['__artifact_kind__'] = np.array(SERIALIZATION_ARTIFACT_KIND)
     state['__artifact_scope__'] = np.array(SERIALIZATION_ARTIFACT_SCOPE)
     state['__default_load_mode__'] = np.array(SERIALIZATION_DEFAULT_LOAD_MODE)
     state['__contains_training_state__'] = np.array(SERIALIZATION_CONTAINS_TRAINING_STATE)
     state['__device_policy__'] = np.array(SERIALIZATION_DEVICE_POLICY)
     state['__dtype_policy__'] = np.array(SERIALIZATION_DTYPE_POLICY)
     state['__recommended_loader__'] = np.array(SERIALIZATION_RECOMMENDED_LOADER)
     state['__compile_contract_policy__'] = np.array(SERIALIZATION_COMPILE_CONTRACT_POLICY)
     state['__tensor_names__'] = np.array(json.dumps(tensor_names))
     np.savez(filename, **state)

 def _normalize_loaded_module_for_inference(module, device=None):
     if device is not None:
         module.to(device)
     module.eval()
     return module

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
         opts = _tensor_options_for_dtype(cfg.get('dtype', 'float32'))
         if t == 'Sequential': return munet.nn.Sequential([build_module(c) for c in cfg['layers']])
         elif t == 'Linear': return munet.nn.Linear(cfg['in_features'], cfg['out_features'], cfg['bias'], opts)
         elif t == 'Conv2d': return munet.nn.Conv2d(cfg['in_channels'], cfg['out_channels'], cfg['kernel_size'], cfg['stride'], cfg['padding'], opts)
         elif t == 'MaxPool2d': return munet.nn.MaxPool2d(cfg['kernel_size'], cfg['stride'], cfg['padding'])
         elif t == 'BatchNorm2d': return munet.nn.BatchNorm2d(cfg['num_features'], cfg['eps'], cfg['momentum'], opts)
         elif t == 'Upsample': return munet.nn.Upsample(cfg['scale_factor'])
         elif t == 'GlobalAvgPool2d': return munet.nn.GlobalAvgPool2d()
         elif t == 'ReLU': return munet.nn.ReLU()
         elif t == 'Sigmoid': return munet.nn.Sigmoid()
         elif t == 'Tanh': return munet.nn.Tanh()
         elif t == 'GELU': return munet.nn.GELU()
         elif t == 'LeakyReLU': return munet.nn.LeakyReLU(cfg.get('negative_slope', 0.01))
         elif t == 'Dropout': return munet.nn.Dropout(cfg.get('p', 0.5))
         elif t == 'Embedding': return munet.nn.Embedding(cfg['num_embeddings'], cfg['embedding_dim'], opts)
         elif t == 'LayerNorm': return munet.nn.LayerNorm(cfg['normalized_shape'], cfg.get('eps', 1e-5), opts)
         elif t == 'MultiHeadAttention': return munet.nn.MultiHeadAttention(cfg['embed_dim'], cfg['num_heads'], cfg.get('causal', True), opts)
         elif t == 'Flatten': return munet.nn.Flatten()
         else:
             raise ValueError(f"Unsupported saved module type: {t}")

     def copy_numpy_into_tensor(t, arr):
         m = __import__("munet")
         req = bool(t.requires_grad)
         target = t.device
         src = m.from_numpy(np.ascontiguousarray(arr))
         if src.dtype != t.dtype:
             src = src.to(t.dtype)
         if target.type != m.DeviceType.CPU:
             src = src.to(target)
         t.replace_(src)
         t.requires_grad = req

     def apply_state(module, state):
         for name, p in _iter_named_tensors(module):
             if name in state:
                 copy_numpy_into_tensor(p, state[name])
         return module

     if filename is None:
         with np.load(arg, allow_pickle=True) as state:
             _validate_serialization_metadata(state)
             if '__config__' not in state:
                 raise ValueError("File does not contain architecture config. Use `load(module, filename)` for weights-only restore.")

             config = json.loads(str(state['__config__']))
             module = build_module(config)
             return apply_state(module, state)
     else:
         module = arg
         with np.load(filename, allow_pickle=True) as state:
             _validate_serialization_metadata(state)
             return apply_state(module, state)

 def load_for_inference(arg, filename=None, device=None):
     """Load a deploy artifact and normalize the result for inference execution.

     Usage:
       - load_for_inference("model.npz", device=None) -> reconstruct + eval-safe module.
       - load_for_inference(module, "model.npz", device=None) -> apply state into existing module, move if requested, then eval().
     """
     import numpy as np
     import munet

     path = arg if filename is None else filename
     with np.load(path, allow_pickle=True) as state:
         metadata = _validate_serialization_metadata(state)
         if metadata["default_load_mode"] != SERIALIZATION_DEFAULT_LOAD_MODE:
             raise ValueError(
                 f"Unsupported deploy load mode: {metadata['default_load_mode']!r}. "
                 f"Expected {SERIALIZATION_DEFAULT_LOAD_MODE!r}."
             )

     module = munet.load(arg, filename) if filename is not None else munet.load(arg)
     return _normalize_loaded_module_for_inference(module, device)

 def load_weights(module, filename):
     """Alias for `load(module, filename)` to explicitly do weights-only restore."""
     m = __import__("munet")
     return m.load(module, filename)

 def load_weights_for_inference(module, filename, device=None):
     """Weights-only restore that also normalizes the module for inference execution."""
     m = __import__("munet")
     m.load(module, filename)
     return _normalize_loaded_module_for_inference(module, device)

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
