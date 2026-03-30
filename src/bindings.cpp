#include "autograd/engine.hpp"
#include "core/op_dispatch.hpp"
#include "inference.hpp"
#include "nn.hpp"
#include "nn/module.hpp"
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
#include <unordered_map>

#ifdef MUNET_USE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef MUNET_USE_VULKAN
#include <vulkan/vulkan.h>
#endif

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
  if (buf.itemsize == static_cast<py::ssize_t>(sizeof(int8_t)) &&
      buf.format == py::format_descriptor<int8_t>::format()) {
    return DataType::Int8;
  }
  if (buf.itemsize == 2 && buf.format.find('e') != std::string::npos) {
    return DataType::Float16;
  }
  if (buf.itemsize == 2 && buf.format.find('H') != std::string::npos) {
    return DataType::BFloat16;
  }
  throw std::runtime_error(
      "Unsupported NumPy dtype; expected float32, float16/bfloat16, int32, or int8");
}

std::string numpy_format_for_dtype(DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return py::format_descriptor<float>::format();
  case DataType::Float16:
    return "e";
  case DataType::BFloat16:
    return "H";
  case DataType::Int32:
    return py::format_descriptor<int32_t>::format();
  case DataType::Int8:
    return py::format_descriptor<int8_t>::format();
  default:
    throw std::runtime_error("Unsupported tensor dtype for NumPy conversion");
  }
}

py::array ensure_c_contiguous(py::array input) {
  py::array contiguous = py::array::ensure(input, py::array::c_style);
  if (!contiguous) {
    throw std::runtime_error(
        "Expected a NumPy array that can be viewed as contiguous");
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

  Tensor converted =
      (source.dtype() == tensor.dtype()) ? source : source.to(tensor.dtype());
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

py::dict accelerator_probe_dict(DeviceType type, const std::string &name,
                                bool compiled,
                                const std::string &version = "") {
  py::dict out;
  out["name"] = py::cast(name);
  out["type"] = py::cast(type);
  out["compiled"] = py::cast(compiled);
  out["version"] = py::cast(version);

  py::list devices;
  std::string detail;
  bool available = false;
  if (compiled) {
    constexpr int kMaxProbeDevices = 32;
    for (int index = 0; index < kMaxProbeDevices; ++index) {
      try {
        Device candidate{type, index};
        auto backend = BackendManager::get(candidate);
        backend->synchronize();
        devices.append(py::cast(candidate));
        available = true;
      } catch (const std::exception &e) {
        if (index == 0) {
          detail = e.what();
        }
        break;
      } catch (...) {
        if (index == 0) {
          detail = "Unknown runtime probe failure.";
        }
        break;
      }
    }

    if (!available && detail.empty()) {
      detail = "No runtime devices detected.";
    }
  } else {
    detail = "Backend not compiled into this build.";
  }

  out["available"] = py::cast(available);
  out["devices"] = std::move(devices);
  out["detail"] = py::cast(detail);
  return out;
}

// Helper to apply a single index/slice to a tensor
Tensor apply_index(const Tensor &t, py::object idx, int dim) {
  if (py::isinstance<py::int_>(idx)) {
    // Integer index - select single element along this dimension
    int i = idx.cast<int>();
    if (i < 0)
      i += t.shape()[dim];
    return t.narrow(dim, i, 1);
  } else if (py::isinstance<py::slice>(idx)) {
    // Slice - extract a range
    py::slice sl = idx.cast<py::slice>();
    size_t start, stop, step, slicelength;
    if (!sl.compute(t.shape()[dim], &start, &stop, &step, &slicelength))
      throw py::error_already_set();
    if (step != 1)
      throw std::runtime_error("Only step=1 slices are supported");
    return t.narrow(dim, static_cast<int>(start), static_cast<int>(stop - start));
  } else if (idx.is_none()) {
    // None means keep the whole dimension
    return t;
  } else {
    throw std::runtime_error("Unsupported index type: " +
                             std::string(py::str(idx.get_type())));
  }
}

// __getitem__ implementation
Tensor tensor_getitem(const Tensor &t, py::object key) {
  if (!t.impl_)
    throw std::runtime_error("Cannot index an uninitialized tensor");
  
  if (py::isinstance<py::tuple>(key)) {
    py::tuple indices = key.cast<py::tuple>();
    Tensor result = t;
    std::vector<int> new_shape;
    
    for (size_t i = 0; i < indices.size(); ++i) {
      auto idx = indices[i];
      if (py::isinstance<py::int_>(idx)) {
        int i_val = idx.cast<int>();
        if (i_val < 0) i_val += result.shape()[i];
        result = result.narrow(static_cast<int>(i), i_val, 1);
      } else if (py::isinstance<py::slice>(idx)) {
        py::slice sl = idx.cast<py::slice>();
        size_t start, stop, step, slicelength;
        if (!sl.compute(result.shape()[i], &start, &stop, &step, &slicelength))
          throw py::error_already_set();
        if (step != 1)
          throw std::runtime_error("Only step=1 slices are supported");
        result = result.narrow(static_cast<int>(i), static_cast<int>(start), static_cast<int>(stop - start));
      }
    }
    return result;
  } else {
    // Single index
    bool is_int = py::isinstance<py::int_>(key);
    Tensor result = apply_index(t, key, 0);
    if (is_int) {
      // Squeeze the first dimension for integer indexing
      std::vector<int> new_shape(result.shape().begin() + 1, result.shape().end());
      return result.reshape(new_shape);
    }
    return result;
  }
}

ops::OpId parse_op_id_name(const std::string &name) {
  static const std::unordered_map<std::string, ops::OpId> kNameToOp = {
      {"add", ops::OpId::Add},
      {"sub", ops::OpId::Sub},
      {"mul", ops::OpId::Mul},
      {"div", ops::OpId::Div},
      {"masked_fill", ops::OpId::MaskedFill},
      {"matmul", ops::OpId::Matmul},
      {"relu", ops::OpId::Relu},
      {"sigmoid", ops::OpId::Sigmoid},
      {"exp", ops::OpId::Exp},
      {"log", ops::OpId::Log},
      {"sqrt", ops::OpId::Sqrt},
      {"rsqrt", ops::OpId::Rsqrt},
      {"sin", ops::OpId::Sin},
      {"cos", ops::OpId::Cos},
      {"softmax", ops::OpId::Softmax},
      {"log_softmax", ops::OpId::LogSoftmax},
      {"cat", ops::OpId::Cat},
      {"sum", ops::OpId::Sum},
      {"sum_to_shape", ops::OpId::SumToShape},
      {"mean", ops::OpId::Mean},
      {"reshape", ops::OpId::Reshape},
      {"conv2d", ops::OpId::Conv2D},
      {"max_pool2d", ops::OpId::MaxPool2D},
      {"upsample2d", ops::OpId::Upsample2D},
      {"batch_norm", ops::OpId::BatchNorm},
      {"layer_norm", ops::OpId::LayerNorm},
      {"mse_loss", ops::OpId::MSELoss},
      {"cross_entropy", ops::OpId::CrossEntropy},
      {"transpose", ops::OpId::Transpose},
      {"narrow", ops::OpId::Narrow},
      {"zeros", ops::OpId::Zeros},
  };
  auto it = kNameToOp.find(name);
  if (it == kNameToOp.end()) {
    throw std::runtime_error("Unknown op name for dispatch debug dump: " +
                             name);
  }
  return it->second;
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
      .value("BFloat16", DataType::BFloat16)
      .value("Int32", DataType::Int32)
      .value("Int8", DataType::Int8)
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
      .def("to_string", &Device::to_string,
           "Returns a stable device string like 'cpu:0' or 'cuda:1'.")
      .def("__repr__", &Device::to_string)
      .def("__str__", &Device::to_string)
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
      .def("__getitem__", &tensor_getitem, py::arg("key"),
           "Get item or slice from tensor using Python-style indexing.")
      .def("numel", [](const Tensor &t) { return t.impl_ ? t.size() : 0; },
          "Returns the total number of elements in the tensor.")
      .def("detach", &Tensor::detach,
           "Returns a new Tensor, detached from the current autograd graph.")
      .def("clone", &Tensor::clone,
           "Returns a deep copy of this tensor.")
      .def("__repr__",
           [](const Tensor &t) {
             if (!t.impl_)
               return std::string("Tensor(uninitialized)");
             return "Tensor(shape=" + to_string(t.shape()) + ", device='" +
                    t.device().to_string() + "'" +
                    (t.requires_grad() ? ", requires_grad=True)" : ")");
           })
      .def("to", py::overload_cast<Device>(&Tensor::to, py::const_),
           py::arg("device"), "Moves the tensor to the specified device.")
      .def("to", py::overload_cast<DataType>(&Tensor::to, py::const_),
           py::arg("dtype"), "Converts the tensor to the specified dtype.")
      .def("to_options",
           py::overload_cast<const TensorOptions &>(&Tensor::to, py::const_),
           py::arg("options"),
           "Converts the tensor using explicit tensor options.")
      .def(
          "copy_from_numpy",
          [](Tensor &t, py::array input) {
            copy_numpy_array_into_tensor(t, input);
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
      .def("narrow", &Tensor::narrow, py::arg("dim"), py::arg("start"),
           py::arg("length"), "Returns a narrowed tensor view.")
      .def("contiguous", &Tensor::contiguous,
           "Returns a contiguous tensor containing the same data as this "
           "tensor.")

      // Autograd
      .def("zero_grad", &Tensor::zero_grad,
           "Clears the gradient of the tensor.")
      .def("has_grad", &Tensor::has_grad,
           "Returns whether this tensor currently has gradient storage.")
      .def(
          "register_gradient_hook",
          [](Tensor &t, py::function hook) {
            t.register_gradient_hook([hook](const Tensor &grad) {
              py::gil_scoped_acquire acquire;
              py::object out = hook(grad);
              if (out.is_none()) {
                return grad;
              }
              return out.cast<Tensor>();
            });
          },
          py::arg("hook"),
          "Registers a gradient hook. Return None to keep the original grad.")
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
      .def(
          "all_reduce",
          [](Tensor &t, std::optional<size_t> num_elements) {
            if (!t.impl_ || !t.impl_->storage) {
              throw std::runtime_error("Cannot all_reduce an uninitialized tensor.");
            }
            if (t.storage_offset() != 0 ||
                t.bytes() != t.impl_->storage->size_bytes()) {
              throw std::runtime_error(
                  "all_reduce currently requires a base tensor (no view/slice offset).");
            }
            const size_t elems = num_elements.value_or(t.size());
            if (elems > t.size()) {
              throw std::runtime_error("all_reduce num_elements exceeds tensor size.");
            }
            t.impl_->backend().all_reduce(*t.impl_->storage, elems);
            return t;
          },
          py::arg("num_elements") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          "Performs in-place all-reduce on this tensor via the active backend.")

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
      .def("__truediv__",
           [](const Tensor &a, const Tensor &b) { return a / b; })
      .def("__truediv__",
           [](const Tensor &a, float b) {
             Tensor bt = make_scalar_tensor(a.device(), a.dtype(), b);
             return a / bt;
           })
      .def("__matmul__",
           [](const Tensor &a, const Tensor &b) { return a.matmul(b); })
      .def("matmul", &Tensor::matmul, py::arg("other"),
           "Matrix multiply with another tensor. Equivalent to `a @ b`.")
      .def("sum", &Tensor::sum,
           "Returns the sum of all elements in the tensor.")
      .def("mean", &Tensor::mean, py::arg("dim") = -1,
           py::arg("keepdim") = false,
           "Returns the mean reduced along the specified dimension.")
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
      .def(
          "fill_",
          [](Tensor &self, float value) -> Tensor & {
            self.fill_(value);
            return self;
          },
          py::arg("value"), py::return_value_policy::reference_internal,
          "In-place fills the tensor with a scalar value and returns self.")
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
      .def("rsqrt", &Tensor::rsqrt,
           "Applies reciprocal square root element-wise.")
      .def("sin", &Tensor::sin, "Applies sine element-wise.")
      .def("cos", &Tensor::cos, "Applies cosine element-wise.")
      .def("softmax", &Tensor::softmax, py::arg("dim") = -1,
           "Applies softmax along a dimension.")
      .def("__getitem__", &tensor_getitem, py::arg("key"),
           "Returns a slice or element of the tensor. Supports integer indexing "
           "and slicing (e.g., t[0], t[0:10], t[0:10, 5:20]).")
      .def("narrow", &Tensor::narrow, py::arg("dim"), py::arg("start"), py::arg("length"))
      .def("contiguous", &Tensor::contiguous)
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
      .def("layer_norm", &Tensor::layer_norm, py::arg("weight"),
           py::arg("bias"), py::arg("eps") = 1e-5f,
           "Applies Layer Normalization over the last tensor dimension.")
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
        const py::ssize_t element_size =
            static_cast<py::ssize_t>(dtype_size(t.dtype()));
        for (size_t i = 0; i < py_shape.size(); ++i) {
          py_strides[i] =
              static_cast<py::ssize_t>(t.strides()[i]) * element_size;
        }

        auto *base_ptr =
            static_cast<char *>(t.data()) +
            static_cast<py::ssize_t>(t.storage_offset()) * element_size;
        return py::buffer_info(base_ptr, dtype_size(t.dtype()),
                               numpy_format_for_dtype(t.dtype()),
                               py_shape.size(), py_shape, py_strides);
      });

  // ============================================================================
  // Factory Functions
  // ============================================================================
  m.def("matmul", &ops::matmul, py::arg("a"), py::arg("b"),
        "Matrix multiply two tensors. Mirrors `torch.matmul(a, b)`.");
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
      "available_accelerators",
      []() {
        py::list out;

        py::dict cpu;
        cpu["name"] = py::cast(std::string("cpu"));
        cpu["type"] = py::cast(DeviceType::CPU);
        cpu["compiled"] = py::cast(true);
        cpu["available"] = py::cast(true);
        cpu["version"] = py::cast(std::string(""));
        py::list cpu_devices;
        cpu_devices.append(py::cast(Device{DeviceType::CPU, 0}));
        cpu["devices"] = std::move(cpu_devices);
        cpu["detail"] = py::cast(std::string("Always available."));
        out.append(std::move(cpu));

#ifdef MUNET_USE_CUDA
        int runtime_version = 0;
        int driver_version = 0;
        std::string cuda_version = "";
        if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
          cuda_version +=
              "runtime=" + std::to_string(runtime_version / 1000) + "." +
              std::to_string((runtime_version % 1000) / 10);
        }
        if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
          if (!cuda_version.empty()) {
            cuda_version += ", ";
          }
          cuda_version +=
              "driver=" + std::to_string(driver_version / 1000) + "." +
              std::to_string((driver_version % 1000) / 10);
        }
        out.append(accelerator_probe_dict(DeviceType::CUDA, "cuda", true,
                                          cuda_version));
#else
        out.append(accelerator_probe_dict(DeviceType::CUDA, "cuda", false));
#endif

#ifdef MUNET_USE_VULKAN
        uint32_t vk_header_version_complete = VK_HEADER_VERSION_COMPLETE;
        std::string vk_version =
            std::to_string(VK_API_VERSION_MAJOR(vk_header_version_complete)) +
            "." +
            std::to_string(VK_API_VERSION_MINOR(vk_header_version_complete)) +
            "." +
            std::to_string(VK_API_VERSION_PATCH(vk_header_version_complete));
        out.append(accelerator_probe_dict(DeviceType::VULKAN, "vulkan", true,
                                          vk_version));
#else
        out.append(accelerator_probe_dict(DeviceType::VULKAN, "vulkan", false));
#endif
        return out;
      },
      "Returns runtime accelerator availability and detected device indices.");

  m.def(
      "available_devices",
      []() {
        py::list devices;
        devices.append(py::cast(Device{DeviceType::CPU, 0}));

#ifdef MUNET_USE_CUDA
        for (int index = 0; index < 32; ++index) {
          try {
            Device d{DeviceType::CUDA, index};
            auto backend = BackendManager::get(d);
            backend->synchronize();
            devices.append(py::cast(d));
          } catch (...) {
            break;
          }
        }
#endif

#ifdef MUNET_USE_VULKAN
        for (int index = 0; index < 32; ++index) {
          try {
            Device d{DeviceType::VULKAN, index};
            auto backend = BackendManager::get(d);
            backend->synchronize();
            devices.append(py::cast(d));
          } catch (...) {
            break;
          }
        }
#endif
        return devices;
      },
      "Returns concrete available devices (CPU plus detected accelerator devices).");

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
          throw std::runtime_error(std::string("copy_from_numpy: ") +
                                   err.what());
        }
      },
      py::arg("tensor"), py::arg("input"),
      "Copies data from a NumPy array into the given CPU tensor.");

  py::class_<core::Module, std::shared_ptr<core::Module>>(m, "_CoreModule");
  py::class_<core::OffloadValidationReport>(m, "OffloadValidationReport")
      .def_readonly("valid", &core::OffloadValidationReport::valid)
      .def_readonly("errors", &core::OffloadValidationReport::errors)
      .def_readonly("warnings", &core::OffloadValidationReport::warnings)
      .def_readonly("estimated_boundaries",
                    &core::OffloadValidationReport::estimated_boundaries)
      .def_readonly("estimated_ping_pong_boundaries",
                    &core::OffloadValidationReport::estimated_ping_pong_boundaries);
  py::class_<core::OffloadTransferTelemetry>(m, "OffloadTransferTelemetry")
      .def_readonly("boundary_transfer_count",
                    &core::OffloadTransferTelemetry::boundary_transfer_count)
      .def_readonly("boundary_transfer_bytes",
                    &core::OffloadTransferTelemetry::boundary_transfer_bytes)
      .def_readonly("direction_counts",
                    &core::OffloadTransferTelemetry::direction_counts);

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
      .def(
          "named_modules",
          [](nn::Module &self, const std::string &prefix) {
            return self.named_modules_typed(prefix);
          },
          py::arg("prefix") = "",
          "Returns an iterator over all nn modules in the network.")
      .def("train", &nn::Module::train, py::arg("mode") = true,
           "Sets the module in training mode.")
      .def("eval", &nn::Module::eval, "Sets the module in evaluation mode.")
      .def("to",
           [](nn::Module &self, Device device) -> nn::Module& {
             self.to(device);
             return self;
           }, py::arg("device"),
           "Moves all parameters and buffers to the specified device. Returns self.")
      .def("to",
           [](nn::Module &self, DataType dtype) -> nn::Module& {
             self.to(dtype);
             return self;
          }, py::arg("dtype"),
          "Converts all parameters and buffers to the specified dtype. Returns self.")
      .def_property_readonly("is_training", &nn::Module::is_training,
           "Returns whether the module is in training mode.")
      .def("to_options",
           [](nn::Module &self, const TensorOptions &options) -> nn::Module& {
             self.to(options);
             return self;
           }, py::arg("options"),
           "Converts all parameters and buffers using explicit tensor options. Returns self.")
      .def(
          "offload",
          [](nn::Module &self, Device device, const std::vector<std::string> &layers)
              -> nn::Module & {
            self.offload(device, layers);
            return self;
          },
          py::arg("device"), py::arg("layers"),
          "Assigns listed module paths to a device and moves their params/buffers.")
      .def("clear_offload", &nn::Module::clear_offload,
           "Clears current model offload placement plan.")
      .def("freeze_offload_plan", &nn::Module::freeze_offload_plan,
           "Returns a persistable layer-path -> device-string plan.")
      .def("apply_offload_plan", &nn::Module::apply_offload_plan, py::arg("plan"),
           "Applies a previously frozen layer-path -> device-string plan.")
      .def(
          "offload_plan",
          [](nn::Module &self, bool explain) -> py::object {
            if (!explain) {
              return py::cast(self.offload_plan());
            }
            py::dict d;
            d["plan"] = py::cast(self.offload_plan());
            py::dict rationale;
            for (const auto &[layer, r] : self.offload_plan_rationale_typed()) {
              py::dict entry;
              entry["source"] = py::cast(r.source);
              entry["strategy"] = py::cast(r.strategy);
              entry["compute_cost"] = py::cast(r.compute_cost);
              entry["param_bytes"] = py::cast(r.param_bytes);
              entry["activation_bytes"] = py::cast(r.activation_bytes);
              entry["transfer_cost"] = py::cast(r.transfer_cost);
              entry["projected_mem_bytes"] = py::cast(r.projected_mem_bytes);
              if (r.budget_bytes.has_value()) {
                entry["budget_bytes"] = py::cast(r.budget_bytes.value());
              } else {
                entry["budget_bytes"] = py::none();
              }
              rationale[py::str(layer)] = std::move(entry);
            }
            d["rationale"] = std::move(rationale);
            return std::move(d);
          },
          py::arg("explain") = false,
          "Returns current module-path -> device placement mapping. If explain=True, returns planner rationale.")
      .def("auto_offload", &nn::Module::auto_offload, py::arg("devices"),
           py::arg("strategy") = "balanced", py::arg("sample_input"),
           py::arg("memory_budgets_bytes") = std::map<std::string, size_t>{},
           "Automatically generates and applies an offload plan.")
      .def("validate_offload_plan", &nn::Module::validate_offload_plan,
           py::arg("sample_input"),
           "Validates current offload plan and returns a typed report.")
      .def("set_offload_warnings", &nn::Module::set_offload_warnings,
           py::arg("enabled") = true,
           "Enables/disables runtime offload transfer warnings.")
      .def("set_offload_warning_threshold_bytes",
           &nn::Module::set_offload_warning_threshold_bytes,
           py::arg("threshold_bytes"),
           "Sets warning threshold for small offload transfers.")
      .def("offload_telemetry_snapshot", &nn::Module::offload_telemetry_snapshot,
           "Returns runtime offload transfer telemetry snapshot.")
      .def("reset_offload_telemetry", &nn::Module::reset_offload_telemetry,
           "Resets runtime offload transfer telemetry.")
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
      .def(py::init<int, int, int, int, int, TensorOptions>(),
           py::arg("in_channels"), py::arg("out_channels"),
           py::arg("kernel_size"), py::arg("stride") = 1,
           py::arg("padding") = 0, py::arg("options") = TensorOptions{})
      .def_readonly("stride", &nn::Conv2d::stride_)
      .def_readonly("padding", &nn::Conv2d::padding_)
      .def_readonly("weight", &nn::Conv2d::weight)
      .def_readonly("bias", &nn::Conv2d::bias);

  py::class_<nn::BatchNorm2d, nn::Module, std::shared_ptr<nn::BatchNorm2d>>(
      nn, "BatchNorm2d", "Applies Batch Normalization over a 4D input.")
      .def(py::init<int, float, float, TensorOptions>(),
           py::arg("num_features"), py::arg("eps") = 1e-5f,
           py::arg("momentum") = 0.1f, py::arg("options") = TensorOptions{})
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

  py::class_<nn::Softmax, nn::Module, std::shared_ptr<nn::Softmax>>(
     nn, "Softmax", "Applies the Softmax function over the specified dimension.")
     .def(py::init<int>(), py::arg("dim") = -1)
     .def_readonly("dim", &nn::Softmax::dim_);

  py::class_<nn::Tanh, nn::Module, std::shared_ptr<nn::Tanh>>(
      nn, "Tanh", "Applies the element-wise Tanh function.")
      .def(py::init<>());

  py::class_<nn::GELU, nn::Module, std::shared_ptr<nn::GELU>>(
      nn, "GELU", "Applies a fast GELU approximation: x * sigmoid(1.702*x).")
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
      "Embedding lookup for [B,T] token ids and projection for [B,T,V] "
      "one-hot/probability inputs.")
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

  py::class_<nn::RMSNorm, nn::Module, std::shared_ptr<nn::RMSNorm>>(
      nn, "RMSNorm",
      "Applies RMS normalization over the last tensor dimension.")
      .def(py::init<int, float, TensorOptions>(), py::arg("normalized_shape"),
           py::arg("eps") = 1e-5f, py::arg("options") = TensorOptions{})
      .def_readonly("normalized_shape", &nn::RMSNorm::normalized_shape_)
      .def_readonly("eps", &nn::RMSNorm::eps_)
      .def_readonly("weight", &nn::RMSNorm::weight);

  py::class_<nn::MultiHeadAttention, nn::Module,
             std::shared_ptr<nn::MultiHeadAttention>>(
      nn, "MultiHeadAttention",
      "Applies causal/non-causal multi-head self-attention over [B,T,E].")
      .def(py::init<int, int, bool, TensorOptions>(), py::arg("embed_dim"),
           py::arg("num_heads"), py::arg("causal") = true,
           py::arg("options") = TensorOptions{})
      .def_readonly("embed_dim", &nn::MultiHeadAttention::embed_dim_)
      .def_readonly("num_heads", &nn::MultiHeadAttention::num_heads_)
      .def_readonly("causal", &nn::MultiHeadAttention::causal_);

  py::class_<nn::GlobalAvgPool2d, nn::Module,
             std::shared_ptr<nn::GlobalAvgPool2d>>(
      nn, "GlobalAvgPool2d",
      "Applies global average pooling over spatial dimensions.")
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
      .def(py::init<>(), "Default constructor")
      .def("add", &nn::Sequential::add, py::arg("module"),
           "Appends a module to the sequence.")
      .def(py::init([](py::args args, py::kwargs kwargs) {
             auto seq = std::make_shared<nn::Sequential>();
             
             // Handle positional arguments: Sequential(layer1, layer2, ...)
             for (auto it : *args) {
               seq->add(it.cast<std::shared_ptr<nn::Module>>());
             }
             
             // Handle keyword argument: Sequential(layers=[...])
             if (kwargs.contains("layers")) {
               auto layers_list = kwargs["layers"].cast<std::vector<std::shared_ptr<nn::Module>>>();
               for (auto l : layers_list) {
                 seq->add(l);
               }
             }
             
             return seq;
           }),
           "Construct from layers: Sequential(layer1, layer2, ...) or Sequential(layers=[...])")
      .def(py::init([](const std::vector<std::shared_ptr<nn::Module>> &layers) {
             auto seq = std::make_shared<nn::Sequential>();
             for (auto l : layers)
               seq->add(l);
             return seq;
           }),
           py::arg("layers"),
           "Construct from a list of layers.")
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
      .def(py::init<inference::EngineConfig>(),
           py::arg("config") = inference::EngineConfig{})
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
           &inference::Engine::set_capture_profiler_memory, py::arg("enabled"))
      .def("capture_profiler_memory",
           &inference::Engine::capture_profiler_memory)
      .def("set_lean_mode", &inference::Engine::set_lean_mode,
           py::arg("enabled"))
      .def("lean_mode", &inference::Engine::lean_mode)
      .def("set_prepared_input_cache_entries",
           &inference::Engine::set_prepared_input_cache_entries,
           py::arg("entries"))
      .def("prepared_input_cache_entries_limit",
           &inference::Engine::prepared_input_cache_entries_limit)
      .def("set_prepared_input_cache_max_bytes",
           &inference::Engine::set_prepared_input_cache_max_bytes,
           py::arg("bytes"))
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
      .def(
          "compile",
          [](inference::Engine &self, const Tensor &example_input,
             const std::optional<std::vector<int>> &expected_input_shape,
             const std::optional<std::vector<int>> &expected_output_shape) {
            self.compile(example_input,
                         expected_input_shape.value_or(std::vector<int>{}),
                         expected_output_shape.value_or(std::vector<int>{}));
          },
          py::arg("example_input"),
          py::arg("expected_input_shape") = py::none(),
          py::arg("expected_output_shape") = py::none())
      .def("prepare", &inference::Engine::prepare, py::arg("example_input"))
      .def("prepare_batch", &inference::Engine::prepare_batch,
           py::arg("inputs"))
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
           "Clears the gradients of all optimized Tensors.")
      .def("grad_global_norm", &optim::Optimizer::grad_global_norm,
           "Computes the global L2 norm of all gradients.")
      .def("clip_grad_norm", &optim::Optimizer::clip_grad_norm,
           py::arg("max_norm"),
           "Clips gradients to the provided global L2 norm and returns the "
           "pre-clip norm.")
      .def("apply_weight_decay", &optim::Optimizer::apply_weight_decay,
           py::arg("weight_decay"),
           "Applies simple decoupled weight decay to all managed parameters.")
      .def_property("lr", &optim::Optimizer::lr, &optim::Optimizer::set_lr,
                    "Gets or sets the optimizer learning rate.");

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
  m.def(
      "dispatch_policy_snapshot", &ops::dispatch_policy_snapshot,
      "Returns the active dispatch fallback-rule matrix as a text snapshot.");
  m.def(
      "fallback_telemetry_snapshot",
      []() {
        const auto snapshot = ops::fallback_telemetry_snapshot();
        py::dict out;
        out["accelerator_cpu_fallback_total"] =
            py::int_(snapshot.accelerator_cpu_fallback_total);
        out["accelerator_cpu_fallback_counters"] =
            py::cast(snapshot.accelerator_cpu_fallback_counters);
        return out;
      },
      "Returns dispatch telemetry counters for accelerator->CPU fallbacks.");
  m.def(
      "reset_fallback_telemetry", &ops::reset_fallback_telemetry,
      "Clears dispatch telemetry counters for accelerator->CPU fallbacks.");
  m.def(
      "dispatch_decision_debug_dump",
      [](const std::string &op_name, const Tensor &tensor) {
        return ops::dispatch_decision_debug_dump(parse_op_id_name(op_name),
                                                 tensor);
      },
      py::arg("op_name"), py::arg("tensor"),
      "Returns a structured dispatch-decision line for the provided op name "
      "and tensor context.");

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
            m.DataType.BFloat16: "bfloat16",
            m.DataType.Int32: "int32",
            m.DataType.Int8: "int8"}[t.dtype]

def _dtype_from_name(name):
    m = __import__("munet")
    return {"float32": m.DataType.Float32,
            "float16": m.DataType.Float16,
            "bfloat16": m.DataType.BFloat16,
            "int32": m.DataType.Int32,
            "int8": m.DataType.Int8}[name]

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
SERIALIZATION_HYBRID_FORMAT_TAG = "munet_hybrid_v1"
SERIALIZATION_CHECKPOINT_ARTIFACT_KIND = "training_checkpoint"
SERIALIZATION_CHECKPOINT_ARTIFACT_SCOPE = "training+inference"
SERIALIZATION_CHECKPOINT_DEFAULT_LOAD_MODE = "train"
SERIALIZATION_CHECKPOINT_RECOMMENDED_LOADER = "load_checkpoint"
SERIALIZATION_CHECKPOINT_COMPILE_CONTRACT_POLICY = "dynamic"
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

def _maybe_source_for_class(cls):
    import inspect
    import textwrap
    try:
        src = inspect.getsource(cls)
        if isinstance(src, str) and src.strip():
            return textwrap.dedent(src)
        return None
    except Exception:
        return None

def _extract_tensors_and_shell(obj, path=""):
    import pickle
    import munet

    tensors = {}
    counter = [0]

    def extract(value, value_path):
        if isinstance(value, munet.Tensor):
            name = f"__tensor_{counter[0]}__"
            counter[0] += 1
            tensors[name] = _tensor_to_numpy(value)
            return {"__tensor_ref__": name, "__path__": value_path}

        if isinstance(value, dict):
            return {k: extract(v, f"{value_path}.{k}" if value_path else str(k)) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            seq = [extract(v, f"{value_path}[{i}]" if value_path else str(i)) for i, v in enumerate(value)]
            return tuple(seq) if isinstance(value, tuple) else seq

        if hasattr(value, "__dict__") and not isinstance(value, (type, type(lambda: None))):
            cls = type(value)
            shell = {
                "__class_module__": cls.__module__,
                "__class_qualname__": cls.__qualname__,
                "__class_name__": cls.__name__,
            }
            class_source = _maybe_source_for_class(cls)
            if class_source is not None:
                shell["__class_source__"] = class_source
            for k, v in value.__dict__.items():
                shell[k] = extract(v, f"{value_path}.{k}" if value_path else k)
            return shell

        if isinstance(value, (int, float, str, bool, type(None))):
            return value

        try:
            pickle.dumps(value)
            return value
        except Exception:
            return repr(value)

    return tensors, extract(obj, path)

def _resolve_class_from_shell(class_module, class_qualname, class_source=None):
    import importlib
    import types

    try:
        module = importlib.import_module(class_module)
        cls = module
        for part in class_qualname.split("."):
            cls = getattr(cls, part)
        return cls
    except Exception:
        pass

    if class_source is not None:
        dynamic_module_name = f"__munet_dynamic_{class_module.replace('.', '_')}__"
        dynamic_module = types.ModuleType(dynamic_module_name)
        namespace = dynamic_module.__dict__
        namespace["munet"] = __import__("munet")
        exec(class_source, namespace, namespace)
        cls = dynamic_module
        try:
            for part in class_qualname.split("."):
                cls = getattr(cls, part)
            return cls
        except AttributeError:
            leaf_name = class_qualname.split(".")[-1]
            if hasattr(dynamic_module, leaf_name):
                return getattr(dynamic_module, leaf_name)
            raise

    raise RuntimeError(
        f"Failed to resolve class {class_module}.{class_qualname}. "
        "No importable definition or embedded class source is available."
    )

def _rebuild_from_shell(shell, tensors, device=None):
    import munet

    def rebuild(value):
        if isinstance(value, dict):
            if "__tensor_ref__" in value:
                tensor_name = value["__tensor_ref__"]
                if tensor_name not in tensors:
                    raise ValueError(f"Missing tensor payload for {tensor_name}")
                t = munet.from_numpy(tensors[tensor_name])
                if device is not None:
                    t = t.to(device)
                return t

            if "__class_module__" in value and "__class_qualname__" in value:
                cls = _resolve_class_from_shell(
                    value["__class_module__"],
                    value["__class_qualname__"],
                    value.get("__class_source__"),
                )
                try:
                    instance = cls()
                except Exception:
                    instance = cls.__new__(cls)
                for k, v in value.items():
                    if k not in ("__class_module__", "__class_qualname__", "__class_name__", "__class_source__"):
                        setattr(instance, k, rebuild(v))
                return instance

            return {k: rebuild(v) for k, v in value.items()}

        if isinstance(value, list):
            return [rebuild(v) for v in value]
        if isinstance(value, tuple):
            return tuple(rebuild(v) for v in value)
        return value

    return rebuild(shell)

def _get_config(m):
    """Get config for built-in modules."""
    name = type(m).__name__
    if name == 'Sequential':
        return {'type': name, 'layers': [_get_config(child) for child in m]}
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
    elif name == 'RMSNorm':
        return {'type': name, 'normalized_shape': m.normalized_shape, 'eps': m.eps, 'dtype': _tensor_dtype_name(m.weight)}
    elif name == 'MultiHeadAttention':
        return {'type': name, 'embed_dim': m.embed_dim, 'num_heads': m.num_heads, 'causal': bool(m.causal), 'dtype': _tensor_dtype_name(m.q_proj.weight)}
    else:
        # For custom modules, return a marker that indicates we need to use hybrid format
        return None

def _get_config_with_custom(module, allow_custom, include_source):
    name = type(module).__name__
    if name == 'Sequential':
        return {'type': name, 'layers': [_get_config_with_custom(child, allow_custom, include_source) for child in module]}
    elif name == 'Linear':
        has_bias = hasattr(module, 'bias') and getattr(module, 'bias') is not None and getattr(module, 'bias').numel() > 0
        return {'type': name, 'in_features': module.weight.shape[0], 'out_features': module.weight.shape[1], 'bias': has_bias, 'dtype': _tensor_dtype_name(module.weight)}
    elif name == 'Conv2d':
        return {'type': name, 'in_channels': module.weight.shape[1], 'out_channels': module.weight.shape[0], 'kernel_size': module.weight.shape[2], 'stride': module.stride, 'padding': module.padding, 'dtype': _tensor_dtype_name(module.weight)}
    elif name == 'MaxPool2d':
        return {'type': name, 'kernel_size': module.kernel_size, 'stride': module.stride, 'padding': module.padding}
    elif name == 'BatchNorm2d':
        return {'type': name, 'num_features': module.weight.shape[0], 'eps': module.eps, 'momentum': module.momentum, 'dtype': _tensor_dtype_name(module.weight)}
    elif name == 'Upsample':
        return {'type': name, 'scale_factor': module.scale_factor}
    elif name == 'GlobalAvgPool2d':
        return {'type': name}
    elif name in ('ReLU', 'Sigmoid', 'Tanh', 'GELU', 'Flatten'):
        return {'type': name}
    elif name == 'LeakyReLU':
        return {'type': name, 'negative_slope': module.negative_slope}
    elif name == 'Dropout':
        return {'type': name, 'p': module.p}
    elif name == 'Embedding':
        return {'type': name, 'num_embeddings': module.num_embeddings, 'embedding_dim': module.embedding_dim, 'dtype': _tensor_dtype_name(module.weight)}
    elif name == 'LayerNorm':
        return {'type': name, 'normalized_shape': module.normalized_shape, 'eps': module.eps, 'dtype': _tensor_dtype_name(module.weight)}
    elif name == 'RMSNorm':
        return {'type': name, 'normalized_shape': module.normalized_shape, 'eps': module.eps, 'dtype': _tensor_dtype_name(module.weight)}
    elif name == 'MultiHeadAttention':
        return {'type': name, 'embed_dim': module.embed_dim, 'num_heads': module.num_heads, 'causal': bool(module.causal), 'dtype': _tensor_dtype_name(module.q_proj.weight)}

    if not allow_custom:
        raise ValueError(
            f"Custom module '{type(module).__qualname__}' is not supported in deploy artifacts. "
            "Use save_checkpoint(...) for custom classes."
        )

    cls = type(module)
    cfg = {'type': '__custom__', 'module': cls.__module__, 'qualname': cls.__qualname__}
    if include_source:
        source = _maybe_source_for_class(cls)
        if source is not None:
            cfg['source'] = source
    return cfg

def _build_module_from_config(cfg, *, trusted=False):
    import importlib
    import munet

    t = cfg['type']
    opts = _tensor_options_for_dtype(cfg.get('dtype', 'float32'))
    if t == 'Sequential':
        return munet.nn.Sequential(*[_build_module_from_config(c, trusted=trusted) for c in cfg['layers']])
    elif t == 'Linear':
        return munet.nn.Linear(cfg['in_features'], cfg['out_features'], cfg['bias'], opts)
    elif t == 'Conv2d':
        return munet.nn.Conv2d(cfg['in_channels'], cfg['out_channels'], cfg['kernel_size'], cfg['stride'], cfg['padding'], opts)
    elif t == 'MaxPool2d':
        return munet.nn.MaxPool2d(cfg['kernel_size'], cfg['stride'], cfg['padding'])
    elif t == 'BatchNorm2d':
        return munet.nn.BatchNorm2d(cfg['num_features'], cfg['eps'], cfg['momentum'], opts)
    elif t == 'Upsample':
        return munet.nn.Upsample(cfg['scale_factor'])
    elif t == 'GlobalAvgPool2d':
        return munet.nn.GlobalAvgPool2d()
    elif t == 'ReLU':
        return munet.nn.ReLU()
    elif t == 'Sigmoid':
        return munet.nn.Sigmoid()
    elif t == 'Tanh':
        return munet.nn.Tanh()
    elif t == 'GELU':
        return munet.nn.GELU()
    elif t == 'LeakyReLU':
        return munet.nn.LeakyReLU(cfg.get('negative_slope', 0.01))
    elif t == 'Dropout':
        return munet.nn.Dropout(cfg.get('p', 0.5))
    elif t == 'Embedding':
        return munet.nn.Embedding(cfg['num_embeddings'], cfg['embedding_dim'], opts)
    elif t == 'LayerNorm':
        return munet.nn.LayerNorm(cfg['normalized_shape'], cfg.get('eps', 1e-5), opts)
    elif t == 'RMSNorm':
        return munet.nn.RMSNorm(cfg['normalized_shape'], cfg.get('eps', 1e-5), opts)
    elif t == 'MultiHeadAttention':
        return munet.nn.MultiHeadAttention(cfg['embed_dim'], cfg['num_heads'], cfg.get('causal', True), opts)
    elif t == 'Flatten':
        return munet.nn.Flatten()
    elif t == '__custom__':
        module_path = cfg.get('module', '')
        class_qualname = cfg.get('qualname', '')
        class_source = cfg.get('source')
        if not module_path or not class_qualname:
            raise ValueError("Custom module saved without class reference. Use load_weights_checkpoint(...) for restore into an existing model.")
        try:
            mod = importlib.import_module(module_path)
            cls = mod
            for part in class_qualname.split('.'):
                cls = getattr(cls, part)
        except (ImportError, AttributeError) as e:
            if not trusted:
                raise ValueError(
                    f"Untrusted custom artifact cannot execute embedded source for '{class_qualname}'. "
                    "Re-run with trusted=True or use load_weights_checkpoint(...) into an existing model."
                ) from e
            if class_source is None:
                raise ValueError(
                    f"Could not reconstruct custom module '{class_qualname}' from module '{module_path}': {e}. "
                    "Use load_weights_checkpoint(...) with an in-code model definition."
                ) from e
            import types
            dynamic_module = types.ModuleType(f"__munet_dynamic_{module_path.replace('.', '_')}__")
            namespace = dynamic_module.__dict__
            namespace["munet"] = munet
            exec(class_source, namespace, namespace)
            leaf_name = class_qualname.split('.')[-1]
            if not hasattr(dynamic_module, leaf_name):
                raise ValueError(f"Embedded source did not define class '{leaf_name}'.")
            cls = getattr(dynamic_module, leaf_name)
        try:
            return cls()
        except Exception as ctor_err:
            raise ValueError(
                f"Custom module '{class_qualname}' must be default-constructible for full reconstruction: {ctor_err}. "
                "Use load_weights_checkpoint(...) for restore into existing model."
            )
    else:
        raise ValueError(f"Unsupported saved module type: {t}")

def _apply_state(module, state):
    for name, p in _iter_named_tensors(module):
        if name in state:
            _copy_numpy_into_tensor(p, state[name])
    return module

def _save_artifact(module, filename, *, artifact_kind):
    import json
    import numpy as np
    import pickle

    is_deploy = artifact_kind == SERIALIZATION_ARTIFACT_KIND
    config = _get_config_with_custom(module, allow_custom=not is_deploy, include_source=not is_deploy)
    use_hybrid_shell = (not is_deploy) and config.get('type') == '__custom__'

    state = {}
    for name, tensor in _iter_named_tensors(module):
        state[name] = _tensor_to_numpy(tensor)
    tensor_names = sorted(state.keys())

    state['__config__'] = np.array(json.dumps(config))
    state['__format_name__'] = np.array(SERIALIZATION_FORMAT_NAME)
    state['__format_revision__'] = np.array(SERIALIZATION_FORMAT_REVISION)
    state['__format_version__'] = np.array(SERIALIZATION_LEGACY_TAG)
    state['__producer__'] = np.array('munet')
    state['__contains_training_state__'] = np.array(False)
    state['__device_policy__'] = np.array(SERIALIZATION_DEVICE_POLICY)
    state['__dtype_policy__'] = np.array(SERIALIZATION_DTYPE_POLICY)
    state['__tensor_names__'] = np.array(json.dumps(tensor_names))

    if is_deploy:
        state['__artifact_kind__'] = np.array(SERIALIZATION_ARTIFACT_KIND)
        state['__artifact_scope__'] = np.array(SERIALIZATION_ARTIFACT_SCOPE)
        state['__default_load_mode__'] = np.array(SERIALIZATION_DEFAULT_LOAD_MODE)
        state['__recommended_loader__'] = np.array(SERIALIZATION_RECOMMENDED_LOADER)
        state['__compile_contract_policy__'] = np.array(SERIALIZATION_COMPILE_CONTRACT_POLICY)
    else:
        state['__artifact_kind__'] = np.array(SERIALIZATION_CHECKPOINT_ARTIFACT_KIND)
        state['__artifact_scope__'] = np.array(SERIALIZATION_CHECKPOINT_ARTIFACT_SCOPE)
        state['__default_load_mode__'] = np.array(SERIALIZATION_CHECKPOINT_DEFAULT_LOAD_MODE)
        state['__recommended_loader__'] = np.array(SERIALIZATION_CHECKPOINT_RECOMMENDED_LOADER)
        state['__compile_contract_policy__'] = np.array(SERIALIZATION_CHECKPOINT_COMPILE_CONTRACT_POLICY)

    if use_hybrid_shell:
        shell_tensors, shell = _extract_tensors_and_shell(module)
        state['__format__'] = np.array(SERIALIZATION_HYBRID_FORMAT_TAG)
        state['__shell__'] = np.frombuffer(pickle.dumps(shell, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8)
        for name, arr in shell_tensors.items():
            state[name] = arr

    np.savez(filename, **state)

def save_deploy(module, filename):
    """Save runtime/deploy artifact. Custom classes are not permitted."""
    _save_artifact(module, filename, artifact_kind=SERIALIZATION_ARTIFACT_KIND)

def save_checkpoint(module, filename):
    """Save training checkpoint artifact with optional hybrid class/source payload for custom classes."""
    _save_artifact(module, filename, artifact_kind=SERIALIZATION_CHECKPOINT_ARTIFACT_KIND)

def _validate_checkpoint_metadata(state):
    metadata = _serialization_metadata_from_state(state)
    if metadata["format_name"] != SERIALIZATION_FORMAT_NAME:
        raise ValueError(f"Unsupported serialization format name: {metadata['format_name']!r}.")
    if metadata["format_revision"] != SERIALIZATION_FORMAT_REVISION:
        raise ValueError(f"Unsupported serialization format revision: {metadata['format_revision']!r}.")
    if metadata["legacy_tag"] not in (None, SERIALIZATION_LEGACY_TAG):
        raise ValueError(f"Unsupported legacy serialization tag: {metadata['legacy_tag']!r}.")
    if metadata["artifact_kind"] not in (SERIALIZATION_CHECKPOINT_ARTIFACT_KIND, SERIALIZATION_ARTIFACT_KIND):
        raise ValueError(f"Unsupported checkpoint artifact kind: {metadata['artifact_kind']!r}.")
    return metadata

def _normalize_loaded_module_for_inference(module, device=None):
    if device is not None:
        module.to(device)
    module.eval()
    return module

def load_checkpoint(arg, filename=None, device=None, trusted=False):
    """
    Load checkpoint artifact.
    trusted=False forbids executing embedded source fallback for custom classes.
    """
    import json
    import numpy as np
    import pickle

    if filename is None:
        with np.load(arg, allow_pickle=True) as state:
            _validate_checkpoint_metadata(state)
            if '__config__' not in state:
                if '__shell__' in state:
                    if not trusted:
                        raise ValueError("Checkpoint contains shell/source payload; trusted=True is required for shell reconstruction.")
                    shell = pickle.loads(state['__shell__'].tobytes())
                    shell_tensors = {
                        name: state[name]
                        for name in state.files
                        if name.startswith('__tensor_') and name.endswith('__')
                    }
                    return _rebuild_from_shell(shell, shell_tensors, device=device)
                raise ValueError("File does not contain architecture config. Use load_weights_checkpoint(...) for weights-only restore.")
            config = json.loads(str(state['__config__']))
            module = _build_module_from_config(config, trusted=trusted)
            return _apply_state(module, state)
    else:
        module = arg
        with np.load(filename, allow_pickle=True) as state:
            _validate_checkpoint_metadata(state)
            return _apply_state(module, state)

def load_deploy(arg, filename=None, device=None):
    """
    Load deploy artifact only (strict runtime metadata).
    """
    import json
    import numpy as np

    if filename is None:
        with np.load(arg, allow_pickle=True) as state:
            _validate_serialization_metadata(state)
            if '__config__' not in state:
                raise ValueError("Deploy artifact missing __config__.")
            config = json.loads(str(state['__config__']))
            if config.get('type') == '__custom__':
                raise ValueError("Deploy artifacts do not support custom class reconstruction. Use load_checkpoint(..., trusted=True).")
            module = _build_module_from_config(config, trusted=False)
            return _apply_state(module, state)
    else:
        module = arg
        with np.load(filename, allow_pickle=True) as state:
            _validate_serialization_metadata(state)
            return _apply_state(module, state)

def load_for_inference(arg, filename=None, device=None):
    """Load a deploy artifact and normalize the result for inference execution."""
    module = load_deploy(arg, filename, device=device) if filename is not None else load_deploy(arg, device=device)
    return _normalize_loaded_module_for_inference(module, device)

def load_weights_checkpoint(module, filename):
    """Checkpoint weights-only restore into existing architecture."""
    return load_checkpoint(module, filename)

def load_weights_deploy(module, filename):
    """Deploy weights-only restore into existing architecture."""
    return load_deploy(module, filename)

def load_weights_for_inference(module, filename, device=None):
    """Deploy weights-only restore + eval normalization."""
    load_weights_deploy(module, filename)
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
