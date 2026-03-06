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

  py::enum_<DeviceType>(m, "DeviceType")
      .value("CPU", DeviceType::CPU)
      .value("CUDA", DeviceType::CUDA)
      .value("VULKAN", DeviceType::VULKAN)
      .export_values();

  py::enum_<DataType>(m, "DataType")
      .value("Float32", DataType::Float32)
      .value("Float16", DataType::Float16)
      .value("Int32", DataType::Int32)
      .export_values();

  py::class_<Device>(m, "Device")
      .def(py::init<DeviceType, int>(), py::arg("type") = DeviceType::CPU,
           py::arg("index") = 0)
      .def("__repr__", &Device::to_string)
      .def_readwrite("type", &Device::type)
      .def_readwrite("index", &Device::index);

  // --- Factory Functions ---
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
      py::arg("shape"), py::arg("device") = py::none(),
      py::arg("requires_grad") = false);

  m.def(
      "ones",
      [](Shape shape, std::optional<Device> device, bool requires_grad) {
        Device dev = device.value_or(Device{DeviceType::CPU, 0});
        Tensor t(shape, dev, DataType::Float32, requires_grad);
        t.uniform_(1.0f, 1.0f);
        return t;
      },
      py::arg("shape"), py::arg("device") = py::none(),
      py::arg("requires_grad") = false);

  m.def(
      "rand",
      [](Shape shape, std::optional<Device> device, bool requires_grad) {
        Device dev = device.value_or(Device{DeviceType::CPU, 0});
        Tensor t(shape, dev, DataType::Float32, requires_grad);
        t.uniform_(0.0f, 1.0f);
        return t;
      },
      py::arg("shape"), py::arg("device") = py::none(),
      py::arg("requires_grad") = false);

  py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
      .def(py::init<Shape, Device, DataType, bool>(), py::arg("shape"),
           py::arg("device") = Device{DeviceType::CPU, 0},
           py::arg("dtype") = DataType::Float32,
           py::arg("requires_grad") = false)
      .def_property(
          "name",
          [](const Tensor &t) { return t.impl_ ? t.name() : "uninitialized"; },
          [](Tensor &t, const std::string &n) {
            if (t.impl_)
              t.set_name(n);
          })
      .def_property_readonly(
          "shape",
          [](const Tensor &t) { return t.impl_ ? t.shape() : Shape{}; })
      .def_property_readonly("device",
                             [](const Tensor &t) {
                               return t.impl_ ? t.device()
                                              : Device{DeviceType::UNKNOWN, 0};
                             })
      .def_property_readonly("dtype",
                             [](const Tensor &t) {
                               return t.impl_ ? t.dtype() : DataType::Float32;
                             })
      .def("__len__",
           [](const Tensor &t) {
             return (!t.impl_ || t.shape().empty()) ? 0 : t.shape()[0];
           })
      .def("numel", [](const Tensor &t) { return t.impl_ ? t.size() : 0; })
      .def("__repr__",
           [](const Tensor &t) {
             if (!t.impl_)
               return std::string("Tensor(uninitialized)");
             return "Tensor(shape=" + to_string(t.shape()) + ", device='" +
                    t.device().to_string() + "'" +
                    (t.requires_grad() ? ", requires_grad=True)" : ")");
           })
      .def_property(
          "requires_grad",
          [](const Tensor &t) { return t.impl_ ? t.requires_grad() : false; },
          [](Tensor &t, bool req) {
            if (t.impl_)
              t.set_requires_grad(req);
          })
      .def_property_readonly("grad", &Tensor::grad)
      .def("zero_grad", &Tensor::zero_grad)
      .def("copy_from_numpy",
           [](Tensor &t, py::array_t<float> input) {
             if (t.device().type != DeviceType::CPU) {
               throw std::runtime_error(
                   "copy_from_numpy: Target tensor must be on CPU.");
             }
             py::buffer_info buf = input.request();
             size_t bytes = buf.size * sizeof(float);
             if (bytes != t.bytes()) {
               throw std::runtime_error(
                   "copy_from_numpy: Size mismatch. Tensor bytes: " +
                   std::to_string(t.bytes()) +
                   ", Numpy bytes: " + std::to_string(bytes));
             }
             std::memcpy(t.data(), buf.ptr, bytes);
           })

      .def("to", &Tensor::to, py::arg("device"))

      // Expose both overloads to Python
      .def("backward", [](Tensor &t) { t.backward(); })
      .def(
          "backward", [](Tensor &t, const Tensor &grad) { t.backward(grad); },
          py::arg("grad"))
      .def("__add__", [](const Tensor &a, const Tensor &b) { return a + b; })
      .def("__sub__", [](const Tensor &a, const Tensor &b) { return a - b; })
      .def("__mul__", [](const Tensor &a, const Tensor &b) { return a * b; })
      .def("__matmul__",
           [](const Tensor &a, const Tensor &b) { return a.matmul(b); })
      .def("relu", &Tensor::relu)
      .def("sigmoid", &Tensor::sigmoid)
      .def("softmax", &Tensor::softmax)
      .def("sum", &Tensor::sum)
      .def("reshape", &Tensor::reshape, py::arg("shape"))
      .def("conv2d", &Tensor::conv2d, py::arg("weight"),
           py::arg("bias") = Tensor(), py::arg("stride") = 1,
           py::arg("padding") = 0)
      .def("max_pool2d", &Tensor::max_pool2d, py::arg("kernel_size"),
           py::arg("stride"), py::arg("padding") = 0)
      .def("upsample2d", &Tensor::upsample2d, py::arg("scale_factor"))
      .def("batch_norm", &Tensor::batch_norm, py::arg("running_mean"),
           py::arg("running_var"), py::arg("weight"), py::arg("bias"),
           py::arg("training"), py::arg("momentum") = 0.1,
           py::arg("eps") = 1e-5)
      .def("mse_loss", &Tensor::mse_loss, py::arg("target"))
      .def("cross_entropy", &Tensor::cross_entropy, py::arg("target"))
      .def("uniform_", &Tensor::uniform_, py::arg("low") = -1.0f,
           py::arg("high") = 1.0f)
      .def("step", &Tensor::step, py::arg("lr"))
      .def("replace_",
           [](Tensor &self, const Tensor &other) { self.impl_ = other.impl_; })

      .def_buffer([](Tensor &t) -> py::buffer_info {
        // Safety Check: Prevent NumPy from segfaulting by accessing GPU memory
        // directly
        if (t.device().type != DeviceType::CPU) {
          throw std::runtime_error(
              "Cannot convert GPU tensor to NumPy array directly. Call "
              "`.to(Device(DeviceType.CPU))` first.");
        }

        std::vector<py::ssize_t> strides(t.shape().size());
        py::ssize_t stride = dtype_size(t.dtype());
        for (int i = (int)t.shape().size() - 1; i >= 0; --i) {
          strides[i] = stride;
          stride *= t.shape()[i];
        }

        return py::buffer_info(t.data(), dtype_size(t.dtype()),
                               py::format_descriptor<float>::format(),
                               t.shape().size(), t.shape(), strides);
      });

  // --- Tensor Helpers ---
  m.def("copy_from_numpy", [](Tensor &t, py::array_t<float> input) {
    if (t.device().type != DeviceType::CPU) {
      throw std::runtime_error(
          "copy_from_numpy: Target tensor must be on CPU.");
    }
    py::buffer_info buf = input.request();
    size_t bytes = buf.size * sizeof(float);
    if (bytes != t.bytes()) {
      throw std::runtime_error(
          "copy_from_numpy: Size mismatch. Tensor bytes: " +
          std::to_string(t.bytes()) +
          ", Numpy bytes: " + std::to_string(bytes));
    }
    std::memcpy(t.data(), buf.ptr, bytes);
  });

  m.def("from_numpy", [](py::array_t<float> input) {
    py::buffer_info buf = input.request();
    std::vector<int> shape;
    for (auto s : buf.shape)
      shape.push_back((int)s);
    Tensor t(shape, Device{DeviceType::CPU, 0});
    std::memcpy(t.data(), buf.ptr, t.bytes());
    return t;
  });

  // --- NN Submodule ---
  auto nn = m.def_submodule("nn", "Neural Network Layers");

  py::class_<nn::Module, std::shared_ptr<nn::Module>, PyModule>(nn, "Module")
      .def(py::init<>())
      .def("forward", &nn::Module::forward)
      .def("parameters", &nn::Module::parameters)
      .def("named_parameters", &nn::Module::named_parameters,
           py::arg("prefix") = "")
      .def("named_modules", &nn::Module::named_modules, py::arg("prefix") = "")
      .def("train", &nn::Module::train, py::arg("mode") = true)
      .def("eval", &nn::Module::eval)
      .def("to", &nn::Module::to)
      .def("zero_grad", &nn::Module::zero_grad)
      .def("__call__", &nn::Module::forward)
      .def("__setattr__", [](py::object self, const std::string &name,
                             py::object value) {
        auto &mod = self.cast<nn::Module &>();

        // 1. If value is a Module, auto-register it
        if (py::isinstance<nn::Module>(value)) {
          mod.register_module(name, value.cast<std::shared_ptr<nn::Module>>());
        }
        // 2. If value is a Tensor, auto-register as parameter or buffer
        else if (py::isinstance<Tensor>(value)) {
          auto &t = value.cast<Tensor &>();
          if (t.requires_grad()) {
            mod.register_parameter(name, t);
          } else {
            mod.register_buffer(name, t);
          }
        }

        // 3. Always perform the standard attribute assignment
        // This ensures the attribute is actually available on the Python object
        auto dict = self.attr("__dict__");
        dict[py::cast(name)] = value;
      });

  py::class_<nn::Linear, nn::Module, std::shared_ptr<nn::Linear>>(nn, "Linear")
      .def(py::init<int, int, bool>(), py::arg("in_features"),
           py::arg("out_features"), py::arg("bias") = true)
      .def_readonly("weight", &nn::Linear::weight)
      .def_readonly("bias", &nn::Linear::bias);

  py::class_<nn::Conv2d, nn::Module, std::shared_ptr<nn::Conv2d>>(nn, "Conv2d")
      .def(py::init<int, int, int, int, int>(), py::arg("in_channels"),
           py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0)
      .def_readonly("stride", &nn::Conv2d::stride_)
      .def_readonly("padding", &nn::Conv2d::padding_)
      .def_readonly("weight", &nn::Conv2d::weight);

  py::class_<nn::BatchNorm2d, nn::Module, std::shared_ptr<nn::BatchNorm2d>>(
      nn, "BatchNorm2d")
      .def(py::init<int, float, float>(), py::arg("num_features"),
           py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f)
      .def_readonly("eps", &nn::BatchNorm2d::eps_)
      .def_readonly("momentum", &nn::BatchNorm2d::momentum_)
      .def_readonly("weight", &nn::BatchNorm2d::weight);

  py::class_<nn::Flatten, nn::Module, std::shared_ptr<nn::Flatten>>(nn,
                                                                    "Flatten")
      .def(py::init<>());

  py::class_<nn::ReLU, nn::Module, std::shared_ptr<nn::ReLU>>(nn, "ReLU")
      .def(py::init<>());

  py::class_<nn::Sigmoid, nn::Module, std::shared_ptr<nn::Sigmoid>>(nn,
                                                                    "Sigmoid")
      .def(py::init<>());

  py::class_<nn::MaxPool2d, nn::Module, std::shared_ptr<nn::MaxPool2d>>(
      nn, "MaxPool2d")
      .def(py::init<int, int, int>(), py::arg("kernel_size"),
           py::arg("stride") = 2, py::arg("padding") = 0)
      .def_readonly("kernel_size", &nn::MaxPool2d::k_)
      .def_readonly("stride", &nn::MaxPool2d::s_)
      .def_readonly("padding", &nn::MaxPool2d::p_);

  py::class_<nn::Upsample, nn::Module, std::shared_ptr<nn::Upsample>>(
      nn, "Upsample")
      .def(py::init<int>(), py::arg("scale_factor"))
      .def_readonly("scale_factor", &nn::Upsample::scale_);

  py::class_<nn::Sequential, nn::Module, std::shared_ptr<nn::Sequential>>(
      nn, "Sequential")
      .def(py::init<>())
      .def("add", &nn::Sequential::add)
      .def(py::init([](const std::vector<std::shared_ptr<nn::Module>> &layers) {
        auto seq = std::make_shared<nn::Sequential>();
        for (auto l : layers)
          seq->add(l);
        return seq;
      }))
      .def(
          "__iter__",
          [](nn::Sequential &s) {
            return py::make_iterator(s.ordered_modules_.begin(),
                                     s.ordered_modules_.end());
          },
          py::keep_alive<0, 1>());

  // --- Optim Submodule ---
  auto optim = m.def_submodule("optim", "Optimizers");

  py::class_<optim::Optimizer, std::shared_ptr<optim::Optimizer>>(optim,
                                                                  "Optimizer")
      .def("step", &optim::Optimizer::step)
      .def("zero_grad", &optim::Optimizer::zero_grad);

  py::class_<optim::SGD, optim::Optimizer, std::shared_ptr<optim::SGD>>(optim,
                                                                        "SGD")
      .def(py::init<std::vector<Tensor>, float>(), py::arg("params"),
           py::arg("lr"));

  m.def("print_profiler_stats", []() { Profiler::get().print_summary(); });
  m.def(
      "reset_profiler", []() { Profiler::get().reset(); },
      "Clears all collected performance statistics and resets peak memory "
      "tracking.");

  py::exec(
      R"(
def save(module, filename):
    import numpy as np
    import json

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
        elif name in ('ReLU', 'Sigmoid', 'Flatten'):
            return {'type': name}
        else:
            raise ValueError(f"Unknown module type {name}")

    config = get_config(module)
    state = {name: np.array(p, copy=False) for name, p in module.named_parameters().items()}
    state['__config__'] = np.array(json.dumps(config))
    np.savez(filename, **state)

def load(arg, filename=None):
    import numpy as np
    import json
    import munet

    if filename is None:
        state = np.load(arg, allow_pickle=True)
        if '__config__' not in state:
            raise ValueError("File does not contain architecture config. Load into a module using `load(module, filename)` instead.")

        config = json.loads(str(state['__config__']))

        def build_module(cfg):
            t = cfg['type']
            if t == 'Sequential': return munet.nn.Sequential([build_module(c) for c in cfg['layers']])
            elif t == 'Linear': return munet.nn.Linear(cfg['in_features'], cfg['out_features'], cfg['bias'])
            elif t == 'Conv2d': return munet.nn.Conv2d(cfg['in_channels'], cfg['out_channels'], cfg['kernel_size'], cfg['stride'], cfg['padding'])
            elif t == 'MaxPool2d': return munet.nn.MaxPool2d(cfg['kernel_size'], cfg['stride'], cfg['padding'])
            elif t == 'BatchNorm2d': return munet.nn.BatchNorm2d(cfg['num_features'], cfg['eps'], cfg['momentum'])
            elif t == 'Upsample': return munet.nn.Upsample(cfg['scale_factor'])
            elif t == 'ReLU': return munet.nn.ReLU()
            elif t == 'Sigmoid': return munet.nn.Sigmoid()
            elif t == 'Flatten': return munet.nn.Flatten()

        module = build_module(config)
        for name, p in module.named_parameters().items():
            if name in state: p.copy_from_numpy(state[name])
        return module
    else:
        module = arg
        state = np.load(filename, allow_pickle=True)
        for name, p in module.named_parameters().items():
            if name in state: p.copy_from_numpy(state[name])
        return module
)",
      py::globals(), m.attr("__dict__"));
}
