#include "ops.hpp"
#include "tensor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
using namespace munet;

PYBIND11_MODULE(munet, m) {
  m.doc() = "MuNet: C++ Machine Learning Framework";

  py::enum_<DeviceType>(m, "DeviceType")
      .value("CPU", DeviceType::CPU)
      .value("CUDA", DeviceType::CUDA)
      .value("VULKAN", DeviceType::VULKAN)
      .value("METAL", DeviceType::METAL)
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

  py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
      .def(py::init<Shape, Device, DataType, bool>(), py::arg("shape"),
           py::arg("device") = Device{DeviceType::CPU, 0},
           py::arg("dtype") = DataType::Float32,
           py::arg("requires_grad") = false)
      .def_property("name", &Tensor::name, &Tensor::set_name)
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("device", &Tensor::device)
      .def_property_readonly("dtype", &Tensor::dtype)
      .def_property("requires_grad", &Tensor::requires_grad,
                    &Tensor::set_requires_grad)

      .def_property_readonly("grad", &Tensor::grad)
      .def("zero_grad", &Tensor::zero_grad)

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
      .def("sum", &Tensor::sum)
      .def("reshape", &Tensor::reshape, py::arg("shape"))
      .def("conv2d", &Tensor::conv2d, py::arg("weight"),
           py::arg("bias") = Tensor(), py::arg("stride") = 1,
           py::arg("padding") = 0)
      .def("max_pool2d", &Tensor::max_pool2d, py::arg("kernel_size"),
           py::arg("stride"), py::arg("padding") = 0)
      .def("upsample2d", &Tensor::upsample2d, py::arg("scale_factor"))
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
}
