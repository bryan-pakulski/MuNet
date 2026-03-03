#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "munet.hpp"

namespace py = pybind11;
using namespace munet;

PYBIND11_MODULE(munet, m) {
    m.doc() = "MuNet: A lightweight C++ AI framework with Python bindings";

    // 1. Register Enums FIRST so they can be used as default arguments
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();

    // 2. Register Tensor
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>, Device>(), py::arg("shape"), py::arg("device") = Device::CPU)
        .def("to_cpu", &Tensor::to_cpu)
        .def("to_gpu", &Tensor::to_gpu)
        .def("shape", &Tensor::shape)
        .def("numpy", [](Tensor& t) {
            if (t.device_ == Device::CUDA) t.to_cpu();
            return py::array_t<float>(t.shape(), static_cast<float*>(t.data()));
        })
        .def_static("from_numpy", [](py::array_t<float> input) {
            py::buffer_info buf = input.request();
            std::vector<int> shape;
            for (auto s : buf.shape) shape.push_back(s);
            Tensor t(shape, Device::CPU);
            std::memcpy(t.data(), buf.ptr, t.bytes());
            return t;
        })
        .def("copy_from_numpy", [](Tensor& t, py::array_t<float> input) {
            py::buffer_info buf = input.request();
            size_t bytes = buf.size * sizeof(float);
            if (bytes != t.bytes()) throw std::runtime_error("Size mismatch in copy_from_numpy");
            t.copy_from(buf.ptr, bytes);
        });


    // 3. Register Layers
    py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer");
    
    py::class_<Linear, Layer, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int>());
        
    py::class_<ReLU, Layer, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>());
        
    py::class_<Softmax, Layer, std::shared_ptr<Softmax>>(m, "Softmax")
        .def(py::init<>());

    py::class_<Flatten, Layer, std::shared_ptr<Flatten>>(m, "Flatten")
        .def(py::init<>());

    py::class_<Conv2D, Layer, std::shared_ptr<Conv2D>>(m, "Conv2D")
        .def(py::init<int, int, int, int, int>(), 
             py::arg("in_channels"), py::arg("out_channels"), 
             py::arg("kernel_size"), py::arg("stride") = 1, py::arg("padding") = 0);

    py::class_<MaxPool2D, Layer, std::shared_ptr<MaxPool2D>>(m, "MaxPool2D")
        .def(py::init<int, int>(), py::arg("kernel_size"), py::arg("stride"));

    py::class_<Dropout, Layer, std::shared_ptr<Dropout>>(m, "Dropout")
        .def(py::init<float>(), py::arg("p") = 0.5f);

    // 4. Register Model
    py::class_<Model>(m, "Model")
        .def(py::init<>())
        .def("add", &Model::add)
        .def("forward", &Model::forward)
        .def("backward", &Model::backward)
        // FIX: Use reference policy. The Tensors are owned by the C++ Layers, not Python.
        .def("parameters", &Model::parameters, py::return_value_policy::reference) 
        .def("train", &Model::train)
        .def("eval", &Model::eval)
        .def("save_weights", &Model::save_weights)
        .def("load_weights", &Model::load_weights);

    // 5. Register Optimizer
    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<Tensor*>, float>(), py::arg("params"), py::arg("lr"))
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad);

    // 6. Loss Helper
    m.def("cross_entropy_loss", [](const Tensor& logits, const Tensor& targets) {
        // Output gradient will live on same device as logits
        auto grad_out = std::make_shared<Tensor>(logits.shape(), logits.device_);
        
        // No more explicit to_cpu() calls here!
        // The C++ function now handles dispatching to CUDA kernel or CPU loop.
        
        // Auto-handle device mismatch:
        // If logits are on GPU but targets are on CPU (common case),
        // temporarily move targets to GPU for the calculation.
        if (logits.device_ != targets.device_) {
            Tensor targets_tmp = targets.clone();
            if (logits.device_ == Device::CUDA) targets_tmp.to_gpu();
            else targets_tmp.to_cpu();
            float loss = cross_entropy_loss(logits, targets_tmp, *grad_out);
            return std::make_pair(loss, grad_out);
        }

        float loss = cross_entropy_loss(logits, targets, *grad_out);
        return std::make_pair(loss, grad_out);

    });
}
