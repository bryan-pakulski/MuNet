#include "core.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py=pybind11;
using namespace munet;
PYBIND11_MODULE(_native,m) {
  m.def("vulkan_built",&vulkan_built);
  m.def("devices",&vulkan_devices);
  py::class_<Graph>(m,"Graph")
    .def(py::init<>())
    .def("leaf",&Graph::leaf,py::arg("kind"),py::arg("name"),py::arg("shape"),py::arg("data")=std::vector<float>{})
    .def("op",&Graph::op,py::arg("kind"),py::arg("inputs"),py::arg("attrs")=Shape{})
    .def("gradients",&Graph::gradients)
    .def("shape",[](const Graph& g,Id i){return g.at(i).shape;})
    .def("nodes",[](const Graph& g){py::list out;for(auto& n:g.nodes){py::dict d;d["op"]=n.op;d["name"]=n.name;d["inputs"]=n.inputs;d["shape"]=n.shape;d["attrs"]=n.attrs;d["data"]=n.data;out.append(d);}return out;});
  py::class_<Plan>(m,"Plan")
    .def(py::init<const Graph&,const std::vector<Id>&,const std::vector<std::pair<Id,Id>>&,bool>(),py::arg("graph"),py::arg("outputs"),py::arg("updates"),py::arg("fuse")=true)
    .def_readonly("inputs",&Plan::inputs)
    .def_readonly("outputs",&Plan::outputs)
    .def_readonly("updates",&Plan::updates)
    .def_property_readonly("graph",[](const Plan& p){return p.graph;})
    .def("shaders",[](const Plan& p){std::vector<std::string> out;for(auto& k:p.kernels)out.push_back(k.source);return out;})
    .def("enable_vulkan",&Plan::enable_vulkan,py::call_guard<py::gil_scoped_release>())
    .def("run",[](Plan& p,const std::vector<py::array_t<float,py::array::c_style>>& feeds){
      std::vector<std::vector<float>> values;for(const auto& a:feeds)values.emplace_back(a.data(),a.data()+a.size());
      py::gil_scoped_release release;p.run(values);
    })
    .def("read",[](Plan& p,Id id){std::vector<float> data;{py::gil_scoped_release release;data=p.read(id);}
      py::array_t<float> result(p.graph.at(id).shape);std::copy(data.begin(),data.end(),result.mutable_data());return result;})
    .def("write",[](Plan& p,Id id,py::array_t<float,py::array::c_style> a){std::vector<float> v(a.data(),a.data()+a.size());py::gil_scoped_release release;p.write(id,v);})
    .def("synchronize",&Plan::synchronize,py::call_guard<py::gil_scoped_release>())
    .def("stats",&Plan::stats)
    .def("device_name",&Plan::device_name);
}
