#include "core.hpp"
#include "inference.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py=pybind11;
using namespace munet;
PYBIND11_MODULE(_native,m) {
  m.def("vulkan_built",&vulkan_built);
  m.def("devices",&vulkan_devices);
  // Internal test/interop bridge. Applications use the installed C++ Model API directly.
  py::class_<Model>(m,"InferenceModel")
    .def(py::init([](const std::string& path,const std::string& device,uint64_t limit){
      py::gil_scoped_release release;return std::make_unique<Model>(path,ModelOptions{device,limit});
    }),py::arg("path"),py::arg("device")="cpu",py::arg("max_memory_bytes")=uint64_t{2}*1024*1024*1024)
    .def("inputs",[](const Model& model){py::list out;for(const auto& s:model.inputs()){py::dict d;d["name"]=s.name;d["shape"]=s.shape;out.append(d);}return out;})
    .def("outputs",[](const Model& model){py::list out;for(const auto& s:model.outputs()){py::dict d;d["name"]=s.name;d["shape"]=s.shape;out.append(d);}return out;})
    .def("run",[](Model& model,const std::vector<py::array_t<float,py::array::c_style>>& inputs){
      std::vector<munet::Tensor> values;
      for(const auto& a:inputs)values.push_back({Shape(a.shape(),a.shape()+a.ndim()),std::vector<float>(a.data(),a.data()+a.size())});
      std::vector<munet::Tensor> outputs;{py::gil_scoped_release release;outputs=model.run(values);}
      py::list result;for(const auto& value:outputs){py::array_t<float> a(value.shape);std::copy(value.data.begin(),value.data.end(),a.mutable_data());result.append(a);}return result;
    })
    .def("stats",&Model::stats)
    .def("device_name",&Model::device_name);
  py::class_<Graph>(m,"Graph")
    .def(py::init<>())
    .def("leaf",&Graph::leaf,py::arg("kind"),py::arg("name"),py::arg("shape"),py::arg("data")=std::vector<float>{})
    .def("leaf_array",[](Graph& g,const std::string& kind,const std::string& name,py::array_t<float,py::array::c_style|py::array::forcecast> a){Shape shape(a.shape(),a.shape()+a.ndim());return g.leaf(kind,name,shape,std::vector<float>(a.data(),a.data()+a.size()));})
    .def_property_readonly("size",[](const Graph& g){return g.nodes.size();})
    .def("name",[](const Graph& g,Id i){return g.at(i).name;})
    .def("op",&Graph::op,py::arg("kind"),py::arg("inputs"),py::arg("attrs")=Shape{})
    .def("gradients",&Graph::gradients)
    .def("shape",[](const Graph& g,Id i){return g.at(i).shape;})
    .def("nodes",[](const Graph& g,bool data){py::list out;for(auto& n:g.nodes){py::dict d;d["op"]=n.op;d["name"]=n.name;d["inputs"]=n.inputs;d["shape"]=n.shape;d["attrs"]=n.attrs;if(data)d["data"]=n.data;else d["data"]=py::list();out.append(d);}return out;},py::arg("data")=true);
  py::class_<Plan>(m,"Plan")
    .def(py::init<const Graph&,const std::vector<Id>&,const std::vector<std::pair<Id,Id>>&,bool>(),py::arg("graph"),py::arg("outputs"),py::arg("updates"),py::arg("fuse")=true)
    .def_readonly("inputs",&Plan::inputs)
    .def_readonly("outputs",&Plan::outputs)
    .def_readonly("updates",&Plan::updates)
    .def_property_readonly("graph",[](Plan& p)->Graph&{return p.graph;},py::return_value_policy::reference_internal)
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
