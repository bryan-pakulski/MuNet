#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/tensor.hpp"
#include "core/device.hpp"

namespace py = pybind11;

namespace munet {
namespace serialization {

// ============================================================================
// Hybrid Serialization Format: Pickle + Binary Tensors
// ============================================================================
// 
// Format:
//   [8 bytes]  Magic: "MUNETPKG"
//   [8 bytes]  Version (uint64_t, little-endian)
//   [8 bytes]  Pickle size (uint64_t, little-endian)
//   [N bytes]  Pickled Python state (with tensor placeholders)
//   [8 bytes]  Tensor count (uint64_t, little-endian)
//   For each tensor:
//     [8 bytes]  Name length (uint64_t)
//     [N bytes]  Name (UTF-8)
//     [8 bytes]  Dims count (uint64_t)
//     [dims_count * 8 bytes] Dims (int64_t each)
//     [8 bytes]  Dtype (uint64_t, cast from DataType enum)
//     [8 bytes]  Data size (uint64_t)
//     [N bytes]  Raw tensor data
//
// This format allows:
//   - Saving custom Python modules without registration (like PyTorch)
//   - Efficient binary tensor storage (C++ backend)
//   - Cross-device tensor restoration
//   - Backward compatibility with NPZ format
// ============================================================================

inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        default: return 4;
    }
}

// Extracted tensor info
struct ExtractedTensor {
    std::string path;      // e.g., "layers.0.weight"
    Tensor* tensor;
};

// Result of extracting tensors from a Python object
struct ExtractionResult {
    py::object placeholder_state;  // Python state with tensor placeholders
    std::vector<ExtractedTensor> tensors;
};

// Recursively extract tensors and replace with placeholders
inline ExtractionResult extract_tensors_with_placeholders(py::object obj, const std::string& prefix = "") {
    ExtractionResult result;
    
    // If obj is a Tensor, mark it and record
    try {
        if (py::isinstance<Tensor>(obj)) {
            ExtractedTensor et;
            et.path = prefix.empty() ? "tensor" : prefix;
            et.tensor = obj.cast<Tensor&>().ptr();
            result.tensors.push_back(et);
            // Return a placeholder dict
            py::dict placeholder;
            placeholder["__munet_tensor_placeholder__"] = py::cast(true);
            placeholder["__path__"] = et.path;
            placeholder["__shape__"] = py::cast(et.tensor->shape());
            placeholder["__dtype__"] = py::cast(static_cast<int>(et.tensor->dtype()));
            result.placeholder_state = placeholder;
            return result;
        }
    } catch (...) {
        // Not a tensor, continue
    }
    
    // If obj is a dict, recurse
    if (py::isinstance<py::dict>(obj)) {
        py::dict new_dict;
        for (auto item : py::cast<py::dict>(obj)) {
            auto key = item.first;
            std::string key_str;
            try {
                if (py::isinstance<py::str>(key)) {
                    key_str = py::cast<std::string>(key);
                }
            } catch (...) {}
            
            auto sub_result = extract_tensors_with_placeholders(
                item.second, 
                prefix.empty() ? key_str : prefix + "." + key_str
            );
            new_dict[key] = sub_result.placeholder_state;
            for (auto& t : sub_result.tensors) {
                result.tensors.push_back(t);
            }
        }
        result.placeholder_state = new_dict;
        return result;
    }
    
    // If obj is a list, recurse
    if (py::isinstance<py::list>(obj)) {
        py::list new_list;
        py::list old_list = py::cast<py::list>(obj);
        for (size_t i = 0; i < old_list.size(); ++i) {
            auto sub_result = extract_tensors_with_placeholders(
                old_list[i],
                prefix + "[" + std::to_string(i) + "]"
            );
            new_list.append(sub_result.placeholder_state);
            for (auto& t : sub_result.tensors) {
                result.tensors.push_back(t);
            }
        }
        result.placeholder_state = new_list;
        return result;
    }

    // If obj is a tuple, recurse
    if (py::isinstance<py::tuple>(obj)) {
        py::list new_list;
        py::tuple old_tuple = py::cast<py::tuple>(obj);
        for (size_t i = 0; i < old_tuple.size(); ++i) {
            auto sub_result = extract_tensors_with_placeholders(
                old_tuple[i],
                prefix + "[" + std::to_string(i) + "]"
            );
            new_list.append(sub_result.placeholder_state);
            for (auto& t : sub_result.tensors) {
                result.tensors.push_back(t);
            }
        }
        result.placeholder_state = py::tuple(new_list);
        return result;
    }

    // For other types, return as-is (they can be pickled directly)
    result.placeholder_state = obj;
    return result;
}

// Write a tensor to binary stream
inline void write_tensor_binary(std::ostream& out, const std::string& name, Tensor* t) {
    // Write name length and name
    uint64_t name_len = name.size();
    out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
    out.write(name.data(), name_len);
    
    // Write shape
    uint64_t dims_count = t->shape().size();
    out.write(reinterpret_cast<const char*>(&dims_count), sizeof(dims_count));
    for (int64_t dim : t->shape()) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    
    // Write dtype
    uint64_t dtype = static_cast<uint64_t>(t->dtype());
    out.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
    
    // Write data size and data
    uint64_t data_size = t->numel() * dtype_size(t->dtype());
    out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    
    // Copy tensor data to CPU if needed
    Tensor* cpu_tensor = t;
    std::unique_ptr<Tensor> cpu_copy;
    if (t->device().type != DeviceType::CPU) {
        cpu_copy = std::make_unique<Tensor>(t->shape(), t->dtype(), Device(DeviceType::CPU, 0));
        cpu_copy->copy_from(*t);
        cpu_tensor = cpu_copy.get();
    }
    
    out.write(reinterpret_cast<const char*>(cpu_tensor->data_ptr()), data_size);
}

// Read tensor result
struct ReadTensorResult {
    std::string name;
    Tensor tensor;
};

// Read a tensor from binary stream
inline ReadTensorResult read_tensor_binary(std::istream& in) {
    ReadTensorResult result;
    
    // Read name
    uint64_t name_len;
    in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
    result.name.resize(name_len);
    in.read(&result.name[0], name_len);
    
    // Read shape
    uint64_t dims_count;
    in.read(reinterpret_cast<char*>(&dims_count), sizeof(dims_count));
    std::vector<int64_t> shape(dims_count);
    for (size_t i = 0; i < dims_count; ++i) {
        in.read(reinterpret_cast<char*>(&shape[i]), sizeof(shape[i]));
    }
    
    // Read dtype
    uint64_t dtype;
    in.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
    
    // Read data
    uint64_t data_size;
    in.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    
    result.tensor = Tensor(shape, static_cast<DataType>(dtype), Device(DeviceType::CPU, 0));
    in.read(reinterpret_cast<char*>(result.tensor.data_ptr()), data_size);
    
    return result;
}

// Restore tensors into placeholder state recursively
inline py::object restore_tensors_to_state(
    py::object state,
    std::vector<ReadTensorResult>& tensors,
    size_t& tensor_idx
) {
    // If this is a placeholder, restore the tensor
    if (py::isinstance<py::dict>(state)) {
        py::dict d = py::cast<py::dict>(state);
        if (d.contains("__munet_tensor_placeholder__")) {
            if (tensor_idx >= tensors.size()) {
                throw std::runtime_error("Tensor index out of bounds during restoration");
            }
            // Return a copy of the tensor (on CPU for now)
            return py::cast(Tensor(tensors[tensor_idx++].tensor));
        }
        
        // Otherwise, recurse into dict
        py::dict new_dict;
        for (auto item : d) {
            new_dict[item.first] = restore_tensors_to_state(item.second, tensors, tensor_idx);
        }
        return new_dict;
    }
    
    // If list, recurse
    if (py::isinstance<py::list>(state)) {
        py::list new_list;
        for (auto item : py::cast<py::list>(state)) {
            new_list.append(restore_tensors_to_state(item, tensors, tensor_idx));
        }
        return new_list;
    }
    
    // If tuple, recurse
    if (py::isinstance<py::tuple>(state)) {
        py::list new_list;
        for (auto item : py::cast<py::tuple>(state)) {
            new_list.append(restore_tensors_to_state(item, tensors, tensor_idx));
        }
        return py::tuple(new_list);
    }
    
    // Otherwise return as-is
    return state;
}

// Save model with hybrid pickle + binary tensors format
inline void save_model_hybrid(py::object model, const std::string& filename) {
    // Extract tensors and create placeholder state
    ExtractionResult extracted = extract_tensors_with_placeholders(model);
    
    // Create state dict for pickling
    py::dict state;
    state["__munet_format__"] = "hybrid_v1";
    state["model_state"] = extracted.placeholder_state;
    state["tensor_count"] = py::cast(extracted.tensors.size());
    
    // Pickle the state
    py::module pickle = py::module::import("pickle");
    py::bytes pickled = pickle.attr("dumps")(state, py::arg("protocol") = 4);
    std::string pickled_str = pickled;
    
    // Write to file
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Write header
    out.write("MUNETPKG", 8);
    uint64_t version = 1;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write pickle
    uint64_t pickle_size = pickled_str.size();
    out.write(reinterpret_cast<const char*>(&pickle_size), sizeof(pickle_size));
    out.write(pickled_str.data(), pickle_size);
    
    // Write tensors
    uint64_t tensor_count = extracted.tensors.size();
    out.write(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));
    
    for (auto& et : extracted.tensors) {
        write_tensor_binary(out, et.path, et.tensor);
    }
    
    out.close();
}

// Load model from hybrid format
// If existing_model is provided (not None), load weights into it
// Otherwise, reconstruct the full model from the pickled state
inline py::object load_model_hybrid(const std::string& filename, py::object existing_model = py::none()) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    // Read and verify header
    char magic[8];
    in.read(magic, 8);
    if (std::string(magic, 8) != "MUNETPKG") {
        throw std::runtime_error("Invalid file format: expected MUNETPKG header");
    }
    
    uint64_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw std::runtime_error("Unsupported MUNETPKG version: " + std::to_string(version));
    }
    
    // Read pickle
    uint64_t pickle_size;
    in.read(reinterpret_cast<char*>(&pickle_size), sizeof(pickle_size));
    std::string pickled_str(pickle_size, '\0');
    in.read(&pickled_str[0], pickle_size);
    
    // Unpickle state
    py::module pickle = py::module::import("pickle");
    py::bytes pickled_bytes(pickled_str);
    py::dict state = pickle.attr("loads")(pickled_bytes);
    
    // Verify format
    if (!state.contains("__munet_format__")) {
        throw std::runtime_error("Invalid state format: missing __munet_format__");
    }
    std::string format = py::cast<std::string>(state["__munet_format__"]);
    if (format != "hybrid_v1") {
        throw std::runtime_error("Unsupported state format: " + format);
    }
    
    // Read tensors
    uint64_t tensor_count;
    in.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
    std::vector<ReadTensorResult> tensors;
    for (size_t i = 0; i < tensor_count; ++i) {
        tensors.push_back(read_tensor_binary(in));
    }
    
    in.close();
    
    // Get model state
    py::object model_state = state["model_state"];
    
    if (existing_model.is_none()) {
        // Full model reconstruction: restore tensors into the pickled state
        // The model_state should be the model object itself (or a reconstruction dict)
        size_t tensor_idx = 0;
        py::object restored = restore_tensors_to_state(model_state, tensors, tensor_idx);
        return restored;
    } else {
        // Weights-only: load tensors into existing model
        // Build a map from name to tensor
        std::unordered_map<std::string, Tensor*> name_to_tensor;
        
        // Collect named parameters/buffers from existing model
        if (py::hasattr(existing_model, "named_parameters")) {
            py::dict params = existing_model.attr("named_parameters")();
            for (auto item : params) {
                std::string name = py::cast<std::string>(item.first);
                name_to_tensor[name] = item.second.cast<Tensor&>().ptr();
            }
        }
        if (py::hasattr(existing_model, "named_buffers")) {
            py::dict buffers = existing_model.attr("named_buffers")();
            for (auto item : buffers) {
                std::string name = py::cast<std::string>(item.first);
                name_to_tensor[name] = item.second.cast<Tensor&>().ptr();
            }
        }
        
        // Copy loaded tensors into model
        for (auto& rt : tensors) {
            auto it = name_to_tensor.find(rt.name);
            if (it != name_to_tensor.end()) {
                it->second->copy_from(rt.tensor);
            }
        }
        
        return existing_model;
    }
}

// Detect file format (hybrid vs legacy NPZ)
inline bool is_hybrid_format(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        return false;
    }
    char magic[8];
    in.read(magic, 8);
    in.close();
    return std::string(magic, 8) == "MUNETPKG";
}

} // namespace serialization
} // namespace munet