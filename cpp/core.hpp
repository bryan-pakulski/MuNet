#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>

namespace munet {
using Shape = std::vector<int64_t>;
using Id = int;
size_t numel(const Shape& shape);
struct Node {
  std::string op, name;
  std::vector<Id> inputs;
  Shape shape, attrs;
  std::vector<float> data;
};
class Graph {
 public:
  std::vector<Node> nodes;
  Id leaf(const std::string& kind, const std::string& name, const Shape& shape,
          const std::vector<float>& data = {});
  Id op(const std::string& kind, const std::vector<Id>& inputs, const Shape& attrs = {});
  std::vector<Id> gradients(Id loss, const std::vector<Id>& wrt);
  const Node& at(Id id) const;
 private:
  Id scalar(float v);
  Id sum_to(Id value, const Shape& shape);
};
struct Kernel { Id root; size_t count; std::string source; };
struct Counters {
  uint64_t runs=0, submissions=0, dispatches=0, upload_bytes=0, download_bytes=0, waits=0;
};
class DeviceExecutor {
 public:
  virtual ~DeviceExecutor() = default;
  virtual void run(const std::vector<std::pair<size_t,std::vector<float>>>& feeds)=0;
  virtual std::vector<float> read(size_t offset,size_t count)=0;
  virtual void write(size_t offset,const std::vector<float>& data)=0;
  virtual void synchronize()=0;
  virtual std::string name() const=0;
};
std::unique_ptr<DeviceExecutor> make_vulkan(
    const std::vector<float>& initial,const std::vector<Kernel>& kernels,
    const std::vector<std::vector<uint32_t>>& spirv,
    const std::vector<std::pair<size_t,size_t>>& input_ranges,
    const std::vector<std::pair<size_t,std::pair<size_t,size_t>>>& updates,
    unsigned device_index,Counters& counters);
bool vulkan_built();
std::vector<std::string> vulkan_devices();
std::map<std::string,uint64_t> vulkan_device_limits(unsigned index);

class Plan {
 public:
  Plan(const Graph& graph,const std::vector<Id>& outputs,
       const std::vector<std::pair<Id,Id>>& updates,bool fuse=true);
  ~Plan();
  void enable_vulkan(const std::vector<std::vector<uint32_t>>& spirv,unsigned device_index);
  void run(const std::vector<std::vector<float>>& feeds);
  std::vector<float> read(Id id);
  void write(Id id,const std::vector<float>& data);
  void synchronize();
  std::map<std::string,uint64_t> stats() const;
  std::string device_name() const;
  Graph graph;
  std::vector<Id> inputs,outputs;
  std::vector<std::pair<Id,Id>> updates;
  std::vector<Kernel> kernels;
 private:
  std::vector<float> arena_;
  std::vector<size_t> offsets_;
  std::vector<bool> inline_, live_;
  size_t unfused_nodes_=0, naive_floats_=0, arena_floats_=0;
  Counters counters_;
  std::unique_ptr<DeviceExecutor> device_;
  float read_value(Id id,size_t index) const;
  float calculate(Id id,size_t index) const;
  std::string expr(Id id,const std::string& index,bool force=false) const;
  std::string shader(Id root) const;
};
}
