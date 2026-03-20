#pragma once

#include "../core/grad_mode.hpp"
#include "../tensor.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace munet {

struct Node;

struct Edge {
  std::shared_ptr<Node> node;
  int input_nr = 0;
  std::string label;
};

struct SavedTensor {
  std::weak_ptr<TensorImpl> weak_impl;
  std::shared_ptr<Storage> storage;
  Shape shape;
  Strides strides;
  size_t storage_offset;
  uint64_t version = 0;
  std::string debug_name;

  SavedTensor(const Tensor &t, std::string name = "")
      : weak_impl(t.impl_), storage(t.impl_ ? t.impl_->storage : nullptr),
        shape(t.impl_ ? t.shape() : Shape{}),
        strides(t.impl_ ? t.strides() : Strides{}),
        storage_offset(t.impl_ ? t.storage_offset() : 0),
        version(t.impl_ ? t.version() : 0), debug_name(std::move(name)) {}

  Tensor unpack() const {
    if (storage == nullptr)
      return Tensor();

    if (auto impl = weak_impl.lock()) {
      if (impl->version_counter != version) {
        const std::string name =
            debug_name.empty() ? "saved tensor" : debug_name;
        throw std::runtime_error("In-place mutation detected for " + name +
                                 " after it was captured for backward");
      }
    }

    // Create a new implementation wrapper that shares the same storage.
    // This breaks circular shared_ptr references to the original TensorImpl.
    Tensor restored;
    restored.impl_ =
        std::make_shared<TensorImpl>(storage, shape, strides, storage_offset);
    return restored;
  }
};

struct AutogradExtension {
  virtual ~AutogradExtension() = default;
  virtual void on_graph_start() {}
  virtual void on_node_ready(const Node &, const Tensor &) {}
  virtual Tensor process_gradient(const Node &, const Tensor &grad) {
    return grad;
  }
  virtual void on_node_complete(const Node &, const std::vector<Tensor> &) {}
  virtual void on_graph_complete() {}
};

struct Node {
  using GradientHook = Tensor::GradientHook;

  virtual ~Node() = default;

  virtual std::string name() const { return "Node"; }
  virtual std::vector<Tensor> apply(const std::vector<Tensor> &grads) = 0;
  virtual void release_resources() {
    saved_tensors.clear();
    next_edges.clear();
    gradient_hooks.clear();
  }

  void save_tensor(const Tensor &tensor, const std::string &debug_name = "") {
    if (!tensor.impl_) {
      return;
    }
    saved_tensors.emplace_back(tensor, debug_name);
  }

  Tensor saved_tensor(size_t index) const {
    if (index >= saved_tensors.size()) {
      throw std::out_of_range("Saved tensor index out of range");
    }
    return saved_tensors[index].unpack();
  }

  void register_gradient_hook(GradientHook hook) {
    gradient_hooks.push_back(std::move(hook));
  }

  Tensor apply_gradient_hooks(const Tensor &grad) const {
    Tensor current = grad;
    for (const auto &hook : gradient_hooks) {
      if (hook) {
        current = hook(current);
      }
    }
    return current;
  }

  void mark_released() {
    graph_released = true;
    release_resources();
  }

  void ensure_graph_available() const {
    if (graph_released) {
      throw std::runtime_error(
          "Autograd graph has been released. Pass retain_graph=true to "
          "backward() if you need to run it more than once.");
    }
  }

  std::vector<Edge> next_edges;
  std::vector<SavedTensor> saved_tensors;
  std::vector<GradientHook> gradient_hooks;
  bool graph_released = false;
};

struct AccumulateGrad : public Node {
  std::weak_ptr<TensorImpl> variable_;

  explicit AccumulateGrad(std::shared_ptr<TensorImpl> var) : variable_(var) {}

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    if (grads.empty()) {
      return {};
    }

    if (auto var = variable_.lock()) {
      NoGradGuard guard; // Crucial: ensure gradient math doesn't track history
      if (!var->grad) {
        var->grad = std::make_shared<TensorImpl>(
            var->shape, var->storage->device(), var->storage->dtype(), false);
        var->grad->storage->zero_();
      }

      Tensor current_grad_wrap;
      current_grad_wrap.impl_ = var->grad;
      // Use detached addition to be doubly sure
      Tensor updated = current_grad_wrap + grads[0].detach();
      var->grad = updated.impl_;
      var->grad->bump_version();
    }

    return {};
  }

  std::string name() const override { return "AccumulateGrad"; }
};

struct BackwardRequest {
  std::shared_ptr<Node> root_node;
  Tensor root_grad;
  bool retain_graph = false;
  bool create_graph = false;
  std::vector<std::shared_ptr<AutogradExtension>> extensions;
};

class GradientBufferPool {
public:
  std::vector<Tensor> acquire() {
    if (free_lists_.empty()) {
      return {};
    }
    std::vector<Tensor> result = std::move(free_lists_.back());
    free_lists_.pop_back();
    return result;
  }

  void release(std::vector<Tensor> &&buffer) {
    buffer.clear();
    free_lists_.push_back(std::move(buffer));
  }

private:
  std::vector<std::vector<Tensor>> free_lists_;
};

class ExecutionEngine {
public:
  void execute(const BackwardRequest &request) {
    if (!request.root_node) {
      throw std::runtime_error(
          "backward() requires a valid autograd root node");
    }
    if (request.create_graph) {
      throw std::runtime_error(
          "create_graph is reserved for higher-order gradients but is not "
          "implemented yet");
    }
    request.root_node->ensure_graph_available();

    GraphState state;
    build_graph(request.root_node, state);
    state.ready_queue.push_back(request.root_node.get());
    state.grad_buffers[request.root_node.get()] = pool_.acquire();
    state.grad_buffers[request.root_node.get()].push_back(request.root_grad);

    for (const auto &extension : request.extensions) {
      if (extension) {
        extension->on_graph_start();
      }
    }

    size_t processed = 0;
    while (processed < state.ready_queue.size()) {
      Node *task = state.ready_queue[processed++];
      auto buffer_it = state.grad_buffers.find(task);
      if (buffer_it == state.grad_buffers.end()) {
        continue;
      }

      auto node_inputs = std::move(buffer_it->second);
      state.grad_buffers.erase(buffer_it);
      if (node_inputs.empty()) {
        pool_.release(std::move(node_inputs));
        continue;
      }

      task->ensure_graph_available();
      Tensor accumulated_grad = accumulate_gradients(node_inputs);
      pool_.release(std::move(node_inputs));

      if (!accumulated_grad.impl_) {
        continue;
      }

      accumulated_grad = task->apply_gradient_hooks(accumulated_grad);
      for (const auto &extension : request.extensions) {
        if (extension) {
          extension->on_node_ready(*task, accumulated_grad);
          accumulated_grad =
              extension->process_gradient(*task, accumulated_grad);
        }
      }

      auto output_grads = task->apply({accumulated_grad});
      for (const auto &extension : request.extensions) {
        if (extension) {
          extension->on_node_complete(*task, output_grads);
        }
      }

      schedule_next_edges(*task, output_grads, state);
    }

    if (!request.retain_graph) {
      release_graph(state.graph_nodes);
    }

    for (const auto &extension : request.extensions) {
      if (extension) {
        extension->on_graph_complete();
      }
    }
  }

private:
  struct GraphState {
    std::unordered_map<Node *, int> dependencies;
    std::unordered_map<Node *, std::vector<Tensor>> grad_buffers;
    std::vector<Node *> ready_queue;
    std::vector<std::shared_ptr<Node>> graph_nodes;
  };

  void build_graph(const std::shared_ptr<Node> &root, GraphState &state) {
    std::vector<std::shared_ptr<Node>> bfs_queue = {root};
    std::unordered_set<Node *> seen = {root.get()};
    state.graph_nodes.push_back(root);

    size_t head = 0;
    while (head < bfs_queue.size()) {
      auto task = bfs_queue[head++];
      task->ensure_graph_available();

      for (const auto &edge : task->next_edges) {
        if (!edge.node) {
          continue;
        }

        Node *child = edge.node.get();
        state.dependencies[child]++;
        if (seen.insert(child).second) {
          bfs_queue.push_back(edge.node);
          state.graph_nodes.push_back(edge.node);
        }
      }
    }

    state.dependencies[root.get()] = 0;
  }

  Tensor accumulate_gradients(const std::vector<Tensor> &inputs) {
    if (inputs.empty())
      return Tensor();
    if (inputs.size() == 1)
      return inputs[0];

    NoGradGuard guard; // Ensure accumulation doesn't create new graph nodes
    Tensor accumulated_grad = inputs.front().clone().detach();

    for (size_t i = 1; i < inputs.size(); ++i) {
      if (inputs[i].impl_) {
        accumulated_grad = accumulated_grad + inputs[i].detach();
      }
    }
    return accumulated_grad;
  }

  void schedule_next_edges(const Node &task,
                           const std::vector<Tensor> &output_grads,
                           GraphState &state) {
    for (size_t i = 0; i < task.next_edges.size(); ++i) {
      const auto &edge = task.next_edges[i];
      if (!edge.node) {
        continue;
      }

      Node *parent = edge.node.get();
      if (i < output_grads.size() && output_grads[i].impl_) {
        auto grad_buffer_it = state.grad_buffers.find(parent);
        if (grad_buffer_it == state.grad_buffers.end()) {
          grad_buffer_it =
              state.grad_buffers.emplace(parent, pool_.acquire()).first;
        }
        grad_buffer_it->second.push_back(output_grads[i]);

        if (--state.dependencies[parent] == 0) {
          state.ready_queue.push_back(parent);
        }
      }
    }
  }

  void release_graph(const std::vector<std::shared_ptr<Node>> &graph_nodes) {
    for (const auto &node : graph_nodes) {
      if (node) {
        node->mark_released();
      }
    }
  }

  GradientBufferPool pool_;
};

class Engine {
public:
  static ExecutionEngine &get_default() {
    static thread_local ExecutionEngine engine;
    return engine;
  }
};

} // namespace munet
