#pragma once
#include "../tensor.hpp"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace munet {

class GradMode {
public:
  static bool is_enabled() { return enabled_; }
  static void set_enabled(bool enabled) { enabled_ = enabled; }

private:
  inline static thread_local bool enabled_ = true;
};

struct Edge {
  std::shared_ptr<Node> node;
  int input_nr;
};

struct Node {
  virtual ~Node() = default;

  virtual std::string name() const { return "Node"; }

  // receives gradients from outputs
  // returns gradients for inputs
  virtual std::vector<Tensor> apply(const std::vector<Tensor> &grads) = 0;

  std::vector<Edge> next_edges;
};

struct AccumulateGrad : public Node {
  std::weak_ptr<TensorImpl> variable_;

  explicit AccumulateGrad(std::shared_ptr<TensorImpl> var) : variable_(var) {}

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    if (grads.empty())
      return {};

    if (auto var = variable_.lock()) {

      if (!var->grad) {
        var->grad = std::make_shared<TensorImpl>(
            var->shape, var->storage->device(), var->storage->dtype(), false);

        var->grad->storage->zero_();
      }
      auto info = compute_broadcast(var->grad->shape, var->grad->strides,
                                    grads[0].shape(), grads[0].strides());
      var->backend().add(*var->grad->storage, *grads[0].impl_->storage,
                         *var->grad->storage, info);
    }

    return {};
  }

  std::string name() const override { return "AccumulateGrad"; }
};

class Engine {
public:
  static void execute(Node *root_node, Tensor root_grad) {

    std::unordered_map<Node *, int> dependencies;
    std::unordered_map<Node *, std::vector<Tensor>> grad_buffer;

    std::vector<Node *> queue;

    // ------------------------------------------------------------------
    // Build dependency counts (BFS over graph)
    // ------------------------------------------------------------------

    std::vector<Node *> bfs_queue = {root_node};
    std::unordered_set<Node *> seen = {root_node};

    size_t head = 0;

    while (head < bfs_queue.size()) {
      Node *task = bfs_queue[head++];

      for (auto &edge : task->next_edges) {

        if (!edge.node)
          continue;

        Node *child = edge.node.get();

        dependencies[child]++;

        if (seen.insert(child).second)
          bfs_queue.push_back(child);
      }
    }

    dependencies[root_node] = 0;

    // ------------------------------------------------------------------
    // Seed root gradient
    // ------------------------------------------------------------------

    queue.push_back(root_node);
    grad_buffer[root_node].push_back(root_grad);

    // ------------------------------------------------------------------
    // Backward execution
    // ------------------------------------------------------------------

    size_t processed = 0;

    while (processed < queue.size()) {

      Node *task = queue[processed++];

      auto it = grad_buffer.find(task);
      if (it == grad_buffer.end())
        continue;

      auto &inputs = it->second;
      if (inputs.empty())
        continue;

      Tensor accumulated_grad =
          inputs.size() == 1 ? inputs[0] : inputs[0].clone();

      for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].impl_) {
          auto info = compute_broadcast(accumulated_grad.shape(),
                                        accumulated_grad.strides(),
                                        inputs[i].shape(), inputs[i].strides());
          accumulated_grad.impl_->backend().add(
              *accumulated_grad.impl_->storage, *inputs[i].impl_->storage,
              *accumulated_grad.impl_->storage, info);
        }
      }

      if (!accumulated_grad.impl_) {
        grad_buffer.erase(task);
        continue;
      }

      auto output_grads = task->apply({accumulated_grad});

      for (size_t i = 0; i < task->next_edges.size(); ++i) {

        auto &edge = task->next_edges[i];
        if (!edge.node)
          continue;

        Node *parent = edge.node.get();

        if (i < output_grads.size() && output_grads[i].impl_) {

          grad_buffer[parent].push_back(output_grads[i]);

          if (--dependencies[parent] == 0)
            queue.push_back(parent);
        }
      }

      grad_buffer.erase(task);
    }
  }
};

} // namespace munet
