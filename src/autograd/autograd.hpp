#pragma once
#include "../tensor.hpp"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace munet {

struct Edge {
  std::shared_ptr<Node> node;
  int input_nr;
};

struct Node {
  virtual ~Node() = default;
  virtual std::string name() const { return "Node"; }

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
      var->backend().add(*var->grad->storage, *grads[0].impl_->storage,
                         *var->grad->storage, numel(var->shape));
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
    std::vector<Node *> bfs_queue = {root_node};
    std::unordered_set<Node *> seen = {root_node};

    size_t head = 0;
    while (head < bfs_queue.size()) {
      Node *task = bfs_queue[head++];
      for (const auto &edge : task->next_edges) {
        if (edge.node) {
          dependencies[edge.node.get()]++;
          if (seen.insert(edge.node.get()).second) {
            bfs_queue.push_back(edge.node.get());
          }
        }
      }
    }

    queue.push_back(root_node);
    grad_buffer[root_node].push_back(root_grad);

    int processed = 0;
    while (processed < queue.size()) {
      Node *task = queue[processed++];

      auto &inputs = grad_buffer[task];
      if (inputs.empty())
        continue;

      Tensor accumulated_grad = inputs[0];
      for (size_t i = 1; i < inputs.size(); ++i) {
        Tensor sum_t(accumulated_grad.shape(), accumulated_grad.device(),
                     accumulated_grad.dtype());
        accumulated_grad.impl_->backend().add(
            *accumulated_grad.impl_->storage, *inputs[i].impl_->storage,
            *sum_t.impl_->storage, accumulated_grad.size());
        accumulated_grad = sum_t;
      }

      auto input_grads = task->apply({accumulated_grad});

      for (size_t i = 0; i < input_grads.size(); ++i) {
        if (i >= task->next_edges.size())
          break;
        auto &edge = task->next_edges[i];
        if (edge.node) {
          grad_buffer[edge.node.get()].push_back(input_grads[i]);
          if (--dependencies[edge.node.get()] == 0) {
            queue.push_back(edge.node.get());
          }
        }
      }
      grad_buffer.erase(task);
    }
  }
};

} // namespace munet
