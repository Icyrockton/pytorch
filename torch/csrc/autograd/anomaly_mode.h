#pragma once

#include <torch/csrc/Export.h>
#include <memory>
#include <string>

namespace torch {
namespace autograd {

// forward declaration of Node from function.h
struct Node;

struct TORCH_API AnomalyMode {
  static bool is_enabled() {
    return _enabled;
  }
  static void set_enabled(bool enabled) {
    _enabled = enabled;
  }

 private:
  static bool _enabled;
};

/// A RAII guard that enables Anomaly Detection Mode.
///
/// Anomaly detection mode is useful for debugging problems happening
/// in the backward, such as unexpectedly modified tensors or NaNs
/// occuring in the backward.
/// 异常检测模式对于调试向后发生的问题很有用，比如向后发生的意外修改张量或nan。
///
/// The enabling of anomaly mode is global - as soon as there is one
/// such guard, it is enabled for all computation and threads. It also
/// comes with a significant performance penalty.
/// 异常模式的启用是全局的——只要有一个这样的保护，它就会对所有的计算和线程启用。它还带来了严重的性能损失。
///
/// Example:
/// @code
/// auto x = torch::tensor({1.}, torch::requires_grad());
/// {
///   torch::autograd::DetectAnomalyGuard detect_anomaly;
///   auto x = torch::tensor({5.0}, torch::requires_grad());
///   auto y = x * x;
///   auto z = y * y;
///   y += 1;
///   z.backward();
/// }
/// @endcode
class TORCH_API DetectAnomalyGuard {
 public:
  DetectAnomalyGuard();
  ~DetectAnomalyGuard();
};

struct TORCH_API AnomalyMetadata {
  virtual ~AnomalyMetadata();
  virtual void store_stack();
  virtual void print_stack(const std::string& current_node_name);
  virtual void assign_parent(const std::shared_ptr<Node>& parent_node);

 private:
  std::string traceback_;
  std::shared_ptr<Node> parent_;
};

} // namespace autograd
} // namespace torch
