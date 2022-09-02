#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include <c10/util/hash.h>

namespace torch {
namespace autograd {

struct Node;

/// Represents a particular input of a function.    表示函数的特定输入。
/// Edge将所有的Node链接起来, Edge都通过函数 gradient_edge 创建
struct Edge {
  Edge() noexcept : function(nullptr), input_nr(0) {}

  Edge(std::shared_ptr<Node> function_, uint32_t input_nr_) noexcept
      : function(std::move(function_)), input_nr(input_nr_) {}

  /// 测试一条Edge是否有效的方便方法
  bool is_valid() const noexcept {
    return function != nullptr;
  }

  // Required for use in associative containers.
  bool operator==(const Edge& other) const noexcept {
    return this->function == other.function && this->input_nr == other.input_nr;
  }

  bool operator!=(const Edge& other) const noexcept {
    return !(*this == other);
  }

  /// The function this `Edge` points to.
  std::shared_ptr<Node> function;

  /// The identifier of a particular input to the function.
  /// forward function中 output输出的索引序号         , backward之后 flip名称 output->input
  uint32_t input_nr;
};
} // namespace autograd
} // namespace torch

// The idiomatic way of enabling use of a custom type as the key of hash
// containers in C++11. This method removes the requirement of having to pass
// a custom hasher to std::unordered_{map, set}.
// See http://en.cppreference.com/w/cpp/utility/hash for more information.
namespace std {
template <>
struct hash<torch::autograd::Edge> {
  // These type aliases are required by the standard.
  using argument_type = torch::autograd::Edge;
  using return_type = size_t;
  return_type operator()(const argument_type& edge) const noexcept {
    return c10::get_hash(edge.function, edge.input_nr);
  }
};
} // namespace std
