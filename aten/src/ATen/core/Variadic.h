#pragma once

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <c10/util/ArrayRef.h>
#include <ATen/core/List.h>

namespace at {

// This class allows you to write variadic functions which
// call a (possibly overloaded) function on each argument,
// in order.  This is most commonly used in autogenerated code,
// where it is convenient to have a function that can uniformly
// take arguments of different types.  If your arguments
// are homogenous consider using a std::initializer_list instead.
//
// 这个类允许您编写可变参数函数，按顺序对每个参数调用一个(可能是重载的)函数。
// 这通常在自动生成代码中使用，在自动生成代码中，有一个函数可以统一地接受不同类型的参数，这很方便。
// 如果参数是同质的，请考虑使用std::initializer_list。
//
// For examples of this in use, see torch/csrc/utils/variadic.h
template <typename F>
struct IterArgs {
  template <typename... Args>
  inline F& apply() {
    return self();
  }

  // NB: Use perfect forwarding here, otherwise we'll make value
  // copies of all arguments!
  template <typename T, typename... Args>
  inline F& apply(T&& arg, Args&&... args) {
    self()(std::forward<T>(arg));
    if (self().short_circuit()) {
      return self();
    } else {
      return apply(std::forward<Args>(args)...);
    }
  }

  // Here are some handy overloads which provide sensible
  // defaults for container-like structures that one might
  // be interested in recursing into.  You can enable them
  // by adding:
  //
  //    using IterArgs<YourStructName>::operator()
  //
  // to your struct.  These are not enabled by default because
  // you may be able to process these structures more efficiently
  // than handling them one-by-one.

  template <typename T>
  void operator()(at::ArrayRef<T> args) {
    for (const auto& arg : args) {
      self()(arg);
      if (self().short_circuit())
        return;
    }
  }

  template <typename T>
  void operator()(const torch::List<T>& args) {
    for (const auto& arg : args) {
      self()(arg);
      if (self().short_circuit())
        return;
    }
  }

  // NB: we need to specify std::vector manually as C++ won't
  // do an implicit conversion to make a template deduction go through.
  template <typename T>
  void operator()(const std::vector<T>& args) {
    self()(at::ArrayRef<T>{args});
  }

  constexpr bool short_circuit() const {
    return false;
  }

 private:
  inline F& self() {
    return *static_cast<F*>(this);
  }
};

} // namespace torch
