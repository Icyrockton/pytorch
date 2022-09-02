#pragma once

#include <c10/util/TypeTraits.h>

namespace c10 {

namespace impl {
  namespace detail {
    template<class FuncType, class ReturnType, class ParameterList>
    class WrapFunctionIntoRuntimeFunctor_ {};

    template<class FuncType, class ReturnType, class... Parameters>
    class WrapFunctionIntoRuntimeFunctor_<FuncType, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
    public:
      template<class FuncType_>
      explicit WrapFunctionIntoRuntimeFunctor_(FuncType_&& kernel_func)
      : kernel_func_(std::forward<FuncType_>(kernel_func)) {}

      decltype(auto) operator()(Parameters... args) {
        return kernel_func_(std::forward<Parameters>(args)...);
      }

    private:
      FuncType kernel_func_;
    };
  }

  // WrapFunctionIntoRuntimeFunctor: Wraps any runtime functor into a functor that
  // inherits from c10::OperatorKernel, so it can be used as a c10 kernel.
  // This can, for example, be used for lambdas, functors or even function pointers.
  // In the case of function pointers, since it is a runtime function pointer,
  // there is an overhead for calling it whenever the kernel is invoked.
  // WrapFunctionIntoRuntimeFunctor:将任何运行时函数封装到一个继承自c10::OperatorKernel的函数中，因此它可以用作c10内核。
  // 例如，这可以用于lambdas、仿函数甚至函数指针。
  // 在函数指针的情况下，由于它是一个运行时函数指针，每当调用内核时调用它都会有开销。

  // 将一个函数封装为仿函数
  template<class FuncType>
  using WrapFunctionIntoRuntimeFunctor = detail::WrapFunctionIntoRuntimeFunctor_<
      FuncType,
      typename guts::infer_function_traits_t<FuncType>::return_type,
      typename guts::infer_function_traits_t<FuncType>::parameter_types
  >;
}

}
