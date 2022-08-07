#pragma once

#include <ATen/core/stack.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/TypeList.h>

namespace c10 {

using Stack = torch::jit::Stack; // TODO Instead of this, move torch::jit::Stack to the c10 namespace.

class OperatorHandle;
struct OperatorKernel;

// This kernel implements the behavior of falling through to the next available
// registered dispatch key.  The implementation of this function is FAST; it is
// no overhead to fallthrough to the next key.  See cpp file for some more
// implementation notes; notably, this does NOT actually go through the
// boxing/unboxing codepath.
TORCH_API void fallthrough_kernel(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*);

// Note [Ambiguity in AutogradOther kernel]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This error-reporting kernel is registered to the AutogradOther entry in the
// dispatch table when there is both a CompositeImplicitAutograd kernel and a
// backend kernel for ANY backend that maps to AutogradOther.  To see why
// this is necessary in the AutogradOther case, it's helpful to first see
// why everything works out fine for a backend that has a reserved Autograd
// entry (see rule 2.2 in [Note] DispatchTable computation):
//
//    CPU   AutogradCPU
//    reg?  registers with...
//    -------------------------------------------------
//    y     Autograd registration takes precedence
//          over CompositeImplicitAutograd.
//          This is good, because the CPU specific backend
//          implementation is more specialized and typically better;
//          if we used the composite, we would bypass it.
//          (NB: the Autograd key is guaranteed to exist because
//          the autograd codegen requires it!)
//
//    n     CompositeImplicitAutograd takes precedence.
//          This is also good, because the Autograd
//          registration (if it exists) would try to redispatch
//          to the (non-existent) CPU implementation; by
//          using the composite, we ensure the operator
//          actually works.
//
// As you can see, when we have a specific Autograd key (AutogradCPU), we can
// decide whether or not to use the CompositeImplicitAutograd kernel or the
// Autograd kernel based on whether or not the backend kernel exists.
//
// However, for AutogradOther (which is the catchall autograd kernel for
// everything that doesn't have a specific Autograd key), we can't do this
// trick because there isn't any unique backend to peek at to disambiguate;
// if there are some backends that have implementations they prefer Autograd,
// but unimplemented backends would prefer CompositeImplicitAutograd.  Rather
// than arbitrarily pick one or the other, we just register a kernel that raises
// an error and let the user decide how to proceed.
TORCH_API void ambiguous_autogradother_kernel(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*);

// Note [named_not_supported_kernel]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This kernel implements reporting an error message saying that named tensor is
// not supported.  This kernel doesn't rely on the Stack, and so it is special
// cased in the dispatcher to be triggered before we attempt boxing (so we can
// give a good error message in cases when boxing is not supported).  When
// boxing is universally supported this can be removed.
[[noreturn]] TORCH_API void named_not_supported_kernel(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*);

/**
 * KernelFunction类似于std::function，但存储的是一个核函数。
 * 您可以从已装箱或开箱的函数function/functor/lambda创建一个KernelFunction，并以装箱或开箱的方式调用它。
 * 如果创建它的方式与调用它的方式不匹配，它将根据需要进行装箱或开箱。
 */
class TORCH_API KernelFunction final {
public:
  // 装箱的kernel实际存储的样子
  //
  // Note [Plumbing Keys Through The Dispatcher]
  // Benchmarks have shown that it is expensive for the dispatcher to read from thread-local storage (TLS)
  // upon every dispatch call into order to compute which kernel to dispatch to.
  //
  // To mitigate this, we've updated the calling convention inside the dispatcher to expect every kernel that it stores
  // to have a first argument of type DispatchKeySet.
  //
  // What are the invariants of the DispatchKeySet when it gets passed to a kernel?
  // - All keys to the left of the current dispatch key have been masked out.
  //   (e.g. a Tracing kernel that takes in the DispatchKeySet will expect the highest bit to be DispatchKey::Tracer)
  // - All other keys that dispatcher normally would have computed through TLS + global state + op arguments
  //   are still in the set.
  //
  // Kernels can then opt into using this keyset to save the dispatcher from doing repeated work during redispatches:
  // recalculating the highest-priority dispatch key, which involves reading from TLS. Instead, the kernels that opt in will
  // calculate an updated DispatchKeySet directly from the old one, and pass the updated set directly into the dispatcher
  // upon redispatching.
  //
  // This is an opt-in mechanism: Kernels can automatically opt in by setting the first argument in their signature
  // to be of type DispatchKeySet. See the kernels in VariableTypeEverything.cpp and TraceTypeEverything.cpp for examples.
  //
  // The mechanism for optionally passing that DispatchKeySet into the kernel lives in make_boxed_from_unboxed_functor.h.
  // See Note [Plumbing Keys Through The Dispatcher 2] for details.
  using InternalBoxedKernelFunction = void(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*);
  // This is the public API for how boxed kernels are defined
  using BoxedKernelFunction = void(const OperatorHandle&, Stack*);
  using BoxedKernelFunction_withDispatchKeys = void(const OperatorHandle&, DispatchKeySet, Stack*);

  KernelFunction(); /* 构造函数 */

  // Fast path for dispatch to allow not touching the boxed kernel in
  // the common case where unboxed is available.
  // 快速调度路径，允许在未装箱的情况下不接触已装箱的内核。
  bool isValidUnboxed() const;
  bool isValid() const;
  bool isFallthrough() const;

  /**
   * 以装箱的方式调用函数。如果kernel函数是用一个开箱的函数创建的，这将调用一个开箱包装器，然后调用该未装箱的函数。
   *
   * Example:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
   * > Tensor result = func.callBoxed(stack);
   *
   * Or, with an unboxed implementation:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   * > Tensor result = func.callBoxed(stack);
   *
   * 以装箱的方式调用
   */
  void callBoxed(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Stack* stack) const;

  /**
   * Call the function in an unboxed way.
   * If the kernel function was created with a boxed function,
   * this will box all inputs and then call into that boxed function.
   * 以开箱方式调用函数。
   * 如果kernel是用一个装箱的函数创建的，那么这将装箱所有的输入，然后调用这个装箱的函数。
   * 注意，这还不能适用于所有类型。
   *
   * Note that this doesn't work for all types yet.
   *
   * Example:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   * > Tensor result = func.call<Tensor, Tensor, bool>(tensor1, true);
   *
   * Or, with a boxed implementation:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
   * > Tensor result = func.call<Tensor, Tensor, bool>(tensor1, true);
   *
   * 以开箱方式调用
   */
  template<class Return, class... Args>
  Return call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const;

  /**
   * 从装箱函数创建KernelFunction
   *
   * Example:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction<&boxed_func>();
   */
  template<BoxedKernelFunction* func>
  static KernelFunction makeFromBoxedFunction();

  /**
   * TODO: This will only be useful if we write a backend fallback that plumbs dispatch keys (currently there are none)
   * See Note [Plumbing Keys Through The Dispatcher] for details.
   */
  template<BoxedKernelFunction_withDispatchKeys* func>
  static KernelFunction makeFromBoxedFunction();

  /**
   * Create a KernelFunction from an unboxed functor.
   *
   * Example:
   *
   * > class MyFunctor final : public c10::OperatorKernel {
   * >   public:
   * >     Tensor operator()(Tensor a, Tensor b) {...}
   * > };
   * > KernelFunction func = KernelFunction::makeFromUnboxedFunctor<MyFunctor>(std::make_unique<MyFunctor>());
   */
  template<bool AllowLegacyTypes = false, class KernelFunctor>
  static KernelFunction makeFromUnboxedFunctor(std::unique_ptr<OperatorKernel> kernelFunctor);

  /**
   * Create a KernelFunction from a boxed functor.
   *
   * Example:
   *
   * > class MyFunctor final : public c10::OperatorKernel {
   * >   public:
   * >     void operator()(const OperatorHandle&, DispatchKeySet, Stack*) {...}
   * > };
   * > KernelFunction func = KernelFunction::makeFromBoxedFunctor(std::make_unique<MyFunctor>());
   */
  template<class KernelFunctor>
  static KernelFunction makeFromBoxedFunctor(std::unique_ptr<KernelFunctor> kernelFunctor);

  /**
   * Create a KernelFunction from an unboxed function.
   * This is usually better than KernelFunction::makeFromUnboxedRuntimeFunction
   * because knowing the function pointer as a template argument (i.e. at
   * compile time) allows the compiler to inline the function into its
   * unboxing wrapper and yields better performance when calling the function.
   * 从一个未装箱的函数创建一个KernelFunction。
   * 这通常比KernelFunction::makeFromUnboxedRuntimeFunction更好，
   * 因为知道函数指针作为模板参数(即在编译时)允许编译器将函数内联到它的开箱包装器中，并在调用函数时产生更好的性能。
   *
   * Example:
   *
   * > Tensor unboxed_func(Tensor a, Tensor b) {...}
   * > KernelFunction func = KernelFunction::makeFromUnboxedFunction<decltype(unboxed_func), &unboxed_func>();
   */
  template<class FuncPtr, bool AllowLegacyTypes = false>
  static KernelFunction makeFromUnboxedFunction(FuncPtr);

  /**
   * Create a KernelFunction from an unboxed function.
   * KernelFunction::makeFromUnboxedFunction is usually a better choice than
   * this if you know the function pointer at compile time, see doc comment
   * there for an explanation.
   *
   * Example:
   *
   * > Tensor unboxed_func(Tensor a, Tensor b) {...}
   * > KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&unboxed_func);
   */
  template<bool AllowLegacyTypes = false, class FuncType>
  static KernelFunction makeFromUnboxedRuntimeFunction(FuncType* func);

  static KernelFunction makeFallthrough();
  static KernelFunction makeAmbiguousAutogradOther();
  static KernelFunction makeNamedNotSupported();

  template<BoxedKernelFunction* func>
  static void make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, DispatchKeySet, Stack* stack);

  template<BoxedKernelFunction_withDispatchKeys* func>
  static void make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, DispatchKeySet, Stack* stack);

  /**
   * Create a KernelFunction from an unboxed lambda.
   *
   * Example:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   */
  template<bool AllowLegacyTypes = false, class Lambda>
  static std::enable_if_t<guts::is_stateless_lambda<std::decay_t<Lambda>>::value, KernelFunction> makeFromUnboxedLambda(Lambda&& lambda);
  template<bool AllowLegacyTypes = false, class Lambda>
  static std::enable_if_t<!guts::is_stateless_lambda<std::decay_t<Lambda>>::value, KernelFunction> makeFromUnboxedLambda(Lambda&& lambda);

  std::string dumpState() const;
  // For testing internal invariants only
  bool _equalsBoxedAndUnboxed(const KernelFunction&) const;

private:

  explicit KernelFunction(std::unique_ptr<OperatorKernel> functor, InternalBoxedKernelFunction* boxed_kernel_func, void* unboxed_kernel_func);

  OperatorKernel* getFunctor_() const;

  c10::intrusive_ptr<OperatorKernel> functor_;  /* 仿函数 */

  InternalBoxedKernelFunction* boxed_kernel_func_;    // 装箱的函数指针
  void* unboxed_kernel_func_;                         // 未装箱的函数指针
};

}

#include <ATen/core/boxing/KernelFunction_impl.h>
