#pragma once

#include <ATen/DimVector.h>
#include <ATen/core/Dimname.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/strides.h>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wdeprecated-copy-dtor")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wdeprecated-copy-dtor")
#endif

namespace at {

class Tensor;

namespace impl {

// 使用它来定义meta function的原型。有两个版本;一种接受一个参数(只有operator名称)，
// 或者接受两个参数(operator名称和overload名称)的FUNC2变体。
//
// Example usage:
//
//    TORCH_META_FUNC2(add, Tensor) (
//      const Tensor& self, const Tensor& other
//    ) {
//      ... compute sizes and options ...
//      set_output(sizes, options);
//    }
//
#define TORCH_META_FUNC(name) void structured_##name::meta
#define TORCH_META_FUNC2(name, overload) \
  void structured_##name##_##overload::meta

// These are versions of TORCH_META_FUNC(2) that include a precompute_out struct
// as a return value. They should be used when the kernel in question has
// precomputed values declared in native_functions.yaml and the corresponding
// implementation should return an instance of the aforementioned struct.
#define TORCH_PRECOMPUTE_META_FUNC(name) \
  structured_##name::meta_return_ty structured_##name::meta
#define TORCH_PRECOMPUTE_META_FUNC2(name, overload) \
  structured_##name##_##overload::meta_return_ty    \
      structured_##name##_##overload::meta

// Use this to create a precompute struct in a meta function.
#define TORCH_PRECOMPUTE_STRUCT(name) structured_##name::precompute_out<>
#define TORCH_PRECOMPUTE_STRUCT2(name, overload) \
  structured_##name##_##overload::precompute_out<>

// Use this to define the prototype for an implementation.  This takes only
// one argument, which is the name of the dispatch key entry you're
// implementing.
// 使用它来定义实现的原型。这只需要一个参数，即您正在实现的调度键条目的名称。
//
// Example usage:
//
//    TORCH_IMPL_FUNC(add_cpu) (
//      Tensor& result, const Tensor& self, const Tensor& other
//    ) {
//      ... do the actual implementation ...
//    }
//
#define TORCH_IMPL_FUNC(name) void structured_##name::impl

// Base class for all structured kernel classes.  The set_output virtual
// method is varied depending whether or not the operator is
// functional/out/inplace, and could also be specialized for CPU/CUDA/etc
// (although presently it isn't).
// 所有结构化内核类的基类。
// set_output虚方法根据操作符是否是functional/out/inplace而变化，也可以专门用于CPU/CUDA/etc(尽管目前它不是)。
//
// A notable subclass of this interface is TensorIteratorBase.
struct TORCH_API MetaBase {
  virtual const Tensor& maybe_get_output(int64_t output_idx) = 0;

  // See: https://github.com/pytorch/pytorch/issues/69813
  // Whenever defining the output properties in the META function of a
  // structured kernel (what was usually done with `set_output`), use one of
  // these 3 variants, instead. In order to decide which variant to use, check
  // the following decision tree:
  // 当在结构化内核的META函数中定义输出属性时(通常用' set_output '来做)，使用这3个变体中的一个来代替。
  // 为了决定使用哪个变体，请检查以下决策树:
  //
  // - Can the kernel you are going to implement support output tensors
  //   with arbitrary strides?
  //   您要实现的kernel能够支持任意步长的输出张量吗?
  //     |
  //     -- YES: `set_output_raw_strided`
  //     |
  //     -- NO: Should the output tensor strides be contiguous? 输出张量步长应该是连续的吗?
  //         |
  //         -- YES: `set_output_contiguous`
  //         |
  //         -- NO: `set_output_strided`
  //
  // Use this function whenever the kernel requires specific strides for the
  // output. If `strides` does not match the given output strides, proxy outputs
  // will be created and passed to the IMPL function.
  // 当内核要求输出特定的步长时，使用此函数。如果' stride '与给定的输出跨距不匹配，将创建代理输出并传递给IMPL函数。
  virtual void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names = {}) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }

  // Use this function whenever the kernel knows how to handle arbitrary strided
  // outputs. This function has the same behavior as the old `set_output`: it
  // will only re-stride if the given output was resized.
  // 只要内核知道如何处理任意步长输出，就可以使用这个函数。
  // 这个函数与旧的 `set_output` 具有相同的行为:只有当给定的输出被调整大小时，它才会重新re-stride。
  virtual void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides_hint,
      TensorOptions options,
      DimnameList names = {}) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }

  // Use this function if the kernel requires contiguous strides.
  // Alias for `set_output_strided`, but with contiguous strides.
  void set_output_contiguous(
      int64_t output_idx,
      IntArrayRef sizes,
      TensorOptions options,
      DimnameList names = {}) {
    auto strides = c10::contiguous_strides(sizes);
    set_output_strided(output_idx, sizes, strides, options, names);
  }

  // Returns a reference to an undefined tensor if there is no presupplied
  // output
  const Tensor& maybe_get_output() {
    return maybe_get_output(0);
  }
  virtual ~MetaBase() {}
};

} // namespace impl

} // namespace at

C10_CLANG_DIAGNOSTIC_POP()
