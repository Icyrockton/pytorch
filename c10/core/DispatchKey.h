#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <ostream>
#include <string>
#include <vector>

namespace c10 {

//
// 从语义上说，BackendComponent的每个值都为我们的分派标识了一个“后端”。
// 我们可以分派的一些功能，允许为每个后端注册不同的处理程序。然后使用BackendComponent来确定要分派到哪个后端实现。

// In implementation terms, the backend component identifies a specific "bit" in
// a DispatchKeySet. The bits in the DispatchKeySet are split between the bottom
// ~12 "BackendComponent" bits, while the remaining upper bits are assigned to
// functionalities.

// 在实现术语中，BackendComponent标识DispatchKeySet中的特定“位”。
// DispatchKeySet的低12位标识了“BackendComponent”，而剩余的高位分配给functionalities
//
// When we encounter a functionality bit that is known to be
// customizeable per-backend, then we also look at the lower BackendComponent
// bits and take the highest bit to determine which backend's implementation to
// use.
//
// 当我们遇到已知的每个后端可定制的功能位时，我们还会查看较低的BackendComponent位，并取最高的位来确定使用哪个后端实现。

enum class BackendComponent : uint8_t {

  // A "backend" is colloquially used to refer to handlers for dispatch
  // which actually implement the numerics of an operation in question.
  // “后端”通常指的是分派的处理程序，它实际实现了相关op的实际操作。
  //
  // Due to the nature of the enum, these backends are specified in
  // an ordered way, but for most backends this order is not semantically
  // meaningful (e.g., it's valid to reorder these backends without changing
  // semantics).  The only situation when backend ordering is meaningful
  // is when the backend participates in multiple dispatch with another
  // backend; e.g., CPU and CUDA (cuda must have higher priority).
  // 由于枚举的性质，这些后端是以有序的方式指定的，但对于大多数后端，这个顺序在语义上没有意义
  // (例如，在不改变语义的情况下重新排序这些后端是有效的)。
  //
  //

  // These keys don't correspond to individual kernels.
  // Instead, they represent the backends that are allowed to override specific
  // pieces of functionality:
  // - dense kernels (e.g. DispatchKey::CPU)
  // - sparse kernels (e.g. DispatchKey::SparseCPU)
  // - quantized kernels (e.g. DispatchKey::QuantizedCPU)
  // - autograd kernels (e.g. DispatchKey::AutogradCPU)
  // We reserve space in the runtime operator table for this full cross product
  // of
  // [backends in this enum] x [keys below that are explicitly marked as having
  // per-backend functionality]

  InvalidBit = 0,
  CPUBit,
  CUDABit,
  HIPBit,
  XLABit,
  MPSBit,
  IPUBit,
  XPUBit,
  HPUBit,
  VEBit,
  LazyBit,
  // A meta tensor is a tensor without any data associated with it.  (They
  // have also colloquially been referred to as tensors on the "null" device).
  // 元张量是一个没有任何数据关联的张量。(它们也被通俗地称为“null”设备上的张量)。
  // A meta tensor can be used to dry run operators without actually doing any
  // computation, e.g., add on two meta tensors would give you another meta
  // tensor with the output shape and dtype, but wouldn't actually add anything.
  // 一个元张量可以用来干运算，而不需要实际做任何计算，
  // 例如，将两个元张量相加会得到另一个具有输出形状和dtype的元张量，但实际上不会添加任何东西。
  MetaBit,
  PrivateUse1Bit,
  PrivateUse2Bit,
  PrivateUse3Bit,
  // Define an alias to represent end of backend dispatch keys.
  // 定义一个别名来表示backend dispatch key的末尾
  // If you add new backend keys after PrivateUse3, please also update it here.
  // (But you shouldn't: private use keys should have higher precedence than
  // all built-in keys)
  EndOfBackendKeys = PrivateUse3Bit,
};

// Semantically, a dispatch key identifies a possible "level" in our
// dispatch, for which a handler may be registered. Each handler corresponds
// to a type of functionality.
// 从语义上讲，dispatch key标识分派中可能的“级别”，可以为该级别注册处理程序。每个处理程序对应于一种类型的功能。
//
// In implementation terms, the dispatch key identifies a specific "bit" in a
// DispatchKeySet.
// 在实现术语中，分派键标识DispatchKeySet中的特定“位”。
// Higher bit indexes get handled by dispatching first (because
// we "count leading zeros" when we extract the highest priority dispatch
// key.)
// 较高位的索引 优先调度处理(因为我们在提取最高优先级的调度键时“计算前导零”)。
//
// Note [DispatchKey分类]
// ***************************这个enum实际上包含几种类型的key，下面会详细说明: ******************************
// QA: 什么叫 functionalities？  Dense,Sparse, 这些叫functionalities
// (1) non-customizable backends (e.g. FPGA)
// (2) non-customizable functionalities (e.g. Functionalize)
// (3) functionalized that are customizable per backend (e.g. Dense, Sparse, AutogradFunctionality)
//     每个后端可定制的功能
// (4) per-backend instances of customizable functionalities (e.g. CPU, SparseCPU, AutogradCPU)
//     可定制功能的每个后端实例
// (5) alias keys (e.g. CompositeImplicitAutograd)
//
// Of the categories above, it's important to note:
// 在上述类别中，需要注意的是:
// (a) which keys are assigned individual bits in a DispatchKeySet
//     DispatchKeySet中的哪些键被分配到单独的位
//
// (b) which keys are assigned individual slots in the runtime operator table
// ("Runtime keys")
//
// (1), (2) and (3) all get their own dedicated bits in the DispatchKeySet.
// (1), (2) and (4) all get their own dedicated slots in the runtime operator
// table.

// See Note [DispatchKeySet Internal Representation] for more details.
//
// NOTE: Keep the list in sync with `DispatchKey` in torchgen/model.py
enum class DispatchKey : uint16_t {

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ UNDEFINED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // 这不是一个“真正的”functionality，但它的存在是为了给我们一个“nullopt”元素，
  // 当DispatchKeySet不包含任何元素时，我们可以返回它。你可以认为DispatchKey在语义上更准确的定义是:
  //
  //    using DispatchKey = optional<RealDispatchKey>
  //
  // and Undefined == nullopt.  We didn't actually represent
  // it this way because optional<RealDispatchKey> would take two
  // words, when DispatchKey fits in eight bits.

  Undefined = 0,

  // 为Undefined定义一个别名来表示CatchAll(从长远来看，这将被消除，但现在它很方便)
  CatchAll = Undefined,

  // ~~~~~~~~~~~~~~~~~~~~~下面的是 Functionality Keys ~~~~~~~~~~~~~~~~~~~~~~ //
  // 枚举中的每一个值(直到EndOfFunctionalityKeys)都对应着一个可以被分派的“功能”。
  // 这在DispatchKeySet中通过将这些枚举值分配给每个剩余的(64 - len(BackendComponent))位来表示。
  //
  // Most of these functionalities have a single handler assigned to them,
  // making them "runtime keys".
  // 这些功能中的大多数都有一个分配给它们的处理程序，使它们成为“运行时键”。
  // That map to a single slot in the runtime operator table.
  // 映射到运行时操作符表中的单个槽。
  //
  // A few functionalities are allowed to be customizable per backend.
  // See [Note: Per-Backend Functionality Dispatch Keys] for details.

  // See [Note: Per-Backend Functionality Dispatch Keys]
  Dense,

  // 下面是不可扩展的后端。这些后端目前没有自己的Autograd/Sparse/Quantized内核覆盖
  // and we therefore don't waste space in the runtime operator table allocating
  // space for them.
  // If any of these backends ever need to customize, e.g., Autograd, then we'll
  // need to add a DispatchKey::*Bit for them.
  // 因此，我们不会浪费运行时操作符表的空间来为它们分配空间。
  // 如果这些后端需要自定义，例如Autograd，那么我们需要为它们添加一个DispatchKey::Bit。

  FPGA, // Xilinx support lives out of tree at
  // https://gitlab.com/pytorch-complex/vitis_kernels

  // ONNX Runtime, lives out of tree at https://github.com/pytorch/ort and
  // https://github.com/microsoft/onnxruntime, and is also used to test general
  // backend/extension machinery in the core. cf:
  // - test/cpp_extensions/ort_extension.cpp
  // - test/test_torch.py
  // - aten/src/ATen/test/extension_backend_test.cpp
  ORT,

  Vulkan,
  Metal,

  // See [Note: Per-Backend Functionality Dispatch Keys]
  Quantized,

  // This backend is to support custom RNGs; it lets you go
  // to a different kernel if you pass in a generator that is not a
  // traditional CPUGeneratorImpl/CUDAGeneratorImpl.  To make use of this
  // key:
  //  1) set it as a second parameter of at::Generator constructor call in
  //     the user-defined PRNG class.
  //  2) use it as a dispatch key while registering custom kernels
  //     (templatized kernels specialized for user-defined PRNG class)
  // intended for out of tree use; tested by aten/src/ATen/test/rng_test.cpp
  CustomRNGKeyId,

  // Here are backends which specify more specialized operators
  // based on the layout of the tensor.  Note that the sparse backends
  // are one case where ordering matters: sparse multi-dispatches with
  // the corresponding dense tensors, and must be handled before them.
  MkldnnCPU, // registered at build/aten/src/ATen/RegisterMkldnnCPU.cpp
  // NB: not to be confused with MKLDNN, which is Caffe2 only

  // See [Note: Per-Backend Functionality Dispatch Keys]
  Sparse,

  SparseCsrCPU,
  SparseCsrCUDA,

  // Note [Non-Customizable Backend Keys]
  // Every key above here is considered a "non-customizable backend".
  // These are backends that will work correctly with autograd, but
  // but currently don't require separate implementations
  // for autograd sparse or quantized kernels.
  // 上面的每个键都被认为是“不可定制的后端”。
  // 这些后端将正确地与autograd一起工作，但但目前不需要为autograd稀疏或量化的核单独实现。
  // Any new backends that don't need to be customized should go above here.
  // If an existing backend needs to e.g. override autograd, then we can
  // consider promoting it into the "BackendComponent" enum
  //
  // For all intents and purposes from the perspective of DispatchKeySet,
  // "non-customizable backend" keys are treated the same way
  // as other functionality keys
  EndOfNonCustomizableBackends = SparseCsrCUDA,

  NestedTensor,

  // In some situations, it is not immediately obvious what the correct
  // backend for function is, because the function in question doesn't
  // have any "tensor" arguments.  In this case, a BackendSelect function
  // can be registered to implement the custom determination of the
  // correct backend.
  // 在某些情况下，函数的正确后端是什么并不是很明显，因为所讨论的函数没有任何“tensor”参数。
  // 在这种情况下，可以注册BackendSelect函数来实现正确后端的自定义确定。
  BackendSelect,

  Python,

  // Out-of-core key for Fake Tensor in torchdistx.
  // See https://pytorch.org/torchdistx/latest/fake_tensor.html
  Fake,
  // See Note [Out-of-tree vmap+grad prototype]. The purpose of this key
  // is to insert code after the "autograd subsystem" runs, so this key should
  // be directly after ADInplaceOrView and all of the autograd keys.
  FuncTorchDynamicLayerBackMode,

  // Alias and mutation removal.
  // If some backends want to opt into only alias removal or only mutation
  // removal,
  // we can consider adding separate keys dedicated to those individual passes.
  // See Note [Functionalization Pass In Core] for details.
  Functionalize,

  // The named dispatch key is set for any tensors with named dimensions.
  // Although we have a dispatch key for named tensors, for historical reasons,
  // this dispatch key doesn't do any of the substantive functionality for named
  // tensor (though, hypothetically, it could!)  At the moment, it's just
  // responsible for letting us give good error messages when operations
  // don't support named tensors.
  //
  // NB: If you ever consider moving named tensor functionality into
  // this dispatch key, note that it might be necessary add another dispatch
  // key that triggers before composite operators, in case a composite operator
  // has named dimension propagation that doesn't match that of its
  // constituent parts.
  Named,

  // The Conjugate dispatch key is set for any tensors that need to perform
  // conjugation
  // This is implemented at a dispatch level right before any backends run
  Conjugate,

  // The Negative dispatch key is set for any tensors that need to perform
  // negation
  // This is implemented at a dispatch level right before any backends run
  Negative,

  ZeroTensor, // registered at build/aten/src/ATen/RegisterZeroTensor.cpp

  // Note [ADInplaceOrView key]
  // ADInplaceOrView key is used by inplace or view ops to register a kernel
  // that does additional setup for future autograd computation.
  //
  // 1. For inplace ops this kernel does version bump
  // 2. For view ops this kernel does `as_view` setup where we properly setup
  //    DifferentiableViewMeta on the view tensors.
  //
  // For other ops it's fallthrough kernel since there's no extra
  // work to do.
  //
  // Note [Dream: skip VariableType kernel when requires_grad=false]
  //
  // In an ideal world where we can skip VariableType kernel for inputs
  // with requires_grad=false, instead of a fallthrough kernel, we'll
  // register a kernel shown below to all functional ops as well:
  // torch::Tensor my_functional_op(...) {
  //   {
  //     // Note for every op in VariableType, you need to go through
  //     // `AutoDispatchBelowADInplaceOrView` guard exactly once to add the
  //     // key to TLS excluded set. If you don't go through it at all,
  //     // inplace/view ops called through `at::` inside your backend
  //     // kernel will dispatch to ADInplaceOrView kernels and do a lot
  //     // of extra work.
  //     at::AutoDispatchBelowADInplaceOrView guard;
  //     at::redispatch::my_functional_op(...);
  //   }
  // }
  // But this work is currently blocked since it adds an extra dispatch
  // for all ops and it's non-trivial overhead at model level(a few percents).
  // Thus our current approach takes advantage of the fact every kernel go
  // through VariableType kernel first and pulls the
  // `at::AutoDispatchBelowADInplaceOrView` guard of functional ops
  // up to the `VariableType` kernel. Thus we only add the extra dispatch
  // to view/inplace ops to minimize its perf impact to real models.
  ADInplaceOrView,
  // Note [Alias Dispatch Key : Autograd]
  // All backends are oblivious to autograd; autograd is handled as a
  // layer which happens on top of all backends. It inspects the autograd
  // metadata of all inputs, determines what autograd metadata should be
  // constructed by the output, and otherwise defers to the backend to
  // actually do the numeric computation.  Autograd contains
  // the bulk of this logic.
  // 所有后端都是无关的autograd;
  // Autograd作为发生在所有后端之上的层次来进行处理。
  // 它检查所有输入的自动的元数据，确定输出应该构造什么元数据，然后让后端执行实际的数值计算。Autograd包含这个逻辑的大部分

  // Autograd is now an alias dispatch key which by default maps to all
  // backend-specific autograd keys.
  // Backend-specific allow backends to override the default kernel registered
  // to Autograd key as needed.
  // Autograd现在是一个别名分派键，默认情况下映射到所有后端特定的Autograd键。
  // 后端特定的允许后端根据需要覆盖注册到Autograd键的默认内核。
  //
  // For example, XLA wants to define autograd for einsum directly.
  // Registering a custom autograd implementation at the XLA key won't work
  // because we process Autograd before XLA.  This key has higher priority and
  // gets processed first.  You generally should NOT redispatch after handling
  // autograd here (since that would result in execution of the Autograd
  // operator, which you're trying to skip).
  // 例如，XLA希望直接为einsum定义autograd。
  // 在XLA键上注册一个自定义autograd实现是行不通的，
  // 因为我们在XLA之前处理autograd。此键具有更高的优先级，并优先处理。
  // 在处理autograd之后，通常不应该重新分派(因为这将导致执行autograd操作符，这是您试图跳过的)。
  //
  // In AutogradXLA implementations,
  // you are responsible for handling autograd yourself, or deferring to other
  // operators which support autograd.

  // Currently we only have backend-specific autograd keys for CPU/CUDA/XLA and
  // reserved user-defined backends. All other in-tree backends share the
  // AutogradOther key. We can add specific autograd key for those backends
  // upon request.
  // 目前我们只有后端特定的CPU/CUDA/XLA和保留用户自定义后端autograd键。
  // 所有其他树中的后端共享AutogradOther。我们可以根据要求为那些后端添加特定的autograd键。
  AutogradOther,

  // See [Note: Per-Backend Functionality Dispatch Keys]
  AutogradFunctionality,

  // NestedTensor is an example of something that isn't a "real backend"
  // (because it mostly consists of redispatching kernels)
  // but it would like to override autograd functionality in C++.
  // We can handle cases like this by adding an extra functionality key
  // exclusively for handling autograd for NestedTensor.
  // lives out of tree at
  // https://github.com/pytorch/nestedtensor
  AutogradNestedTensor,

  Tracer,

  // Autocasting precedes VariableTypeId, to ensure casts are autograd-exposed
  // and inputs are saved for backward in the post-autocast type.
  AutocastCPU,
  AutocastXPU,
  // Naughtily, AutocastCUDA is also being used for XLA.  In the terminal state,
  // it probably should get its own Autocast key
  AutocastCUDA,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ WRAPPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // There are a number of alternative modes which may want to handle before
  // autograd; for example, error checking, tracing, profiling or vmap.  They
  // go here.
  // 有许多其他的模式，在autograd之前处理;例如，错误检查、跟踪、性能分析或vmap。

  FuncTorchBatched, // See Note [Out-of-tree vmap+grad prototype]
  FuncTorchVmapMode, // See Note [Out-of-tree vmap+grad prototype]

  // 这是 BatchedTensorImpl 的分派键，用于实现vmap的批处理规则。
  Batched,

  // When we are inside a vmap, all tensors dispatch on this key.
  // 当我们在vmap中，所有张量都分配到这个键上。
  // See Note: [DispatchKey::VmapMode usage] for more details.
  VmapMode,

  FuncTorchGradWrapper, // See Note [Out-of-tree vmap+grad prototype]

  // Out-of-core key for Deferred Module Initialization in torchdistx.
  // See https://pytorch.org/torchdistx/latest/deferred_init.html
  DeferredInit,

  // Used by Python key logic to know the set of tls on entry to the dispatcher
  // This kernel assumes it is the top-most non-functorch-related DispatchKey.
  // If you add a key above, make sure to update the fallback implementation for
  // this.
  PythonTLSSnapshot,

  // This key should be at the very top of the dispatcher
  // 这个key位于Dispatcher的最顶端
  FuncTorchDynamicLayerFrontMode, // See Note [Out-of-tree vmap+grad prototype]

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a single
  // process test.  Use it by creating a TensorImpl with this DispatchKey, and
  // then registering operators to operate on this type id.  See
  // aten/src/ATen/core/dispatch/backend_fallback_test.cpp for a usage example.
  TESTING_ONLY_GenericWrapper,

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a ingle
  // process test.  Use it by toggling the mode on and off via
  // TESTING_ONLY_tls_generic_mode_set_enabled and then registering operators
  // to operate on this type id.  See
  // aten/src/ATen/core/dispatch/backend_fallback_test.cpp
  // for a usage example
  TESTING_ONLY_GenericMode,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  EndOfFunctionalityKeys, // End of functionality keys.    functionality keys结束

  // ~~~~~~~~~~~~~~ "Dense" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~~~~~ //
  // ~~~~~~~~~~~~~~ "Dense" backend 的 Dispatch keys ~~~~~~~~~~~~~~~~~~~~ //
  // Here are backends which you think of as traditionally specifying
  // how to implement operations on some device.

  // See Note [The Ordering of Per-Backend Dispatch Keys Matters!]
  StartOfDenseBackends,
  CPU, // registered at build/aten/src/ATen/RegisterCPU.cpp
  CUDA, // registered at build/aten/src/ATen/RegisterCUDA.cpp
  HIP, // NB: I think this is not actually used, due to Note [Masquerading as
  // CUDA]
  XLA, // lives out of tree at https://github.com/pytorch/xla
  MPS, // registered at build/aten/src/ATen/RegisterMPS.cpp
  IPU, // lives out of tree at https://github.com/graphcore/poptorch
  XPU, // For out of tree Intel's heterogeneous computing plug-in
  HPU, // For out of tree & closed source integration of HPU / Habana
  VE, // For out of tree & closed source integration of SX-Aurora / NEC
  Lazy, // For lazy tensor backends
  Meta,
  // Here are reserved backends for user-defined backends, see Note [Private use
  // DispatchKey]
  // To see some example about how to use this, check out ORT
  PrivateUse1,
  PrivateUse2,
  PrivateUse3,
  EndOfDenseBackends = PrivateUse3,

  // ~~~~~~~~~~~~~~ "Quantized" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~ //
  // 以_开头的键目前没有使用
  // but are needed to ensure that every backend is indexed correctly.

  // See Note [The Ordering of Per-Backend Dispatch Keys Matters!]
  StartOfQuantizedBackends,
  QuantizedCPU, // registered at build/aten/src/ATen/RegisterQuantizedCPU.cpp
  QuantizedCUDA, // registered at build/aten/src/ATen/RegisterQuantizedCUDA.cpp
  _QuantizedHIP,
  _QuantizedXLA,
  _QuantizedMPS,
  _QuantizedIPU,
  QuantizedXPU, // For out of tree Intel's heterogeneous computing plug-in
  _QuantizedHPU,
  _QuantizedVE,
  _QuantizedLazy,
  _QuantizedMeta,
  _QuantizedPrivateUse1,
  _QuantizedPrivateUse2,
  _QuantizedPrivateUse3,
  EndOfQuantizedBackends = _QuantizedPrivateUse3,

  // ~~~~~~~~~~~~~~ "Sparse" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~~~~ //
  // keys starting with an _ are not currently used,
  // but are needed to ensure that every backend is indexed correctly.

  // See Note [The Ordering of Per-Backend Dispatch Keys Matters!]
  StartOfSparseBackends,
  SparseCPU, // registered at build/aten/src/ATen/RegisterSparseCPU.cpp
  SparseCUDA, // registered at build/aten/src/ATen/RegisterSparseCUDA.cpp
  SparseHIP, // TODO: I think this is not actually used, due to Note
  // [Masquerading as CUDA]
  _SparseXLA,
  _SparseMPS,
  _SparseIPU,
  SparseXPU, // For out of tree Intel's heterogeneous computing plug-in
  _SparseHPU,
  SparseVE, // For out of tree & closed source integration of SX-Aurora / NEC
  _SparseLazy,
  _SparseMeta,
  _SparsePrivateUse1,
  _SparsePrivateUse2,
  _SparsePrivateUse3,
  EndOfSparseBackends = _SparsePrivateUse3,

  // ~~~~~~~~~~~~~~ "NestedTensor" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~~~~
  // //
  // keys starting with an _ are not currently used,
  // but are needed to ensure that every backend is indexed correctly.

  // See Note [The Ordering of Per-Backend Dispatch Keys Matters!]
  StartOfNestedTensorBackends,
  // registered at build/aten/src/ATen/RegisterNestedTensorCPU.cpp
  NestedTensorCPU,
  // registered at build/aten/src/ATen/RegisterNestedTensorCUDA.cpp
  NestedTensorCUDA,
  _NestedTensorHIP,
  _NestedTensorXLA,
  _NestedTensorMPS,
  _NestedTensorIPU,
  _NestedTensorXPU,
  _NestedTensorHPU,
  _NestedTensorVE,
  _NestedTensorLazy,
  _NestedTensorMeta,
  _NestedTensorPrivateUse1,
  _NestedTensorPrivateUse2,
  _NestedTensorPrivateUse3,
  EndOfNestedTensorBackends = _NestedTensorPrivateUse3,

  // ~~~~~~~~~~~~~~ "Autograd" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~~ //
  // keys starting with an _ are not currently used,
  // but are needed to ensure that every backend is indexed correctly.

  // See Note [The Ordering of Per-Backend Dispatch Keys Matters!]
  StartOfAutogradBackends,
  AutogradCPU,
  AutogradCUDA,
  _AutogradHIP,
  AutogradXLA,
  AutogradMPS,
  AutogradIPU,
  AutogradXPU,
  AutogradHPU,
  _AutogradVE,
  AutogradLazy,
  AutogradMeta,
  // Here are some reserved pre-autograd keys for user-defined backends, see
  // Note [Private use DispatchKey]
  AutogradPrivateUse1,
  AutogradPrivateUse2,
  AutogradPrivateUse3,
  EndOfAutogradBackends = AutogradPrivateUse3,
  // If we add a new per-backend functionality key that has higher priority
  // than Autograd, then this key should be updated.
  EndOfRuntimeBackendKeys = EndOfAutogradBackends,

  // ~~~~~~~~~~~~~~~~~~~~~~ Alias Dispatch Keys ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // ~~~~~~~~~~~~~~~~~~~~~~ 别名 Dispatch Keys ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Note [Alias Dispatch Keys]
  // Alias dispatch keys are synthetic dispatch keys which map to multiple
  // runtime dispatch keys. Alisa keys have precedence, but they are always
  // lower precedence than runtime keys. You can register a kernel to an
  // alias key, the kernel might be populated to the mapped runtime keys
  // during dispatch table computation.
  // 别名分派键是映射到多个运行时分派键的合成分派键。Alisa键有优先级，
  // 但它们的优先级总是低于运行时键。您可以将内核注册到别名键，在分派表计算期间，内核可能被填充到映射的运行时键
  // If a runtime dispatch key has multiple kernels from alias keys, which
  // kernel wins is done based on the precedence of alias keys (but runtime
  // keys always have precedence over alias keys).
  // Alias keys won't be directly called during runtime.
  // 如果一个运行时分派键有多个来自别名键的内核，
  // 那么哪个内核胜出将基于别名键的优先级(但运行时键总是优先于别名键)。在运行时不会直接调用别名键。

  // See Note [Alias Dispatch Key : Autograd]
  Autograd,
  CompositeImplicitAutograd, // registered at
  // build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
  CompositeExplicitAutograd, // registered at
  // build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp
  // See Note [CompositeExplicitAutogradNonFunctional Key]
  CompositeExplicitAutogradNonFunctional, // registered at
  // build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp

  // Define an alias key to represent end of alias dispatch keys.
  // If you add new alias keys after Autograd, please also update it here.
  StartOfAliasKeys = Autograd,
  EndOfAliasKeys = CompositeExplicitAutogradNonFunctional, //

  // ~~~~~~~~~~~~~~~~~~~~~~~~~ BC ALIASES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // The aliases exist for backwards compatibility reasons, they shouldn't
  // be used
  CPUTensorId = CPU,
  CUDATensorId = CUDA,
  DefaultBackend = CompositeExplicitAutograd,
  PrivateUse1_PreAutograd = AutogradPrivateUse1,
  PrivateUse2_PreAutograd = AutogradPrivateUse2,
  PrivateUse3_PreAutograd = AutogradPrivateUse3,
  Autocast = AutocastCUDA,
};

// Note [Private use DispatchKey]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Private use tensor IDs are preallocated tensor type IDs for use in user
// applications.  Similar to private use fields in HTTP, they can be used
// by end users for experimental or private applications, without needing
// to "standardize" the tensor ID (which would be done by submitting a PR
// to PyTorch to add your type ID).
//
// Private use tensor IDs are appropriate to use if you want to experiment
// with adding a new tensor type (without having to patch PyTorch first) or
// have a private, non-distributed application that needs to make use of a
// new tensor type.  Private use tensor IDs are NOT appropriate to use for
// libraries intended to be distributed to further users: please contact
// the PyTorch developers to get a type ID registered in this case.
//
// We provide two classes of private user tensor id: regular DispatchKeys
// and Autograd DispatchKeys.  DispatchKeys serve the role of ordinary "backend"
// DispatchKeys; if you were adding support for a new type of accelerator, you
// would use a backend DispatchKey, and ideally automatically reuse
// AutogradOther definitions already defined in PyTorch.  AutogradPrivateUse
// DispatchKeys serve as "wrapper" DispatchKeys: they are only necessary for
// tensors that compose multiple internal tensors, and for cases when the
// built-in autograd formulas for operators are not appropriate.

static_assert(
    (static_cast<uint8_t>(BackendComponent::EndOfBackendKeys) +
     static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys)) <= 64,
    "The BackendComponent and DispatchKey enums (below EndOfFunctionalityKeys)"
    " both map to backend and functionality bits"
    " into a 64-bit bitmask; you must have less than 64 total entries between them");

// Check if a DispatchKey is an alias mapping to other runtime keys.
// 检查DispatchKey是别名键?
constexpr bool isAliasDispatchKey(DispatchKey k) {
  return k >= DispatchKey::StartOfAliasKeys && k <= DispatchKey::EndOfAliasKeys;
}

// [Note: Per-Backend Functionality Dispatch Keys]
// Check if a DispatchKey is a per-backend functionality key
// Any functionalities that can be customized per-backend should be added here.
// These keys correspond to functionalities that can be customized indivually
// per backend.
// 这些键对应于每个后端可以单独定制的功能
// While they only take up one bit in the `DispatchKeySet` bitset,
// they map to (# backends) slots in the operator table.
// 虽然它们只占用' DispatchKeySet ' bitset中的一个位，但它们映射到操作符表中的(后端)槽。
// Each of these keys also has a separate set of "runtime keys" in the dispatch
// key enum, per backend, which *do* map to the individual operator table slots.
// For example, the "Sparse" key maps to an individual bit in the
// DispatchKeySet, while `SparseCPU`, `SparseCUDA`, etc all map to individual
// slots in the runtime operator table.
// 例如，“Sparse” key映射到DispatchKeySet中的单个bit，
//      而“SparseCPU”、“SparseCUDA”等都映射到运行时操作符表中的单个槽。
//
// DispatchKey是否是每个backend都有的 (CPU,SparseCPU,QuantizedCPU, AutogradCPU)
constexpr bool isPerBackendFunctionalityKey(DispatchKey k) {  // 每个后端都有的功能键，Dense,Quantized....
  if (k == DispatchKey::Dense || k == DispatchKey::Quantized ||
      k == DispatchKey::Sparse || k == DispatchKey::AutogradFunctionality ||
      k == DispatchKey::NestedTensor) {
    return true;
  } else {
    return false;
  }
}

// Note that this includes Undefined in the total count.
// BUT EndOfFunctionalityKeys is its own (placeholder) key.
// e.g. Undefined=0, Dense=1, Sparse=2, EndOfFunctionalityKeys=3.
// In the above example, there are 3 total functionality keys.
//
// 注意，这Undefined括在总数中。
// 例如:Undefined=0, Dense=1, Sparse=2, EndOfFunctionalityKeys=3。
// 在上面的例子中，总共有3个功能键。
constexpr uint8_t num_functionality_keys =
    static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys);

constexpr uint8_t num_backends =
    static_cast<uint8_t>(BackendComponent::EndOfBackendKeys);

// Note [No More Than 16 Backends]
// Search for this note to find places in the code where the "no more than 16
// backends" invariant is baked in.
static_assert(
    static_cast<uint8_t>(BackendComponent::EndOfBackendKeys) <= 16,
    "BackendComponent currently only supports <= 16 backends. If we really need to extend this, \
there are a few places where this invariant is baked in");

/**
 * @return 返回 5
 */
constexpr uint8_t numPerBackendFunctionalityKeys() {
  uint8_t count = 0;
  for (uint8_t k = 0; k <= num_functionality_keys; ++k) {
    if (isPerBackendFunctionalityKey(static_cast<DispatchKey>(k)))
      ++count;
  }
  return count;
}

#if defined(C10_MOBILE_TRIM_DISPATCH_KEYS)
// See [Note: Trimmed Mobile Dispatch Keys]
constexpr uint16_t num_runtime_entries = 8;
#else
/*
 * 运行时键
 * Dense | Quantized |  Sparse |  AutogradFunctionality |  NestedTensor
 * 这5个functionality 再乘以 后端的个数(13)  得到per-backend的个数
 * 再加上不能自定义的functionality
 *
 * 40 + 13 × 5 = 105
 */
constexpr uint16_t num_runtime_entries = num_functionality_keys +
    (numPerBackendFunctionalityKeys() * (num_backends - 1));
#endif

// See Note [No More Than 16 Backends]
// backend的mask
constexpr uint16_t full_backend_mask =
    (static_cast<uint16_t>(1) << num_backends) - 1;

C10_API const char* toString(DispatchKey);
C10_API const char* toString(BackendComponent);
C10_API std::ostream& operator<<(std::ostream&, DispatchKey);
C10_API std::ostream& operator<<(std::ostream&, BackendComponent);

C10_API DispatchKey getAutogradKeyFromBackend(BackendComponent k);

// Parses a string into a dispatch key.
// If the string cannot be correctly parsed, throws an exception.
C10_API c10::DispatchKey parseDispatchKey(const std::string& k);

// These are some convenience identifiers for dispatch keys which are
// shorter to type than their long counterparts.  Note that some of these
// dispatch keys directly correspond to DeviceType; and most APIs that
// accept DispatchKey also accept DeviceType; e.g.,
// torch::dispatch(torch::kCPU, ...) is also valid.
constexpr DispatchKey kAutograd = DispatchKey::Autograd;

// See Note [The Ordering of Per-Backend Dispatch Keys Matters!]
// 将per-backend的DispatchKey 转换成 BackendComponent
constexpr BackendComponent toBackendComponent(DispatchKey k) {
  if (k >= DispatchKey::StartOfDenseBackends &&
      k <= DispatchKey::EndOfDenseBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfDenseBackends));
  } else if (
      k >= DispatchKey::StartOfQuantizedBackends &&
      k <= DispatchKey::EndOfQuantizedBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfQuantizedBackends));
  } else if (
      k >= DispatchKey::StartOfSparseBackends &&
      k <= DispatchKey::EndOfSparseBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfSparseBackends));
  } else if (
      k >= DispatchKey::StartOfNestedTensorBackends &&
      k <= DispatchKey::EndOfNestedTensorBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfNestedTensorBackends));
  } else if (
      k >= DispatchKey::StartOfAutogradBackends &&
      k <= DispatchKey::EndOfAutogradBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfAutogradBackends));
  } else {
    return BackendComponent::InvalidBit;
  }
}

constexpr DispatchKey toFunctionalityKey(DispatchKey k) {
  if (k <= DispatchKey::EndOfFunctionalityKeys) {
    return k;
  } else if (k <= DispatchKey::EndOfDenseBackends) {
    return DispatchKey::Dense;
  } else if (k <= DispatchKey::EndOfQuantizedBackends) {
    return DispatchKey::Quantized;
  } else if (k <= DispatchKey::EndOfSparseBackends) {
    return DispatchKey::Sparse;
  } else if (k <= DispatchKey::EndOfNestedTensorBackends) {
    return DispatchKey::NestedTensor;
  } else if (k <= DispatchKey::EndOfAutogradBackends) {
    return DispatchKey::AutogradFunctionality;
  } else {
    return DispatchKey::Undefined;
  }
}

// 转换成runtime-key
// 给定 (DispatchKey::Dense, BackendComponent::CUDABit) 返回 DispatchKey::CUDA
constexpr DispatchKey toRuntimePerBackendFunctionalityKey(
    DispatchKey functionality_k,
    BackendComponent backend_k) {
  if (functionality_k == DispatchKey::Dense) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfDenseBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::Sparse) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfSparseBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::Quantized) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfQuantizedBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::NestedTensor) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfNestedTensorBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::AutogradFunctionality) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfAutogradBackends) +
        static_cast<uint8_t>(backend_k));
  }
  return DispatchKey::Undefined;
}

} // namespace c10

namespace torch {
// Expose the constant, but not the TYPE (DispatchKey is an implementation
// detail!)
using c10::kAutograd;
} // namespace torch

// NB: You really shouldn't use this instance; this enum is guaranteed
// to be pretty small so a regular array should be acceptable.
namespace std {
template <>
struct hash<c10::DispatchKey> {
  typedef size_t result_type;
  typedef c10::DispatchKey argument_type;

  size_t operator()(c10::DispatchKey x) const {
    return static_cast<size_t>(x);
  }
};
} // namespace std
