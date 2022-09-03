#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/op_registration/infer_schema.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/dispatch/ObservedOperators.h>

namespace c10 {
namespace impl {

namespace {
  std::string toString(c10::optional<DispatchKey> k) {
    if (k.has_value()) {
      return toString(*k);
    } else {
      return "(catch all)";
    }
  }
}

OperatorEntry::OperatorEntry(OperatorName&& operator_name)
: name_(std::move(operator_name))
, schema_()
#ifndef C10_MOBILE
, tags_()
#endif
, dispatchTable_()
, dispatchKeyExtractor_(DispatchKeyExtractor::makeUninitialized())
, kernels_()
, cpp_signature_()
, is_observed_(ObservedOperators::isObserved(name_))
{
  // Pick up any backend fallbacks that were registered prior to this
  // OperatorEntry being created
  updateDispatchTableFull_(c10::Dispatcher::singleton());
}

namespace {
  void checkSchema(const OperatorName& name, const FunctionSchema& from_def, const std::string& from_def_debug, const FunctionSchema& inferred, const std::string& inferred_debug) {
    c10::optional<std::string> schema_difference = findSchemaDifferences(from_def, inferred);
    if (schema_difference.has_value()) {
      TORCH_CHECK(false,
        "Inferred operator schema for a C++ kernel function doesn't match the expected function schema.\n"
        "  operator: ", toString(name), "\n",
        "  expected schema: ", toString(from_def), "\n",
        "    ", from_def_debug, "\n",
        "  inferred schema: ", toString(inferred), "\n",
        "    ", inferred_debug, "\n",
        "  reason: ", *schema_difference);
    }
  }
} // anonymous namespace

const AnnotatedKernel& OperatorEntry::missingKernel() const {
  static AnnotatedKernel kernel;
  return kernel;
}

const AnnotatedKernel& OperatorEntry::ambiguousAutogradOtherKernel() const {
  static AnnotatedKernel kernel(
    c10::KernelFunction::makeAmbiguousAutogradOther(), nullptr, "ambiguous_autogradother");
  return kernel;
}

void OperatorEntry::registerSchema(FunctionSchema&& schema, std::string&& debug, std::vector<at::Tag> tags) {
  TORCH_INTERNAL_ASSERT(!schema_.has_value());
  for (const auto& kernel : kernels_) {
    for (const auto &j : kernel.second) {
      if (j.inferred_function_schema != nullptr) {
        checkSchema(name_, schema, debug, *j.inferred_function_schema, j.debug);
      }
    }
  }
  // NB: don't register schema until after we've checked everything!
  dispatchKeyExtractor_.registerSchema(schema);
  schema_ = AnnotatedSchema(std::move(schema), std::move(debug));
  #ifndef C10_MOBILE
    tags_ = std::move(tags);
  #endif
}

void OperatorEntry::deregisterSchema() {
  TORCH_INTERNAL_ASSERT(schema_.has_value());
  schema_ = c10::nullopt;
  dispatchKeyExtractor_.deregisterSchema();
}

/* 注册operator对应的kernel */
OperatorEntry::AnnotatedKernelContainerIterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  // NB: cpp_signature doesn't get cleared even after the kernel that populated
  // it is deleted.  This means you could poison the value of cpp_signature_
  // with a bad signature value, and then it would permanently stay there until
  // you deregister the schema.  This can't really be fixed, because we
  // only do a typed() test once in the lifetime of a TypedOperatorHandle,
  // which means if you could validly change the type of a cpp_signature, then
  // that would also invalidate the old TypedOperatorHandles.
  if (cpp_signature.has_value()) {
    if (cpp_signature_.has_value()) {
      // kernel不匹配
      TORCH_CHECK(*cpp_signature == cpp_signature_->signature,
        "\nMismatch in kernel C++ signatures\n",
        "  operator: ", (this->schema_.has_value() ? toString(this->schema_->schema) : toString(name_)), "\n",
        "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
        "  kernel 1: ", cpp_signature_->signature.name(), "\n",
        "    dispatch key: ", toString(cpp_signature_->dispatch_key), "\n",
        "    ", cpp_signature_->debug, "\n",
        "  kernel 2: ", cpp_signature->name(), "\n",
        "    dispatch key: ", toString(dispatch_key), "\n",
        "    ", debug, "\n"
      );
    } else {
      cpp_signature_ = CppSignatureWithDebug { *cpp_signature, debug, dispatch_key };
    }
  }

  if (schema_ && inferred_function_schema) {    //  这个inferred_function_schema 是根据传递而来的function推断出来的
    checkSchema(name_, schema_->schema, schema_->debug, *inferred_function_schema, debug);
  }

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  // Redirect catchAll registrations to CompositeImplicitAutograd.
  // 将内核添加到内核列表中，如果这是第一个内核，可能会创建这个列表。
  // 重定向catchAll注册到CompositeImplicitAutograd。
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];

#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  if (k[0].kernel.isValid()) {
#else
  if (k.size() > 0) {
#endif
    TORCH_WARN("Overriding a previously registered kernel for the same operator and the same dispatch key\n",
               "  operator: ", (schema_.has_value() ? toString(schema_->schema) : toString(name_)), "\n",
               "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
               "  dispatch key: ", toString(dispatch_key), "\n",
               "  previous kernel: ", (cpp_signature_.has_value() ? cpp_signature_->debug : "no debug info"), "\n",
               "       new kernel: ", debug
    );
  }

#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  k[0].kernel = std::move(kernel);
  k[0].inferred_function_schema = std::move(inferred_function_schema);
  k[0].debug = std::move(debug);
#else
  // 添加到 kernels_ 里面
  k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));
#endif
  AnnotatedKernelContainerIterator inserted = k.begin();
  // 更新 dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel    更新dispatch table，即重新建立dispatch table指向最新的kernel
  if (dispatch_key.has_value()) {
    updateDispatchTable_(dispatcher, *dispatch_key);
  } else {
    updateDispatchTableFull_(dispatcher);
  }
  return inserted;
}

void OperatorEntry::deregisterKernel_(
  const c10::Dispatcher& dispatcher,
  c10::optional<DispatchKey> dispatch_key,
  AnnotatedKernelContainerIterator kernel
) {
  // Redirect catchAll deregistrations to CompositeImplicitAutograd.
  DispatchKey dk = dispatch_key.has_value() ? *dispatch_key : DispatchKey::CompositeImplicitAutograd;
  auto found = kernels_.find(dk);
  TORCH_INTERNAL_ASSERT(found != kernels_.end(), "Tried to deregister a kernel for dispatch key ", toString(dispatch_key), " but there are no kernels registered for this dispatch key. The operator is ", toString(name_));
  auto& k = found->second;
#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  // We are about to remove the array from the map, no need to do anything.
#else
  k.erase(kernel);
#endif
  if (k.empty()) {
    // the invariant says we don't want empty lists but instead remove the list from the map
    kernels_.erase(found);
  }
  updateDispatchTable_(dispatcher, dk);
}

void OperatorEntry::updateFallback(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  updateDispatchTable_(dispatcher, dispatch_key);
}

const KernelFunction& OperatorEntry::computeDispatchTableEntry(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const {
  return computeDispatchTableEntryWithDebug(dispatcher, dispatch_key).first.kernel;
}

bool OperatorEntry::hasKernelForAnyDispatchKey(DispatchKeySet ks) const {
  TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end());
  for (auto& kv : kernels_) {
    // Note [No Alias Keys in DispatchKeySet]
    if (!isAliasDispatchKey(kv.first) && ks.has(kv.first)) return true;
  }
  return false;
}

bool OperatorEntry::hasKernelForDispatchKey(DispatchKey k) const {
  TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end());
  for (auto& kv : kernels_) {
    if (k == kv.first) return true;
  }
  return false;
}

const AnnotatedKernel* OperatorEntry::getKernelForDispatchKey(DispatchKey dispatch_key) const{
  auto kern_it = kernels_.find(dispatch_key);
  if (kern_it != kernels_.end()) {
    TORCH_INTERNAL_ASSERT(!kern_it->second.empty());
    TORCH_INTERNAL_ASSERT(kern_it->second.front().kernel.isValid());
    return &kern_it->second.front();
  }
  return nullptr;
}

const std::vector<at::Tag>& OperatorEntry::getTags() const {
  #if defined C10_MOBILE
    TORCH_CHECK(false, "tags are not saved for Mobile");
  #else
    return tags_;
  #endif
}

std::pair<const AnnotatedKernel&, const char*> OperatorEntry::computeDispatchTableEntryWithDebug(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const {
  // [Note] DispatchTable的计算
  // dispatchTable contains entries for runtime dispatch keys.
  // dispatchTable包含运行时分派键的条目。
  // 对于任何dispatch key，它将使用以下顺序选择一个kernel
  //  (1) 如果kernel直接注册到这个key，则使用这个kernel
  //  (2) Handle runtime keys that have kernels available from alias keys
  //  (2) 处理从别名键中获得内核的运行时键
  //    (2.1) Use kernel from DispatchKey::CompositeExplicitAutogradNonFunctional if available.
  //          This is used to register a kernel that works for all backends in inference, except "functional" backends
  //          like LazyTensor/XLA. But it requires separate registration for Autograd keys to support training.
  //          如果可用的话，使用从DispatchKey::CompositeExplicitAutogradNonFunctional的内核。
  //          这是用来注册一个内核，它适用于推断中的所有后端，除了像 LazyTensor/XLA 这样的“功能性”后端。但是它需要单独注册Autograd键来支持训练。
  //
  //    (2.2) Use kernel from DispatchKey::CompositeExplicitAutograd if available.
  //          This is used to register a kernel that works for all backend in inference. But it requires
  //          separate registration for Autograd keys to support training.
  //          如果可用的话，使用DispatchKey::CompositeExplicitAutograd的内核。
  //          这用于注册一个用于推断中的所有后端的内核。但是它需要单独注册Autograd键来支持培训。
  //
  //    (2.3) Use kernel from DispatchKey::CompositeImplicitAutograd if available.
  //          For autograd keys, we only use kernel from CompositeImplicitAutograd when there's no direct registration
  //          to its corresponding backend key or CompositeExplicitAutograd. See Note [CompositeExplicitAutograd and CompositeImplicitAutograd].
  //          For AutogradOther, we eagerly return ambiguousAutogradOtherKernel() if there's registration to any of
  //          its backends and ask backend extender to request a decicated Autograd key for the backend.
  //          See Note [Ambiguity in AutogradOther kernel] for more details.
  //          A CompositeExplicitAutograd kernel prevents CompositeImplicitAutograd kernel being used for Autograd keys, but it doesn't
  //          cause confusion for AutogradOther. It's pretty straightforward to use Autograd (if available)
  //          in this case.

  //    (2.4) Use kernel from DispatchKey::Autograd if available
  //    The implementation of (2.2) relies on the invariant that for a given backend,
  //    `computeDispatchTableEntryWithDebug()` will be called for that backend's autograd key after the
  //    backend key. See Note [Refresh Runtime Autograd entries in dispatchTable_]
  //  (3) Use fallthrough kernel that are registered as fallback.
  // Alias Key Precedence:
  //   CompositExplicitAutogradNonFunctional > CompositeExplicitAutograd > CompositeImplicitAutograd > Autograd
  // Note [CompositeExplicitAutograd and CompositeImplicitAutograd]
  //   When there're registrations to both CompositeExplicitAutograd & CompositeImplicitAutograd & Autograd, from (2.2) we know CompositeExplicitAutograd
  //   and Autograd kernels will be picked up and CompositeImplicitAutograd is overriden.
  //   This is fine and in practice CompositeExplicitAutograd and CompositeImplicitAutograd shouldn't co-exist for an op.
  // TODO: Update alias key precedence after we add new alias keys AutogradDispatchCPUOrCUDA .

  // 1. Operator registration   直接注册到这个DispatchKey的kernel
  if (auto direct_registration = getKernelForDispatchKey(dispatch_key)) {
    return {*direct_registration, "kernel"};
  }

  // 2.1 使用 CompositeExplicitAutogradNonFunctional 如果可用的话
  //     See Note [Undefined in dispatchTable_] for the special handling for Undefined.
  if (dispatch_key == DispatchKey::Undefined || isIncludedInAlias(dispatch_key, DispatchKey::CompositeExplicitAutogradNonFunctional)) {
    if (auto default_backend_registration = getKernelForDispatchKey(DispatchKey::CompositeExplicitAutogradNonFunctional)) {
      return {*default_backend_registration, "default backend kernel"};
    }
  }

  // 2.2 使用 CompositeExplicitAutograd ，如果可用的话
  //     See Note [Undefined in dispatchTable_] for the special handling for Undefined.
  if (dispatch_key == DispatchKey::Undefined || isIncludedInAlias(dispatch_key, DispatchKey::CompositeExplicitAutograd)) {
    if (auto default_backend_registration = getKernelForDispatchKey(DispatchKey::CompositeExplicitAutograd)) {
      return {*default_backend_registration, "default backend kernel"};
    }
  }

  // Note when there's direct registration to CompositeExplicitAutograd, this code path will only be hit by
  // non backend keys (e.g AutogradXXX, Batched etc) due to (2.1).
  // 注意，当有直接注册到CompositeExplicitAutograd，该代码路径将只会被非后端键(例如AutogradXXX，Batched等)由于(2.1)。
  bool has_backend_kernel =
    hasKernelForAnyDispatchKey(getBackendKeySetFromAutograd(dispatch_key)) ||
    // See Note [No Alias Keys in DispatchKeySet]
    hasKernelForDispatchKey(DispatchKey::CompositeExplicitAutograd); // 如果有 CompositeExplicitAutograd 的kernel，不再设置 CompositeImplicitAutograd 的kernel了

  // 2.3. Use CompositeImplicitAutograd kernel if available. For autograd keys, we only use kernel from CompositeImplicitAutograd
  //      when there's no direct registration to its corresponding backend key or CompositeExplicitAutograd.
  //      For AutogradOther, we return ambiguousAutogradOtherKernel() if there's registration
  //      to any of its backends.
  //      如果可用的话，使用CompositeImplicitAutograd内核。
  //      对于autograd键，我们只在没有直接注册到对应的后端键或CompositeExplicitAutograd时使用内核。
  //      对于AutogradOther，如果注册到任何一个后端，我们返回ambiguousAutogradOtherKernel()。
  //      See Note [Undefined in dispatchTable_] for the special handling for Undefined.
  if (dispatch_key == DispatchKey::Undefined || isIncludedInAlias(dispatch_key, DispatchKey::CompositeImplicitAutograd)) {
    if (auto math_registration = getKernelForDispatchKey(DispatchKey::CompositeImplicitAutograd)) {
      if (dispatch_key == DispatchKey::AutogradOther
          && hasKernelForAnyDispatchKey(c10::autogradother_backends)) {
        return {ambiguousAutogradOtherKernel(), "ambiguous autogradother"};
      } else if (!has_backend_kernel) {
        return {*math_registration, "math kernel"};
      }
    }
  }

  // 2.4. For autograd backend keys, use kernel from DispatchKey::Autograd if available
  if (isIncludedInAlias(dispatch_key, DispatchKey::Autograd)) {
    if (auto autograd_registration = getKernelForDispatchKey(DispatchKey::Autograd)) {
      return {*autograd_registration, "autograd kernel"};
    }
  }

  // 3. Backend fallback
  auto dispatch_ix = getDispatchTableIndexForDispatchKey(dispatch_key);
  if (dispatch_ix < 0) {
    return {missingKernel(), "backend fallback not registered on mobile"};
  }
  if (dispatcher.backendFallbackKernels_[dispatch_ix].kernel.isValid()) {
    return {dispatcher.backendFallbackKernels_[dispatch_ix], "backend fallback"};
  }

  // 4. Default to error
  return {missingKernel(), "missing"};
}

// synchronizes the dispatch table entry for a given dispatch key
// with the current state of kernel registrations in the dispatcher.
// note that this is not a complete update, due to relationships between
// dispatch keys (e.g. runtime keys and their associated autograd keys,
// or alias keys and their associated keysets).
// This function should be considered a private helper for updateDispatchTable_()
// 将给定分派键的分派表条目与Dispatcher中内核注册的当前状态同步。
// 注意，这不是一个完整的更新，由于分派键之间的关系
// (例如，运行时键和它们关联的自动grad键，或别名键和它们关联的键集)。
// 这个函数应该被认为是updateDispatchTable_()的私有助手。
void OperatorEntry::updateDispatchTableEntry_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  const auto dispatch_ix = getDispatchTableIndexForDispatchKey(dispatch_key);
  if (C10_UNLIKELY(dispatch_ix == -1)) {
    return;
  }
  dispatchTable_[dispatch_ix] = computeDispatchTableEntry(dispatcher, dispatch_key);  // 设置分派表的kernel
  dispatchKeyExtractor_.setOperatorHasFallthroughForKey(dispatch_key, dispatchTable_[dispatch_ix].isFallthrough());
}

// synchronizes the dispatch table entries for a given dispatch key *and its
// associated keys* with the current state of kernel registrations in the
// dispatcher.
// 根据给定的dispatch key和dispatcher里面相关的key  同步dispatch table的项
// After a kernel has been registered to a dispatch key, a call to this
// function will synchronize the dispatcher state. See e.g. registerKernel()
void OperatorEntry::updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  // Handle Undefined separately since it isn't a runtime key but we have an entry in dispatchTable_.
  // See Note [Undefined in dispatchTable_]
  if (dispatch_key == DispatchKey::Undefined) {
    updateDispatchTableEntry_(dispatcher, dispatch_key);
    return;
  }
  for (auto k : c10::getRuntimeDispatchKeySet(dispatch_key)) {
    updateDispatchTableEntry_(dispatcher, k);
  }
  // Registration to CompositeExplicitAutogradNonFunctional, CompositeExplicitAutograd and CompositeImplicitAutograd should be populated to Undefined.
  // We cannot do this above since Undefined cannot be represented in DispatchKeySet.
  if (dispatch_key == DispatchKey::CompositeImplicitAutograd
   || dispatch_key == DispatchKey::CompositeExplicitAutograd
   || dispatch_key == DispatchKey::CompositeExplicitAutogradNonFunctional) {
    updateDispatchTableEntry_(dispatcher, DispatchKey::Undefined);
  }
  // Note [Refresh Runtime Autograd entries in dispatchTable_]
  // Registering to backend key might affect computed entry at its Autograd backend key due to (2.1) & (2.3).
  // In theory, we should only have to check if the given runtime key has "dense" functionality,
  // e.g. DispatchKey::CPU (which is composed of DispatchKey::Dense and BackendComponent::CPUBit).
  // However, there are some backends that should be included in this set that don't have the dense key set.
  // E.g. DispatchKey::Meta, DispatchKey::ORT.
  if (c10::isBackendDispatchKey(dispatch_key)) {  // if语句里面 CPU -> AutogradCPU
    DispatchKey autograd_key = getAutogradKeyFromBackend(toBackendComponent(dispatch_key));
    updateDispatchTableEntry_(dispatcher, autograd_key);
  }
}

// does a complete update of the dispatch table, synchronizing all
// runtime dispatch keys with the current state of kernel registrations
// in the dispatcher.
// Note that we use updateDispatchTable_() to perform our per-key updating,
// even though that function is equipped to handle out-of-order updates and
// alias key updates, neither of which we send it. This is deliberate - the
// current design is more tractable with all updates funneled through a single
// per-key update mechanism, than with multiple variations that assume different
// invariants.
// 注意，我们使用updateDispatchTable_()来执行每个键的更新，尽管该函数用于处理乱序更新和别名键更新，但我们都没有发送它。
// 这是有意为之的——当前的设计通过单个每个键的更新机制来进行所有更新，比假设不同不变量的多个变体更容易处理。
//
void OperatorEntry::updateDispatchTableFull_(const c10::Dispatcher& dispatcher) {
  // Note [Undefined in dispatchTable_]
  // DispatchKey Undefined在运行时使用:
  // (1) it gives people place to specify functionality that should run when there are no dispatch keys,
  //     e.g., an op without Tensor inputs or empty TensorList arguments
  //     它给人们提供了一个地方来指定在*没有分派键时*应该运行的功能，例如，一个没有张量输入或空的TensorList参数的操作
  // (2) it would let us remove the explicit error checking code in the dispatch hotpath, and so when
  //     no dispatch keys are available we just slide into the undefined handler which would then raise
  //     the error message.
  //     它将允许我们删除分派热路径中显式的错误检查代码，因此当没有可用的分派键时，我们只需滑动到未定义的处理程序，然后抛出错误消息。
  // In the old world of catchAll, the only way to "register" a kernel to Undefined is by registering it to
  // catchAll. After catchAllKernel_ is removed, Undefined now can get a kernel from either CompositeExplicitAutograd,
  // or CompositeImplicitAutograd alias key so that we don't break the support. Ideally isIncludedInAlias(Undefined, CompositeImplicitAutograd)
  // should return true, it returns false because Undefined cannot be represented in a DispatchKeySet.
  updateDispatchTable_(dispatcher, DispatchKey::Undefined);
  for (auto k : DispatchKeySet(DispatchKeySet::FULL)) { // 为什么要上面这行？  因为 DispatchKeySet(DispatchKeySet::FULL) 不包括 DispatchKey::Undefined
    updateDispatchTable_(dispatcher, k);
  }
}

void OperatorEntry::checkInvariants() const {
  if (schema_) {
    TORCH_INTERNAL_ASSERT(schema_->schema.operator_name() == name_, dumpState());
    dispatchKeyExtractor().checkInvariants(schema_->schema);
  }
  TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end(), dumpState());
  for (const auto& kv : kernels_) {
    TORCH_INTERNAL_ASSERT(kv.second.size() > 0, dumpState());
  }
  for (auto k : DispatchKeySet(DispatchKeySet::FULL)) {
    auto expected_k = computeDispatchTableEntry(c10::Dispatcher::singleton(), k);
    auto idx = getDispatchTableIndexForDispatchKey(k);
    if (C10_UNLIKELY(idx == -1)) {
      continue;
    }
    TORCH_INTERNAL_ASSERT(expected_k._equalsBoxedAndUnboxed(dispatchTable_[idx]),
      "Canonical state\n~~~~~~~~~~~\n", dumpState(), "\n\n"
      "Computed table:\n~~~~~~~~~~~\n", dumpComputedTable());
  }
}

std::string OperatorEntry::listAllDispatchKeys() const {
  std::ostringstream str;
  str << "[";

  bool has_kernels = false;
  for (auto k : DispatchKeySet(DispatchKeySet::FULL)) {
    auto iter = getDispatchTableIndexForDispatchKey(k);
    if (iter == -1 || !dispatchTable_[iter].isValid()) {
      continue;
    }
    if (has_kernels) {
      str << ", ";
    }
    str << static_cast<DispatchKey>(iter);
    has_kernels = true;
  }
  str << "]";
  return str.str();
}

void OperatorEntry::reportSignatureError(const CppSignature call_signature) const {
  TORCH_CHECK(false,
        "\nTried to access or call an operator with a wrong signature.\n",
        "  operator: ", (schema_.has_value() ? toString(schema_->schema) : toString(name_)), "\n",
        "    ", (schema_.has_value() ? schema_->debug : "unknown debug info"), "\n",
        "  correct signature:  ", cpp_signature_->signature.name(), "\n",
        "    ", cpp_signature_->debug, "\n",
        "  accessed/called as: ", call_signature.name(), "\n",
        "This likely happened in a call to OperatorHandle::typed<Return (Args...)>(). ",
        "Please make sure that the function signature matches the signature in the operator registration call."
  );
};

void OperatorEntry::reportError(DispatchKey dispatchKey) const {
  // If there is an invariant problem, report it now.
  checkInvariants();

  if (dispatchKey == DispatchKey::Undefined) {
    TORCH_CHECK_NOT_IMPLEMENTED(false,
          "There were no tensor arguments to this function (e.g., you passed an "
          "empty list of Tensors), but no fallback function is registered for schema ", name_,
          ".  This usually means that this function requires a non-empty list of Tensors, "
          "or that you (the operator writer) forgot to register a fallback function.  "
          "Available functions are ", listAllDispatchKeys(), ".\n\n", dumpComputedTable())
  }

  TORCH_CHECK_NOT_IMPLEMENTED(false, "Could not run '", name_, "' with arguments",
          " from the '", toString(dispatchKey), "' backend. This could be because "
          "the operator doesn't exist for this backend, or was omitted during ",
          "the selective/custom build process (if using custom build). If you are a ",
          "Facebook employee using PyTorch on mobile, please visit ",
          "https://fburl.com/ptmfixes for possible resolutions. '",
          name_, "' is only available for these backends: ",
          listAllDispatchKeys(), ".\n\n", dumpComputedTable());
}


// 检查计算的分派表将是什么(例如，updateDispatchTableFull_将更新分派表为什么)
std::string OperatorEntry::dumpComputedTable() const {
  std::ostringstream oss;
  // 需要单独处理Undefined，因为它是一个运行时键，不能在DispatchKeySet中表示。
  std::vector<DispatchKey> runtime_keys = {DispatchKey::Undefined};
  for (auto k : DispatchKeySet(DispatchKeySet::FULL)) runtime_keys.push_back(k);

  for (auto k : runtime_keys) {
    auto kernel_prov = computeDispatchTableEntryWithDebug(c10::Dispatcher::singleton(), k);
    if (kernel_prov.first.kernel.isValid()) {
      oss << toString(k) << ": "
          << (kernel_prov.first.kernel.isFallthrough() ? "fallthrough " : "")
          << kernel_prov.first.debug << " [" << kernel_prov.second << "]\n";
    }
  }
  return oss.str();
}

// Inspect the "canonical" information in OperatorEntry.  This only prints out
// *non-derived* information including kernels registered to alias dispatch keys;
// i.e., what the source of truth says about the operator.  This dumping function
// is appropriate for expect tests.
// This WON'T report backend fallbacks.
std::string OperatorEntry::dumpState() const {
  std::ostringstream oss;
  oss << "name: " << name_ << "\n";
  if (schema_) {
    oss << "schema: " << schema_->schema << "\n";
    oss << "debug: " << schema_->debug << "\n";
    oss << "alias analysis kind: " << toString(schema_->schema.aliasAnalysis())
        << (schema_->schema.isDefaultAliasAnalysisKind() ? " (default)" : "") << "\n";
  } else {
    oss << "schema: (none)\n";
  }

  auto print_kernel = [&](const char* k_desc, const AnnotatedKernelContainer& jts, bool is_alias_key=false) {
    int64_t i = 0;
    for (const auto& jt : jts) {
      oss << k_desc
          << (is_alias_key ? "[alias]" :  "")
          << (i > 0 ? " (inactive)" : "")
          << ": "
          << jt.debug << " :: "
          << (jt.inferred_function_schema ? toString(*jt.inferred_function_schema) : "(none)")
          << " [ " << jt.kernel.dumpState() << "]\n";
      i++;
    }
  };

  // Iterate over DispatchKey, not the flat hash map, so we have a stable order
  for (uint8_t i = 0; i <= static_cast<uint8_t>(DispatchKey::EndOfAliasKeys); i++) {
    auto k = static_cast<DispatchKey>(i);
    auto it = kernels_.find(k);
    if (it != kernels_.end()) {
      print_kernel(toString(k), it->second, c10::isAliasDispatchKey(k));
    }
  }
  return oss.str();
}

}
}
