#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>

#include <c10/util/intrusive_ptr.h>

namespace c10 {

// A storage represents the underlying backing data buffer for a
// tensor.  This concept was inherited from the original Torch7
// codebase; we'd kind of like to get rid of the concept
// (see https://github.com/pytorch/pytorch/issues/14797) but
// it's hard work and no one has gotten around to doing it.
// storage表示张量的底层后备数据缓冲区。这个概念是从最初的《Torch7》代码库继承而来的;
// 我们有点想摆脱这个概念(参见 https://github.com/pytorch/pytorch/issues/14797)，但这是一项艰苦的工作，没有人抽时间去做它。
//
// NB: storage is supposed to uniquely own a data pointer; e.g.,
// two non-null data pointers alias if and only if they are from
// the same storage.  Technically you can violate this invariant
// (e.g., you can create a non-owning StorageImpl with at::from_blob)
// but a lot of things won't work correctly, including:
// 注: storage应该唯一拥有一个数据指针;例如，两个非空数据指针别名当且仅当它们来自相同的存储。
// 从技术上讲，你可以违反这个不变量(例如，你可以用at::from_blob创建一个非所有的StorageImpl)，但是很多事情都不能正常工作，包括:
//
// - An ordinary deleter on such a storage is wrong, because normal deleters
//   assume unique ownership, but if you have two storages at the same data,
//   that implies there is some sort of shared ownership. So your deleter would
//   have to actually be internally doing some sort of refcount thing
//   这种存储上的N个普通删除器是错误的，因为普通删除器假定所有权是唯一的，但如果对同一数据有两个存储，这意味着存在某种共享所有权。你的删除器会在内部做一些refcount之类的事情
// - Deepcopy in Python side relies on storage equality and not data pointer
//   equality; so if there are two separate storages pointing to the same data,
//   the data will actually get duplicated in that case (one data ptr before,
//   two data ptrs after)
// - Version counts won't work correctly, because we do all VC tracking at the
//   level of storages (unless you explicitly disconnect the VC with detach);
//   mutation because data pointers are the same are totally untracked
struct C10_API StorageImpl : public c10::intrusive_ptr_target {
 public:
  struct use_byte_size_t {};

  StorageImpl(
      use_byte_size_t /*use_byte_size*/,
      SymInt size_bytes,
      at::DataPtr data_ptr, // 实际的数据指针
      at::Allocator* allocator, // 内存分配器
      bool resizable)
      : data_ptr_(std::move(data_ptr)),
        size_bytes_(size_bytes),
        resizable_(resizable),
        received_cuda_(false),
        allocator_(allocator) {
    if (resizable) {
      TORCH_INTERNAL_ASSERT(
          allocator_, "For resizable storage, allocator must be provided");
    }
  }

  StorageImpl(
      use_byte_size_t /*use_byte_size*/,
      SymInt size_bytes,
      at::Allocator* allocator,
      bool resizable)
      : StorageImpl(
            use_byte_size_t(),
            size_bytes,
            size_bytes.is_symbolic()
                ? allocator->allocate(0)
                : allocator->allocate(size_bytes.as_int_unchecked()),
            allocator,
            resizable) {}

  StorageImpl& operator=(StorageImpl&& other) = default;
  StorageImpl& operator=(const StorageImpl&) = delete;
  StorageImpl() = delete;
  StorageImpl(StorageImpl&& other) = default;
  StorageImpl(const StorageImpl&) = delete;
  ~StorageImpl() override = default;

  void reset() {
    data_ptr_.clear();
    size_bytes_ = 0;
  }

  template <typename T>
  inline T* data() const {
    return unsafe_data<T>();
  }

  template <typename T>
  inline T* unsafe_data() const {
    return static_cast<T*>(this->data_ptr_.get());
  }

  // Destructor doesn't call release_resources because it's
  // unnecessary; don't forget to change that if needed!
  void release_resources() override {
    data_ptr_.clear();
  }

  size_t nbytes() const {
    return size_bytes_.expect_int();
  }

  SymInt sym_nbytes() const {
    return size_bytes_;
  }

  // TODO: remove later
  void set_nbytes(size_t size_bytes) {
    size_bytes_ = size_bytes;
  }

  bool resizable() const {
    return resizable_;
  };

  at::DataPtr& data_ptr() {
    return data_ptr_;
  };

  const at::DataPtr& data_ptr() const {
    return data_ptr_;
  };

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
    at::DataPtr old_data_ptr(std::move(data_ptr_));
    data_ptr_ = std::move(data_ptr);
    return old_data_ptr;
  };

  void set_data_ptr_noswap(at::DataPtr&& data_ptr) {
    data_ptr_ = std::move(data_ptr);
  }

  // TODO: Return const ptr eventually if possible
  void* data() {
    return data_ptr_.get();
  }

  void* data() const {
    return data_ptr_.get();
  }

  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }

  at::Allocator* allocator() {
    return allocator_;
  }

  const at::Allocator* allocator() const {
    return allocator_;
  };

  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  void set_allocator(at::Allocator* allocator) {
    allocator_ = allocator;
  }

  Device device() const {
    return data_ptr_.device();
  }

  void set_resizable(bool resizable) {
    if (resizable) {
      // We need an allocator to be resizable
      AT_ASSERT(allocator_);
    }
    resizable_ = resizable;
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      void* src,
      size_t size_bytes,
      DeleterFnPtr d = nullptr) {
    UniqueStorageShareExternalPointer(
        at::DataPtr(src, src, d, data_ptr_.device()), size_bytes);
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      size_t size_bytes) {
    data_ptr_ = std::move(data_ptr);
    size_bytes_ = size_bytes;
    allocator_ = nullptr;
    resizable_ = false;
  }

  // This method can be used only after storage construction and cannot be used
  // to modify storage status
  void set_received_cuda(bool received_cuda) {
    received_cuda_ = received_cuda;
  }

  bool received_cuda() {
    return received_cuda_;
  }

 private:
  DataPtr data_ptr_;
  SymInt size_bytes_;
  bool resizable_;
  // Identifies that Storage was received from another process and doesn't have
  // local to process cuda memory allocation
  // 标识从另一个进程接收的存储，没有本地处理cuda内存分配
  bool received_cuda_;
  Allocator* allocator_;
};
} // namespace c10
