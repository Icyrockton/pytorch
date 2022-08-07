#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/in_place.h>

#include <type_traits>

namespace c10 {

/// MaybeOwnedTraits<T> describes how to borrow from T.  Here is how we
/// can implement borrowing from an arbitrary type T using a raw
/// pointer to const:
// MaybeOwnedTraits<T>描述了如何从T借。下面是如何使用指向const的原始指针实现从任意类型T借位:
template <typename T>
struct MaybeOwnedTraitsGenericImpl {
  using owned_type = T;
  using borrow_type = const T*;

  static borrow_type createBorrow(const owned_type& from) {
    return &from;
  }

  static void assignBorrow(borrow_type& lhs, borrow_type rhs) {
    lhs = rhs;
  }

  static void destroyBorrow(borrow_type& /*toDestroy*/) {}

  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return *borrow;
  }

  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  static bool debugBorrowIsValid(const borrow_type& borrow) {
    return borrow != nullptr;
  }
};

/// It is possible to eliminate the extra layer of indirection for
/// borrows for some types that we control. For examples, see
/// intrusive_ptr.h and TensorBody.h.

template <typename T>
struct MaybeOwnedTraits;

// Explicitly enable MaybeOwned<shared_ptr<T>>, rather than allowing
// MaybeOwned to be used for any type right away.
template <typename T>
struct MaybeOwnedTraits<std::shared_ptr<T>>
    : public MaybeOwnedTraitsGenericImpl<std::shared_ptr<T>> {};

/// A smart pointer around either a borrowed or owned T. When
/// constructed with borrowed(), the caller MUST ensure that the
/// borrowed-from argument outlives this MaybeOwned<T>.
/// 一个围绕借用或拥有的T的智能指针。使用borrow()构造时，调用者必须确保从某个参数借用的寿命超过这个MaybeOwned<T>。
/// Compare to
/// Rust's std::borrow::Cow
/// (https://doc.rust-lang.org/std/borrow/enum.Cow.html), but note
/// that it is probably not suitable for general use because C++ has
/// no borrow checking. Included here to support
/// Tensor::expect_contiguous.
template <typename T>
class MaybeOwned final {
  using borrow_type = typename MaybeOwnedTraits<T>::borrow_type;
  using owned_type = typename MaybeOwnedTraits<T>::owned_type;

  bool isBorrowed_;
  union {
    borrow_type borrow_;
    owned_type own_;
  };

  /// 别用这个;  用 borrowed()         borrow_type是owned_type的指针类型
  explicit MaybeOwned(const owned_type& t)
      : isBorrowed_(true), borrow_(MaybeOwnedTraits<T>::createBorrow(t)) {}

  /// 别用这个; 用owned()
  explicit MaybeOwned(T&& t) noexcept(
      std::is_nothrow_move_constructible<T>::value)
      : isBorrowed_(false), own_(std::move(t)) {}

  /// 别用这个; 用owned()
  template <class... Args>
  explicit MaybeOwned(in_place_t, Args&&... args)
      : isBorrowed_(false), own_(std::forward<Args>(args)...) {}

 public:
  explicit MaybeOwned() : isBorrowed_(true), borrow_() {}

  // Copying a borrow yields another borrow of the original, as with a
  // T*. Copying an owned T yields another owned T for safety: no
  // chains of borrowing by default! (Note you could get that behavior
  // with MaybeOwned<T>::borrowed(*rhs) if you wanted it.)
  // 复制一个借来的东西会产生另一个原来的东西，就像复制一个拥有的T会产生另一个拥有的T一样，
  // 出于安全考虑:默认情况下没有借贷链!(注意，如果你想要，你可以通过MaybeOwned<T>::borrow (rhs)获得该行为。)
  MaybeOwned(const MaybeOwned& rhs) : isBorrowed_(rhs.isBorrowed_) {
    if (C10_LIKELY(rhs.isBorrowed_)) {
      MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
    } else {
      new (&own_) T(rhs.own_);
    }
  }

  MaybeOwned& operator=(const MaybeOwned& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_UNLIKELY(!isBorrowed_)) {
      if (rhs.isBorrowed_) {
        own_.~T();
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
        isBorrowed_ = true;
      } else {
        own_ = rhs.own_;
      }
    } else {
      if (C10_LIKELY(rhs.isBorrowed_)) {
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
      } else {
        MaybeOwnedTraits<T>::destroyBorrow(borrow_);
        new (&own_) T(rhs.own_);
        isBorrowed_ = false;
      }
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
    return *this;
  }

  MaybeOwned(MaybeOwned&& rhs) noexcept(
      std::is_nothrow_move_constructible<T>::value)
      : isBorrowed_(rhs.isBorrowed_) {
    if (C10_LIKELY(rhs.isBorrowed_)) {
      MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
    } else {
      new (&own_) T(std::move(rhs.own_));
    }
  }

  MaybeOwned& operator=(MaybeOwned&& rhs) noexcept(
      std::is_nothrow_move_assignable<T>::value) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_UNLIKELY(!isBorrowed_)) {
      if (rhs.isBorrowed_) {
        own_.~T();
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
        isBorrowed_ = true;
      } else {
        own_ = std::move(rhs.own_);
      }
    } else {
      if (C10_LIKELY(rhs.isBorrowed_)) {
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
      } else {
        MaybeOwnedTraits<T>::destroyBorrow(borrow_);
        new (&own_) T(std::move(rhs.own_));
        isBorrowed_ = false;
      }
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
    return *this;
  }

  static MaybeOwned borrowed(const T& t) {
    return MaybeOwned(t);
  }

  static MaybeOwned owned(T&& t) noexcept(
      std::is_nothrow_move_constructible<T>::value) {
    return MaybeOwned(std::move(t));
  }

  template <class... Args>
  static MaybeOwned owned(in_place_t, Args&&... args) {
    return MaybeOwned(in_place, std::forward<Args>(args)...);
  }

  ~MaybeOwned() {
    if (C10_UNLIKELY(!isBorrowed_)) {
      own_.~T();
    } else {
      MaybeOwnedTraits<T>::destroyBorrow(borrow_);
    }
  }

  // This is an implementation detail!  You should know what you're doing
  // if you are testing this.  If you just want to guarantee ownership move
  // this into a T
  bool unsafeIsBorrowed() const {
    return isBorrowed_;
  }

  const T& operator*() const& {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
    }
    return C10_LIKELY(isBorrowed_)
        ? MaybeOwnedTraits<T>::referenceFromBorrow(borrow_)
        : own_;
  }

  const T* operator->() const {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
    }
    return C10_LIKELY(isBorrowed_)
        ? MaybeOwnedTraits<T>::pointerFromBorrow(borrow_)
        : &own_;
  }

  // If borrowed, copy the underlying T. If owned, move from
  // it. borrowed/owned state remains the same, and either we
  // reference the same borrow as before or we are an owned moved-from
  // T.
  T operator*() && {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
      return MaybeOwnedTraits<T>::referenceFromBorrow(borrow_);
    } else {
      return std::move(own_);
    }
  }
};

} // namespace c10
