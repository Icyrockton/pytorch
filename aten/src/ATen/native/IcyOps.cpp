//
// Created by icyrockton on 22-8-7.
//

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/IcyOps.h>

namespace at{




namespace native{

Tensor& icy_abs_out(const Tensor& self, Tensor& result) {  // torch.icy_abs(out=xxx)
    auto iter = TensorIterator::unary_float_op(result, self);
    icy_abs_stub(iter.device_type(),iter);
    return result;
}

Tensor icy_abs(const Tensor& self) {
  Tensor result = at::empty({0},self.options());
  return at::icy_abs_out(result,self);
}

Tensor& icy_abs_(Tensor& self) {
    return at::icy_abs_out(self,self);    // 这个是 at namespace下的
}

DEFINE_DISPATCH(icy_abs_stub);


}
}