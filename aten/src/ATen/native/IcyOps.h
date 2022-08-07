//
// Created by icyrockton on 22-8-7.
//

#pragma once

#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>


namespace at {
    class Tensor;
    class TensorBase;
    struct TensorIteratorBase;
}


namespace at{
namespace native{

using icy_fn = void(*)(TensorIteratorBase&);
DECLARE_DISPATCH(icy_fn,icy_abs_stub);

}


}