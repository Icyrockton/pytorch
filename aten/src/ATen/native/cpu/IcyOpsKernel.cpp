//
// Created by icyrockton on 22-8-7.
//

#include <ATen/native/IcyOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/Math.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

namespace at{
namespace native{

inline namespace CPU_CAPABILITY{

    static void icy_abs_kernel(TensorIteratorBase& iter){
      auto device = iter.device();
        AT_DISPATCH_ALL_TYPES(iter.dtype(),"icy_abs_kernel",[&]() {

            cpu_kernel_vec(iter,
                           [=](scalar_t a ) { return abs_impl(a);  },
                           [=](Vectorized<scalar_t> a) { return a.abs();  }
                           );

        });
    }



}


REGISTER_DISPATCH(icy_abs_stub, &CPU_CAPABILITY::icy_abs_kernel);

}
}