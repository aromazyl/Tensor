#include "tensor_traits.hpp"
#include "check.h"

namespace tensor {



// cuda function support implementation
namespace {
}

template <int k, typename T>
struct cuTensor {
  typedef TensorTraits<k, T, cuTensor> traits_type;
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;

  size_t n_[k] = {0};
  pointer data_;
  bool wrapped_;

  constexpr static int rand() { return k; };
  pointer data() { return data_; }
  const_pointer data() { return data_; }
  cuTensor() : data_(NULL), wrapped_() {}

  cuTensor(const cuTensor& a);

  cuTensor(cuTensor&& src);

  ~cuTensor() {
    if (!wrapped_) {
      checkCudaErrors(cudaFree(data_));
    }
  }

  cuTensor &operator=(const cuTensor& src);

  cuTensor &operator=(cuTensor&& src);

  size_t (*init_dim)(void);

};

}
