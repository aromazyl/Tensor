#pragma once

#include "tensor_traits.hpp"
#include "return_type.hpp"
#include "check.h"

namespace tensor {



// cuda function support implementation
namespace {
template <typename A, typename B> struct SameClass {
  enum { result = false };
};

template <typename A> struct SameClass<A, A> {
  enum { result = true };
};


template <int k>
struct IntType {
  enum { Val = k };
};

typedef IntType<1> TrueType;
typedef IntType<0> FalseType;

template <typename Cond, typename If, typename Else>
struct IfThenElse;

template <typename If, typename Else>
struct IfThenElse<TrueType, If, Else> {
  typedef Result If;
};

template <typename If, typename Else>
struct IfThenElse<FalseType, If, Else> {
  typedef Result Else;
};



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
  pointer d_data_;
  bool wrapped_;

  constexpr static int rand() { return k; };
  pointer data() { return data_; }
  const_pointer data() { return data_; }
  cuTensor() : data_(NULL), d_data_(NULL), wrapped_() {}

  cuTensor(const cuTensor& a);

  cuTensor(cuTensor&& src);

  ~cuTensor() {
    if (!wrapped_) {
      checkCudaErrors(cudaFree(d_data_));
      delete [] data_;
    }
  }

  cuTensor &operator=(const cuTensor& src);

  cuTensor &operator=(cuTensor&& src);
  
  size_t init_dim() {
    size_t s = n_[0];
    for (size_t i = 1; i < k; ++i) {
      if (n_[i] == 0)
        n_[i] = n_[i-1];
      s *= n_[i];
    }
    return s;
  }

  template <int d, typename U, typename... Args>
  typename std::enable_if<
  std::is_integral<U>::value and !std::is_pointer<U>::value and
  d<k, void>::type init(U i, Args&& ... args) {
    assert(i != 0);
    n_[d] = i;
    init<d+1>(args...);
  }

  template <int d> void init(value_type v = value_type()) {
    size_t s = init_dim();
    data_ = new value_type[s];
    std::fill_n(data_, s, v);
    checkCudaErrors(cudaMalloc(&d_data_, sizeof(v) * s));
    checkCudaErrors(cudaMemcpy(d_data_, data_, sizeof(v) * s, cudaMemcpyHostToDevice));
  }

  template <int d, typename P, typename d_P, typename... Args>
  typename std::enable_if<std::is_pointer<P>::value, void>::type
  init(P p, d_P d_p, Args&&... args) {
    init_dim();
    wrapped_ = true;
    data_ = p;
    d_data_ = d_p;
  }

  template <int d, class functor>
  typename std::enable_if<
  !std::is_integral<functor>::value and !std::pointer<functor>::value,
  void>::type
  init(functor fn) {
    size_t s = init_dim();
    data_ = new value_type[s];
    this->fill(fn);
    checkCudaErrors(cudaMalloc(&d_data_, sizeof(value_type) * s));
    checkCudaErrors(cudaMemcpy(d_data_, data_, sizeof(value_type) * s, cudaMemcpyHostToDevice));
  }

public:

  template <typename... Args>
  cuTensor(const Args&... args)
  : data_(NULL), wrapped_() {
    static_assert(sizeof...(Args) <= k+1,
        "*** ERROR *** Wrong number of arguments for tensor");
    init<0>(args...);
  }

  template <int d, typename U> struct Initializer_list {
    typedef std::initializer_list<
      typename Initializer_list<d-1, U>::list_type> list_type;

    static void process(list_type l, cuTensor& a, size_t s, size_t idx) {
      a.n_[k - d] = l.size();
      size_t j = 0;
      for (const auto& r : l) {
        Initializer_list<d-1, U>::process(r, a, s*l.size(), idx);
      }
    }
  };

  template <typename U> struct Initializer_list<1, U> {
    typedef std::initializer_list<U> list_type;
    static void process(list_type l, cuTensor& a, size_t s, size_t idx) {
      a.n_[k-1] = l.size();
      size_t j = 0;
      if (!a.data_) {
        a.data_ = new value_type[s * l.size()];
      }
      for (const auto& r : l) {
        a.data_[idx+s*j++] = r;
      }
    }
  };

  typedef typename Initializer_list<k, T>::list_type initializer_type;

  cuTensor(initializer_type l) : data_(NULL), d_data_(NULL), wrapped_(false) {
    Initializer_list<k, T>::process(l, *this, 1, 0);
    int s = this->size();
    checkCudaErrors(cudaMalloc(&d_data_, sizeof(value_type) * s));
    checkCudaErrors(cudaMemcpy(d_data_, data_, sizeof(value_type) * s, cudaMemcpyHostToDevice));
  }

  template <class A> cuTensor(const Expr<A>& expr) : data_(NULL), d_data_(NULL), wrapped_(false) {
    static_assert(SameClass<cuTensor<k, T>, typename A::result_type>::result,
        "");
    *this = expr();
  }

  size_t size() const {
    size_t n = 1;
    for (size_t i = 0; i < k; ++i)
      n *= n_[i];
    return n;
  }

  size_t size(size_t i) const {
    return n_[i];
  }

private:
  template <typename pack_type>
  pack_type index(pack_type indices[]) const {
    pack_type i = indices[0]; s = 1;
    for (int j = 1; j < k; ++j) {
      assert(indices[j] >= 0);
      assert(static_assert<size_t>(indices[j]) < n_[j]);
      s *= n_[j - 1];
      i += s * indices[j];
    }
    return i;
  }


};

}
