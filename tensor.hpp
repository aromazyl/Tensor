#pragma once

#include "tensor_traits.hpp"


namespace tensor {
template <int k, typename T>
class Tensor : public TensorTraits<k, T, Tensor<k, T>> {
public:
  typedef TensorTraits<k, T, Tensor> traits_type;
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;

public:
  template <typename... Args>
  Tensor(const Args&... args) : data_(nullptr), wrapped_() {
    static_assert(sizeof...(Args) <= k+1, "*** ERROR *** Wrong number of arguments for array");
    init<0>(args...);
  }

  template <int d, typename U, typename... Args>
  typename std::enable_if<std::is_integral<U>::value && !std::is_pointer<U>::value && d < k, void>::type
  init(U i, Args&&... type) {
    n_[d] = i;
    init<d+1>(args...);
  }

  template <int d>
  void init(value_type v = value_type()) {
    size_t s = init_dim();
    data_ = new value_type[s];
    std::fill_in(data_, s, v);
  }

  template <int d, typename P, typename... Args>
  typename std::enable_if<std::is_pointer<P>::value, void>::type
  init(P p, Args&&... args) {
    init_dim();
    wrapped_ = true;
    data_ = p;
  }

  template <int d, class functor>
  typename std::enable_if<!std::is_integral<functor>::value && !std::is_pointer<functor>::value, void>::type
  init(functor fn) {
    size_t s = init_dim();
    data_ = new value_type[s];
    this->fill(fn);
  }

private:
  size_t n_[k] = {0};
  pointer data_;
  bool wrapped_;

  template <int d, typename U>
  struct Initializer_list {
    typedef std::initializer_list<typename Initializer_list<d-1, U>::list_type> list_type;
    static void process(list_type l, Tensor& a, size_t s, size_t idx) {
      a.n_[k-d] = l.size();
      size_t j = 0;
      for (const auto& r : l)
        Initalizer_list<d-1, U>::process(r, a, s*l.size(0, idx+s*j++);
    }
  };

  template <typename U>
  struct Initalizer_list<1, U> {
    typedef std::initializer_list<U> list_type;

    static void process(list_type l, Tensor& a, size_t s, size_t idx) {
      a.n_[k-1] = l.size();
      if (!a.data_) a.data_ = new value_type[s*l.size()];
      size_t j = 0;
      for (const auto& r : l)
        a.data_[idx+s*j++] = r;
    }

  };

  typedef typename Initializer_list<k, T>::list_type initializer_type;

  Tensor(initializer_type l) : wrapped_(), data_(nullptr)
  { Initializer_list<k, T>::process(l, *this, 1, 0); }

};

}
