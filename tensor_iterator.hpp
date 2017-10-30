#pragma once

#include "tensor.hpp"
#include "tensor_proxy_traits.hpp"
#include "tensor_traits.hpp"

namespace tensor {
template <typename T, typename P, int d>
struct TensorIterator : public std::iterator<std::random_access_iterator_tag, T, ptrdiff_t, P> {
  typedef P pointer;
  TensorIterator(T* x, size_t str) : p_(x), str_(str) {}

  TensorIterator& operator++()
  { p_ += str_; return *this; }

  TensorIterator operator++(int)
  { TensorIterator it(p_); p_ += str_; return it; }

  template <int d>
  using diterator = TensorIterator<value_type, pointer, d>;

  template <int d> diterator<d> dbegin()
  { return diterator<d>(data_, stride(d)); }

  template <int d> diterator<d> dend()
  { size_t s = stride(d); return diterator<d>(data_ + stride(d+1), s); }

  template <int d, typename iterator>
  diterator<d> dbegin(iterator it) { return diterator<d>(&*it, stride(d)); }

  template <int d, typename iterator> diterator<d> dend(iterator it)
  { size_t s = stride(d); return diterator<d>(&*it + stride(d+1), s); }


private:
  pointer p_;
  size_t str_;

};

}
