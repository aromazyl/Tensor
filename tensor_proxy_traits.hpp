#pragma once

template <int d, class Tensor>
struct TensorProxy;

template <int d, class Tensor>
struct TensorProxyTraits {
  typedef TensorProxy<d-1, Tensor> reference;
  typedef const TensorProxy<d-1, Tensor> value_type;

  static reference get_reference(Tensor& a, size_t i) {
    return reference(a, i);
  }

  static value_type value(Tensor& a, size_t i) {
    return value_type(a, i);
  }
};

template <class Tensor>
struct TensorProxyTraits<1, Tensor> {
  typedef typename Tensor::value_type primitive_type;
  typedef primitive_type& reference;
  typedef primitive_type const & value_type;

  static reference get_reference(Tensor& a, size_t i) {
    return a.data_[i];
  }

  static value_type value(Tensor& a, size_t i)
  { return a.data_[i]; }

};

template <int d, class Tensor>
struct TensorProxy {
  typedef const TensorProxy<d-1, Tensor> value_type;
  typedef TensorProxy<d-1, Tensor> reference;

  explicit TensorProxy(const Tensor& a, size_t i)
    : a_(a), i_(i), s_(a.n_[0]) {}
  template <int c> TensorProxy(const TensorProxy<c, Tensor>& a, size_t i)
    : a_(a.a_), i_(a.i_ + a.s_ * i), s_(a.s_ * a.a_.n_[Tensor::rank() - c]) {}

  reference operator[](size_t i)
  { return reference(*this, i); }
  value_type operator[](size_t i) const
  { return value_type(*this, i); }
  const Tensor& a_;
  size_t i_, s_;
};

template <class Tensor>
struct TensorProxy<1, Tensor> {
  typedef typename Tensor::reference reference;
  typedef typename Tensor::value_type value_type;

  explicit TensorProxy(const Tensor& a, size_t i)
    : a_(a), i_(i), s_(a.n_[0]) {}
  explicit TensorProxy(const TensorProxy<c, Tensor>& a, size_t i)
    : a_(a.a_), i_(a.i_ + a.s_ * i), s_(a.s_ * a.a_.n_[Tensor::rank() - c]) {}

  reference operator[](size_t i)
  { return a_.data_[i_ + i * s_]; }
  value_type operator[](size_t i) const
  { return a_.data_[i_ + i * s_]; }

private:
  const Tensor& a_;
  size_t i_, s_;

};
