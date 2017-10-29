#pragma once

namespace tensor {
template <int k, typename T, class TensorType>
class TensorTraits;

template <typename T, class TensorType>
class TensorTraits<1, T, TensorType> {
public:
  T norm() const;
};

template <typename T, class TensorType>
class TensorTraits<2, T, TensorType> {
public:
  size_t rows() const;
  size_t columns() const;
};
}
