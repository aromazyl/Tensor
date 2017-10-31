#pragma once

namespace tensor {
template <int k, typename T, class TensorType>
class TensorTraits;

template <typename T, class TensorType>
class TensorTraits<1, T, TensorType> {
public:
  size_t dim() const;
};

template <typename T, class TensorType>
class TensorTraits<2, T, TensorType> {
public:
  size_t rows() const;
  size_t columns() const;
};

template <typename T, class TensorType>
struct TensorTraits<4, T, TensorType> {
  size_t batch_size() const;
  size_t rows() const;
  size_t columns() const;
  size_t depth() const;
};

template <typename T, class TensorType>
struct TensorTraits<3, T, TensorType> {
  size_t rows() const;
  size_t columns() const;
  size_t depth() const;
};

struct False {
  enum { val = 0 };
}

template <int k, typename T, class TensorType>
struct TensorTraits : False {
  static_assert(val, "Cannot support tensor dim other than {1 2 3 4}.");
};

}
