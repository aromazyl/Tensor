#pragma once

namespace tensor {
template <int k, typename T, class TensorType>
class TensorTraits;

template <typename T, class TensorType>
class TensorTraits<1, T, TensorType> {
public:
  size_t dim() const {
    return static_cast<TensorType&>(*this).n_[0];
  }
};

template <typename T, class TensorType>
class TensorTraits<2, T, TensorType> {
public:
  size_t rows() const {
    return static_cast<TensorType&>(*this).n_[0];
  }
  size_t columns() const {
    return static_cast<TensorType&>(*this).n_[1];
  }
};

template <typename T, class TensorType>
struct TensorTraits<3, T, TensorType> {
  size_t rows() const {
    return static_cast<TensorType&>(*this).n_[0];
  }

  size_t columns() const {
    return static_cast<TensorType&>(*this).n_[1];
  }

  size_t depth() const {
    return static_cast<TensorType&>(*this).n_[2];
  }
};

template <typename T, class TensorType>
struct TensorTraits<4, T, TensorType> {
  size_t batch_size() const {
    return static_cast<TensorType&>(*this).n_[0];
  }
  size_t rows() const {
    return static_cast<TensorType&>(*this).n_[1];
  }

  size_t columns() const {
    return static_cast<TensorType&>(*this).n_[2];
  }
  size_t depth() const {
    return static_cast<TensorType&>(*this).n_[3];
  }
};

struct False {
  enum { val = 0 };
}

template <int k, typename T, class TensorType>
struct TensorTraits : False {
  static_assert(val, "Cannot support tensor dim other than {1 2 3 4}.");
};

}
