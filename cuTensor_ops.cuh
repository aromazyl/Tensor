#pragma once

#include "cuTensor.cuh"
#include "operators_expr.hpp"

namespace tensor {

class cuApAdd {
public:
  template <int d, typename T>
  static cuTensor<d, T>& apply(cuTensor<d, T>& a, const cuTensor<d, T>& b)
};
}
