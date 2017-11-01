/*
 * check.h
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CHECK_H
#define CHECK_H

#include <cuda.h>
#include <iostream>

namespace tensor {
template <typename first_type, typename... Rest>
struct CheckInternal {
  typedef first_type pack_type;
  enum { tmp = std::is_integral<first_type>::value };
  enum { value = tmp && CheckInternal<Rest...>::value };
  static_assert (value, "*** ERROR *** Non-internal type parameter found.");
};

template <typename last_type>
struct CheckInternal<last_type> {
  typedef last_type pack_type;
  enum { value = std::is_integral<last_type>::value };
};

#define checkCudaErrors(val) check(val, #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* func, char* filename, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << filename << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

}

#endif /* !CHECK_H */
