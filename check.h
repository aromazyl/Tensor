/*
 * check.h
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CHECK_H
#define CHECK_H

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
}

#endif /* !CHECK_H */
