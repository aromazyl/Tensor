#pragma once

#include "return_type.hpp"

namespace tensor {
template <class A>
class Expr {
  A a_;

public:
  typedef A expression_type;
  typedef typename A::result_type result_type;
  Expr() : a_() {}
  Expr(const A& x) : a_(x) {}

  auto left() const -> decltype(a_.left()) { return a_.left(); }
  auto right() const -> decltype(a_.right()) { return a_.right(); }

  operator result_type() { return a_(); }
  result_type operator()() const { return a_(); }
  friend inline std::ostream& operator<<(std::ostream& os, const Expr<A>& expr);
};

template <class A, class B, class Op>
class BinExprOp {
  typename ExprTraits<A>::type a_;
  typename ExprTraits<B>::type b_;

public:
  typedef A left_type;
  typedef B right_type;
  typedef Op operator_type;
  typedef typename ReturnType<left_type, right_type, operator_type>::result_type result_type;
  BinExprOp(const A& a, const B& b) : a_(a), b_(b) {}
  auto left() const -> decltype(a_) { return a_; }
  auto right() const -> decltype(b_) { return b_; }
  auto operator()() const -> decltype(Op::apply(a_, b_))
  { return operator_type::apply(a_, b_); }
  auto operator()() const -> decltype(Op::apply(a_, b_))
  { return operator_type::apply(a_, b_); }
};

template <typename T>
using SMm = SAm<2, T>;

class ApMul {
public:
  template <typename T>
  static matrix_type<T> apply(const SMn<T>& x, const SMm<T>& y) {
    const matrix_type<T>& a = x.right();
    const matrix_type<T>& b = y.right();
    matrix_type<T> r(a.rows(), b.columns());
    // TODO {CUDA}
    return r;
  }
};

template <class A, class B>
static typename ReturnType<Expr<A>, Expr<B>, ApMul>::result_type
apply(const Expr<A>& a, const Expr<B>& b)
{ return a() * b(); }

template <class A, class B>
typename enable_if<!is_arithmetic<A>::value && !is_arithmetic<B>::value, A&>::type
operator+=(A& a, const B& b) {
  typedef RefBinExpr<A, B, ApAdd> ExprT;
  return Expr<ExprT>(ExprT(a, b))();
};

template <typename T>
using SMmSMmm = Expr<BinExprOp<SMm<T>, SMm<T>, ApMul>>;

class ApAdd {
public:
  template <typename T>
  static matrix_type<T>& apply(matrix_type<T>& c, const SMmSMmm<T>& y) {
    const matrix_type<T>& a = y.left().right();
    const matrix_type<T>& b = y.right().right();
    // TODO {CUDA}
    return c;
  }
};

}
