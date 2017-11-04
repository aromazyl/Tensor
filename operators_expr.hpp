#pragma once

#include <iostream>
#include <typeinfo>
#include <utility>
#include <type_traits>

#include "return_type.hpp"

namespace tensor {

template <typename T>
struct ExprIdentity {
  typedef T value_type;
  typedef T result_type;
  value_type operator()(T x) const
  { return x; }
};

template <typename T>
struct ExprLiteral {
  typedef T value_type;
  ExprLiteral(value_type value) : value_(value) {}
  template <typename... Args>
  value_type operator()(Args... params) const {
    return value_;
  }

private:
  value_type value_;
};

template <class A>
class Expr {
  A a_;

public:
  typedef A expression_type;
  typedef typename A::result_type result_type;
  Expr() : a_() {}
  Expr(const A& x) : a_(x) {}

  const A& expr() const { return this->a_; }

  auto left() const -> decltype(a_.left()) { return a_.left(); }
  auto right() const -> decltype(a_.right()) { return a_.right(); }

  operator result_type() { return a_(); }
  result_type operator()() const { return a_(); }
  // friend inline std::ostream& operator<<(std::ostream& os, const Expr<A>& expr);
};

template <typename T>
struct ExprTraits;

template <class A>
struct ExprTraits<Expr<A>> {
  typedef Expr<A> type;
};

template <typename T>
struct ExprTraits<ExprLiteral<T>> {
  typedef ExprLiteral<T> type;
};

template <int k, typename T>
struct Tensor;

template <int k, typename T>
struct cuTensor;

template <int d, typename T, template <int k, typename T> TensorType>
struct ExprTraits<TensorType<d, T>> {
  typedef const TensorType<d, T>& type;
};

template <>
struct ExprTraits<EmptyType> {
  typedef EmptyType type;
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

template <class A, class B, class Op>
struct RefBinExprOp {
  typedef A& reference_type;
  reference_type a_;
  B b_;

  typedef reference_type left_type;
  typedef B right_type;
  typedef reference_type result_type;

  RefBinExprOp(reference_type a, const B& b)
    : a_(a), b_(b) {}

  left_type left() const
  { return a_; }
  const right_type& right() const
  { return b_; }

  reference_type operator()() const {
    return Op::apply(a_, b_);
  }

  reference_type operator()(double x) const {
    return Op::apply(a_(x), b_(x));
  }
};

template <class A, class B>
static typename ReturnType<Expr<A>, Expr<B>, ApMul>::result_type
apply(const Expr<A>& a, const Expr<B>& b)
{ return a() * b(); }

// template <class A, class B>
// typename enable_if<!is_arithmetic<A>::value && !is_arithmetic<B>::value, A&>::type
// operator+=(A& a, const B& b) {
//   typedef RefBinExpr<A, B, ApAdd> ExprT;
//   return Expr<ExprT>(ExprT(a, b))();
// };
// 
// template <typename T>
// using SMmSMmm = Expr<BinExprOp<SMm<T>, SMm<T>, ApMul>>;
// 
// class ApAdd {
// public:
//   template <typename T>
//   static matrix_type<T>& apply(matrix_type<T>& c, const SMmSMmm<T>& y) {
//     const matrix_type<T>& a = y.left().right();
//     const matrix_type<T>& b = y.right().right();
//     // TODO {CUDA}
//     return c;
//   }
// };

}
