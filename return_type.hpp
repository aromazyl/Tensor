#pragma once
#include "tensor.hpp"
namespace tensor {
template <typename... Params>
struct ReturnType;

template <class A>
struct Expr;

template <typename A, typename B, class Op>
struct BinExprOp;

template <typename A, typename B, class Op>
struct ReturnType<Expr<BinExprOp<A, B, Op>>> {
  typedef typename ReturnType<A>::result_type left_result;
  typedef typename ReturnType<B>::result_type right_result;
  typedef typename ReturnType<left_result, right_result, Op>::result_type result_type;
};

template <typename A, typename B, class Op>
struct ReturnType<A, Expr<B>, Op> {
  typedef typename ReturnType<typename B::left_type,
         typename B::right_type, typename B::operator_type>::result_type right_result;

  typedef typename ReturnType<A, right_result, Op>::result_type result_type;
};

template <typename A, typename B, class Op>
struct ReturnType<Expr<A>, B, Op> {
  typedef typename ReturnType<typename A::left_type,
          typename A::right_type, typename A::operator_type>::result_type left_result;
  typedef typename ReturnType<left_result, B, Op>::result_type result_type;
};

template <typename A, typename B, class Op>
struct ReturnType<Expr<A>, Expr<B>, Op> {
  typedef typename ResultType<typename A::left_type,
          typename A::right_type, typename A::operator_type> left_type;
  typedef typename ResultType<typename B::right_type,
          typename B::right_type, typename B::operator_type> right_type;
  typedef typename ResultType<left_result, right_type, Op>::result_type result_type;
};

template <int d, typename T, class Op>
struct ReturnType<ExprLiteral<T>, Tensor<d, T>, Op> {
  typedef Tensor<d, T> result_type;
};

template <typename T>
struct ReturnType<BinExprOp<Tensor<1, T>, EmptyType, ApTr>, Tensor<1, T>, ApMul> {
  typedef T result_type;
};

template <typename T>
struct ReturnType<Tensor<Tensor<1, T>, BinExprOp<Tensor<1, T>, EmptyType, ApTr>, ApMul> {
  typedef Tensor<2, T> result_type;
};

}
