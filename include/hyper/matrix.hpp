/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

#include "hyper/definitions.hpp"

namespace hyper {

constexpr auto DefaultStorageOption(const int rows, const int cols) -> int {
  return Eigen::AutoAlign | ((rows == 1 && cols != 1) ? Eigen::RowMajor : ((cols == 1 && rows != 1) ? Eigen::ColMajor : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION));  // NOLINT
}

template <int TNumRows>
using Vector = Eigen::Matrix<Scalar, TNumRows, 1>;

template <typename TDerived>
using VectorN = Vector<TDerived::SizeAtCompileTime>;

using VectorX = Vector<Eigen::Dynamic>;

template <int TNumRows, int TNumCols = TNumRows, int TOptions = DefaultStorageOption(TNumRows, TNumCols)>
using Matrix = Eigen::Matrix<Scalar, TNumRows, TNumCols, TOptions>;

template <typename TDerived, typename TOtherDerived = TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, TOtherDerived::SizeAtCompileTime)>
using MatrixNM = Matrix<TDerived::SizeAtCompileTime, TOtherDerived::SizeAtCompileTime, TOptions>;

template <typename TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, Eigen::Dynamic)>
using MatrixNX = Matrix<TDerived::SizeAtCompileTime, Eigen::Dynamic, TOptions>;

template <typename TOtherDerived, int TOptions = DefaultStorageOption(Eigen::Dynamic, TOtherDerived::SizeAtCompileTime)>
using MatrixXN = Matrix<Eigen::Dynamic, TOtherDerived::SizeAtCompileTime, TOptions>;

template <int TOptions = DefaultStorageOption(Eigen::Dynamic, Eigen::Dynamic)>
using MatrixX = Matrix<Eigen::Dynamic, Eigen::Dynamic, TOptions>;

template <int TNumRows, int TNumCols = TNumRows, int TOptions = DefaultStorageOption(TNumRows, TNumCols)>
using Jacobian = Matrix<TNumRows, TNumCols, TOptions>;

template <typename TDerived, typename TOtherDerived = TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, TOtherDerived::SizeAtCompileTime)>
using JacobianNM = MatrixNM<TDerived, TOtherDerived, TOptions>;

template <typename TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, Eigen::Dynamic)>
using JacobianNX = MatrixNX<TDerived, TOptions>;

template <typename TOtherDerived, int TOptions = DefaultStorageOption(Eigen::Dynamic, TOtherDerived::SizeAtCompileTime)>
using JacobianXN = MatrixXN<TOtherDerived, TOptions>;

template <int TOptions = DefaultStorageOption(Eigen::Dynamic, Eigen::Dynamic)>
using JacobianX = MatrixX<TOptions>;

}  // namespace hyper
