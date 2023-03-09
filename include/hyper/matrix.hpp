/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

namespace hyper {

constexpr auto DefaultStorageOption(const int rows, const int cols) -> int {
  return Eigen::AutoAlign | ((rows == 1 && cols != 1) ? Eigen::RowMajor : ((cols == 1 && rows != 1) ? Eigen::ColMajor : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION));  // NOLINT
}

template <typename TScalar, int TNumRows, int TNumCols = TNumRows, int TOptions = DefaultStorageOption(TNumRows, TNumCols)>
using Matrix = Eigen::Matrix<TScalar, TNumRows, TNumCols, TOptions>;

template <typename TDerived, typename TOtherDerived = TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, TOtherDerived::SizeAtCompileTime)>
using MatrixNM = Matrix<typename TDerived::Scalar, TDerived::SizeAtCompileTime, TOtherDerived::SizeAtCompileTime, TOptions>;

template <typename TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, Eigen::Dynamic)>
using MatrixNX = Matrix<typename TDerived::Scalar, TDerived::SizeAtCompileTime, Eigen::Dynamic, TOptions>;

template <typename TOtherDerived, int TOptions = DefaultStorageOption(Eigen::Dynamic, TOtherDerived::SizeAtCompileTime)>
using MatrixXN = Matrix<typename TOtherDerived::Scalar, Eigen::Dynamic, TOtherDerived::SizeAtCompileTime, TOptions>;

template <typename TScalar, int TOptions = DefaultStorageOption(Eigen::Dynamic, Eigen::Dynamic)>
using MatrixX = Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic, TOptions>;

}  // namespace hyper
