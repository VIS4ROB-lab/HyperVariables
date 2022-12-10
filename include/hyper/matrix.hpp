/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

namespace hyper {

// clang-format off

constexpr auto DefaultMatrixStorageOption(const int rows, const int cols) -> int {
  return Eigen::AutoAlign | ((rows == 1 && cols != 1) ? Eigen::RowMajor : ((cols == 1 && rows != 1) ? Eigen::ColMajor : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION)); // NOLINT
}

template <typename TScalar, int TNumRows, int TNumCols = TNumRows, int TOptions = DefaultMatrixStorageOption(TNumRows, TNumCols)>
using TMatrix = Eigen::Matrix<TScalar, TNumRows, TNumCols, TOptions>;

template <typename TDerived, typename TOtherDerived = TDerived, int TOptions = DefaultMatrixStorageOption(TDerived::SizeAtCompileTime, TOtherDerived::SizeAtCompileTime)>
using TMatrixNN = TMatrix<typename TDerived::Scalar, TDerived::SizeAtCompileTime, TOtherDerived::SizeAtCompileTime, TOptions>;

template <typename TDerived, int TOptions = DefaultMatrixStorageOption(TDerived::SizeAtCompileTime, Eigen::Dynamic)>
using TMatrixNX = TMatrix<typename TDerived::Scalar, TDerived::SizeAtCompileTime, Eigen::Dynamic, TOptions>;

template <typename TOtherDerived, int TOptions = DefaultMatrixStorageOption(Eigen::Dynamic, TOtherDerived::SizeAtCompileTime)>
using TMatrixXN = TMatrix<typename TOtherDerived::Scalar, Eigen::Dynamic, TOtherDerived::SizeAtCompileTime, TOptions>;

template <typename TScalar, int TOptions = DefaultMatrixStorageOption(Eigen::Dynamic, Eigen::Dynamic)>
using TMatrixX = TMatrix<TScalar, Eigen::Dynamic, Eigen::Dynamic, TOptions>;

// clang-format on

} // namespace hyper
