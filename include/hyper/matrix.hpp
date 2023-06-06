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

template <int TNumRows, int TNumCols = TNumRows>
using Matrix = Eigen::Matrix<Scalar, TNumRows, TNumCols, DefaultStorageOption(TNumRows, TNumCols)>;

template <typename TDerived, typename TOtherDerived = TDerived>
using MatrixNM = Matrix<TDerived::SizeAtCompileTime, TOtherDerived::SizeAtCompileTime>;

template <typename TDerived>
using MatrixNX = Matrix<TDerived::SizeAtCompileTime, Eigen::Dynamic>;

template <typename TOtherDerived>
using MatrixXN = Matrix<Eigen::Dynamic, TOtherDerived::SizeAtCompileTime>;

using MatrixX = Matrix<Eigen::Dynamic, Eigen::Dynamic>;

template <int TNumRows, int TNumCols = TNumRows>
using Jacobian = Matrix<TNumRows, TNumCols>;

template <typename TDerived, typename TOtherDerived = TDerived>
using JacobianNM = MatrixNM<TDerived, TOtherDerived>;

template <typename TDerived>
using JacobianNX = MatrixNX<TDerived>;

template <typename TOtherDerived>
using JacobianXN = MatrixXN<TOtherDerived>;

using JacobianX = MatrixX;

}  // namespace hyper
