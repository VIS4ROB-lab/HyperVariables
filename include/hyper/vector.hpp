/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

namespace hyper {

constexpr auto DefaultVectorStorageOption(const int rows, const int cols) -> int {
  return Eigen::AutoAlign | ((rows == 1 && cols != 1) ? Eigen::RowMajor : ((cols == 1 && rows != 1) ? Eigen::ColMajor : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION));  // NOLINT
}

template <typename TScalar, int TNumRows, int TOptions = DefaultVectorStorageOption(TNumRows, 1)>
using Vector = Eigen::Matrix<TScalar, TNumRows, 1, TOptions>;

template <typename TDerived, int TOptions = DefaultVectorStorageOption(TDerived::SizeAtCompileTime, 1)>
using VectorN = Vector<typename TDerived::Scalar, TDerived::SizeAtCompileTime, TOptions>;

template <typename TScalar, int TOptions = DefaultVectorStorageOption(Eigen::Dynamic, 1)>
using VectorX = Vector<TScalar, Eigen::Dynamic, TOptions>;

}  // namespace hyper
