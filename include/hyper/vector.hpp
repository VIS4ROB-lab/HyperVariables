/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

namespace hyper {

// clang-format off

constexpr auto DefaultVectorStorageOption(const int rows, const int cols) -> int {
  return Eigen::AutoAlign | ((rows == 1 && cols != 1) ? Eigen::RowMajor : ((cols == 1 && rows != 1) ? Eigen::ColMajor : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION)); // NOLINT
}

// clang-format on

template <typename TScalar, int TNumRows, int TOptions = DefaultVectorStorageOption(TNumRows, 1)>
using TVector = Eigen::Matrix<TScalar, TNumRows, 1, TOptions>;

template <typename TDerived, int TOptions = DefaultVectorStorageOption(TDerived::SizeAtCompileTime, 1)>
using TVectorN = TVector<typename TDerived::Scalar, TDerived::SizeAtCompileTime, TOptions>;

template <typename TScalar, int TOptions = DefaultVectorStorageOption(Eigen::Dynamic, 1)>
using TVectorX = TVector<TScalar, TOptions>;

} // namespace hyper
