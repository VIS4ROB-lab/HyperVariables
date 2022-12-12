/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/vector.hpp"

namespace hyper {

// clang-format off

constexpr auto DefaultResidualStorageOption(const int rows, const int cols) -> int {
  return Eigen::AutoAlign | ((rows == 1 && cols != 1) ? Eigen::RowMajor : ((cols == 1 && rows != 1) ? Eigen::ColMajor : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION)); // NOLINT
}

// clang-format on

template <typename TScalar, int TNumRows, int TOptions = DefaultResidualStorageOption(TNumRows, 1)>
using TResidual = TVector<TScalar, TNumRows, TOptions>;

template <typename TDerived, int TOptions = DefaultResidualStorageOption(TDerived::SizeAtCompileTime, 1)>
using TResidualN = TVectorN<TDerived, TOptions>;

template <typename TScalar, int TOptions = DefaultResidualStorageOption(Eigen::Dynamic, 1)>
using TResidualX = TVectorX<TScalar, TOptions>;

} // namespace hyper