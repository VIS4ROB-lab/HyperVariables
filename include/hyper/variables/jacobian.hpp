/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper {

constexpr auto DefaultMatrixStorageOptionOrder(const int rows, const int cols) -> int {
  return Eigen::AutoAlign | ((rows == 1 && cols != 1)      ? Eigen::RowMajor
                                : (cols == 1 && rows != 1) ? Eigen::ColMajor // NOLINT
                                                           : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION);
}

template <typename TScalar, int NumRows, int NumCols = NumRows, int TOptions = DefaultMatrixStorageOptionOrder(NumRows, NumCols)> // NOLINT
using SizedJacobian = Eigen::Matrix<TScalar, NumRows, NumCols, TOptions>;

template <typename TDerived, typename TOtherDerived = TDerived, int TOptions = DefaultMatrixStorageOptionOrder(Traits<TDerived>::kNumParameters, Traits<TOtherDerived>::kNumParameters), typename TScalar = typename Traits<TDerived>::Scalar> // NOLINT
using Jacobian = SizedJacobian<TScalar, Traits<TDerived>::kNumParameters, Traits<TOtherDerived>::kNumParameters, TOptions>;

template <typename TDerived, int TOptions = DefaultMatrixStorageOptionOrder(Traits<TDerived>::kNumParameters, Eigen::Dynamic), typename TScalar = typename Traits<TDerived>::Scalar> // NOLINT
using DynamicInputJacobian = SizedJacobian<TScalar, Traits<TDerived>::kNumParameters, Eigen::Dynamic, TOptions>;

template <typename TDerived, int TOptions = DefaultMatrixStorageOptionOrder(Eigen::Dynamic, Traits<TDerived>::kNumParameters), typename TScalar = typename Traits<TDerived>::Scalar> // NOLINT
using DynamicOutputJacobian = SizedJacobian<TScalar, Eigen::Dynamic, Traits<TDerived>::kNumParameters, TOptions>;

template <typename TScalar, int TOptions = DefaultMatrixStorageOptionOrder(Eigen::Dynamic, Eigen::Dynamic)> // NOLINT
using DynamicJacobian = SizedJacobian<TScalar, Eigen::Dynamic, Eigen::Dynamic, TOptions>;

} // namespace hyper
