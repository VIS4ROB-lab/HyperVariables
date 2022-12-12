/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/matrix.hpp"

namespace hyper {

// clang-format off

template <typename TScalar, int TNumRows, int TNumCols = TNumRows, int TOptions = DefaultMatrixStorageOption(TNumRows, TNumCols)>
using TJacobian = TMatrix<TScalar, TNumRows, TNumCols, TOptions>;

template <typename TDerived, typename TOtherDerived = TDerived, int TOptions = DefaultMatrixStorageOption(TDerived::SizeAtCompileTime, TOtherDerived::SizeAtCompileTime)>
using TJacobianNM = TMatrixNN<TDerived, TOtherDerived, TOptions>;

template <typename TDerived, int TOptions = DefaultMatrixStorageOption(TDerived::SizeAtCompileTime, Eigen::Dynamic)>
using TJacobianNX = TMatrixNX<TDerived, TOptions>;

template <typename TOtherDerived, int TOptions = DefaultMatrixStorageOption(Eigen::Dynamic, TOtherDerived::SizeAtCompileTime)>
using TJacobianXN = TMatrixXN<TOtherDerived, TOptions>;

template <typename TScalar, int TOptions = DefaultMatrixStorageOption(Eigen::Dynamic, Eigen::Dynamic)>
using TJacobianX = TMatrixX<TScalar, TOptions>;

// clang-format on

} // namespace hyper