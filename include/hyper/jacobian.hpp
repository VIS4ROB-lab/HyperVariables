/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/matrix.hpp"

namespace hyper {

template <typename TScalar, int TNumRows, int TNumCols = TNumRows, int TOptions = DefaultStorageOption(TNumRows, TNumCols)>
using Jacobian = Matrix<TScalar, TNumRows, TNumCols, TOptions>;

template <typename TDerived, typename TOtherDerived = TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, TOtherDerived::SizeAtCompileTime)>
using JacobianNM = MatrixNM<TDerived, TOtherDerived, TOptions>;

template <typename TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, Eigen::Dynamic)>
using JacobianNX = MatrixNX<TDerived, TOptions>;

template <typename TOtherDerived, int TOptions = DefaultStorageOption(Eigen::Dynamic, TOtherDerived::SizeAtCompileTime)>
using JacobianXN = MatrixXN<TOtherDerived, TOptions>;

template <typename TScalar, int TOptions = DefaultStorageOption(Eigen::Dynamic, Eigen::Dynamic)>
using JacobianX = MatrixX<TScalar, TOptions>;

}  // namespace hyper