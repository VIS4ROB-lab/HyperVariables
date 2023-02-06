/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/matrix.hpp"

namespace hyper::variables {

template <typename TScalar, int TDim, int TOptions = DefaultStorageOption(TDim, TDim)>
using Covariance = Matrix<TScalar, TDim, TDim, TOptions>;

template <typename TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, TDerived::SizeAtCompileTime)>
using CovarianceN = MatrixNN<TDerived, TDerived, TOptions>;

template <typename TScalar, int TOptions = DefaultStorageOption(Eigen::Dynamic, Eigen::Dynamic)>
using CovarianceX = MatrixX<TScalar, TOptions>;

template <typename TScalar, int TDim, int TOptions = DefaultStorageOption(TDim, TDim)>
using Precision = Matrix<TScalar, TDim, TDim, TOptions>;

template <typename TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, TDerived::SizeAtCompileTime)>
using PrecisionN = MatrixNN<TDerived, TDerived, TOptions>;

template <typename TScalar, int TOptions = DefaultStorageOption(Eigen::Dynamic, Eigen::Dynamic)>
using PrecisionX = MatrixX<TScalar, TOptions>;

}  // namespace hyper::variables