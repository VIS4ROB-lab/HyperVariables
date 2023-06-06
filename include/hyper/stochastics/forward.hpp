/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#include "hyper/matrix.hpp"

namespace hyper::stochastics {

enum class GaussianType { STANDARD, INFORMATION };

template <int TOrder>
class Uncertainty;

template <int TOrder, GaussianType>
class Gaussian;

template <int TOrder>
class DualGaussian;

template <int TSize, int TOptions = DefaultStorageOption(TSize, TSize)>
using Covariance = Matrix<TSize, TSize, TOptions>;

template <typename TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, TDerived::SizeAtCompileTime)>
using CovarianceN = MatrixNM<TDerived, TDerived, TOptions>;

template <int TOptions = DefaultStorageOption(Eigen::Dynamic, Eigen::Dynamic)>
using CovarianceX = MatrixX<TOptions>;

template <int TSize, int TOptions = DefaultStorageOption(TSize, TSize)>
using Precision = Matrix<TSize, TSize, TOptions>;

template <typename TDerived, int TOptions = DefaultStorageOption(TDerived::SizeAtCompileTime, TDerived::SizeAtCompileTime)>
using PrecisionN = MatrixNM<TDerived, TDerived, TOptions>;

template <int TOptions = DefaultStorageOption(Eigen::Dynamic, Eigen::Dynamic)>
using PrecisionX = MatrixX<TOptions>;

}  // namespace hyper::stochastics
