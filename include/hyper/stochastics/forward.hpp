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

template <int TSize>
using Covariance = Matrix<TSize, TSize>;

template <typename TDerived>
using CovarianceN = MatrixNM<TDerived, TDerived>;

using CovarianceX = MatrixX;

template <int TSize>
using Precision = Matrix<TSize, TSize>;

template <typename TDerived>
using PrecisionN = MatrixNM<TDerived, TDerived>;

using PrecisionX = MatrixX;

template <int TOrder>
using StandardGaussian = Gaussian<TOrder, GaussianType::STANDARD>;

template <int TOrder>
using InformationGaussian = Gaussian<TOrder, GaussianType::INFORMATION>;

}  // namespace hyper::stochastics
