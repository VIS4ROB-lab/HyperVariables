/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#include "hyper/matrix.hpp"

namespace hyper::stochastics {

enum class GaussianType { STANDARD, INFORMATION };

template <int TOrder>
class Uncertainty;

template <typename TDerived>
using UncertaintyN = Uncertainty<TDerived::kNumParameters>;

using UncertaintyX = Uncertainty<Eigen::Dynamic>;

template <int TOrder, GaussianType>
class Gaussian;

template <typename TDerived, GaussianType gaussianType>
using GaussianN = Gaussian<TDerived::kNumParameters, gaussianType>;

template <GaussianType gaussianType>
using GaussianX = Gaussian<Eigen::Dynamic, gaussianType>;

template <int TOrder>
class DualGaussian;

template <typename TDerived>
using DualGaussianN = DualGaussian<TDerived::kNumParameters>;

using DualGaussianX = DualGaussian<Eigen::Dynamic>;

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

template <typename TDerived>
using StandardGaussianN = StandardGaussian<TDerived::kNumParameters>;

template <typename TDerived>
using InformationGaussianN = InformationGaussian<TDerived::kNumParameters>;

using StandardGaussianX = StandardGaussian<Eigen::Dynamic>;

using InformationGaussianX = InformationGaussian<Eigen::Dynamic>;

template <int TOrder>
using Prior = InformationGaussian<TOrder>;

template <typename TDerived>
using PriorN = InformationGaussianN<TDerived>;

using PriorX = InformationGaussianX;

}  // namespace hyper::stochastics
