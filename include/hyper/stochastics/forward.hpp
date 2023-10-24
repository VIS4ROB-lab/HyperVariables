/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#include "hyper/matrix.hpp"

namespace hyper {

namespace stochastics {

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
class Uncertainty;

template <typename TDerived>
using UncertaintyN = Uncertainty<TDerived::kNumParameters>;

using UncertaintyX = Uncertainty<Eigen::Dynamic>;

template <int TOrder>
class Gaussian;

template <typename TDerived>
using GaussianN = Gaussian<TDerived::kNumParameters>;

using GaussianX = Gaussian<Eigen::Dynamic>;

template <int TOrder>
class InverseGaussian;

template <typename TDerived>
using InverseGaussianN = InverseGaussian<TDerived::kNumParameters>;

using InverseGaussianX = InverseGaussian<Eigen::Dynamic>;

}  // namespace stochastics

template <int TOrder>
struct Traits<stochastics::Gaussian<TOrder>> {
  // Constants.
  static constexpr auto kOrder = TOrder;

  // Definitions.
  using Base = Matrix<TOrder, (TOrder != Eigen::Dynamic) ? (TOrder + 1) : Eigen::Dynamic>;
};

template <int TOrder>
struct Traits<stochastics::InverseGaussian<TOrder>> {
  // Constants.
  static constexpr auto kOrder = TOrder;

  // Definitions.
  using Base = Matrix<TOrder, (TOrder != Eigen::Dynamic) ? (TOrder + 1) : Eigen::Dynamic>;
};

}  // namespace hyper
