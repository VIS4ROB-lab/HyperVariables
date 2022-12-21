/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper {

template <typename, int>
class Gaussian;

template <typename TScalar, int TOrder>
struct Traits<Gaussian<TScalar, TOrder>> {
  using Base = Eigen::Matrix<TScalar, TOrder, (TOrder != Eigen::Dynamic) ? (TOrder + 1) : Eigen::Dynamic>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::Gaussian, int)

template <typename, int>
class CanonicalGaussian;

template <typename TScalar, int TOrder>
struct Traits<CanonicalGaussian<TScalar, TOrder>> {
  using Base = Eigen::Matrix<TScalar, TOrder, (TOrder != Eigen::Dynamic) ? (TOrder + 1) : Eigen::Dynamic>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::CanonicalGaussian, int)

} // namespace hyper
