/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper::metrics {

template <typename TScalar>
class Metric;

template <typename TScalar, int TSize>
class AngularMetric;

template <typename TScalar, int TSize>
class CartesianMetric;

template <typename TManifold>
class ManifoldMetric;

template <typename TScalar>
using PixelMetric = CartesianMetric<TScalar, variables::Pixel<TScalar>::kNumParameters>;

template <typename TScalar>
using BearingMetric = AngularMetric<TScalar, variables::Bearing<TScalar>::kNumParameters>;

template <typename TManifold>
using TangentMetric = CartesianMetric<typename TManifold::Scalar, variables::Tangent<TManifold>::kNumParameters>;

template <typename TManifold>
using InertialMetric = TangentMetric<TManifold>;

}  // namespace hyper::metrics