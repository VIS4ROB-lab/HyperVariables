/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/groups/forward.hpp"

namespace hyper::metrics {

template <typename TScalar>
class Metric;

template <typename TScalar, int TDim>
class AngularMetric;

template <typename TScalar, int TDim>
class CartesianMetric;

template <typename TManifold>
class ManifoldMetric;

}  // namespace hyper::metrics