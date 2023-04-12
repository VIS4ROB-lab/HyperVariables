/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper::metrics {

template <typename TScalar>
class Metric;

template <typename TInput>
class AngularMetric;

template <typename TInput, typename TOutput = TInput>
class EuclideanMetric;

template <typename TInput, typename TOutput = variables::Tangent<TInput>>
class ManifoldMetric;

}  // namespace hyper::metrics