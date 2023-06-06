/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper::metrics {

class Metric;

template <typename TInput>
class AngularMetric;

template <typename TInput>
class EuclideanMetric;

class SU2Metric;
class SE3Metric;

}  // namespace hyper::metrics