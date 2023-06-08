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

template <typename TGroup>
class GroupMetric;

}  // namespace hyper::metrics