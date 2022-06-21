/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

namespace hyper {

struct MetricShape {
  using Size = int;
  Size num_inputs;
  Size num_outputs;
};

template <typename>
class AbstractMetric;

template <typename>
class CartesianMetric;

template <typename>
class AngularMetric;

template <typename>
class ManifoldMetric;

} // namespace hyper
