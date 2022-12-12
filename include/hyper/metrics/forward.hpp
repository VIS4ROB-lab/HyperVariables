/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/groups/forward.hpp"

namespace hyper::metrics {

template <typename TScalar>
class TMetric;

template <typename TScalar, int TDim>
class TAngularMetric;

template <typename TScalar, int TDim>
class TCartesianMetric;

template <typename TScalar, variables::ManifoldEnum TManifoldEnum>
class TManifoldMetric;

} // namespace hyper::metrics