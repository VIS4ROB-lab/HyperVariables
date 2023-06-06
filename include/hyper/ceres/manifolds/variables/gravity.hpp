/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#if HYPER_COMPILE_WITH_CERES

#include "hyper/ceres/manifolds/variables/bearing.hpp"
#include "hyper/variables/gravity.hpp"

namespace hyper::ceres::manifolds {

template <>
class Manifold<variables::Gravity> final : public Manifold<variables::Bearing> {
 public:
  // Definitions.
  using Bearing = variables::Bearing;
  using Gravity = variables::Gravity;

  /// Constructor from constancy flag.
  /// \param constant Constancy flag.
  explicit Manifold(bool constant = false);
};

}  // namespace hyper::ceres::manifolds

#endif
