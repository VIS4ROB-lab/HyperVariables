/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#ifdef HYPER_COMPILE_WITH_CERES

#include "hyper/manifolds/ceres/bearing.hpp"

namespace hyper::manifolds::ceres {

template <>
class Manifold<variables::Gravity<double>> final : public Manifold<variables::Bearing<double>> {
 public:
  // Definitions.
  using Bearing = variables::Bearing<Scalar>;
  using Gravity = variables::Gravity<Scalar>;

  /// Constructor from constancy flag.
  /// \param constant Constancy flag.
  explicit Manifold(bool constant = false);
};

}  // namespace hyper::manifolds::ceres

#endif
