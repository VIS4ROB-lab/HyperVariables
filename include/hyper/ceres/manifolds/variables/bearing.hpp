/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#ifdef HYPER_COMPILE_WITH_CERES

#include "hyper/ceres/manifolds/wrapper.hpp"
#include "hyper/variables/bearing.hpp"

namespace hyper::ceres::manifolds {

template <>
class Manifold<variables::Bearing<double>> : public ManifoldWrapper {
 public:
  // Definitions.
  using Bearing = variables::Bearing<Scalar>;

  /// Constructor from constancy flag.
  /// \param constant Constancy flag.
  explicit Manifold(const bool constant = false) : ManifoldWrapper{CreateManifold(constant)} {}

 private:
  /// Creates a constant or non-constant manifold.
  /// \param constant Constancy flag.
  /// \return Manifold.
  static auto CreateManifold(bool constant) -> std::unique_ptr<::ceres::Manifold>;
};

}  // namespace hyper::ceres::manifolds

#endif
