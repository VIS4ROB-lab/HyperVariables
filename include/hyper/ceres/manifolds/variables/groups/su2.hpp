/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#ifdef HYPER_COMPILE_WITH_CERES

#include "hyper/ceres/manifolds/wrapper.hpp"
#include "hyper/variables/groups/su2.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper::ceres::manifolds {

template <>
class Manifold<variables::SU2<double>> final : public ManifoldWrapper {
 public:
  // Definitions.
  using SU2 = variables::SU2<Scalar>;

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
