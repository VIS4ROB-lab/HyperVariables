/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#ifdef HYPER_COMPILE_WITH_CERES

#include "hyper/variables/groups/forward.hpp"

#include "hyper/manifolds/ceres/su2.hpp"

namespace hyper::manifolds::ceres {

template <>
class Manifold<variables::SE3<double>> final : public ManifoldWrapper {
 public:
  // Definitions.
  using SE3 = variables::SE3<Scalar>;

  /// Constructor from constancy flags.
  /// \param rotation_constant Rotation constancy flag.
  /// \param translation_constant Translation constancy flag.
  explicit Manifold(const bool rotation_constant = false, const bool translation_constant = false) : ManifoldWrapper{CreateManifold(rotation_constant, translation_constant)} {}

 private:
  /// Creates a (partially) constant or non-constant manifold.
  /// \param rotation_constant Rotation constancy flag.
  /// \param translation_constant Translation constancy flag.
  /// \return Manifold.
  static auto CreateManifold(bool rotation_constant, bool translation_constant) -> std::unique_ptr<::ceres::Manifold>;
};

}  // namespace hyper::manifolds::ceres

#endif