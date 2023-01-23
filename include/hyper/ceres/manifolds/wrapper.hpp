/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#ifdef HYPER_COMPILE_WITH_CERES

#include <ceres/manifold.h>

#include "hyper/ceres/manifolds/variables/forward.hpp"

namespace hyper::ceres::manifolds {

/// @class Manifold wrapper for Ceres.
/// This wrapper is required due to some
/// manifolds being marked as final within
/// Ceres, rendering them non-inheritable.
/// We refer to the Ceres documentation
/// for all inherited and overloaded
/// functions in this class.
class ManifoldWrapper : public ::ceres::Manifold {
 public:
  using Scalar = double;  // Ceres scalar.

  /// Creates a constancy mask (i.e. all parameters are held constant).
  /// \param num_parameters Number of parameters.
  /// \return Constancy mask.
  static auto ConstancyMask(int num_parameters) -> std::vector<int>;

  [[nodiscard]] auto AmbientSize() const -> int final;

  [[nodiscard]] auto TangentSize() const -> int final;

  auto Plus(const Scalar* x, const Scalar* delta, Scalar* x_plus_delta) const -> bool final;

  auto PlusJacobian(const Scalar* x, Scalar* jacobian) const -> bool final;

  auto RightMultiplyByPlusJacobian(const Scalar* x, int num_rows, const Scalar* ambient_matrix, Scalar* tangent_matrix) const -> bool final;

  auto Minus(const Scalar* y, const Scalar* x, Scalar* y_minus_x) const -> bool final;

  auto MinusJacobian(const Scalar* x, Scalar* jacobian) const -> bool final;

 protected:
  /// Protected constructor from input manifold.
  /// \param manifold Input manifold.
  explicit ManifoldWrapper(std::unique_ptr<::ceres::Manifold>&& manifold);

 private:
  std::unique_ptr<::ceres::Manifold> manifold_;  ///< Manifold.
};

}  // namespace hyper::ceres::manifolds

#endif
