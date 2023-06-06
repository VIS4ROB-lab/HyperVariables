/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#if HYPER_COMPILE_WITH_CERES

#include <ceres/manifold.h>

#include "hyper/ceres/manifolds/forward.hpp"

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
  /// Creates a constancy mask (i.e. all parameters are held constant).
  /// \param num_parameters Number of parameters.
  /// \return Constancy mask.
  static auto ConstancyMask(int num_parameters) -> std::vector<int>;

  [[nodiscard]] auto AmbientSize() const -> int final;

  [[nodiscard]] auto TangentSize() const -> int final;

  auto Plus(const double* x, const double* delta, double* x_plus_delta) const -> bool final;

  auto PlusJacobian(const double* x, double* jacobian) const -> bool final;

  auto RightMultiplyByPlusJacobian(const double* x, int num_rows, const double* ambient_matrix, double* tangent_matrix) const -> bool final;

  auto Minus(const double* y, const double* x, double* y_minus_x) const -> bool final;

  auto MinusJacobian(const double* x, double* jacobian) const -> bool final;

 protected:
  /// Protected constructor from input manifold.
  /// \param manifold Input manifold.
  explicit ManifoldWrapper(std::unique_ptr<::ceres::Manifold>&& manifold);

 private:
  std::unique_ptr<::ceres::Manifold> manifold_;  ///< Manifold.
};

}  // namespace hyper::ceres::manifolds

#endif
