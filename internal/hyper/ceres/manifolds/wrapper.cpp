/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#if HYPER_COMPILE_WITH_CERES

#include <numeric>

#include "hyper/ceres/manifolds/wrapper.hpp"

namespace hyper::ceres::manifolds {

auto ManifoldWrapper::ConstancyMask(const int num_parameters) -> std::vector<int> {
  std::vector<int> mask;
  mask.resize(num_parameters);
  std::iota(mask.begin(), mask.end(), 0);
  return mask;
}

auto ManifoldWrapper::AmbientSize() const -> int {
  return manifold_->AmbientSize();
}

auto ManifoldWrapper::TangentSize() const -> int {
  return manifold_->TangentSize();
}

auto ManifoldWrapper::Plus(const double* x, const double* delta, double* x_plus_delta) const -> bool {
  return manifold_->Plus(x, delta, x_plus_delta);
}

auto ManifoldWrapper::PlusJacobian(const double* x, double* jacobian) const -> bool {
  return manifold_->PlusJacobian(x, jacobian);
}

auto ManifoldWrapper::RightMultiplyByPlusJacobian(const double* x, const int num_rows, const double* ambient_matrix, double* tangent_matrix) const -> bool {
  return manifold_->RightMultiplyByPlusJacobian(x, num_rows, ambient_matrix, tangent_matrix);
}

auto ManifoldWrapper::Minus(const double* y, const double* x, double* y_minus_x) const -> bool {
  return manifold_->Minus(y, x, y_minus_x);
}

auto ManifoldWrapper::MinusJacobian(const double* x, double* jacobian) const -> bool {
  return manifold_->MinusJacobian(x, jacobian);
}

ManifoldWrapper::ManifoldWrapper(std::unique_ptr<::ceres::Manifold>&& manifold) : manifold_{std::move(manifold)} {}

}  // namespace hyper::ceres::manifolds

#endif
