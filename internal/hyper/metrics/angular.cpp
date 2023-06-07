/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <Eigen/Geometry>

#include "hyper/metrics/angular.hpp"

namespace hyper::metrics {

using namespace variables;

template <int N>
auto AngularMetric<Rn<N>>::Evaluate(const Scalar* lhs, const Scalar* rhs, Scalar* d, Scalar* J_lhs, Scalar* J_rhs) -> void {
  const auto lhs_ = Eigen::Map<const Input>{lhs};
  const auto rhs_ = Eigen::Map<const Input>{rhs};

  const auto cross = lhs_.cross(rhs_).eval();
  const auto ncross = cross.norm();
  const auto dot = lhs_.dot(rhs_);

  if (J_lhs || J_rhs) {
    if (ncross < Eigen::NumTraits<Scalar>::epsilon()) {
      if (J_lhs) {
        Eigen::Map<Jacobian>{J_lhs}.setZero();
      }
      if (J_rhs) {
        Eigen::Map<Jacobian>{J_rhs}.setZero();
      }
    } else {
      const auto a = ncross * ncross + dot * dot;
      const auto b = (dot / (a * ncross));
      const auto c = (ncross / a);
      if (J_lhs) {
        Eigen::Map<Jacobian>{J_lhs}.noalias() = (b * rhs_.cross(cross) - c * rhs_).transpose();
      }
      if (J_rhs) {
        Eigen::Map<Jacobian>{J_rhs}.noalias() = (b * cross.cross(lhs_) - c * lhs_).transpose();
      }
    }
  }

  Eigen::Map<Output>{d}[0] = std::atan2(ncross, dot);
}

template <int N>
auto AngularMetric<Rn<N>>::Evaluate(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, Scalar* J_lhs, Scalar* J_rhs) -> Rn<1> {
  Output output;
  Evaluate(lhs.data(), rhs.data(), output.data(), J_lhs, J_rhs);
  return output;
}

template <int N>
auto AngularMetric<Rn<N>>::evaluate(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs, Scalar* J_rhs) -> void {
  Evaluate(lhs, rhs, output, J_lhs, J_rhs);
}

template class AngularMetric<R3>;

}  // namespace hyper::metrics
