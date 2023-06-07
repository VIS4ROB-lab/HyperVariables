/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/metrics/euclidean.hpp"

namespace hyper::metrics {

using namespace variables;

template <int N>
auto EuclideanMetric<Rn<N>>::Evaluate(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs, Scalar* J_rhs) -> void {
  if (J_lhs) {
    Eigen::Map<Jacobian>{J_lhs}.setIdentity();
  }

  if (J_rhs) {
    Eigen::Map<Jacobian>{J_rhs}.noalias() = Scalar{-1} * Jacobian::Identity();
  }

  Eigen::Map<Output>{output}.noalias() = Eigen::Map<const Input>{lhs} - Eigen::Map<const Input>{rhs};
}

template <int N>
auto EuclideanMetric<Rn<N>>::Evaluate(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, Scalar* J_lhs, Scalar* J_rhs) -> Rn<N> {
  Output output;
  Evaluate(lhs.data(), rhs.data(), output.data(), J_lhs, J_rhs);
  return output;
}

template <int N>
auto EuclideanMetric<Rn<N>>::evaluate(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs, Scalar* J_rhs) -> void {
  Evaluate(lhs, rhs, output, J_lhs, J_rhs);
}

template class EuclideanMetric<R2>;
template class EuclideanMetric<R3>;
template class EuclideanMetric<R6>;

}  // namespace hyper::metrics
