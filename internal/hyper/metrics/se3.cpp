/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/metrics/se3.hpp"

namespace hyper::metrics {

auto SE3Metric::Evaluate(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs, Scalar* J_rhs) -> void {
  const auto lhs_ = Eigen::Map<const Input>{lhs};
  const auto rhs_ = Eigen::Map<const Input>{rhs};
  auto output_ = Eigen::Map<Output>{output};

  if (!J_lhs && !J_rhs) {
    output_ = lhs_.gPlus(rhs_.gInv()).gLog();
  } else if (J_lhs && J_rhs) {
    Jacobian J_t_p, J_p_l, J_p_ir, J_ir_r;
    output_ = lhs_.gPlus(rhs_.gInv(J_ir_r.data()), J_p_l.data(), J_p_ir.data()).gLog(J_t_p.data());
    Eigen::Map<Jacobian>{J_lhs}.noalias() = J_t_p * J_p_l;
    Eigen::Map<Jacobian>{J_rhs}.noalias() = J_t_p * J_p_ir * J_ir_r;
  } else if (J_lhs) {
    Jacobian J_t_p, J_p_l;
    output_ = lhs_.gPlus(rhs_.gInv(), J_p_l.data()).gLog(J_t_p.data());
    Eigen::Map<Jacobian>{J_lhs}.noalias() = J_t_p * J_p_l;
  } else {
    Jacobian J_t_p, J_p_ir, J_ir_r;
    output_ = lhs_.gPlus(rhs_.gInv(J_ir_r.data()), nullptr, J_p_ir.data()).gLog(J_t_p.data());
    Eigen::Map<Jacobian>{J_rhs}.noalias() = J_t_p * J_p_ir * J_ir_r;
  }
}

auto SE3Metric::Evaluate(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, Scalar* J_lhs, Scalar* J_rhs) -> Output {
  Output output;
  Evaluate(lhs.data(), rhs.data(), output.data(), J_lhs, J_rhs);
  return output;
}

auto SE3Metric::evaluate(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs, Scalar* J_rhs) -> void {
  Evaluate(lhs, rhs, output, J_lhs, J_rhs);
}

}  // namespace hyper::metrics
