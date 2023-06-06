/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/metrics/metric.hpp"
#include "hyper/variables/se3.hpp"

namespace hyper::metrics {

class SE3Metric final : public Metric {
 public:
  // Definitions.
  using Input = variables::SE3<Scalar>;
  using Output = variables::Tangent<variables::SE3<Scalar>>;
  using InputTangent = variables::Tangent<Input>;
  using OutputTangent = variables::Tangent<Output>;
  using Jacobian = hyper::Jacobian<Scalar, OutputTangent::kNumParameters, InputTangent::kNumParameters>;

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  static auto Evaluate(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs = nullptr, Scalar* J_rhs = nullptr) -> void {
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

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  /// \return Distance between elements.
  static auto Evaluate(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, Scalar* J_lhs = nullptr, Scalar* J_rhs = nullptr) -> Output {
    Output output;
    Evaluate(lhs.data(), rhs.data(), output.data(), J_lhs, J_rhs);
    return output;
  }

  /// Retrieves the ambient input size.
  /// \return Ambient input size.
  [[nodiscard]] constexpr auto ambientInputSize() const -> int final { return Input::kNumParameters; }

  /// Retrieves the ambient output size.
  /// \return Ambient output size.
  [[nodiscard]] constexpr auto ambientOutputSize() const -> int final { return Output::kNumParameters; }

  /// Retrieves the tangent input size.
  /// \return Tangent input size.
  [[nodiscard]] constexpr auto tangentInputSize() const -> int final { return InputTangent::kNumParameters; }

  /// Retrieves the tangent output size.
  /// \return Tangent output size.
  [[nodiscard]] constexpr auto tangentOutputSize() const -> int final { return OutputTangent::kNumParameters; }

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  auto evaluate(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs, Scalar* J_rhs) -> void final { Evaluate(lhs, rhs, output, J_lhs, J_rhs); }
};

}  // namespace hyper::metrics
