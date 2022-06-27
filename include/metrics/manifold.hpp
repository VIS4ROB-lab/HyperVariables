/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "metrics/abstract.hpp"
#include "variables/groups/se3.hpp"

namespace hyper {

template <typename TScalar>
class ManifoldMetric<SE3<TScalar>> final
    : public AbstractMetric<TScalar> {
 public:
  using Scalar = TScalar;
  using Input = SE3<Scalar>;
  using Output = Tangent<Input>;

  ///
  /// \param coupled Compute SE3 instead of SU2 x R3 Jacobians.
  /// \param frame
  explicit ManifoldMetric(const bool coupled = false, const Frame frame = Frame::DEFAULT)
      : coupled_{coupled},
        frame_{frame} {}

  /// Computes the distance between elements.
  /// \param lhs Left input element.
  /// \param raw_rhs Right input element.
  /// \param raw_J_lhs Jacobian w.r.t. left input.
  /// \param raw_J_rhs Jacobian w.r.t. right input.
  /// \return Distance between elements.
  static auto Distance(
      const Eigen::Ref<const Input>& lhs,
      const Eigen::Ref<const Input>& rhs,
      Scalar* raw_J_lhs = nullptr,
      Scalar* raw_J_rhs = nullptr,
      const bool coupled = false,
      const Frame frame = Frame::DEFAULT) -> Output {
    using Jacobian = Jacobian<Output, Tangent<Input>>;

    if (raw_J_lhs || raw_J_rhs) {
      if (raw_J_lhs && raw_J_rhs) {
        Jacobian J_t_p, J_p_l, J_p_ir, J_ir_r;
        const auto i_rhs = rhs.groupInverse(J_ir_r.data(), coupled, frame);
        const auto lhs_plus_i_rhs = lhs.groupPlus(i_rhs, J_p_l.data(), J_p_ir.data(), coupled, frame);
        auto output = lhs_plus_i_rhs.toTangent(J_t_p.data(), coupled, frame);
        Eigen::Map<Jacobian>{raw_J_lhs}.noalias() = J_t_p * J_p_l;
        Eigen::Map<Jacobian>{raw_J_rhs}.noalias() = J_t_p * J_p_ir * J_ir_r;
        return output;

      } else {
        if (raw_J_lhs) {
          Jacobian J_t_p, J_p_l;
          const auto i_rhs = rhs.groupInverse(nullptr, coupled, frame);
          const auto lhs_plus_i_rhs = lhs.groupPlus(i_rhs, J_p_l.data(), nullptr, coupled, frame);
          auto output = lhs_plus_i_rhs.toTangent(J_t_p.data(), coupled, frame);
          Eigen::Map<Jacobian>{raw_J_lhs}.noalias() = J_t_p * J_p_l;
          return output;
        } else {
          Jacobian J_t_p, J_p_ir, J_ir_r;
          const auto i_rhs = rhs.groupInverse(J_ir_r.data(), coupled, frame);
          const auto lhs_plus_i_rhs = lhs.groupPlus(i_rhs, nullptr, J_p_ir.data(), coupled, frame);
          auto output = lhs_plus_i_rhs.toTangent(J_t_p.data(), coupled, frame);
          Eigen::Map<Jacobian>{raw_J_rhs}.noalias() = J_t_p * J_p_ir * J_ir_r;
          return output;
        }
      }
    } else {
      return (lhs.groupPlus(rhs.groupInverse())).toTangent();
    }
  }

  /// Retrieves the metric shape (i.e. input and output size).
  /// \return Metric shape.
  [[nodiscard]] constexpr auto shape() const -> Shape final {
    return {Traits<Input>::kNumParameters, Traits<Output>::kNumParameters};
  }

  /// Retrieves the Jacobian shape.
  /// \return Jacobian shape.
  [[nodiscard]] constexpr auto jacobianShape() const -> Shape final {
    return {Traits<Tangent<Input>>::kNumParameters, Traits<Output>::kNumParameters};
  }

  /// Computes the distance between elements.
  /// \param raw_output Distance between elements.
  /// \param raw_lhs Left input element.
  /// \param raw_rhs Right input element.
  /// \param raw_J_lhs Jacobian w.r.t. left input.
  /// \param raw_J_rhs Jacobian w.r.t. right input.
  auto distance(Scalar* raw_output, const Scalar* raw_lhs, const Scalar* raw_rhs, Scalar* raw_J_lhs = nullptr, Scalar* raw_J_rhs = nullptr) const -> void final {
    const auto lhs = Eigen::Map<const Input>{raw_lhs};
    const auto rhs = Eigen::Map<const Input>{raw_rhs};
    Eigen::Map<Output>{raw_output}.noalias() = Distance(lhs, rhs, raw_J_lhs, raw_J_rhs, coupled_, frame_);
  }

 private:
  bool coupled_;
  Frame frame_;
};

} // namespace hyper
