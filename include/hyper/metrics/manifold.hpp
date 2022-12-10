/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "abstract.hpp"
#include "hyper/variables/groups/se3.hpp"

namespace hyper {

template <typename TScalar>
class ManifoldMetric<SE3<TScalar>> final
    : public AbstractMetric<TScalar> {
 public:
  using Scalar = TScalar;
  using Input = SE3<Scalar>;
  using Output = Tangent<Input>;

  static constexpr auto kDefaultDerivativesAreGlobal = HYPER_DEFAULT_TO_GLOBAL_LIE_GROUP_DERIVATIVES;
  static constexpr auto kDefaultDerivativesAreCoupled = HYPER_DEFAULT_TO_COUPLED_LIE_GROUP_DERIVATIVES;

  /// Default constructor.
  /// \param global Request global Jacobians flag.
  /// \param coupled Compute SE3 instead of SU2 x R3 Jacobians.
  explicit ManifoldMetric(const bool global = kDefaultDerivativesAreGlobal, const bool coupled = kDefaultDerivativesAreCoupled)
      : global_{global},
        coupled_{coupled} {}

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
      const bool coupled = kDefaultDerivativesAreCoupled,
      const bool global = kDefaultDerivativesAreGlobal)
      -> Output {
    using Jacobian = TJacobianNM<Output, Tangent<Input>>;

    if (raw_J_lhs || raw_J_rhs) {
      if (raw_J_lhs && raw_J_rhs) {
        Jacobian J_t_p, J_p_l, J_p_ir, J_ir_r;
        const auto i_rhs = rhs.groupInverse(J_ir_r.data(), coupled, global);
        const auto lhs_plus_i_rhs = lhs.groupPlus(i_rhs, J_p_l.data(), J_p_ir.data(), coupled, global);
        auto output = lhs_plus_i_rhs.toTangent(J_t_p.data(), coupled, global);
        Eigen::Map<Jacobian>{raw_J_lhs}.noalias() = J_t_p * J_p_l;
        Eigen::Map<Jacobian>{raw_J_rhs}.noalias() = J_t_p * J_p_ir * J_ir_r;
        return output;
      } else if (raw_J_lhs) {
        Jacobian J_t_p, J_p_l;
        const auto i_rhs = rhs.groupInverse(nullptr, coupled, global);
        const auto lhs_plus_i_rhs = lhs.groupPlus(i_rhs, J_p_l.data(), nullptr, coupled, global);
        auto output = lhs_plus_i_rhs.toTangent(J_t_p.data(), coupled, global);
        Eigen::Map<Jacobian>{raw_J_lhs}.noalias() = J_t_p * J_p_l;
        return output;
      } else {
        Jacobian J_t_p, J_p_ir, J_ir_r;
        const auto i_rhs = rhs.groupInverse(J_ir_r.data(), coupled, global);
        const auto lhs_plus_i_rhs = lhs.groupPlus(i_rhs, nullptr, J_p_ir.data(), coupled, global);
        auto output = lhs_plus_i_rhs.toTangent(J_t_p.data(), coupled, global);
        Eigen::Map<Jacobian>{raw_J_rhs}.noalias() = J_t_p * J_p_ir * J_ir_r;
        return output;
      }
    } else {
      return (lhs.groupPlus(rhs.groupInverse())).toTangent();
    }
  }

  /// Retrieves the input size.
  /// \return Input size.
  [[nodiscard]] constexpr auto inputSize() const -> int final {
    return Traits<Input>::kNumParameters;
  }

  /// Retrieves the output size.
  /// \return Output size.
  [[nodiscard]] constexpr auto outputSize() const -> int final {
    return Traits<Output>::kNumParameters;
  }

  /// Computes the distance between inputs.
  /// \param lhs Left input.
  /// \param rhs Right input.
  /// \param J_lhs Jacobian w.r.t. left input.
  /// \param J_rhs Jacobian w.r.t. right input.
  /// \return Distance between inputs.
  auto distance(
      const Eigen::Ref<const DynamicVector<Scalar>>& lhs,
      const Eigen::Ref<const DynamicVector<Scalar>>& rhs,
      TJacobianX<Scalar>* J_lhs,
      TJacobianX<Scalar>* J_rhs) const
      -> DynamicVector<Scalar> final {
    if (J_lhs || J_rhs) {
      if (J_lhs && J_rhs) {
        J_lhs->resize(Traits<Output>::kNumParameters, Traits<Tangent<Input>>::kNumParameters);
        J_rhs->resize(Traits<Output>::kNumParameters, Traits<Tangent<Input>>::kNumParameters);
        return Distance(lhs, rhs, J_lhs->data(), J_rhs->data(), coupled_, global_);
      } else if (J_lhs) {
        J_lhs->resize(Traits<Output>::kNumParameters, Traits<Tangent<Input>>::kNumParameters);
        return Distance(lhs, rhs, J_lhs->data(), nullptr, coupled_, global_);
      } else {
        J_rhs->resize(Traits<Output>::kNumParameters, Traits<Tangent<Input>>::kNumParameters);
        return Distance(lhs, rhs, nullptr, J_rhs->data(), coupled_, global_);
      }
    } else {
      return Distance(lhs, rhs, nullptr, nullptr, coupled_, global_);
    }
  }

 private:
  bool global_;
  bool coupled_;
};

} // namespace hyper
