/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/metrics/metric.hpp"
#include "hyper/variables/groups/se3.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper::metrics {

template <typename TScalar>
class ManifoldMetric<SE3<TScalar>> final : public Metric<TScalar> {
 public:
  // Definitions.
  using Input = SE3<TScalar>;
  using Output = Tangent<Input>;
  using Jacobian = JacobianNM<Output>;

  // Constants.
  static constexpr auto kInputDim = Input::kNumParameters;
  static constexpr auto kOutputDim = Output::kNumParameters;

  static constexpr auto kGlobal = HYPER_DEFAULT_TO_GLOBAL_LIE_GROUP_DERIVATIVES;
  static constexpr auto kCoupled = HYPER_DEFAULT_TO_COUPLED_LIE_GROUP_DERIVATIVES;

  /// Default constructor.
  /// \param global Request global Jacobians flag.
  /// \param coupled Compute SE3 instead of SU2 x R3 Jacobians.
  explicit ManifoldMetric(const bool global = kGlobal, const bool coupled = kCoupled) : global_{global}, coupled_{coupled} {}

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  static auto Distance(const TScalar* lhs, const TScalar* rhs, TScalar* output, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr, const bool global = kGlobal,
                       const bool coupled = kCoupled) -> void {
    const auto lhs_ = Eigen::Map<const Input>{lhs};
    const auto rhs_ = Eigen::Map<const Input>{rhs};
    auto output_ = Eigen::Map<Output>{output};

    if (J_lhs || J_rhs) {
      if (J_lhs && J_rhs) {
        Jacobian J_t_p, J_p_l, J_p_ir, J_ir_r;
        const auto i_rhs = rhs_.groupInverse(J_ir_r.data(), global, coupled);
        const auto lhs_plus_i_rhs = lhs_.groupPlus(i_rhs, J_p_l.data(), J_p_ir.data(), global, coupled);
        output_ = lhs_plus_i_rhs.toTangent(J_t_p.data(), global, coupled);
        Eigen::Map<Jacobian>{J_lhs}.noalias() = J_t_p.lazyProduct(J_p_l);
        Eigen::Map<Jacobian>{J_rhs}.noalias() = J_t_p.lazyProduct(J_p_ir.lazyProduct(J_ir_r));
      } else if (J_lhs) {
        Jacobian J_t_p, J_p_l;
        const auto i_rhs = rhs_.groupInverse(nullptr, global, coupled);
        const auto lhs_plus_i_rhs = lhs_.groupPlus(i_rhs, J_p_l.data(), nullptr, global, coupled);
        output_ = lhs_plus_i_rhs.toTangent(J_t_p.data(), global, coupled);
        Eigen::Map<Jacobian>{J_lhs}.noalias() = J_t_p.lazyProduct(J_p_l);
      } else {
        Jacobian J_t_p, J_p_ir, J_ir_r;
        const auto i_rhs = rhs_.groupInverse(J_ir_r.data(), global, coupled);
        const auto lhs_plus_i_rhs = lhs_.groupPlus(i_rhs, nullptr, J_p_ir.data(), global, coupled);
        output_ = lhs_plus_i_rhs.toTangent(J_t_p.data(), global, coupled);
        Eigen::Map<Jacobian>{J_rhs}.noalias() = J_t_p.lazyProduct(J_p_ir.lazyProduct(J_ir_r));
      }
    } else {
      output_ = lhs_.groupPlus(rhs_.groupInverse()).toTangent();
    }
  }

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  /// \return Distance between elements.
  static auto Distance(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr, const bool global = kGlobal,
                       const bool coupled = kCoupled) -> Output {
    Output output;
    Distance(lhs.data(), rhs.data(), output.data(), J_lhs, J_rhs, global, coupled);
    return output;
  }

  /// Retrieves the input dimension.
  /// \return Input dimension.
  [[nodiscard]] constexpr auto inputDim() const -> int final { return kInputDim; };

  /// Retrieves the output dimension.
  /// \return Output dimension.
  [[nodiscard]] constexpr auto outputDim() const -> int final { return kOutputDim; };

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  auto distance(const TScalar* lhs, const TScalar* rhs, TScalar* output, TScalar* J_lhs, TScalar* J_rhs) -> void final {
    Distance(lhs, rhs, output, J_lhs, J_rhs, global_, coupled_);
  }

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  /// \return Distance between elements.
  auto distance(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) const -> Output {
    return Distance(lhs, rhs, J_lhs, J_rhs, global_, coupled_);
  }

 private:
  bool global_;
  bool coupled_;
};

}  // namespace hyper::metrics
