/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/metrics/metric.hpp"
#include "hyper/variables/groups/groups.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper::metrics {

template <typename TScalar>
class ManifoldMetric<variables::SE3<TScalar>> final : public Metric<TScalar> {
 public:
  // Definitions.
  using Input = variables::SE3<TScalar>;
  using Output = variables::Tangent<Input>;
  using Jacobian = variables::JacobianNM<Output>;

  // Constants.
  static constexpr auto kInputSize = Input::kNumParameters;
  static constexpr auto kOutputSize = Output::kNumParameters;

  /// Default constructor.
  explicit ManifoldMetric() = default;

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  static auto Distance(const TScalar* lhs, const TScalar* rhs, TScalar* output, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) -> void {
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
  static auto Distance(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) -> Output {
    Output output;
    Distance(lhs.data(), rhs.data(), output.data(), J_lhs, J_rhs);
    return output;
  }

  /// Retrieves the input size.
  /// \return Input size.
  [[nodiscard]] constexpr auto inputSize() const -> int final { return kInputSize; };

  /// Retrieves the output size.
  /// \return Output size.
  [[nodiscard]] constexpr auto outputSize() const -> int final { return kOutputSize; };

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  auto distance(const TScalar* lhs, const TScalar* rhs, TScalar* output, TScalar* J_lhs, TScalar* J_rhs) -> void final { Distance(lhs, rhs, output, J_lhs, J_rhs); }

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  /// \return Distance between elements.
  auto distance(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) const -> Output {
    return Distance(lhs, rhs, J_lhs, J_rhs);
  }
};

}  // namespace hyper::metrics
