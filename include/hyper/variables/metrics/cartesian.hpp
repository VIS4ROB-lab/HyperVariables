/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/cartesian.hpp"
#include "hyper/variables/metrics/abstract.hpp"

namespace hyper {

template <typename TInput>
class CartesianMetric final
    : public AbstractMetric<typename Traits<TInput>::Scalar> {
 public:
  using Scalar = typename Traits<TInput>::Scalar;
  using Input = TInput;
  using Output = TInput;

  /// Computes the distance between elements.
  /// \param lhs Left input element.
  /// \param raw_rhs Right input element.
  /// \param raw_J_lhs Jacobian w.r.t. left input.
  /// \param raw_J_rhs Jacobian w.r.t. right input.
  /// \return Distance between elements.
  static auto Distance(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, Scalar* raw_J_lhs = nullptr, Scalar* raw_J_rhs = nullptr) -> Output {
    using Jacobian = Jacobian<Output, Input>;

    if (raw_J_lhs) {
      Eigen::Map<Jacobian>{raw_J_lhs}.setIdentity();
    }

    if (raw_J_rhs) {
      Eigen::Map<Jacobian>{raw_J_rhs}.noalias() = Scalar{-1} * Jacobian::Identity();
    }

    return lhs - rhs;
  }

  /// Retrieves the metric shape (i.e. input and output size).
  /// \return Metric shape.
  [[nodiscard]] constexpr auto shape() const -> Shape final {
    return {Traits<Input>::kNumParameters, Traits<Output>::kNumParameters};
  }

  /// Retrieves the Jacobian shape.
  /// \return Jacobian shape.
  [[nodiscard]] constexpr auto jacobianShape() const -> Shape final {
    return shape();
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
    Eigen::Map<Output>{raw_output}.noalias() = Distance(lhs, rhs, raw_J_lhs, raw_J_rhs);
  }
};

} // namespace hyper
