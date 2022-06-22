/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>

#include "metrics/abstract.hpp"
#include "variables/cartesian.hpp"

namespace hyper {

template <typename TInput>
class AngularMetric final
    : public AbstractMetric<typename TInput::Scalar> {
 public:
  using Scalar = typename Traits<TInput>::Scalar;
  using Input = TInput;
  using Output = Cartesian<Scalar, 1>;

  /// Computes the distance between elements.
  /// \param lhs Left input element.
  /// \param raw_rhs Right input element.
  /// \param raw_J_lhs Jacobian w.r.t. left input.
  /// \param raw_J_rhs Jacobian w.r.t. right input.
  /// \return Distance between elements.
  static auto Distance(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, Scalar* raw_J_lhs = nullptr, Scalar* raw_J_rhs = nullptr) -> Output {
    using Jacobian = Jacobian<Output, Input>;

    const auto nl2 = lhs.squaredNorm();
    const auto nr2 = rhs.squaredNorm();
    const auto n2 = nl2 * nr2;
    DLOG_IF(WARNING, n2 < Eigen::NumTraits<Scalar>::epsilon()) << "Singularity detected.";
    const auto i_n = Scalar{1} / std::sqrt(n2);

    const auto d = lhs.dot(rhs);
    const auto x = i_n * d;

    if (raw_J_lhs || raw_J_rhs) {
      const auto a = (Scalar{1} <= x + Eigen::NumTraits<Scalar>::epsilon()) ? Scalar{0} : Scalar{-i_n} / std::sqrt(Scalar{1} - x * x);
      if (raw_J_lhs) {
        Eigen::Map<Jacobian>{raw_J_lhs}.noalias() = a * (rhs.transpose() - d * lhs.transpose() / nl2);
      }
      if (raw_J_rhs) {
        Eigen::Map<Jacobian>{raw_J_rhs}.noalias() = a * (lhs.transpose() - d * rhs.transpose() / nr2);
      }
    }

    return Output{std::acos(x)};
  }

  /// Retrieves the metric shape (i.e. input and output size).
  /// \return Metric shape.
  [[nodiscard]] constexpr auto shape() const -> Shape final {
    return {Traits<Input>::kNumParameters, 1};
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
