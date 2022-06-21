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
  using Output = Tangent<SE3<Scalar>>;

  /// Computes the distance between elements.
  /// \param lhs Left input element.
  /// \param raw_rhs Right input element.
  /// \param raw_J_lhs Jacobian w.r.t. left input.
  /// \param raw_J_rhs Jacobian w.r.t. right input.
  /// \return Distance between elements.
  static auto Distance(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, Scalar* raw_J_lhs = nullptr, Scalar* raw_J_rhs = nullptr) -> Output {
    //auto output = Eigen::Map<Output>{raw_output};
    lhs.rotation();

    static constexpr auto kNumAngularOutputs = 3;
    static constexpr auto kNumLinearOutputs = 3;
    static constexpr auto kAngularOutputOffset = 0;
    static constexpr auto kLinearOutputOffset = kNumAngularOutputs;

    //const auto tangent = (lhs.rotation() * rhs.rotation().inverse()).log().asTangent();
    //output.segment<kNumAngularOutputs>(kAngularOutputOffset) = tangent;
    //output.segment<kNumLinearOutputs>(kNumLinearOutputs) = lhs.translation() - rhs.translation();

    /* if (raw_J_lhs || raw_J_rhs) {
      const auto J_left_inverse = tangent.leftInverseJacobian();

      if (raw_J_lhs) {
        auto J = Eigen::Map<Jacobian>{raw_J_lhs};
        J.setZero();
        J.block<kNumAngularOutputs, 3>(kAngularOutputOffset, 0) = J_left_inverse;
        J.block<kNumLinearOutputs, 3>(kLinearOutputOffset, 3) = Eigen::Matrix<Scalar, kNumLinearOutputs, 3>::Identity();
      }

      if (raw_J_rhs) {
        auto J = Eigen::Map<Jacobian>{raw_J_rhs};
        J.setZero();
        J.block<kNumAngularOutputs, 3>(kAngularOutputOffset, 0) = J_left_inverse * lhs.rotation().matrix();
        J.block<kNumLinearOutputs, 3>(kLinearOutputOffset, 3) = Scalar{-1} * Eigen::Matrix<Scalar, kNumLinearOutputs, 3>::Identity();
      }
    } */
  }

  /// Retrieves the shape (i.e. input and output size).
  /// \return Metric shape.
  [[nodiscard]] constexpr auto dimensions() const -> MetricShape final {
    return {Traits<Input>::kNumParameters, Traits<Output>::kNumParameters};
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
