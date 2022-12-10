/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>

#include "abstract.hpp"
#include "hyper/variables/cartesian.hpp"

namespace hyper {

template <typename TInput>
class AngularMetric final
    : public AbstractMetric<typename TInput::Scalar> {
 public:
  using Input = TInput;
  using Scalar = typename Traits<Input>::Scalar;
  using Output = Cartesian<Scalar, 1>;

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
      Scalar* raw_J_rhs = nullptr)
      -> Output {
    using Jacobian = TJacobianNM<Output, Input>;

    const auto cross = lhs.cross(rhs).eval();
    const auto ncross = cross.norm();
    const auto dot = lhs.dot(rhs);

    if (raw_J_lhs || raw_J_rhs) {
      if (ncross < Eigen::NumTraits<Scalar>::epsilon()) {
        if (raw_J_lhs) {
          Eigen::Map<Jacobian>{raw_J_lhs}.setZero();
        }
        if (raw_J_rhs) {
          Eigen::Map<Jacobian>{raw_J_rhs}.setZero();
        }
      } else {
        const auto a = ncross * ncross + dot * dot;
        const auto b = (dot / (a * ncross));
        const auto c = (ncross / a);
        if (raw_J_lhs) {
          Eigen::Map<Jacobian>{raw_J_lhs}.noalias() = (b * rhs.cross(cross) - c * rhs).transpose();
        }
        if (raw_J_rhs) {
          Eigen::Map<Jacobian>{raw_J_rhs}.noalias() = (b * cross.cross(lhs) - c * lhs).transpose();
        }
      }
    }

    return Output{std::atan2(ncross, dot)};
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
      const Eigen::Ref<const TVectorX<Scalar>>& lhs,
      const Eigen::Ref<const TVectorX<Scalar>>& rhs,
      TJacobianX<Scalar>* J_lhs = nullptr,
      TJacobianX<Scalar>* J_rhs = nullptr) const
      -> TVectorX<Scalar> final {
    if (J_lhs || J_rhs) {
      if (J_lhs && J_rhs) {
        J_lhs->resize(Traits<Output>::kNumParameters, Traits<Input>::kNumParameters);
        J_rhs->resize(Traits<Output>::kNumParameters, Traits<Input>::kNumParameters);
        return Distance(lhs, rhs, J_lhs->data(), J_rhs->data());
      } else if (J_lhs) {
        J_lhs->resize(Traits<Output>::kNumParameters, Traits<Input>::kNumParameters);
        return Distance(lhs, rhs, J_lhs->data(), nullptr);
      } else {
        J_rhs->resize(Traits<Output>::kNumParameters, Traits<Input>::kNumParameters);
        return Distance(lhs, rhs, nullptr, J_rhs->data());
      }
    } else {
      return Distance(lhs, rhs, nullptr, nullptr);
    }
  }
};

} // namespace hyper
