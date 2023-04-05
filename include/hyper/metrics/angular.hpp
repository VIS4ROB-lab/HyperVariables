/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>

#include "hyper/metrics/metric.hpp"

namespace hyper::metrics {

template <typename TScalar, int TSize>
class AngularMetric final : public Metric<TScalar> {
 public:
  // Constants.
  static constexpr auto kInputSize = TSize;
  static constexpr auto kOutputSize = 1;

  // Definitions.
  using Input = variables::Cartesian<TScalar, kInputSize>;
  using Output = variables::Cartesian<TScalar, kOutputSize>;
  using Jacobian = hyper::JacobianNM<Output, Input>;

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param d Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  static auto Distance(const TScalar* lhs, const TScalar* rhs, TScalar* d, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) -> void {
    const auto lhs_ = Eigen::Map<const Input>{lhs};
    const auto rhs_ = Eigen::Map<const Input>{rhs};

    const auto cross = lhs_.cross(rhs_).eval();
    const auto ncross = cross.norm();
    const auto dot = lhs_.dot(rhs_);

    if (J_lhs || J_rhs) {
      if (ncross < Eigen::NumTraits<TScalar>::epsilon()) {
        if (J_lhs) {
          Eigen::Map<Jacobian>{J_lhs}.setZero();
        }
        if (J_rhs) {
          Eigen::Map<Jacobian>{J_rhs}.setZero();
        }
      } else {
        const auto a = ncross * ncross + dot * dot;
        const auto b = (dot / (a * ncross));
        const auto c = (ncross / a);
        if (J_lhs) {
          Eigen::Map<Jacobian>{J_lhs}.noalias() = (b * rhs_.cross(cross) - c * rhs_).transpose();
        }
        if (J_rhs) {
          Eigen::Map<Jacobian>{J_rhs}.noalias() = (b * cross.cross(lhs_) - c * lhs_).transpose();
        }
      }
    }

    Eigen::Map<Output>{d}[0] = std::atan2(ncross, dot);
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
