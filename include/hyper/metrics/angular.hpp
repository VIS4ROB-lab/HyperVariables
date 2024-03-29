/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>

#include "hyper/metrics/metric.hpp"

namespace hyper::metrics {

template <typename TScalar, int N>
class AngularMetric<variables::Rn<TScalar, N>> final : public Metric<TScalar> {
 public:
  // Definitions.
  using Input = variables::Rn<TScalar, N>;
  using Output = variables::R1<TScalar>;

  // Constants.
  static constexpr auto kAmbientInputSize = Input::kNumParameters;
  static constexpr auto kAmbientOutputSize = Output::kNumParameters;
  static constexpr auto kTangentInputSize = variables::Tangent<Input>::kNumParameters;
  static constexpr auto kTangentOutputSize = variables::Tangent<Output>::kNumParameters;
  using Jacobian = hyper::Jacobian<TScalar, kTangentOutputSize, kTangentInputSize>;

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param d Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  static auto Evaluate(const TScalar* lhs, const TScalar* rhs, TScalar* d, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) -> void {
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
  static auto Evaluate(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) -> Output {
    Output output;
    Evaluate(lhs.data(), rhs.data(), output.data(), J_lhs, J_rhs);
    return output;
  }

  /// Retrieves the ambient input size.
  /// \return Ambient input size.
  [[nodiscard]] constexpr auto ambientInputSize() const -> int final { return kAmbientInputSize; }

  /// Retrieves the ambient output size.
  /// \return Ambient output size.
  [[nodiscard]] constexpr auto ambientOutputSize() const -> int final { return kAmbientOutputSize; }

  /// Retrieves the tangent input size.
  /// \return Tangent input size.
  [[nodiscard]] constexpr auto tangentInputSize() const -> int final { return kTangentInputSize; }

  /// Retrieves the tangent output size.
  /// \return Tangent output size.
  [[nodiscard]] constexpr auto tangentOutputSize() const -> int final { return kTangentOutputSize; }

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  auto evaluate(const TScalar* lhs, const TScalar* rhs, TScalar* output, TScalar* J_lhs, TScalar* J_rhs) -> void final { Evaluate(lhs, rhs, output, J_lhs, J_rhs); }

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  /// \return Distance between elements.
  auto evaluate(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) const -> Output {
    return Evaluate(lhs, rhs, J_lhs, J_rhs);
  }
};

}  // namespace hyper::metrics
