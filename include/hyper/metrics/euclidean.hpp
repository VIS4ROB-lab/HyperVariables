/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/metrics/metric.hpp"
#include "hyper/variables/cartesian.hpp"

namespace hyper::metrics {

template <typename TScalar, int N>
class EuclideanMetric<variables::Rn<TScalar, N>> final : public Metric<TScalar> {
 public:
  // Definitions.
  using Input = variables::Rn<TScalar, N>;
  using Output = variables::Rn<TScalar, N>;

  // Constants.
  static constexpr auto kAmbientInputSize = Input::kNumParameters;
  static constexpr auto kAmbientOutputSize = Output::kNumParameters;
  static constexpr auto kTangentInputSize = variables::Tangent<Input>::kNumParameters;
  static constexpr auto kTangentOutputSize = variables::Tangent<Output>::kNumParameters;
  using Jacobian = hyper::Jacobian<TScalar, kTangentOutputSize, kTangentInputSize>;

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  static auto Evaluate(const TScalar* lhs, const TScalar* rhs, TScalar* output, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) -> void {
    if (J_lhs) {
      Eigen::Map<Jacobian>{J_lhs}.setIdentity();
    }

    if (J_rhs) {
      Eigen::Map<Jacobian>{J_rhs}.noalias() = TScalar{-1} * Jacobian::Identity();
    }

    Eigen::Map<Output>{output}.noalias() = Eigen::Map<const Input>{lhs} - Eigen::Map<const Input>{rhs};
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
