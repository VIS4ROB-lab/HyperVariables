/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/metrics/metric.hpp"
#include "hyper/variables/rn.hpp"

namespace hyper::metrics {

template <int N>
class EuclideanMetric<variables::Rn<N>> final : public Metric {
 public:
  // Definitions.
  using Input = variables::Rn<N>;
  using Output = variables::Rn<N>;
  using InputTangent = variables::Tangent<Input>;
  using OutputTangent = variables::Tangent<Output>;
  using Jacobian = hyper::JacobianNM<OutputTangent, InputTangent>;

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  static auto Evaluate(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs = nullptr, Scalar* J_rhs = nullptr) -> void;

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  /// \return Distance between elements.
  static auto Evaluate(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, Scalar* J_lhs = nullptr, Scalar* J_rhs = nullptr) -> Output;

  /// Retrieves the ambient input size.
  /// \return Ambient input size.
  [[nodiscard]] constexpr auto ambientInputSize() const -> int final { return Input::kNumParameters; }

  /// Retrieves the ambient output size.
  /// \return Ambient output size.
  [[nodiscard]] constexpr auto ambientOutputSize() const -> int final { return Output::kNumParameters; }

  /// Retrieves the tangent input size.
  /// \return Tangent input size.
  [[nodiscard]] constexpr auto tangentInputSize() const -> int final { return InputTangent::kNumParameters; }

  /// Retrieves the tangent output size.
  /// \return Tangent output size.
  [[nodiscard]] constexpr auto tangentOutputSize() const -> int final { return OutputTangent::kNumParameters; }

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  auto evaluate(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs, Scalar* J_rhs) -> void final;
};

}  // namespace hyper::metrics
