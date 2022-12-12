/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/metrics/metric.hpp"

namespace hyper {

template <typename TScalar, int TDim>
class TCartesianMetric final : public TMetric<TScalar> {
 public:
  // Constants.
  static constexpr auto kInputDim = TDim;
  static constexpr auto kOutputDim = TDim;

  // Definitions.
  using Scalar = TScalar;
  using Input = hyper::Cartesian<Scalar, kInputDim>;
  using Output = hyper::Cartesian<Scalar, kOutputDim>;
  using Jacobian = hyper::Jacobian<Scalar, kOutputDim, kInputDim>;

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  static auto Distance(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs = nullptr,
      Scalar* J_rhs = nullptr) -> void {
    if (J_lhs) {
      Eigen::Map<Jacobian>{J_lhs}.setIdentity();
    }

    if (J_rhs) {
      Eigen::Map<Jacobian>{J_rhs}.noalias() = Scalar{-1} * Jacobian::Identity();
    }

    Eigen::Map<Output>{output}.noalias() = Eigen::Map<const Input>{lhs} - Eigen::Map<const Input>{rhs};
  }

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  /// \return Distance between elements.
  static auto Distance(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, Scalar* J_lhs = nullptr,
      Scalar* J_rhs = nullptr) -> Output {
    Output output;
    Distance(lhs.data(), rhs.data(), output.data(), J_lhs, J_rhs);
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
  auto distance(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs, Scalar* J_rhs) -> void final {
    Distance(lhs, rhs, output, J_lhs, J_rhs);
  }

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param J_lhs Jacobian w.r.t. left element (optional).
  /// \param J_rhs Jacobian w.r.t. right element (optional).
  /// \return Distance between elements.
  auto distance(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, Scalar* J_lhs = nullptr,
      Scalar* J_rhs = nullptr) const -> Output {
    return Distance(lhs, rhs, J_lhs, J_rhs);
  }
};

} // namespace hyper
