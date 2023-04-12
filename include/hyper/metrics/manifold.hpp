/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/jacobian.hpp"
#include "hyper/metrics/metric.hpp"
#include "hyper/variables/se3.hpp"

namespace hyper::metrics {

template <typename TScalar>
class ManifoldMetric<variables::SU2<TScalar>> final : public Metric<TScalar> {
 public:
  // Definitions.
  using Input = variables::SU2<TScalar>;
  using Output = variables::Tangent<variables::SU2<TScalar>>;

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
  static auto Evaluate(const Input& lhs, const Input& rhs, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) -> Output {
    Output output;
    Evaluate(lhs.data(), rhs.data(), output.data(), J_lhs, J_rhs);
    return output;
  }

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
  auto evaluate(const Input& lhs, const Input& rhs, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) const -> Output {
    return Evaluate(lhs, rhs, J_lhs, J_rhs);
  }
};

template <typename TScalar>
class ManifoldMetric<variables::SE3<TScalar>> final : public Metric<TScalar> {
 public:
  // Definitions.
  using Input = variables::SE3<TScalar>;
  using Output = variables::Tangent<variables::SE3<TScalar>>;

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
  static auto Evaluate(const Eigen::Ref<const Input>& lhs, const Eigen::Ref<const Input>& rhs, TScalar* J_lhs = nullptr, TScalar* J_rhs = nullptr) -> Output {
    Output output;
    Evaluate(lhs.data(), rhs.data(), output.data(), J_lhs, J_rhs);
    return output;
  }

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
