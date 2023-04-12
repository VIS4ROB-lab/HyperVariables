/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/metrics/forward.hpp"

#include "hyper/jacobian.hpp"
#include "hyper/vector.hpp"

namespace hyper::metrics {

template <typename TScalar>
class Metric {
 public:
  // Definitions.
  using Scalar = TScalar;

  /// Default destructor.
  virtual ~Metric() = default;

  /// Retrieves the ambient input size.
  /// \return Ambient input size.
  [[nodiscard]] virtual auto ambientInputSize() const -> int = 0;

  /// Retrieves the ambient output size.
  /// \return Ambient output size.
  [[nodiscard]] virtual auto ambientOutputSize() const -> int = 0;

  /// Retrieves the tangent input size.
  /// \return Tangent input size.
  [[nodiscard]] virtual auto tangentInputSize() const -> int = 0;

  /// Retrieves the tangent output size.
  /// \return Tangent output size.
  [[nodiscard]] virtual auto tangentOutputSize() const -> int = 0;

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element.
  /// \param J_rhs Jacobian w.r.t. right element.
  virtual auto evaluate(const TScalar* lhs, const TScalar* rhs, TScalar* output, TScalar* J_lhs, TScalar* J_rhs) -> void = 0;
};

}  // namespace hyper::metrics