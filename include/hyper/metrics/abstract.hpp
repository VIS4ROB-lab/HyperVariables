/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/metrics/forward.hpp"

#include "hyper/variables/jacobian.hpp"

namespace hyper {

template <typename TScalar>
class AbstractMetric {
 public:
  /// Default destructor.
  virtual ~AbstractMetric() = default;

  /// Retrieves the metric shape (i.e. input and output size).
  /// \return Metric shape.
  [[nodiscard]] virtual auto shape() const -> Shape = 0;

  /// Retrieves the Jacobian shape.
  /// \return Jacobian shape.
  [[nodiscard]] virtual auto jacobianShape() const -> Shape = 0;

  /// Computes the distance between elements.
  /// \param raw_output Distance between elements.
  /// \param raw_lhs Left input element.
  /// \param raw_rhs Right input element.
  /// \param raw_J_lhs Jacobian w.r.t. left input.
  /// \param raw_J_rhs Jacobian w.r.t. right input.
  virtual auto distance(TScalar* raw_output, const TScalar* raw_lhs, const TScalar* raw_rhs, TScalar* raw_J_lhs, TScalar* raw_J_rhs) const -> void = 0;
};

} // namespace hyper
