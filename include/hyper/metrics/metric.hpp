/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/metrics/forward.hpp"

#include "hyper/variables/definitions/jacobian.hpp"
#include "hyper/vector.hpp"

namespace hyper::metrics {

template <typename TScalar>
class TMetric {
 public:
  // Definitions.
  using Scalar = TScalar;
  using Input = TVectorX<Scalar>;
  using Output = TVectorX<Scalar>;

  /// Default destructor.
  virtual ~TMetric() = default;

  /// Retrieves the input dimension.
  /// \return Input dimension.
  [[nodiscard]] virtual auto inputDim() const -> int = 0;

  /// Retrieves the output dimension.
  /// \return Output dimension.
  [[nodiscard]] virtual auto outputDim() const -> int = 0;

  /// Evaluates the distance between elements.
  /// \param lhs Left element/input vector.
  /// \param rhs Right element/input vector.
  /// \param output Distance between elements.
  /// \param J_lhs Jacobian w.r.t. left element.
  /// \param J_rhs Jacobian w.r.t. right element.
  virtual auto distance(const Scalar* lhs, const Scalar* rhs, Scalar* output, Scalar* J_lhs, Scalar* J_rhs) -> void = 0;
};

} // namespace hyper::metrics