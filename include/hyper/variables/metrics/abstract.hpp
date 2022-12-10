/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/metrics/forward.hpp"

#include "hyper/variables/jacobian.hpp"

namespace hyper {

template <typename TScalar>
class AbstractMetric {
 public:
  /// Default destructor.
  virtual ~AbstractMetric() = default;

  /// Retrieves the input size.
  /// \return Input size.
  [[nodiscard]] virtual auto inputSize() const -> int = 0;

  /// Retrieves the output size.
  /// \return Output size.
  [[nodiscard]] virtual auto outputSize() const -> int = 0;

  /// Computes the distance between inputs.
  /// \param lhs Left input.
  /// \param rhs Right input.
  /// \param J_lhs Jacobian w.r.t. left input.
  /// \param J_rhs Jacobian w.r.t. right input.
  /// \return Distance between inputs.
  virtual auto distance(
      const Eigen::Ref<const DynamicVector<TScalar>>& lhs,
      const Eigen::Ref<const DynamicVector<TScalar>>& rhs,
      TJacobianX<TScalar>* J_lhs,
      TJacobianX<TScalar>* J_rhs) const
      -> DynamicVector<TScalar> = 0;
};

} // namespace hyper
