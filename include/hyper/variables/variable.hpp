/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#include "hyper/vector.hpp"

namespace hyper::variables {

template <typename TScalar>
class ConstVariable {
 public:
  // Definitions.
  using Scalar = TScalar;

  /// Virtual default destructor.
  virtual ~ConstVariable() = default;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() const -> Eigen::Ref<const VectorX<Scalar>> = 0;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() -> Eigen::Ref<const VectorX<TScalar>> = 0;
};

template <typename TScalar>
class Variable {
 public:
  // Definitions.
  using Scalar = TScalar;

  /// Virtual default destructor.
  virtual ~Variable() = default;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() const -> Eigen::Ref<const VectorX<Scalar>> = 0;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() -> Eigen::Ref<VectorX<TScalar>> = 0;
};

}  // namespace hyper::variables
