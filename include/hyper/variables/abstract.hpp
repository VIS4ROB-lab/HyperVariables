/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#include "hyper/vector.hpp"

namespace hyper::variables {

template <typename TScalar>
class ConstAbstractVariable {
 public:
  // Definitions.
  using Scalar = TScalar;

  /// Virtual default destructor.
  virtual ~ConstAbstractVariable() = default;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() const -> Eigen::Map<const TVectorX<Scalar>> = 0;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() -> Eigen::Map<const TVectorX<TScalar>> = 0;
};

template <typename TScalar>
class AbstractVariable {
 public:
  // Definitions.
  using Scalar = TScalar;

  /// Virtual default destructor.
  virtual ~AbstractVariable() = default;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() const -> Eigen::Map<const TVectorX<Scalar>> = 0;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() -> Eigen::Map<TVectorX<TScalar>> = 0;
};

} // namespace hyper::variables
