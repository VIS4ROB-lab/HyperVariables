/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#include "hyper/vector.hpp"

namespace hyper {

template <typename TScalar>
class AbstractVariable {
 public:
  // Definitions.
  using Scalar = std::remove_const_t<TScalar>;
  using VectorXWithConstIfNotLvalue = std::conditional_t<std::is_const_v<TScalar>, const TVectorX<Scalar>, TVectorX<Scalar>>;

  /// Virtual default destructor.
  virtual ~AbstractVariable() = default;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() const -> Eigen::Map<const TVectorX<Scalar>> = 0;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() -> Eigen::Map<VectorXWithConstIfNotLvalue> = 0;
};

} // namespace hyper
