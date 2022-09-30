/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper {

template <typename TScalar>
class AbstractVariable {
 public:
  // Definitions.
  using Scalar = std::remove_const_t<TScalar>;
  using DynamicVectorWithConstIfNotLvalue = std::conditional_t<std::is_const_v<TScalar>, const DynamicVector<Scalar>, DynamicVector<Scalar>>;

  /// Virtual default destructor.
  virtual ~AbstractVariable() = default;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() const -> Eigen::Map<const DynamicVector<Scalar>> = 0;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() -> Eigen::Map<DynamicVectorWithConstIfNotLvalue> = 0;
};

} // namespace hyper
