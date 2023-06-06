/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#include "hyper/matrix.hpp"

namespace hyper::variables {

class ConstVariable {
 public:
  /// Virtual default destructor.
  virtual ~ConstVariable() = default;

  /// Map as Eigen vector.
  /// \return Vector.
  [[nodiscard]] virtual auto asVector() const -> Eigen::Ref<const VectorX> = 0;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() -> Eigen::Ref<const VectorX> = 0;
};

class Variable {
 public:
  /// Virtual default destructor.
  virtual ~Variable() = default;

  /// Map as Eigen vector.
  /// \return Vector.
  [[nodiscard]] virtual auto asVector() const -> Eigen::Ref<const VectorX> = 0;

  /// Map as Eigen vector.
  /// \return Vector.
  virtual auto asVector() -> Eigen::Ref<VectorX> = 0;
};

}  // namespace hyper::variables
