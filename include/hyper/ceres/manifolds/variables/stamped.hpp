/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#ifdef HYPER_COMPILE_WITH_CERES

#include <ceres/product_manifold.h>

#include "hyper/ceres/manifolds/variables/euclidean.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::ceres::manifolds {

template <typename TVariable>
class Manifold<variables::Stamped<TVariable>> final : public ManifoldWrapper {
 public:
  // Definitions.
  using Variable = TVariable;

  using Stamp = variables::Stamp<Scalar>;
  using StampedVariable = variables::Stamped<TVariable>;

  /// Constructor from constancy flags.
  /// \tparam TArgs_ Variadic argument types.
  /// \param time_constant Time constancy flag.
  /// \param args Variadic arguments (i.e. further constancy flags).
  template <typename... TArgs_>
  explicit Manifold(const bool time_constant = true, TArgs_&&... args) : ManifoldWrapper{CreateManifold(time_constant, std::forward<TArgs_>(args)...)} {}

 private:
  /// Creates a (partially) constant or non-constant manifold.
  /// \tparam TArgs_ Variadic argument types.
  /// \param time_constant Time constancy flag.
  /// \param args Variadic arguments (i.e. further constancy flags).
  /// \return Manifold
  template <typename... TArgs_>
  static auto CreateManifold(const bool time_constant, TArgs_&&... args) -> std::unique_ptr<::ceres::Manifold> {
    using StampManifold = Manifold<Stamp>;
    using VariableManifold = Manifold<Variable>;
    using ProductManifold = ::ceres::ProductManifold<VariableManifold, StampManifold>;
    return std::make_unique<ProductManifold>(VariableManifold{std::forward<TArgs_>(args)...}, StampManifold{time_constant});
  }
};

}  // namespace hyper::ceres::manifolds

#endif
