/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>

#include "variables/cartesian.hpp"

namespace hyper {

template <typename TDerived>
class GravityBase
    : public CartesianBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = CartesianBase<TDerived>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(GravityBase)

  /// Checks the norm.
  /// \return True if correct.
  [[nodiscard]] auto checkNorm() const -> bool {
    return Eigen::internal::isApprox(this->norm(), Traits<Gravity<Scalar>>::kNorm);
  }
};

template <typename TScalar>
class Gravity final
    : public GravityBase<Gravity<TScalar>> {
 public:
  using Base = GravityBase<Gravity<TScalar>>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Gravity)

  /// Deleted default constructor.
  Gravity() = delete;

  /// Forwarding constructor with norm check.
  /// \tparam TArgs_ Input argument types.
  /// \param args Inputs arguments.
  template <typename... TArgs_>
  Gravity(TArgs_&&... args) // NOLINT
      : Base{std::forward<TArgs_>(args)...} {
    DCHECK(this->checkNorm());
  }
};

} // namespace hyper

HYPER_DECLARE_EIGEN_INTERFACE(Gravity)
