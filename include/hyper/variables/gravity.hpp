/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>

#include "hyper/variables/rn.hpp"

namespace hyper::variables {

template <typename TDerived>
class GravityBase : public RnBase<TDerived> {
 public:
  // Definitions.
  using Base = RnBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  // Constants.
  static constexpr auto kNorm = Scalar{9.80741};  // Magnitude of local gravity for Zurich in [m/s²].

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(GravityBase)

  /// Checks the norm.
  /// \return True if correct.
  [[nodiscard]] auto checkNorm() const -> bool { return Eigen::internal::isApprox(this->norm(), kNorm); }
};

class Gravity final : public GravityBase<Gravity> {
 public:
  using Base = GravityBase<Gravity>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Gravity)

  /// Default constructor.
  Gravity() = default;

  /// Forwarding constructor with norm check.
  /// \tparam TArgs_ Input argument types.
  /// \param args Inputs arguments.
  template <typename... TArgs_>
  Gravity(TArgs_&&... args)  // NOLINT
      : Base{std::forward<TArgs_>(args)...} {
    DCHECK(this->checkNorm());
  }
};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::Gravity)

}  // namespace hyper::variables

HYPER_DECLARE_EIGEN_INTERFACE(hyper::variables::Gravity)
