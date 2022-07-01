/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/cartesian.hpp"

namespace hyper {

template <typename TDerived>
class PitchYawBase
    : public CartesianBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = CartesianBase<TDerived>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(PitchYawBase)

  /// Pitch accessor.
  /// \return Pitch.
  [[nodiscard]] auto pitch() const -> const Scalar& {
    return this->data()[Traits<TDerived>::kPitchOffset];
  }

  /// Pitch modifier.
  /// \return Pitch.
  auto pitch() -> ScalarWithConstIfNotLvalue& {
    return this->data()[Traits<TDerived>::kPitchOffset];
  }

  /// Yaw accessor.
  /// \return Yaw.
  [[nodiscard]] auto yaw() const -> const Scalar& {
    return this->data()[Traits<TDerived>::kYawOffset];
  }

  /// Yaw modifier.
  /// \return Yaw.
  auto yaw() -> ScalarWithConstIfNotLvalue& {
    return this->data()[Traits<TDerived>::kYawOffset];
  }
};

template <typename TScalar>
class PitchYaw final
    : public PitchYawBase<PitchYaw<TScalar>> {
 public:
  using Base = PitchYawBase<PitchYaw<TScalar>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(PitchYaw)
};

} // namespace hyper

HYPER_DECLARE_EIGEN_INTERFACE(hyper::PitchYaw)
