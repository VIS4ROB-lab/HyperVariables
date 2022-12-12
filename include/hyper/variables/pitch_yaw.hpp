/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/cartesian.hpp"

namespace hyper::variables {

template <typename TDerived>
class PitchYawBase
    : public CartesianBase<TDerived> {
 public:
  // Constants.
  static constexpr auto kPitchOffset = 0;
  static constexpr auto kNumPitchParameters = 1;
  static constexpr auto kYawOffset = kPitchOffset + kNumPitchParameters;
  static constexpr auto kNumYawParameters = 1;

  // Definitions.
  using Base = CartesianBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Scalar>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(PitchYawBase)

  /// Pitch accessor.
  /// \return Pitch.
  [[nodiscard]] auto pitch() const -> const Scalar& {
    return this->data()[kPitchOffset];
  }

  /// Pitch modifier.
  /// \return Pitch.
  auto pitch() -> ScalarWithConstIfNotLvalue& {
    return this->data()[kPitchOffset];
  }

  /// Yaw accessor.
  /// \return Yaw.
  [[nodiscard]] auto yaw() const -> const Scalar& {
    return this->data()[kYawOffset];
  }

  /// Yaw modifier.
  /// \return Yaw.
  auto yaw() -> ScalarWithConstIfNotLvalue& {
    return this->data()[kYawOffset];
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

} // namespace hyper::variables

HYPER_DECLARE_EIGEN_INTERFACE(hyper::variables::PitchYaw)
