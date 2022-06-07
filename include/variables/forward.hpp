/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

#include "variables/macros.hpp"

namespace hyper {

template <typename, typename = Eigen::Index>
struct MemoryBlock;

template <typename, typename = Eigen::Index>
class MemoryBlocks;

template <typename>
class AbstractVariable;

template <typename>
class AbstractStampedVariable;

template <typename>
class StampedVariable;

template <typename>
class CompositeVariable;

template <typename>
struct Traits;

template <typename>
struct NumericVariableTraits;

template <typename, int>
class Cartesian;

template <typename TScalar, int TNumParameters>
struct Traits<Cartesian<TScalar, TNumParameters>> {
  // Constants.
  static constexpr auto kNumParameters = TNumParameters;

  // Definitions.
  using Scalar = TScalar;
  using ScalarWithConstIfNotLvalue = TScalar;
  using Base = Eigen::Matrix<TScalar, TNumParameters, 1>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_MAP_TRAITS(Cartesian, int)

template <typename>
class PitchYaw;

template <typename TScalar>
struct Traits<PitchYaw<TScalar>>
    : public Traits<Cartesian<TScalar, 2>> {
  // Constants.
  static constexpr auto kPitchOffset = 0;
  static constexpr auto kNumPitchParameters = 1;
  static constexpr auto kYawOffset = kPitchOffset + kNumPitchParameters;
  static constexpr auto kNumYawParameters = 1;
  static constexpr auto kNumParameters = kNumPitchParameters + kNumYawParameters;
};

HYPER_DECLARE_EIGEN_MAP_TRAITS(PitchYaw)

template <typename>
class Bearing;

template <typename TScalar>
struct Traits<Bearing<TScalar>>
    : public Traits<Cartesian<TScalar, 3>> {
  // Constants.
  static constexpr auto kNorm = TScalar{1};
};

HYPER_DECLARE_EIGEN_MAP_TRAITS(Bearing)

template <typename>
class Gravity;

template <typename TScalar>
struct Traits<Gravity<TScalar>>
    : public Traits<Cartesian<TScalar, 3>> {
  // Constants.
  static constexpr auto kNorm = TScalar{9.80741}; // Magnitude of local gravity for Zurich in [m/s²].
};

HYPER_DECLARE_EIGEN_MAP_TRAITS(Gravity)

template <typename>
class Intrinsics;

template <typename TScalar>
struct Traits<Intrinsics<TScalar>>
    : public Traits<Cartesian<TScalar, 4>> {
  // Constants.
  static constexpr auto kPrincipalOffset = 0;
  static constexpr auto kPrincipalOffsetX = kPrincipalOffset;
  static constexpr auto kPrincipalOffsetY = kPrincipalOffset + 1;
  static constexpr auto kNumPrincipalParameters = 2;
  static constexpr auto kFocalOffset = kPrincipalOffset + kNumPrincipalParameters;
  static constexpr auto kFocalOffsetX = kFocalOffset;
  static constexpr auto kFocalOffsetY = kFocalOffset + 1;
  static constexpr auto kNumFocalParameters = 2;
};

HYPER_DECLARE_EIGEN_MAP_TRAITS(Intrinsics)

template <typename, int>
class OrthonormalityAlignment;

template <typename TScalar, int TOrder>
struct Traits<OrthonormalityAlignment<TScalar, TOrder>>
    : public Traits<Cartesian<TScalar, TOrder + ((TOrder - 1) * TOrder) / 2>> {
  // Constants.
  static constexpr auto kOrder = TOrder;
  static constexpr auto kNumDiagonalParameters = TOrder;
  static constexpr auto kNumOffDiagonalParameters = ((TOrder - 1) * TOrder) / 2;
  static constexpr auto kNumParameters = kNumDiagonalParameters + kNumOffDiagonalParameters;
};

HYPER_DECLARE_TEMPLATED_EIGEN_MAP_TRAITS(OrthonormalityAlignment, int)

template <typename TVariable>
struct Traits<StampedVariable<TVariable>>
    : public Traits<Cartesian<typename Traits<TVariable>::Scalar, Traits<TVariable>::kNumParameters + 1>> {
  // Constants.
  static constexpr auto kVariableOffset = 0;
  static constexpr auto kNumVariableParameters = Traits<TVariable>::kNumParameters;
  static constexpr auto kStampOffset = kVariableOffset + kNumVariableParameters;
  static constexpr auto kNumStampParameters = 1;

  // Definitions.
  using Stamp = Cartesian<typename Traits<TVariable>::Scalar, kNumStampParameters>;
  using StampWithConstIfNotLvalue = Stamp;
  using Variable = TVariable;
  using VariableWithConstIfNotLvalue = TVariable;
};

template <typename TVariable, int TMapOptions>
struct Traits<Eigen::Map<StampedVariable<TVariable>, TMapOptions>> final
    : public Traits<StampedVariable<TVariable>> {
  using Base = Eigen::Map<typename Traits<StampedVariable<TVariable>>::Base, TMapOptions>;
};

template <typename TVariable, int TMapOptions>
struct Traits<Eigen::Map<const StampedVariable<TVariable>, TMapOptions>> final
    : public Traits<StampedVariable<TVariable>> {
  using ScalarWithConstIfNotLvalue = const typename Traits<StampedVariable<TVariable>>::Scalar;
  using StampWithConstIfNotLvalue = const typename Traits<StampedVariable<TVariable>>::Stamp;
  using VariableWithConstIfNotLvalue = const typename Traits<StampedVariable<TVariable>>::Variable;
  using Base = Eigen::Map<const typename Traits<StampedVariable<TVariable>>::Base, TMapOptions>;
};

template <typename TScalar, int TNumParameters = 2>
using Pixel = Cartesian<TScalar, TNumParameters>;

template <typename TScalar, int TNumParameters = 3>
using Position = Cartesian<TScalar, TNumParameters>;

template <typename TScalar, int TNumParameters = 3>
using Translation = Cartesian<TScalar, TNumParameters>;

template <>
struct NumericVariableTraits<float> {
  static constexpr float kSmallAngleTolerance = 1e-4;
  static constexpr float kDistortionTolerance = 1e-6;
  static constexpr float kDistortionTolerance2 = kDistortionTolerance * kDistortionTolerance;
  static constexpr auto kMaxNumDistortionSteps = 20;
};

template <>
struct NumericVariableTraits<double> {
  static constexpr double kSmallAngleTolerance = 1e-8;
  static constexpr double kDistortionTolerance = 1e-12;
  static constexpr double kDistortionTolerance2 = kDistortionTolerance * kDistortionTolerance;
  static constexpr auto kMaxNumDistortionSteps = 20;
};

} // namespace hyper
