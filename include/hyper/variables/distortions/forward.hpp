/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper {

template <typename TScalar, bool = std::is_const_v<TScalar>>
class AbstractDistortion;

template <typename TDerived, bool = !VariableIsLValue<TDerived>::value>
class DistortionBase;

template <typename, int>
class EquidistantDistortion;

template <typename TScalar, int TOrder>
struct Traits<EquidistantDistortion<TScalar, TOrder>>
    : public Traits<Cartesian<TScalar, TOrder>> {
  // Constants.
  static constexpr auto kRadialOffset = 0;
  static constexpr auto kNumRadialParameters = TOrder;
  static constexpr auto kOrder = TOrder;

  // Definitions.
  using PlainDerivedType = EquidistantDistortion<TScalar, TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::EquidistantDistortion, int)

template <typename, int>
class RadialTangentialDistortion;

template <typename TScalar, int TOrder>
struct Traits<RadialTangentialDistortion<TScalar, TOrder>>
    : public Traits<Cartesian<TScalar, TOrder == Eigen::Dynamic ? TOrder : TOrder + 2>> {
  // Constants.
  static constexpr auto kRadialOffset = 0;
  static constexpr auto kNumRadialParameters = TOrder;
  static constexpr auto kTangentialOffset = kRadialOffset + kNumRadialParameters;
  static constexpr auto kNumTangentialParameters = 2;
  static constexpr auto kOrder = TOrder;

  // Definitions.
  using PlainDerivedType = RadialTangentialDistortion<TScalar, TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::RadialTangentialDistortion, int)

template <typename, int>
class IterativeRadialDistortion;

template <typename TScalar, int TOrder>
struct Traits<IterativeRadialDistortion<TScalar, TOrder>>
    : public Traits<Cartesian<TScalar, TOrder>> {
  // Constants.
  static constexpr auto kRadialOffset = 0;
  static constexpr auto kNumRadialParameters = TOrder;
  static constexpr auto kOrder = TOrder;

  // Definitions.
  using PlainDerivedType = IterativeRadialDistortion<TScalar, TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::IterativeRadialDistortion, int)

} // namespace hyper
