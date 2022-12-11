/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper {

template <typename TScalar, typename TBase>
class AbstractDistortionBase;

template <typename TScalar>
class AbstractDistortion;

template <typename TScalar>
class ConstAbstractDistortion;

template <typename TDerived, typename TBase>
class DistortionBase;

template <typename TDerived>
class Distortion;

template <typename TDerived>
class ConstDistortion;

template <typename, int>
class EquidistantDistortion;

template <typename TScalar, int TOrder>
struct Traits<EquidistantDistortion<TScalar, TOrder>>
    : public Traits<Cartesian<TScalar, TOrder>> {
  // Definitions.
  static constexpr auto kOrder = TOrder;
  using Distortion = EquidistantDistortion<TScalar, TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::EquidistantDistortion, int)

template <typename, int>
class RadialTangentialDistortion;

template <typename TScalar, int TOrder>
struct Traits<RadialTangentialDistortion<TScalar, TOrder>>
    : public Traits<Cartesian<TScalar, TOrder == Eigen::Dynamic ? TOrder : TOrder + 2>> {
  static constexpr auto kOrder = TOrder;
  using Distortion = RadialTangentialDistortion<TScalar, TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::RadialTangentialDistortion, int)

template <typename, int>
class IterativeRadialDistortion;

template <typename TScalar, int TOrder>
struct Traits<IterativeRadialDistortion<TScalar, TOrder>>
    : public Traits<Cartesian<TScalar, TOrder>> {
  static constexpr auto kOrder = TOrder;
  using Distortion = IterativeRadialDistortion<TScalar, TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::IterativeRadialDistortion, int)

} // namespace hyper
