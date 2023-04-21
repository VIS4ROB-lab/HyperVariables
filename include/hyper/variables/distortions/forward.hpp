/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper::variables {

template <typename TDerived>
class Distortion;

template <typename TDerived>
class ConstDistortion;

template <typename, int>
class EquidistantDistortion;

template <typename TScalar, int TOrder>
struct Traits<EquidistantDistortion<TScalar, TOrder>> : public Traits<Rn<TScalar, TOrder>> {
  static constexpr auto kOrder = TOrder;
  using PlainDistortion = EquidistantDistortion<TScalar, TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::EquidistantDistortion, int)

template <typename, int>
class RadialTangentialDistortion;

template <typename TScalar, int TOrder>
struct Traits<RadialTangentialDistortion<TScalar, TOrder>> : public Traits<Rn<TScalar, TOrder + 2>> {
  static constexpr auto kOrder = TOrder;
  using PlainDistortion = RadialTangentialDistortion<TScalar, TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::RadialTangentialDistortion, int)

template <typename, int>
class IterativeRadialDistortion;

template <typename TScalar, int TOrder>
struct Traits<IterativeRadialDistortion<TScalar, TOrder>> : public Traits<Rn<TScalar, TOrder>> {
  static constexpr auto kOrder = TOrder;
  using PlainDistortion = IterativeRadialDistortion<TScalar, TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::IterativeRadialDistortion, int)

}  // namespace hyper::variables
