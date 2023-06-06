/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper::variables {

class Distortion;

class ConstDistortion;

template <int TOrder>
class EquidistantDistortion;

template <int TOrder>
struct Traits<EquidistantDistortion<TOrder>> : public Traits<Rn<TOrder>> {
  static constexpr auto kOrder = TOrder;
  using PlainDistortion = EquidistantDistortion<TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::EquidistantDistortion, int)

template <int TOrder>
class RadialTangentialDistortion;

template <int TOrder>
struct Traits<RadialTangentialDistortion<TOrder>> : public Traits<Rn<TOrder + 2>> {
  static constexpr auto kOrder = TOrder;
  using PlainDistortion = RadialTangentialDistortion<TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::RadialTangentialDistortion, int)

template <int TOrder>
class IterativeRadialDistortion;

template <int TOrder>
struct Traits<IterativeRadialDistortion<TOrder>> : public Traits<Rn<TOrder>> {
  static constexpr auto kOrder = TOrder;
  using PlainDistortion = IterativeRadialDistortion<TOrder>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::IterativeRadialDistortion, int)

}  // namespace hyper::variables
