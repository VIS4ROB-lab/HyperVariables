/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper {

namespace variables {

class Distortion;

class ConstDistortion;

template <int TOrder>
class EquidistantDistortion;

template <int TOrder>
class RadialTangentialDistortion;

template <int TOrder>
class IterativeRadialDistortion;

}  // namespace variables

template <int TOrder>
struct Traits<variables::EquidistantDistortion<TOrder>> : public Traits<variables::Rn<TOrder>> {
  static constexpr auto kOrder = TOrder;
  using PlainDistortion = variables::EquidistantDistortion<TOrder>;
};

template <int TOrder>
struct Traits<variables::RadialTangentialDistortion<TOrder>> : public Traits<variables::Rn<TOrder + 2>> {
  static constexpr auto kOrder = TOrder;
  using PlainDistortion = variables::RadialTangentialDistortion<TOrder>;
};

template <int TOrder>
struct Traits<variables::IterativeRadialDistortion<TOrder>> : public Traits<variables::Rn<TOrder>> {
  static constexpr auto kOrder = TOrder;
  using PlainDistortion = variables::IterativeRadialDistortion<TOrder>;
};

}  // namespace hyper
