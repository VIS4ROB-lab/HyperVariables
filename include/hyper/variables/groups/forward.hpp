/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#define HYPER_DEFAULT_TO_GLOBAL_LIE_GROUP_DERIVATIVES false
#define HYPER_DEFAULT_TO_COUPLED_LIE_GROUP_DERIVATIVES false

namespace hyper {

enum class ManifoldEnum {
  SU2,
  SE3,
};

struct QuaternionOrder {
  static constexpr auto kW = 3;
  static constexpr auto kX = 0;
  static constexpr auto kY = 1;
  static constexpr auto kZ = 2;
};

template <typename>
class Quaternion;

template <typename TScalar>
struct Traits<Quaternion<TScalar>> : public Traits<Cartesian<TScalar, 4>> {
  using Base = Eigen::Quaternion<TScalar>;
};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::Quaternion)

template <typename>
class SU2;

template <typename TScalar>
struct Traits<SU2<TScalar>> : Traits<Quaternion<TScalar>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::SU2)

template <typename>
class SE3;

template <typename TScalar>
struct Traits<SE3<TScalar>> : Traits<Cartesian<TScalar, SU2<TScalar>::SizeAtCompileTime + 3>> {
  static constexpr auto kRotationOffset = 0;
  static constexpr auto kNumRotationParameters = SU2<TScalar>::SizeAtCompileTime;
  static constexpr auto kTranslationOffset = kNumRotationParameters;
  static constexpr auto kNumTranslationParameters = 3;
};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::SE3)

template <typename>
class Algebra;

template <typename TDerived>
struct Traits<Algebra<TDerived>> : Traits<TDerived> {};

template <typename TDerived, int TMapOptions>
struct Traits<Eigen::Map<Algebra<TDerived>, TMapOptions>> : Traits<Algebra<TDerived>> {
  using Base = typename Traits<Eigen::Map<TDerived, TMapOptions>>::Base;
};

template <typename TDerived, int TMapOptions>
struct Traits<Eigen::Map<const Algebra<TDerived>, TMapOptions>> : Traits<Algebra<TDerived>> {
  using Base = typename Traits<Eigen::Map<const TDerived, TMapOptions>>::Base;
  using ScalarWithConstIfNotLvalue = const typename Traits<TDerived>::Scalar;
};

template <typename>
class Tangent;

template <typename TDerived>
struct Traits<Tangent<TDerived>> : Traits<TDerived> {};

template <typename TDerived, int TMapOptions>
struct Traits<Eigen::Map<Tangent<TDerived>, TMapOptions>> : Traits<Tangent<TDerived>> {
  using Base = typename Traits<Eigen::Map<TDerived, TMapOptions>>::Base;
};

template <typename TDerived, int TMapOptions>
struct Traits<Eigen::Map<const Tangent<TDerived>, TMapOptions>> : Traits<Tangent<TDerived>> {
  using Base = typename Traits<Eigen::Map<const TDerived, TMapOptions>>::Base;
};

template <typename TScalar>
struct Traits<Tangent<SU2<TScalar>>> : Traits<Cartesian<TScalar, 3>> {
  static constexpr auto kAngularOffset = 0;
  static constexpr auto kNumAngularParameters = 3;
};

HYPER_DECLARE_TANGENT_MAP_TRAITS(hyper::SU2)

template <typename TScalar>
struct Traits<Tangent<SE3<TScalar>>> : Traits<Cartesian<TScalar, 6>> {
  static constexpr auto kAngularOffset = 0;
  static constexpr auto kNumAngularParameters = 3;
  static constexpr auto kLinearOffset = kAngularOffset + kNumAngularParameters;
  static constexpr auto kNumLinearParameters = 3;
};

HYPER_DECLARE_TANGENT_MAP_TRAITS(hyper::SE3)

} // namespace hyper
