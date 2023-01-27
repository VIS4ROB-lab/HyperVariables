/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#define HYPER_DEFAULT_TO_GLOBAL_MANIFOLD_DERIVATIVES false
#define HYPER_DEFAULT_TO_COUPLED_MANIFOLD_DERIVATIVES false

namespace hyper::variables {

template <typename>
class Quaternion;

template <typename TScalar>
struct Traits<Quaternion<TScalar>> : public Traits<Cartesian<TScalar, 4>> {
  using Base = Eigen::Quaternion<TScalar>;
};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::Quaternion)

template <typename>
class SU2;

template <typename TScalar>
struct Traits<SU2<TScalar>> : Traits<Quaternion<TScalar>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::SU2)

template <typename>
class SE3;

template <typename TScalar>
struct Traits<SE3<TScalar>> : Traits<Cartesian<TScalar, SU2<TScalar>::kNumParameters + 3>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::SE3)

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
struct Traits<Tangent<SU2<TScalar>>> : Traits<Cartesian<TScalar, 3>> {};

HYPER_DECLARE_TANGENT_MAP_TRAITS(hyper::variables::SU2)

template <typename TScalar>
struct Traits<Tangent<SE3<TScalar>>> : Traits<Cartesian<TScalar, 6>> {};

HYPER_DECLARE_TANGENT_MAP_TRAITS(hyper::variables::SE3)

}  // namespace hyper::variables
