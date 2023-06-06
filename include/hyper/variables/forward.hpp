/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

#include "hyper/definitions.hpp"
#include "hyper/variables/macros.hpp"

namespace hyper {}

namespace hyper::variables {

template <typename>
struct Traits;

template <typename>
class Variable;

template <typename>
class ConstVariable;

template <typename>
class Stamped;

template <typename, int>
class Rn;

template <typename TScalar>
using R1 = Rn<TScalar, 1>;

template <typename TScalar>
using R2 = Rn<TScalar, 2>;

template <typename TScalar>
using R3 = Rn<TScalar, 3>;

template <typename TScalar>
using R4 = Rn<TScalar, 4>;

template <typename TScalar>
using R5 = Rn<TScalar, 5>;

template <typename TScalar>
using R6 = Rn<TScalar, 6>;

template <typename TScalar>
using Stamp = R1<TScalar>;

template <typename TScalar>
using Pixel = R2<TScalar>;

template <typename TScalar, int TNumParameters>
struct Traits<Rn<TScalar, TNumParameters>> {
  // Constants.
  static constexpr auto kNumParameters = TNumParameters;

  // Definitions.
  using Base = Eigen::Matrix<TScalar, TNumParameters, 1>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::Rn, int)

template <typename>
class PitchYaw;

template <typename TScalar>
struct Traits<PitchYaw<TScalar>> : public Traits<R2<TScalar>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::PitchYaw)

template <typename>
class Bearing;

template <typename TScalar>
struct Traits<Bearing<TScalar>> : public Traits<R3<TScalar>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::Bearing)

template <typename>
class Gravity;

template <typename TScalar>
struct Traits<Gravity<TScalar>> : public Traits<R3<TScalar>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::Gravity)

template <typename>
class Intrinsics;

template <typename TScalar>
struct Traits<Intrinsics<TScalar>> : public Traits<R4<TScalar>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::Intrinsics)

template <typename, int>
class AxesOffset;

template <typename TScalar, int TOrder>
struct Traits<AxesOffset<TScalar, TOrder>> : public Traits<Rn<TScalar, TOrder * TOrder>> {
  static constexpr auto kOrder = TOrder;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::AxesOffset, int)

template <typename, int>
class OrthonormalityAlignment;

template <typename TScalar, int TOrder>
struct Traits<OrthonormalityAlignment<TScalar, TOrder>> : public Traits<Rn<TScalar, TOrder + ((TOrder - 1) * TOrder) / 2>> {
  static constexpr auto kOrder = TOrder;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::OrthonormalityAlignment, int)

template <typename, int>
class Sensitivity;

template <typename TScalar, int TOrder>
struct Traits<Sensitivity<TScalar, TOrder>> : public Traits<Rn<TScalar, TOrder * TOrder>> {
  static constexpr auto kOrder = TOrder;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::Sensitivity, int)

template <typename>
class Quaternion;

template <typename TScalar>
struct Traits<Quaternion<TScalar>> : public Traits<R4<TScalar>> {
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
struct Traits<SE3<TScalar>> : Traits<Rn<TScalar, SU2<TScalar>::kNumParameters + 3>> {};

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
struct Traits<Tangent<SU2<TScalar>>> : Traits<R3<TScalar>> {};

HYPER_DECLARE_TANGENT_MAP_TRAITS(hyper::variables::SU2)

template <typename TScalar>
struct Traits<Tangent<SE3<TScalar>>> : Traits<R6<TScalar>> {};

HYPER_DECLARE_TANGENT_MAP_TRAITS(hyper::variables::SE3)

template <typename TVariable>
struct Traits<Stamped<TVariable>> : public Traits<Rn<typename TVariable::Scalar, Stamp<typename TVariable::Scalar>::kNumParameters + TVariable::kNumParameters>> {
  using Variable = TVariable;
};

template <typename TVariable, int TMapOptions>
struct Traits<Eigen::Map<Stamped<TVariable>, TMapOptions>> final : public Traits<Stamped<TVariable>> {
  using Base = Eigen::Map<typename Traits<Stamped<TVariable>>::Base, TMapOptions>;
};

template <typename TVariable, int TMapOptions>
struct Traits<Eigen::Map<const Stamped<TVariable>, TMapOptions>> final : public Traits<Stamped<TVariable>> {
  using Base = Eigen::Map<const typename Traits<Stamped<TVariable>>::Base, TMapOptions>;
};

template <typename TDerived>
using DerivedScalar_t = typename Traits<TDerived>::Base::Scalar;

template <typename TDerived>
inline constexpr bool VariableIsLValue_v = (!bool(std::is_const_v<TDerived>)) && bool(Traits<TDerived>::Base::Flags & Eigen::LvalueBit);

template <typename TDerived, typename TValue>
using ConstValueIfVariableIsNotLValue_t = std::conditional_t<VariableIsLValue_v<TDerived>, TValue, const TValue>;

template <typename TDerived, typename TBase, typename TConstBase>
using ConditionalConstBase_t = std::conditional_t<VariableIsLValue_v<TDerived>, TBase, TConstBase>;

}  // namespace hyper::variables
