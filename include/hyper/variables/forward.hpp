/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

#include "hyper/variables/macros.hpp"

namespace hyper::variables {

template <typename>
struct Traits;

template <typename>
class AbstractVariable;

template <typename>
class ConstAbstractVariable;

template <typename>
class AbstractStamped;

template <typename>
class Stamped;

template <typename, int>
class Cartesian;

template <typename TScalar, int TNumParameters>
struct Traits<Cartesian<TScalar, TNumParameters>> {
  using Base = Eigen::Matrix<TScalar, TNumParameters, 1>;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::Cartesian, int)

template <typename>
class PitchYaw;

template <typename TScalar>
struct Traits<PitchYaw<TScalar>> : public Traits<Cartesian<TScalar, 2>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::PitchYaw)

template <typename>
class Bearing;

template <typename TScalar>
struct Traits<Bearing<TScalar>> : public Traits<Cartesian<TScalar, 3>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::Bearing)

template <typename>
class Gravity;

template <typename TScalar>
struct Traits<Gravity<TScalar>> : public Traits<Cartesian<TScalar, 3>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::Gravity)

template <typename>
class Intrinsics;

template <typename TScalar>
struct Traits<Intrinsics<TScalar>> : public Traits<Cartesian<TScalar, 4>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::variables::Intrinsics)

template <typename, int>
class OrthonormalityAlignment;

template <typename TScalar, int TOrder>
struct Traits<OrthonormalityAlignment<TScalar, TOrder>> : public Traits<Cartesian<TScalar, TOrder + ((TOrder - 1) * TOrder) / 2>> {
  static constexpr auto kOrder = TOrder;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::variables::OrthonormalityAlignment, int)

template <typename TScalar>
using Stamp = Cartesian<TScalar, 1>;

template <typename TVariable>
struct Traits<Stamped<TVariable>> : public Traits<Cartesian<typename TVariable::Scalar, Stamp<typename TVariable::Scalar>::kNumParameters + TVariable::kNumParameters>> {
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

template <typename TScalar, int TNumParameters = 2>
using Pixel = Cartesian<TScalar, TNumParameters>;

template <typename TScalar, int TNumParameters = 3>
using Position = Cartesian<TScalar, TNumParameters>;

template <typename TScalar, int TNumParameters = 3>
using Translation = Cartesian<TScalar, TNumParameters>;

template <typename>
struct NumericVariableTraits;

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

template <typename TDerived>
using DerivedScalar_t = typename Traits<TDerived>::Base::Scalar;

template <typename TDerived>
inline constexpr bool VariableIsLValue_v = (!bool(std::is_const_v<TDerived>)) && bool(Traits<TDerived>::Base::Flags & Eigen::LvalueBit);

template <typename TDerived, typename TValue>
using ConstValueIfVariableIsNotLValue_t = std::conditional_t<VariableIsLValue_v<TDerived>, TValue, const TValue>;

template <typename TDerived, typename TBase, typename TConstBase>
using ConditionalConstBase_t = std::conditional_t<VariableIsLValue_v<TDerived>, TBase, TConstBase>;

}  // namespace hyper::variables
