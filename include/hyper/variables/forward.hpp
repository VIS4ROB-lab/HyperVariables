/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

#include "hyper/variables/macros.hpp"

namespace hyper {

template <typename>
struct Traits;

template <typename>
struct NumericVariableTraits;

template <typename>
class AbstractVariable;

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

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::Cartesian, int)

template <typename>
class PitchYaw;

template <typename TScalar>
struct Traits<PitchYaw<TScalar>>
    : public Traits<Cartesian<TScalar, 2>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::PitchYaw)

template <typename>
class Bearing;

template <typename TScalar>
struct Traits<Bearing<TScalar>>
    : public Traits<Cartesian<TScalar, 3>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::Bearing)

template <typename>
class Gravity;

template <typename TScalar>
struct Traits<Gravity<TScalar>>
    : public Traits<Cartesian<TScalar, 3>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::Gravity)

template <typename>
class Intrinsics;

template <typename TScalar>
struct Traits<Intrinsics<TScalar>>
    : public Traits<Cartesian<TScalar, 4>> {};

HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(hyper::Intrinsics)

template <typename, int>
class OrthonormalityAlignment;

template <typename TScalar, int TOrder>
struct Traits<OrthonormalityAlignment<TScalar, TOrder>>
    : public Traits<Cartesian<TScalar, TOrder + ((TOrder - 1) * TOrder) / 2>> {
  static constexpr auto kOrder = TOrder;
};

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(hyper::OrthonormalityAlignment, int)

template <typename TVariable>
struct Traits<Stamped<TVariable>>
    : public Traits<Cartesian<typename TVariable::Scalar, TVariable::SizeAtCompileTime + 1>> {
  // Definitions.
  using Scalar = typename TVariable::Scalar;
  using Stamp = Cartesian<Scalar, 1>;
  using StampWithConstIfNotLvalue = Stamp;
  using Variable = TVariable;
  using VariableWithConstIfNotLvalue = TVariable;
};

template <typename TVariable, int TMapOptions>
struct Traits<Eigen::Map<Stamped<TVariable>, TMapOptions>> final
    : public Traits<Stamped<TVariable>> {
  using Base = Eigen::Map<typename Traits<Stamped<TVariable>>::Base, TMapOptions>;
};

template <typename TVariable, int TMapOptions>
struct Traits<Eigen::Map<const Stamped<TVariable>, TMapOptions>> final
    : public Traits<Stamped<TVariable>> {
  using StampWithConstIfNotLvalue = const typename Traits<Stamped<TVariable>>::Stamp;
  using VariableWithConstIfNotLvalue = const typename Traits<Stamped<TVariable>>::Variable;
  using Base = Eigen::Map<const typename Traits<Stamped<TVariable>>::Base, TMapOptions>;
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

template <typename TDerived>
struct VariableIsLValue_q {
  static constexpr auto kValue = (!bool(std::is_const_v<TDerived>)) && bool(Traits<TDerived>::Base::Flags & Eigen::LvalueBit);
};

template <typename TDerived>
inline constexpr bool VariableIsLValue_v = VariableIsLValue_q<TDerived>::kValue;

template <typename TDerived, typename TValue>
struct ConstValueIfVariableIsNotLValue {
  using Type = std::conditional_t<VariableIsLValue_v<TDerived>, TValue, const TValue>;
};

template <typename TDerived, typename TValue>
using ConstValueIfVariableIsNotLValue_t = ConstValueIfVariableIsNotLValue<TDerived, TValue>::Type;

template <typename TDerived>
using ConstScalarIfVariableIsNotLValue_t = ConstValueIfVariableIsNotLValue_t<TDerived, typename Traits<TDerived>::Base::Scalar>;

} // namespace hyper
