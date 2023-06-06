/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

#include "hyper/matrix.hpp"
#include "hyper/variables/macros.hpp"

namespace hyper {}

namespace hyper::variables {

template <typename>
struct Traits;

class Variable;

class ConstVariable;

template <typename>
class Stamped;

template <int TOrder>
class Rn;

using R1 = Rn<1>;
using R2 = Rn<2>;
using R3 = Rn<3>;
using R4 = Rn<4>;
using R5 = Rn<5>;
using R6 = Rn<6>;

using Stamp = Rn<1>;

template <int TOrder>
struct Traits<Rn<TOrder>> {
  // Constants.
  static constexpr auto kNumParameters = TOrder;

  // Definitions.
  using Base = Matrix<TOrder, 1>;
};

class PitchYaw;

template <>
struct Traits<PitchYaw> : public Traits<R2> {};

class Bearing;

template <>
struct Traits<Bearing> : public Traits<R3> {};

class Gravity;

template <>
struct Traits<Gravity> : public Traits<R3> {};

class Intrinsics;

template <>
struct Traits<Intrinsics> : public Traits<R4> {};

template <int TOrder>
class AxesOffset;

template <int TOrder>
struct Traits<AxesOffset<TOrder>> : public Traits<Rn<TOrder * TOrder>> {
  static constexpr auto kOrder = TOrder;
};

template <int TOrder>
class OrthonormalityAlignment;

template <int TOrder>
struct Traits<OrthonormalityAlignment<TOrder>> : public Traits<Rn<TOrder + ((TOrder - 1) * TOrder) / 2>> {
  static constexpr auto kOrder = TOrder;
};

template <int TOrder>
class Sensitivity;

template <int TOrder>
struct Traits<Sensitivity<TOrder>> : public Traits<Rn<TOrder * TOrder>> {
  static constexpr auto kOrder = TOrder;
};

class Quaternion;

template <>
struct Traits<Quaternion> : public Traits<R4> {
  using Base = Eigen::Quaternion<Scalar>;
};

class SU2;

template <>
struct Traits<SU2> : Traits<Quaternion> {};

class SE3;

template <>
struct Traits<SE3> : Traits<Rn<Traits<SU2>::kNumParameters + Traits<R3>::kNumParameters>> {};

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

template <>
struct Traits<Tangent<SU2>> : Traits<R3> {};

template <>
struct Traits<Tangent<SE3>> : Traits<R6> {};

template <typename TVariable>
struct Traits<Stamped<TVariable>> : public Traits<Rn<TVariable::kNumParameters + Traits<R1>::kNumParameters>> {
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
inline constexpr bool VariableIsLValue_v = (!bool(std::is_const_v<TDerived>)) && bool(Traits<TDerived>::Base::Flags & Eigen::LvalueBit);

template <typename TDerived, typename TValue>
using ConstValueIfVariableIsNotLValue_t = std::conditional_t<VariableIsLValue_v<TDerived>, TValue, const TValue>;

template <typename TDerived, typename TBase, typename TConstBase>
using ConditionalConstBase_t = std::conditional_t<VariableIsLValue_v<TDerived>, TBase, TConstBase>;

}  // namespace hyper::variables
