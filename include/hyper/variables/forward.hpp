/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Core>

#include "hyper/matrix.hpp"
#include "hyper/variables/macros.hpp"

namespace hyper {

namespace variables {

class Variable;

class ConstVariable;

template <int TOrder>
class Rn;

template <typename TVariable>
class Tangent;

template <typename TVariable>
class Stamped;

template <int TOrder>
class AxesOffset;

template <int TOrder>
class OrthonormalityAlignment;

template <int TOrder>
class Sensitivity;

class Bearing;
class Gravity;
class Intrinsics;
class PitchYaw;
class Quaternion;
class SU2;
class SE3;

template <typename TDerived>
inline constexpr bool VariableIsLValue_v = (!bool(std::is_const_v<TDerived>)) && bool(Traits<TDerived>::Base::Flags & Eigen::LvalueBit);

template <typename TDerived, typename TValue>
using ConstValueIfVariableIsNotLValue_t = std::conditional_t<VariableIsLValue_v<TDerived>, TValue, const TValue>;

template <typename TDerived, typename TBase, typename TConstBase>
using ConditionalConstBase_t = std::conditional_t<VariableIsLValue_v<TDerived>, TBase, TConstBase>;

using R1 = Rn<1>;
using R2 = Rn<2>;
using R3 = Rn<3>;
using R4 = Rn<4>;
using R5 = Rn<5>;
using R6 = Rn<6>;
using Rx = Rn<Eigen::Dynamic>;

using Stamp = R1;
using SU2Tangent = Tangent<SU2>;
using SE3Tangent = Tangent<SE3>;

}  // namespace variables

template <int TOrder>
struct Traits<variables::Rn<TOrder>> {
  // Constants.
  static constexpr auto kNumParameters = TOrder;

  // Definitions.
  using Base = Matrix<TOrder, 1>;
};

template <>
struct Traits<variables::PitchYaw> : public Traits<variables::R2> {};

template <>
struct Traits<variables::Bearing> : public Traits<variables::R3> {};

template <>
struct Traits<variables::Gravity> : public Traits<variables::R3> {};

template <>
struct Traits<variables::Intrinsics> : public Traits<variables::R4> {};

template <int TOrder>
struct Traits<variables::AxesOffset<TOrder>> : public Traits<variables::Rn<TOrder * TOrder>> {
  static constexpr auto kOrder = TOrder;
};

template <int TOrder>
struct Traits<variables::OrthonormalityAlignment<TOrder>> : public Traits<variables::Rn<TOrder + ((TOrder - 1) * TOrder) / 2>> {
  static constexpr auto kOrder = TOrder;
};

template <int TOrder>
struct Traits<variables::Sensitivity<TOrder>> : public Traits<variables::Rn<TOrder * TOrder>> {
  static constexpr auto kOrder = TOrder;
};

template <>
struct Traits<variables::Quaternion> : public Traits<variables::R4> {
  using Base = Eigen::Quaternion<Scalar>;
};

template <>
struct Traits<variables::SU2> : Traits<variables::Quaternion> {};

template <>
struct Traits<variables::SE3> : Traits<variables::Rn<Traits<variables::SU2>::kNumParameters + Traits<variables::R3>::kNumParameters>> {};

template <typename TDerived>
struct Traits<variables::Tangent<TDerived>> : Traits<TDerived> {};

template <typename TDerived, int TMapOptions>
struct Traits<Eigen::Map<variables::Tangent<TDerived>, TMapOptions>> : Traits<variables::Tangent<TDerived>> {
  using Base = typename Traits<Eigen::Map<TDerived, TMapOptions>>::Base;
};

template <typename TDerived, int TMapOptions>
struct Traits<Eigen::Map<const variables::Tangent<TDerived>, TMapOptions>> : Traits<variables::Tangent<TDerived>> {
  using Base = typename Traits<Eigen::Map<const TDerived, TMapOptions>>::Base;
};

template <>
struct Traits<variables::Tangent<variables::SU2>> : Traits<variables::R3> {};

template <>
struct Traits<variables::Tangent<variables::SE3>> : Traits<variables::R6> {};

template <typename TVariable>
struct Traits<variables::Stamped<TVariable>> : public Traits<variables::Rn<Traits<TVariable>::kNumParameters + Traits<variables::R1>::kNumParameters>> {
  using Variable = TVariable;
};

}  // namespace hyper
