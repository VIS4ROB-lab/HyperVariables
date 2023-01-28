/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/groups/groups.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper::variables {

namespace internal {

template <typename TVariable>
struct JacobianAdapterImpl {
  // Definitions.
  using Scalar = typename TVariable::Scalar;
  using Group = TVariable;
  using Tangent = variables::Tangent<Group>;

  /// Adapter from group to tangent Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto project(const Scalar* /* values */) -> JacobianNM<Group, Tangent> { return JacobianNM<Group, Tangent>::Identity(); }

  /// Adapter from tangent to group Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto lift(const Scalar* /* values */) -> JacobianNM<Tangent, Group> { return JacobianNM<Tangent, Group>::Identity(); }
};

template <typename TScalar>
struct JacobianAdapterImpl<SU2<TScalar>> {
  // Definitions.
  using Group = variables::SU2<TScalar>;
  using Tangent = variables::Tangent<Group>;

  /// Adapter from group to tangent Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto project(const TScalar* values) -> JacobianNM<Group, Tangent> {
    JacobianNM<Group, Tangent> J;

    using Order = Group::Ordering;
    TScalar tau[SU2<TScalar>::kNumParameters];
    tau[Order::kW] = Group::kiAlpha * values[Order::kW];
    tau[Order::kX] = Group::kiAlpha * values[Order::kX];
    tau[Order::kY] = Group::kiAlpha * values[Order::kY];
    tau[Order::kZ] = Group::kiAlpha * values[Order::kZ];

    J(Order::kX, 0) = tau[Order::kW];
    J(Order::kY, 0) = tau[Order::kZ];
    J(Order::kZ, 0) = -tau[Order::kY];
    J(Order::kW, 0) = -tau[Order::kX];

    J(Order::kX, 1) = -tau[Order::kZ];
    J(Order::kY, 1) = tau[Order::kW];
    J(Order::kZ, 1) = tau[Order::kX];
    J(Order::kW, 1) = -tau[Order::kY];

    J(Order::kX, 2) = tau[Order::kY];
    J(Order::kY, 2) = -tau[Order::kX];
    J(Order::kZ, 2) = tau[Order::kW];
    J(Order::kW, 2) = -tau[Order::kZ];

    return J;
  }

  /// Adapter from tangent to group Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto lift(const TScalar* values) -> JacobianNM<Tangent, Group> {
    JacobianNM<Tangent, Group> J;

    using Order = Group::Ordering;
    TScalar tau[SU2<TScalar>::kNumParameters];
    tau[Order::kW] = Group::kAlpha * values[Order::kW];
    tau[Order::kX] = Group::kAlpha * values[Order::kX];
    tau[Order::kY] = Group::kAlpha * values[Order::kY];
    tau[Order::kZ] = Group::kAlpha * values[Order::kZ];

    J(0, Order::kX) = tau[Order::kW];
    J(1, Order::kX) = -tau[Order::kZ];
    J(2, Order::kX) = tau[Order::kY];

    J(0, Order::kY) = tau[Order::kZ];
    J(1, Order::kY) = tau[Order::kW];
    J(2, Order::kY) = -tau[Order::kX];

    J(0, Order::kZ) = -tau[Order::kY];
    J(1, Order::kZ) = tau[Order::kX];
    J(2, Order::kZ) = tau[Order::kW];

    J(0, Order::kW) = -tau[Order::kX];
    J(1, Order::kW) = -tau[Order::kY];
    J(2, Order::kW) = -tau[Order::kZ];

    return J;
  }
};

template <class TScalar>
struct JacobianAdapterImpl<SE3<TScalar>> {
  // Definitions.
  using Group = variables::SE3<TScalar>;
  using Tangent = variables::Tangent<Group>;

  /// Adapter from group to tangent Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto project(const TScalar* values) -> JacobianNM<Group, Tangent> {
    JacobianNM<Group, Tangent> J;
    Tangent::template AngularJacobian<Group::kNumRotationParameters>(J, Group::kRotationOffset).noalias() =
        JacobianAdapterImpl<SU2<TScalar>>::project(values + Group::kRotationOffset);
    Tangent::template LinearJacobian<Group::kNumRotationParameters>(J, Group::kRotationOffset).setZero();
    Tangent::template AngularJacobian<Group::kNumTranslationParameters>(J, Group::kTranslationOffset).setZero();
    Tangent::template LinearJacobian<Group::kNumTranslationParameters>(J, Group::kTranslationOffset).setIdentity();
    return J;
  }

  /// Adapter from tangent to group Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto lift(const TScalar* values) -> JacobianNM<Tangent, Group> {
    JacobianNM<Tangent, Group> J;
    Group::template RotationJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() =
        JacobianAdapterImpl<SU2<TScalar>>::lift(values + Group::kRotationOffset);
    Group::template TranslationJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
    Group::template RotationJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
    Group::template TranslationJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
    return J;
  }
};

}  // namespace internal

template <typename TVariable>
inline auto JacobianAdapter(const typename TVariable::Scalar* values) -> JacobianNM<Tangent<TVariable>, TVariable> {
  return internal::JacobianAdapterImpl<TVariable>::lift(values);
}

template <typename TVariable>
inline auto InverseJacobianAdapter(const typename TVariable::Scalar* values) -> JacobianNM<TVariable, Tangent<TVariable>> {
  return internal::JacobianAdapterImpl<TVariable>::project(values);
}

}  // namespace hyper::variables
