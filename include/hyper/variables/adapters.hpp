/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/groups/groups.hpp"
#include "hyper/variables/jacobian.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::variables {

namespace internal {

template <typename TVariable>
struct JacobianAdapterImpl {
  // Definitions.
  using Scalar = typename TVariable::Scalar;
  using Group = TVariable;
  using Tangent = variables::Tangent<Group>;

  using GroupToTangentJacobian = JacobianNM<Group, Tangent>;
  using TangentToGroupJacobian = JacobianNM<Tangent, Group>;

  /// Adapter from group to tangent Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto groupToTangentJacobian(const Scalar* /* values */) -> GroupToTangentJacobian { return GroupToTangentJacobian::Identity(); }

  /// Adapter from tangent to group Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto tangentToGroupJacobian(const Scalar* /* values */) -> TangentToGroupJacobian { return TangentToGroupJacobian::Identity(); }
};

template <typename TScalar>
struct JacobianAdapterImpl<SU2<TScalar>> {
  // Definitions.
  using Group = variables::SU2<TScalar>;
  using Tangent = variables::Tangent<Group>;

  using GroupToTangentJacobian = JacobianNM<Group, Tangent>;
  using TangentToGroupJacobian = JacobianNM<Tangent, Group>;

  /// Adapter from group to tangent Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto groupToTangentJacobian(const TScalar* values) -> GroupToTangentJacobian {
    GroupToTangentJacobian J;

    using Order = typename Group::Order;
    TScalar tau[SU2<TScalar>::kNumParameters];
    tau[Order::kW] = Group::kiAlpha * values[Order::kW];
    tau[Order::kX] = Group::kiAlpha * values[Order::kX];
    tau[Order::kY] = Group::kiAlpha * values[Order::kY];
    tau[Order::kZ] = Group::kiAlpha * values[Order::kZ];

#if HYPER_USE_GLOBAL_MANIFOLD_DERIVATIVES
    J(Order::kX, 0) = tau[Order::kW];
    J(Order::kY, 0) = -tau[Order::kZ];
    J(Order::kZ, 0) = tau[Order::kY];
    J(Order::kW, 0) = -tau[Order::kX];

    J(Order::kX, 1) = tau[Order::kZ];
    J(Order::kY, 1) = tau[Order::kW];
    J(Order::kZ, 1) = -tau[Order::kX];
    J(Order::kW, 1) = -tau[Order::kY];

    J(Order::kX, 2) = -tau[Order::kY];
    J(Order::kY, 2) = tau[Order::kX];
    J(Order::kZ, 2) = tau[Order::kW];
    J(Order::kW, 2) = -tau[Order::kZ];
#else
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
#endif
    return J;
  }

  /// Adapter from tangent to group Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto tangentToGroupJacobian(const TScalar* values) -> TangentToGroupJacobian {
    TangentToGroupJacobian J;

    using Order = typename Group::Order;
    TScalar tau[SU2<TScalar>::kNumParameters];
    tau[Order::kW] = Group::kAlpha * values[Order::kW];
    tau[Order::kX] = Group::kAlpha * values[Order::kX];
    tau[Order::kY] = Group::kAlpha * values[Order::kY];
    tau[Order::kZ] = Group::kAlpha * values[Order::kZ];

#if HYPER_USE_GLOBAL_MANIFOLD_DERIVATIVES
    J(0, Order::kX) = tau[Order::kW];
    J(1, Order::kX) = tau[Order::kZ];
    J(2, Order::kX) = -tau[Order::kY];

    J(0, Order::kY) = -tau[Order::kZ];
    J(1, Order::kY) = tau[Order::kW];
    J(2, Order::kY) = tau[Order::kX];

    J(0, Order::kZ) = tau[Order::kY];
    J(1, Order::kZ) = -tau[Order::kX];
    J(2, Order::kZ) = tau[Order::kW];

    J(0, Order::kW) = -tau[Order::kX];
    J(1, Order::kW) = -tau[Order::kY];
    J(2, Order::kW) = -tau[Order::kZ];
#else
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
#endif
    return J;
  }
};

template <class TScalar>
struct JacobianAdapterImpl<SE3<TScalar>> {
  // Definitions.
  using Group = variables::SE3<TScalar>;
  using Tangent = variables::Tangent<Group>;

  using GroupToTangentJacobian = JacobianNM<Group, Tangent>;
  using TangentToGroupJacobian = JacobianNM<Tangent, Group>;

  /// Adapter from group to tangent Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto groupToTangentJacobian(const TScalar* values) -> GroupToTangentJacobian {
    GroupToTangentJacobian J;
    Tangent::template AngularJacobian<Group::kNumRotationParameters>(J, Group::kRotationOffset).noalias() =
        JacobianAdapterImpl<SU2<TScalar>>::groupToTangentJacobian(values + Group::kRotationOffset);
    Tangent::template LinearJacobian<Group::kNumRotationParameters>(J, Group::kRotationOffset).setZero();
    Tangent::template AngularJacobian<Group::kNumTranslationParameters>(J, Group::kTranslationOffset).setZero();
    Tangent::template LinearJacobian<Group::kNumTranslationParameters>(J, Group::kTranslationOffset).setIdentity();
    return J;
  }

  /// Adapter from tangent to group Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto tangentToGroupJacobian(const TScalar* values) -> TangentToGroupJacobian {
    TangentToGroupJacobian J;
    Group::template RotationJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() =
        JacobianAdapterImpl<SU2<TScalar>>::tangentToGroupJacobian(values + Group::kRotationOffset);
    Group::template TranslationJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
    Group::template RotationJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
    Group::template TranslationJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
    return J;
  }
};

template <class TVariable>
struct JacobianAdapterImpl<Stamped<TVariable>> {
  // Definitions.
  using Scalar = typename TVariable::Scalar;
  using Stamp = variables::Stamp<Scalar>;
  using Group = variables::Stamped<TVariable>;
  using Tangent = variables::Stamped<variables::Tangent<TVariable>>;

  using GroupToTangentJacobian = JacobianNM<Group, Tangent>;
  using TangentToGroupJacobian = JacobianNM<Tangent, Group>;

  /// Adapter from stamped group to stamped tangent Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto groupToTangentJacobian(const Scalar* values) -> GroupToTangentJacobian {
    GroupToTangentJacobian J;
    J.template block<TVariable::kNumParameters, variables::Tangent<TVariable>::kNumParameters>(Group::kVariableOffset, Tangent::kVariableOffset).noalias() =
        JacobianAdapterImpl<TVariable>::groupToTangentJacobian(values + Group::kVariableOffset);
    J.template block<Stamp::kNumParameters, variables::Tangent<TVariable>::kNumParameters>(Group::kStampOffset, Tangent::kVariableOffset).setZero();
    J.template block<TVariable::kNumParameters, Stamp::kNumParameters>(Group::kVariableOffset, Tangent::kStampOffset).setZero();
    J.template block<Stamp::kNumParameters, Stamp::kNumParameters>(Group::kStampOffset, Tangent::kStampOffset).setIdentity();
    return J;
  }

  /// Adapter from stamped tangent to stamped group Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto tangentToGroupJacobian(const Scalar* values) -> TangentToGroupJacobian {
    TangentToGroupJacobian J;
    J.template block<variables::Tangent<TVariable>::kNumParameters, TVariable::kNumParameters>(Tangent::kVariableOffset, Group::kVariableOffset).noalias() =
        JacobianAdapterImpl<TVariable>::tangentToGroupJacobian(values + Group::kVariableOffset);
    J.template block<variables::Tangent<TVariable>::kNumParameters, Stamp::kNumParameters>(Tangent::kVariableOffset, Group::kStampOffset).setZero();
    J.template block<Stamp::kNumParameters, TVariable::kNumParameters>(Tangent::kStampOffset, Group::kVariableOffset).setZero();
    J.template block<Stamp::kNumParameters, Stamp::kNumParameters>(Tangent::kStampOffset, Group::kStampOffset).setIdentity();
    return J;
  }
};

}  // namespace internal

template <typename TVariable>
inline auto JacobianAdapter(const typename TVariable::Scalar* values) -> internal::JacobianAdapterImpl<TVariable>::TangentToGroupJacobian {
  return internal::JacobianAdapterImpl<TVariable>::tangentToGroupJacobian(values);
}

template <typename TVariable>
inline auto InverseJacobianAdapter(const typename TVariable::Scalar* values) -> internal::JacobianAdapterImpl<TVariable>::GroupToTangentJacobian {
  return internal::JacobianAdapterImpl<TVariable>::groupToTangentJacobian(values);
}

}  // namespace hyper::variables
