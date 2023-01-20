/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/groups/groups.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper::variables {

namespace internal {

template <typename TVariable>
struct JacobianAdapterImpl;

template <typename TScalar>
struct JacobianAdapterImpl<SU2<TScalar>> {
  // Constants.
  static constexpr auto kAlpha = TScalar{2.0};
  static constexpr auto kiAlpha = 1 / kAlpha;

  // Definitions.
  using Manifold = variables::SU2<TScalar>;
  using Tangent = variables::Tangent<Manifold>;

  /// Adapter from manifold to tangent Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto project(const TScalar* values) -> JacobianNM<Manifold, Tangent> {
    JacobianNM<Manifold, Tangent> J;

    using Order = Manifold::Ordering;
    TScalar tau[SU2<TScalar>::kNumParameters];
    tau[Order::kW] = kiAlpha * values[Order::kW];
    tau[Order::kX] = kiAlpha * values[Order::kX];
    tau[Order::kY] = kiAlpha * values[Order::kY];
    tau[Order::kZ] = kiAlpha * values[Order::kZ];

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

  /// Adapter from tangent to manifold Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto lift(const TScalar* values) -> JacobianNM<Tangent, Manifold> {
    JacobianNM<Tangent, Manifold> J;

    using Order = Manifold::Ordering;
    TScalar tau[SU2<TScalar>::kNumParameters];
    tau[Order::kW] = kAlpha * values[Order::kW];
    tau[Order::kX] = kAlpha * values[Order::kX];
    tau[Order::kY] = kAlpha * values[Order::kY];
    tau[Order::kZ] = kAlpha * values[Order::kZ];

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
  using Manifold = variables::SE3<TScalar>;
  using Tangent = variables::Tangent<Manifold>;

  /// Adapter from manifold to tangent Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto project(const TScalar* values) -> JacobianNM<Manifold, Tangent> {
    JacobianNM<Manifold, Tangent> J;
    Tangent::template AngularJacobian<Manifold::kNumRotationParameters>(J, Manifold::kRotationOffset).noalias() =
        JacobianAdapterImpl<SU2<TScalar>>::project(values + Manifold::kRotationOffset);
    Tangent::template LinearJacobian<Manifold::kNumRotationParameters>(J, Manifold::kRotationOffset).setZero();
    Tangent::template AngularJacobian<Manifold::kNumTranslationParameters>(J, Manifold::kTranslationOffset).setZero();
    Tangent::template LinearJacobian<Manifold::kNumTranslationParameters>(J, Manifold::kTranslationOffset).setIdentity();
    return J;
  }

  /// Adapter from tangent to manifold Jacobian.
  /// \param values Values.
  /// \return Jacobian.
  static auto lift(const TScalar* values) -> JacobianNM<Tangent, Manifold> {
    JacobianNM<Tangent, Manifold> J;
    Manifold::template RotationJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() =
        JacobianAdapterImpl<SU2<TScalar>>::lift(values + Manifold::kRotationOffset);
    Manifold::template TranslationJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
    Manifold::template RotationJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
    Manifold::template TranslationJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
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
