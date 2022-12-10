/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/groups/se3.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper {

/// SU2 Jacobian adapter making
/// internal SU2 Jacobians compatible
/// with the Ceres SU2 manifold.
/// \tparam TScalar Scalar type.
/// \param raw_su2 Raw SU2 input.
/// \return Jacobian adapter.
template <typename TScalar>
auto SU2JacobianAdapter(const TScalar* raw_su2) -> TJacobianNM<Tangent<SU2<TScalar>>, SU2<TScalar>> {
  TJacobianNM<Tangent<SU2<TScalar>>, SU2<TScalar>> J;

  using Order = QuaternionOrder;
  TScalar tau[SU2<TScalar>::SizeAtCompileTime];
  tau[Order::kW] = TScalar{2} * raw_su2[Order::kW];
  tau[Order::kX] = TScalar{2} * raw_su2[Order::kX];
  tau[Order::kY] = TScalar{2} * raw_su2[Order::kY];
  tau[Order::kZ] = TScalar{2} * raw_su2[Order::kZ];

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

/// SE3 Jacobian adapter making
/// internal SE3 Jacobians compatible
/// with the Ceres SE3 manifold.
/// \tparam TScalar Scalar type.
/// \param raw_se3 Raw SE3 input.
/// \return Jacobian adapter.
template <typename TScalar>
auto SE3JacobianAdapter(const TScalar* raw_se3) -> TJacobianNM<Tangent<SE3<TScalar>>, SE3<TScalar>> {
  TJacobianNM<Tangent<SE3<TScalar>>, SE3<TScalar>> J;
  SE3<TScalar>::template RotationJacobian<Tangent<SE3<TScalar>>::kNumAngularParameters>(J, Tangent<SE3<TScalar>>::kAngularOffset).noalias() = SU2JacobianAdapter(raw_se3 + SE3<TScalar>::kRotationOffset);
  SE3<TScalar>::template TranslationJacobian<Tangent<SE3<TScalar>>::kNumAngularParameters>(J, Tangent<SE3<TScalar>>::kAngularOffset).setZero();
  SE3<TScalar>::template RotationJacobian<Tangent<SE3<TScalar>>::kNumLinearParameters>(J, Tangent<SE3<TScalar>>::kLinearOffset).setZero();
  SE3<TScalar>::template TranslationJacobian<Tangent<SE3<TScalar>>::kNumLinearParameters>(J, Tangent<SE3<TScalar>>::kLinearOffset).setIdentity();
  return J;
}

} // namespace hyper
