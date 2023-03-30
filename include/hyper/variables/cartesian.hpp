/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/jacobian.hpp"
#include "hyper/variables/variable.hpp"

namespace hyper::variables {

template <typename TDerived>
class CartesianTangentBase;

template <typename TDerived>
class CartesianBase : public Traits<TDerived>::Base, public ConditionalConstBase_t<TDerived, Variable<DerivedScalar_t<TDerived>>, ConstVariable<DerivedScalar_t<TDerived>>> {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = typename Base::Scalar;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, VectorX<Scalar>>;
  using Base::Base;

  using Tangent = variables::Tangent<Cartesian<Scalar, (int)Base::SizeAtCompileTime>>;

  // Constants.
  static constexpr auto kNumParameters = (int)Base::SizeAtCompileTime;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(CartesianBase)

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() const -> Eigen::Ref<const VectorX<Scalar>> final { return *this; }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Ref<VectorXWithConstIfNotLvalue> final { return *this; }

  /// Casts this to its derived type.
  /// \return Derived type.
  auto derived() const -> const TDerived& { return static_cast<const TDerived&>(*this); }

  /// Casts this to its derived type.
  /// \return Derived type.
  auto derived() -> TDerived& { return const_cast<TDerived&>(std::as_const(*this).derived()); }

  /// Tangent plus.
  /// \tparam TOther_ Other type.
  /// \param other Other element.
  /// \return Cartesian.
  template <typename TOther_>
  auto tPlus(const CartesianTangentBase<TOther_>& other) const -> Cartesian<Scalar, kNumParameters> {
    return *this + other;
  }

  /// Tangent minus.
  /// \tparam TOther_ Other type.
  /// \param other Other element.
  /// \return Tangent.
  template <typename TOther_>
  auto tMinus(const CartesianBase<TOther_>& other) const -> Tangent {
    return *this - other;
  }

  /// Tangent plus Jacobian.
  /// \return Jacobian.
  auto tPlusJacobian() const -> Jacobian<Scalar, kNumParameters, Traits<Tangent>::kNumParameters> {
    return Jacobian<Scalar, kNumParameters, Traits<Tangent>::kNumParameters>::Identity();
  }

  /// Tangent minus Jacobian.
  /// \return Jacobian.
  auto tMinusJacobian() const -> Jacobian<Scalar, Traits<Tangent>::kNumParameters, kNumParameters> {
    return Jacobian<Scalar, Traits<Tangent>::kNumParameters, kNumParameters>::Identity();
  }
};

template <typename TScalar, int TNumParameters>
class Cartesian final : public CartesianBase<Cartesian<TScalar, TNumParameters>> {
 public:
  using Base = CartesianBase<Cartesian<TScalar, TNumParameters>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Cartesian)
};

}  // namespace hyper::variables

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(hyper::variables::Cartesian, int)

namespace hyper::variables {

template <typename TDerived>
class CartesianTangentBase : public CartesianBase<TDerived> {
 public:
  using Base = CartesianBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(CartesianTangentBase)
};

template <typename TDerived>
class Tangent final : public CartesianTangentBase<Tangent<TDerived>> {
 public:
  using Base = CartesianTangentBase<Tangent<TDerived>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Tangent)
};

}  // namespace hyper::variables

namespace Eigen {

using namespace hyper::variables;

template <typename TDerived, int TMapOptions>
class Map<Tangent<TDerived>, TMapOptions> final : public CartesianTangentBase<Map<Tangent<TDerived>, TMapOptions>> {
 public:
  using Base = CartesianTangentBase<Map<Tangent<TDerived>, TMapOptions>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)
};

template <typename TDerived, int TMapOptions>
class Map<const Tangent<TDerived>, TMapOptions> final : public CartesianTangentBase<Map<const Tangent<TDerived>, TMapOptions>> {
 public:
  using Base = CartesianTangentBase<Map<const Tangent<TDerived>, TMapOptions>>;
  using Base::Base;
};

}  // namespace Eigen
