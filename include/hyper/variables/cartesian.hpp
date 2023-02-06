/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

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

  using Index = Eigen::Index;

  // Constants.
  static constexpr auto kNumParameters = (int)Base::SizeAtCompileTime;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(CartesianBase)

  /// Retrieves the manifold size.
  /// \return Manifold size.
  [[nodiscard]] auto manifoldSize() const -> Index override { return kNumParameters; }

  /// Retrieves the tangent size.
  /// \return Tangent size.
  [[nodiscard]] auto tangentSize() const -> Index override { return kNumParameters; }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() const -> Eigen::Ref<const VectorX<Scalar>> final { return *this; }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Ref<VectorXWithConstIfNotLvalue> final { return *this; }

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
  auto tMinus(const CartesianBase<TOther_>& other) const -> Tangent<Cartesian<Scalar, kNumParameters>> {
    return *this - other;
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
