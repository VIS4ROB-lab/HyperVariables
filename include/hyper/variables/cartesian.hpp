/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/abstract.hpp"

namespace hyper {

template <typename TDerived>
class CartesianBase
    : public Traits<TDerived>::Base,
      public AbstractVariable<std::conditional_t<VariableIsLValue<TDerived>::value, typename Traits<TDerived>::Base::Scalar, const typename Traits<TDerived>::Base::Scalar>> {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = Base::Scalar;
  using ScalarWithConstIfNotLvalue = std::conditional_t<VariableIsLValue<TDerived>::value, Scalar, const Scalar>;
  using VectorXWithConstIfNotLvalue = std::conditional_t<VariableIsLValue<TDerived>::value, TVectorX<Scalar>, const TVectorX<Scalar>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(CartesianBase)

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() const -> Eigen::Map<const TVectorX<Scalar>> final {
    return {this->data(), this->size(), 1};
  }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Map<VectorXWithConstIfNotLvalue> final {
    return {this->data(), this->size(), 1};
  }
};

template <typename TScalar, int TNumParameters>
class Cartesian final
    : public CartesianBase<Cartesian<TScalar, TNumParameters>> {
 public:
  using Base = CartesianBase<Cartesian<TScalar, TNumParameters>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Cartesian)
};

} // namespace hyper

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(hyper::Cartesian, int)

namespace hyper {

template <typename TDerived>
class CartesianTangentBase
    : public CartesianBase<TDerived> {
 public:
  using Base = CartesianBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = std::conditional_t<VariableIsLValue<TDerived>::value, Scalar, const Scalar>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(CartesianTangentBase)
};

template <typename TDerived>
class Tangent final
    : public CartesianTangentBase<Tangent<TDerived>> {
 public:
  using Base = CartesianTangentBase<Tangent<TDerived>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Tangent)
};

} // namespace hyper

namespace Eigen {

using namespace hyper;

template <typename TDerived, int TMapOptions>
class Map<Tangent<TDerived>, TMapOptions> final
    : public CartesianTangentBase<Map<Tangent<TDerived>, TMapOptions>> {
 public:
  using Base = CartesianTangentBase<Map<Tangent<TDerived>, TMapOptions>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)
};

template <typename TDerived, int TMapOptions>
class Map<const Tangent<TDerived>, TMapOptions> final
    : public CartesianTangentBase<Map<const Tangent<TDerived>, TMapOptions>> {
 public:
  using Base = CartesianTangentBase<Map<const Tangent<TDerived>, TMapOptions>>;
  using Base::Base;
};

} // namespace Eigen
