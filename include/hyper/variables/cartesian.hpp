/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/abstract.hpp"

namespace hyper {

template <typename TDerived>
class CartesianBase
    : public Traits<TDerived>::Base,
      public AbstractVariable<typename Traits<TDerived>::ScalarWithConstIfNotLvalue> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = typename Traits<TDerived>::Base;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(CartesianBase)

  /// Memory accessor.
  /// \return Memory block.
  [[nodiscard]] auto memory() const -> MemoryBlock<const Scalar> final {
    return {this->data(), this->size()};
  }

  /// Memory modifier.
  /// \return Memory block.
  [[nodiscard]] auto memory() -> MemoryBlock<ScalarWithConstIfNotLvalue> final {
    return {this->data(), this->size()};
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
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = CartesianBase<TDerived>;
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
