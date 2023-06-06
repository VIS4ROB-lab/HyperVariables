/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/matrix.hpp"
#include "hyper/variables/variable.hpp"

namespace hyper::variables {

template <typename TDerived>
class RnTangentBase;

template <typename TDerived>
class RnBase : public Traits<TDerived>::Base, public ConditionalConstBase_t<TDerived, Variable, ConstVariable> {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = typename Base::Scalar;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, VectorX>;
  using Base::Base;

  using Tangent = variables::Tangent<Rn<Scalar, (int)Base::SizeAtCompileTime>>;

  // Constants.
  static constexpr auto kNumParameters = (int)Base::SizeAtCompileTime;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(RnBase)

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() const -> Eigen::Ref<const VectorX> final { return *this; }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Ref<VectorXWithConstIfNotLvalue> final { return *this; }

  /// Casts this to its derived type.
  /// \return Derived type.
  auto derived() const -> const TDerived& { return static_cast<const TDerived&>(*this); }

  /// Casts this to its derived type.
  /// \return Derived type.
  auto derived() -> TDerived& { return const_cast<TDerived&>(std::as_const(*this).derived()); }

  /// Group inverse.
  /// \param J_this Jacobian w.r.t. this.
  /// \return Group element.
  auto gInv(Scalar* J_this = nullptr) const -> Rn<Scalar, kNumParameters> {
    Rn<Scalar, kNumParameters> inv = Scalar{-1} * *this;

    if (!J_this) {
      return inv;
    }

    Eigen::Map<JacobianNM<Tangent>>{J_this} = Scalar{-1} * JacobianNM<Tangent>::Identity();
    return inv;
  }

  /// Group plus.
  /// \tparam TOther_ Other type.
  /// \param other Other element.
  /// \param J_this Jacobian w.r.t. this.
  /// \param J_other Jacobian w.r.t. other.
  /// \return Group element.
  template <typename TOther_>
  auto gPlus(const RnBase<TOther_>& other, Scalar* J_this = nullptr, Scalar* J_other = nullptr) const -> Rn<Scalar, kNumParameters> {
    Rn<Scalar, kNumParameters> plus = *this + other;

    if (!J_this && !J_other) {
      return plus;
    }
    if (J_this) {
      Eigen::Map<JacobianNM<Tangent>>{J_this}.setIdentity();
    }
    if (J_other) {
      Eigen::Map<JacobianNM<Tangent>>{J_other}.setIdentity();
    }
    return plus;
  }

  /// Group logarithm (Rn -> Rn tangent).
  /// \param J_this Jacobian w.r.t. this.
  /// \return Tangent element.
  [[nodiscard]] auto gLog(Scalar* J_this = nullptr) const -> Tangent {
    if (J_this) {
      Eigen::Map<JacobianNM<Tangent>>{J_this}.setIdentity();
    }
    return *this;
  }

  /// Tangent plus.
  /// \tparam TOther_ Other type.
  /// \param other Other element.
  /// \return Element.
  template <typename TOther_>
  auto tPlus(const RnTangentBase<TOther_>& other) const -> Rn<Scalar, kNumParameters> {
    return *this + other;
  }

  /// Tangent minus.
  /// \tparam TOther_ Other type.
  /// \param other Other element.
  /// \return Tangent.
  template <typename TOther_>
  auto tMinus(const RnBase<TOther_>& other) const -> Tangent {
    return *this - other;
  }

  /// Tangent plus Jacobian.
  /// \return Jacobian.
  auto tPlusJacobian() const -> Jacobian<kNumParameters, Traits<Tangent>::kNumParameters> { return Jacobian<kNumParameters, Traits<Tangent>::kNumParameters>::Identity(); }

  /// Tangent minus Jacobian.
  /// \return Jacobian.
  auto tMinusJacobian() const -> Jacobian<Traits<Tangent>::kNumParameters, kNumParameters> { return Jacobian<Traits<Tangent>::kNumParameters, kNumParameters>::Identity(); }
};

template <typename TScalar, int TOrder>
class Rn final : public RnBase<Rn<TScalar, TOrder>> {
 public:
  using Base = RnBase<Rn<TScalar, TOrder>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Rn)
};

}  // namespace hyper::variables

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(hyper::variables::Rn, int)

namespace hyper::variables {

template <typename TDerived>
class RnTangentBase : public RnBase<TDerived> {
 public:
  using Base = RnBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(RnTangentBase)

  /// Group exponential (Rn tangent -> Rn).
  /// \param J_this Jacobian w.r.t. this.
  /// \return Group element.
  auto gExp(Scalar* J_this = nullptr) const -> Rn<Scalar, Base::kNumParameters> {
    if (J_this) {
      Eigen::Map<JacobianNM<Rn<Scalar, Base::kNumParameters>>>{J_this}.setIdentity();
    }
    return *this;
  }
};

template <typename TDerived>
class Tangent final : public RnTangentBase<Tangent<TDerived>> {
 public:
  using Base = RnTangentBase<Tangent<TDerived>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Tangent)
};

}  // namespace hyper::variables

namespace Eigen {

using namespace hyper::variables;

template <typename TDerived, int TMapOptions>
class Map<Tangent<TDerived>, TMapOptions> final : public RnTangentBase<Map<Tangent<TDerived>, TMapOptions>> {
 public:
  using Base = RnTangentBase<Map<Tangent<TDerived>, TMapOptions>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)
};

template <typename TDerived, int TMapOptions>
class Map<const Tangent<TDerived>, TMapOptions> final : public RnTangentBase<Map<const Tangent<TDerived>, TMapOptions>> {
 public:
  using Base = RnTangentBase<Map<const Tangent<TDerived>, TMapOptions>>;
  using Base::Base;
};

}  // namespace Eigen
