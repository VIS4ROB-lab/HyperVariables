/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>
#include <Eigen/Geometry>

#include "hyper/variables/rn.hpp"

namespace hyper::variables {

template <typename TDerived>
class SU2TangentBase;

template <typename TDerived>
class QuaternionBase : public Traits<TDerived>::Base, public ConditionalConstBase_t<TDerived, Variable, ConstVariable> {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Scalar>;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, VectorX>;
  using Base::Base;
  using Base::operator*;

  using Translation = R3;
  using TranslationJacobian = Jacobian<Traits<Translation>::kNumParameters>;

  // Constants.
  static constexpr auto SizeAtCompileTime = (int)Base::Coefficients::SizeAtCompileTime;
  static constexpr auto kNumParameters = (int)Base::Coefficients::SizeAtCompileTime;

  // Order.
  struct Order {
    static constexpr auto kX = 0;
    static constexpr auto kY = 1;
    static constexpr auto kZ = 2;
    static constexpr auto kW = 3;
  };

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(QuaternionBase)

  /// Identity group element.
  /// \return Identity element.
  static auto Identity() -> Quaternion;

  /// Random group element.
  /// \return Random element.
  static auto Random() -> Quaternion;

  /// Sets this to identity.
  /// \return Derived type.
  auto setIdentity() -> TDerived&;

  /// Sets this to random.
  /// \return Derived type.
  auto setRandom() -> TDerived&;

  /// Data accessor.
  /// \return Data.
  [[nodiscard]] auto data() const -> const Scalar* { return this->coeffs().data(); }

  /// Data modifier.
  /// \return Data.
  [[nodiscard]] auto data() -> ScalarWithConstIfNotLvalue* { return this->coeffs().data(); }

  /// Map as Eigen vector.
  /// \return Vector.
  [[nodiscard]] auto asVector() const -> Eigen::Ref<const VectorX> final { return this->coeffs(); }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Ref<VectorXWithConstIfNotLvalue> final { return this->coeffs(); }

  /// Casts this to its derived type.
  /// \return Derived type.
  [[nodiscard]] auto derived() const -> const TDerived& { return static_cast<const TDerived&>(*this); }

  /// Casts this to its derived type.
  /// \return Derived type.
  auto derived() -> TDerived& { return const_cast<TDerived&>(std::as_const(*this).derived()); }

  /// Group inverse.
  /// \return Inverse element.
  [[nodiscard]] auto gInv() const -> Quaternion;

  /// Group plus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \return Group element.
  template <typename Other_>
  auto gPlus(const Eigen::QuaternionBase<Other_>& other) const -> Quaternion;

  /// Group logarithm (quaternion -> quaternion).
  /// \return Group element.
  [[nodiscard]] auto glog() const -> Quaternion;

  /// Group exponential (quaternion -> quaternion).
  /// \return Group element.
  [[nodiscard]] auto gexp() const -> Quaternion;

  /// Group action.
  /// \tparam Other_ Other type.
  /// \param other Other vector.
  /// \return Vector.
  template <typename Other_>
  auto act(const Eigen::MatrixBase<Other_>& other) const -> Translation;
};

template <typename TDerived>
class SU2Base : public QuaternionBase<TDerived> {
 public:
  using Base = QuaternionBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;
  using Base::operator*;

  using Translation = typename Base::Translation;
  using TranslationJacobian = typename Base::TranslationJacobian;
  using Order = typename Base::Order;

  using Tangent = variables::Tangent<SU2>;

  using Adjoint = Matrix<Traits<Tangent>::kNumParameters>;
  using GroupJacobian = Jacobian<Base::kNumParameters>;
  using TangentJacobian = Jacobian<Traits<Tangent>::kNumParameters>;
  using GroupToTangentJacobian = Jacobian<Base::kNumParameters, Traits<Tangent>::kNumParameters>;
  using TangentToGroupJacobian = Jacobian<Traits<Tangent>::kNumParameters, Base::kNumParameters>;
  using ActionJacobian = Jacobian<Traits<Translation>::kNumParameters, Traits<Tangent>::kNumParameters>;

  static constexpr auto kAlpha = Scalar{2.0};
  static constexpr auto kiAlpha = Scalar{1} / kAlpha;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SU2Base)

  /// Identity group element.
  /// \return Identity element.
  static auto Identity() -> SU2;

  /// Random group element.
  /// \return Random element.
  static auto Random() -> SU2;

  /// Group inverse.
  /// \param J_this Jacobian w.r.t. this.
  /// \return Inverse group element.
  [[nodiscard]] auto gInv(Scalar* J_this = nullptr) const -> SU2;

  /// Group plus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \param J_this Jacobian w.r.t. this.
  /// \param J_other Jacobian w.r.t. other.
  /// \return Group element.
  template <typename Other_>
  auto gPlus(const SU2Base<Other_>& other, Scalar* J_this = nullptr, Scalar* J_other = nullptr) const -> SU2;

  /// Group logarithm (SU2 -> SU2 tangent).
  /// \param J_this Jacobian w.r.t. this.
  /// \return Tangent element.
  [[nodiscard]] auto gLog(Scalar* J_this = nullptr) const -> Tangent;

  /// Group exponential (SU2 -> quaternion).
  /// \return Group element.
  [[nodiscard]] auto gExp() const -> Quaternion;

  /// Tangent plus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \return Group element.
  template <typename Other_>
  auto tPlus(const SU2TangentBase<Other_>& other) const -> SU2;

  /// Tangent minus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \return Tangent element.
  template <typename Other_>
  auto tMinus(const SU2Base<Other_>& other) const -> Tangent;

  /// Tangent plus Jacobian.
  /// \return Jacobian.
  [[nodiscard]] auto tPlusJacobian() const -> GroupToTangentJacobian;

  /// Tangent minus Jacobian.
  /// \return Jacobian.
  [[nodiscard]] auto tMinusJacobian() const -> TangentToGroupJacobian;

  /// Group adjoint.
  /// \return Adjoint.
  [[nodiscard]] auto gAdj() const -> Adjoint;

  /// Group action.
  /// \tparam Other_ Other type.
  /// \param other Other vector.
  /// \param J_this Jacobian w.r.t. this.
  /// \param J_other Jacobian w.r.t. other.
  /// \return Vector.
  template <typename Other_>
  auto act(const Eigen::MatrixBase<Other_>& other, Scalar* J_this = nullptr, Scalar* J_other = nullptr) const -> Translation;
};

class Quaternion final : public QuaternionBase<Quaternion> {
 public:
  using Base = QuaternionBase<Quaternion>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Quaternion)
};

HYPER_DECLARE_EIGEN_CLASS_TRAITS(Map, hyper::variables::Quaternion)

class SU2 final : public SU2Base<SU2> {
 public:
  using Base = SU2Base<SU2>;

  /// Default constructor.
  SU2() : Base{Scalar{1}, Scalar{0}, Scalar{0}, Scalar{0}} {}

  /// Perfect forwarding constructor.
  template <typename... TArgs>
  SU2(TArgs&&... args) : Base{std::forward<TArgs>(args)...} {}  // NOLINT

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SU2)
};

HYPER_DECLARE_EIGEN_CLASS_TRAITS(Map, hyper::variables::SU2)

template <typename TDerived>
class SU2TangentBase : public RnBase<TDerived> {
 public:
  // Definitions.
  using Base = RnBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  using Jacobian = hyper::Jacobian<Traits<Tangent<SU2>>::kNumParameters>;

  // Constants.
  static constexpr auto kAngularOffset = 0;
  static constexpr auto kNumAngularParameters = 3;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SU2TangentBase)

  /// Group exponential (SU2 tangent -> SU2).
  /// \param J_this Jacobian w.r.t. this.
  /// \return Group element.
  auto gExp(Scalar* J_this = nullptr) const -> SU2 {
    // Constants.
    constexpr auto kiAlpha = SU2::kiAlpha;

    const auto nw2 = this->squaredNorm();
    const auto nv2 = kiAlpha * kiAlpha * nw2;
    const auto nv = std::sqrt(nv2);

    Scalar s, c;
    if (nv < Eigen::NumTraits<Scalar>::epsilon()) {
      s = Scalar{1};
      c = Scalar{1};
    } else {
      s = std::sin(nv) / nv;
      c = std::cos(nv);
    }

    const auto a = kiAlpha * s;
    SU2 exp = {c, a * this->x(), a * this->y(), a * this->z()};

    if (!J_this) {
      return exp;
    }

    const auto nw = std::sqrt(nw2);
    const auto nw3 = nw * nw2;

    if (nw3 < Eigen::NumTraits<Scalar>::epsilon()) {
      Eigen::Map<Jacobian>{J_this}.setIdentity();
    } else {
      const auto Wx = this->hat();
#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
      Eigen::Map<Jacobian>{J_this}.noalias() = Jacobian::Identity() + (Scalar{1} - std::cos(nw)) / nw2 * Wx + (nw - std::sin(nw)) / nw3 * Wx * Wx;
#else
      Eigen::Map<Jacobian>{J_this}.noalias() = Jacobian::Identity() - (Scalar{1} - std::cos(nw)) / nw2 * Wx + (nw - std::sin(nw)) / nw3 * Wx * Wx;
#endif
    }
    return exp;
  }
};

template <>
class Tangent<SU2> final : public SU2TangentBase<Tangent<SU2>> {
 public:
  using Base = SU2TangentBase<Tangent<SU2>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Tangent)
};

HYPER_DECLARE_TANGENT_MAP_TRAITS(hyper::variables::SU2)

template <typename TDerived>
auto QuaternionBase<TDerived>::Identity() -> Quaternion {
  return Base::Identity();
}

template <typename TDerived>
auto QuaternionBase<TDerived>::Random() -> Quaternion {
  return Eigen::Quaternion<Scalar>::UnitRandom();
}

template <typename TDerived>
auto QuaternionBase<TDerived>::setIdentity() -> TDerived& {
  *this = Identity();
  return this->derived();
}

template <typename TDerived>
auto QuaternionBase<TDerived>::setRandom() -> TDerived& {
  *this = Random();
  return this->derived();
}

template <typename TDerived>
auto QuaternionBase<TDerived>::gInv() const -> Quaternion {
  return this->inverse();
}

template <typename TDerived>
template <typename Other_>
auto QuaternionBase<TDerived>::gPlus(const Eigen::QuaternionBase<Other_>& other) const -> Quaternion {
  return Base::operator*(other);
}

template <typename TDerived>
auto QuaternionBase<TDerived>::glog() const -> Quaternion {
  const auto nv2 = this->vec().squaredNorm();
  const auto w2 = this->w() * this->w();
  const auto nq2 = nv2 + w2;

  const auto nv = std::sqrt(nv2);
  const auto nq = std::sqrt(nq2);

  DLOG_IF(FATAL, nq < Eigen::NumTraits<Scalar>::epsilon()) << "Quaternion norm is zero.";
  const auto a = (nv < Eigen::NumTraits<Scalar>::epsilon()) ? (Scalar{1} / nq) : (std::atan2(nv, this->w()) / nv);
  return {std::log(nq), a * this->x(), a * this->y(), a * this->z()};
}

template <typename TDerived>
auto QuaternionBase<TDerived>::gexp() const -> Quaternion {
  const auto nv2 = this->vec().squaredNorm();
  const auto nv = std::sqrt(nv2);

  Scalar s, c;
  if (nv < Eigen::NumTraits<Scalar>::epsilon()) {
    s = Scalar{1};
    c = Scalar{1};
  } else {
    s = std::sin(nv) / nv;
    c = std::cos(nv);
  }

  const auto exp = std::exp(this->w());
  const auto a = exp * s;
  return {c * exp, a * this->x(), a * this->y(), a * this->z()};
}

template <typename TDerived>
template <typename Other_>
auto QuaternionBase<TDerived>::act(const Eigen::MatrixBase<Other_>& other) const -> Translation {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Other_, 3);
  return *this * other;
}

template <typename TDerived>
auto SU2Base<TDerived>::Identity() -> SU2 {
  return Base::Identity();
}

template <typename TDerived>
auto SU2Base<TDerived>::Random() -> SU2 {
  return Base::UnitRandom();
}

template <typename TDerived>
auto SU2Base<TDerived>::gInv(Scalar* J_this) const -> SU2 {
  auto inv = this->conjugate();
  if (!J_this) {
    return inv;
  }

#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
  Eigen::Map<TangentJacobian>{J_this}.noalias() = Scalar{-1} * inv.matrix();
#else
  Eigen::Map<TangentJacobian>{J_this}.noalias() = Scalar{-1} * this->matrix();
#endif
  return inv;
}

template <typename TDerived>
template <typename Other_>
auto SU2Base<TDerived>::gPlus(const SU2Base<Other_>& other, Scalar* J_this, Scalar* J_other) const -> SU2 {
  if (!J_this && !J_other) {
    return *this * other;
  }

#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
  if (J_this) {
    Eigen::Map<TangentJacobian>{J_this}.setIdentity();
  }
  if (J_other) {
    Eigen::Map<TangentJacobian>{J_other}.noalias() = this->matrix();
  }
#else
  if (J_this) {
    Eigen::Map<TangentJacobian>{J_this}.noalias() = other.inverse().matrix();
  }
  if (J_other) {
    Eigen::Map<TangentJacobian>{J_other}.setIdentity();
  }
#endif
  return *this * other;
}

template <typename TDerived>
auto SU2Base<TDerived>::gLog(Scalar* J_this) const -> Tangent {
  const auto nv2 = this->vec().squaredNorm();
  const auto w2 = this->w() * this->w();
  const auto nq2 = nv2 + w2;

  const auto nv = std::sqrt(nv2);
  const auto nq = std::sqrt(nq2);

  DLOG_IF(FATAL, nq < Eigen::NumTraits<Scalar>::epsilon()) << "Quaternion norm is zero.";
  const auto a = (nv < Eigen::NumTraits<Scalar>::epsilon()) ? (Scalar{1} / nq) : (std::atan2(nv, this->w()) / nv);
  Tangent log = kAlpha * a * this->vec();

  if (!J_this) {
    return log;
  }

  const auto nw2 = kAlpha * kAlpha * a * a * nv2;
  const auto nw = kAlpha * a * nv;
  const auto nw3 = nw * nw2;

  if (nw3 < Eigen::NumTraits<Scalar>::epsilon()) {
    Eigen::Map<TangentJacobian>{J_this}.setIdentity();
  } else {
    const auto Wx = log.hat();
#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
    Eigen::Map<TangentJacobian>{J_this}.noalias() =
        TangentJacobian::Identity() - Scalar{0.5} * Wx + (Scalar{1} / nw2 - (Scalar{1} + std::cos(nw)) / (Scalar{2} * nw * std::sin(nw))) * Wx * Wx;
#else
    Eigen::Map<TangentJacobian>{J_this}.noalias() =
        TangentJacobian::Identity() + Scalar{0.5} * Wx + (Scalar{1} / nw2 - (Scalar{1} + std::cos(nw)) / (Scalar{2} * nw * std::sin(nw))) * Wx * Wx;
#endif
  }
  return log;
}

template <typename TDerived>
auto SU2Base<TDerived>::gExp() const -> Quaternion {
  return Base::gExp();
}

template <typename TDerived>
template <typename Other_>
auto SU2Base<TDerived>::tPlus(const SU2TangentBase<Other_>& other) const -> SU2 {
#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
  return other.gExp().gPlus(*this);
#else
  return this->gPlus(other.gExp());
#endif
}

template <typename TDerived>
template <typename Other_>
auto SU2Base<TDerived>::tMinus(const SU2Base<Other_>& other) const -> Tangent {
#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
  return this->gPlus(other.gInv()).gLog();
#else
  return other.gInv().gPlus(*this).gLog();
#endif
}

template <typename TDerived>
auto SU2Base<TDerived>::tPlusJacobian() const -> GroupToTangentJacobian {
  GroupToTangentJacobian J;
  const auto ptr = this->data();
  Scalar tau[Base::kNumParameters];
  tau[Order::kW] = kiAlpha * ptr[Order::kW];
  tau[Order::kX] = kiAlpha * ptr[Order::kX];
  tau[Order::kY] = kiAlpha * ptr[Order::kY];
  tau[Order::kZ] = kiAlpha * ptr[Order::kZ];
#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
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

template <typename TDerived>
auto SU2Base<TDerived>::tMinusJacobian() const -> TangentToGroupJacobian {
  TangentToGroupJacobian J;
  const auto ptr = this->data();
  Scalar tau[Base::kNumParameters];
  tau[Order::kW] = kAlpha * ptr[Order::kW];
  tau[Order::kX] = kAlpha * ptr[Order::kX];
  tau[Order::kY] = kAlpha * ptr[Order::kY];
  tau[Order::kZ] = kAlpha * ptr[Order::kZ];
#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
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

template <typename TDerived>
auto SU2Base<TDerived>::gAdj() const -> Adjoint {
#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
  return Adjoint::Identity();
#else
  return this->matrix();
#endif
}

template <typename TDerived>
template <typename Other_>
auto SU2Base<TDerived>::act(const Eigen::MatrixBase<Other_>& other, Scalar* J_this, Scalar* J_other) const -> Translation {
  auto x = Base::act(other);
  if (!J_this && !J_other) {
    return x;
  }

  if (J_this) {
#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
    Eigen::Map<ActionJacobian>{J_this}.noalias() = Scalar{-1} * x.hat();
#else
    Eigen::Map<ActionJacobian>{J_this}.noalias() = Scalar{-1} * this->matrix() * other.hat();
#endif
  }
  if (J_other) {
    Eigen::Map<TranslationJacobian>{J_other}.noalias() = this->matrix();
  }
  return x;
}

}  // namespace hyper::variables

HYPER_DECLARE_EIGEN_CLASS(Map, hyper::variables::Quaternion, using Base::Base;)
HYPER_DECLARE_CONST_EIGEN_CLASS(Map, hyper::variables::Quaternion, using Base::Base;)
HYPER_DECLARE_EIGEN_CLASS(Map, hyper::variables::SU2, using Base::Base;)
HYPER_DECLARE_CONST_EIGEN_CLASS(Map, hyper::variables::SU2, using Base::Base;)
HYPER_DECLARE_TANGENT_MAP(hyper::variables::SU2)
