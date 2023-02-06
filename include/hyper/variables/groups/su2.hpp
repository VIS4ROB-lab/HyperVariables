/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>
#include <Eigen/Geometry>

#include "hyper/variables/groups/forward.hpp"

#include "hyper/variables/cartesian.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper::variables {

template <typename TDerived>
class SU2TangentBase;

template <typename TDerived>
class QuaternionBase : public Traits<TDerived>::Base, public ConditionalConstBase_t<TDerived, Variable<DerivedScalar_t<TDerived>>, ConstVariable<DerivedScalar_t<TDerived>>> {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Scalar>;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, VectorX<Scalar>>;
  using Base::Base;
  using Base::operator*;

  using Index = Eigen::Index;
  using Translation = Cartesian<Scalar, 3>;

  // Constants.
  static constexpr auto SizeAtCompileTime = (int)Base::Coefficients::SizeAtCompileTime;
  static constexpr auto kNumParameters = (int)Base::Coefficients::SizeAtCompileTime;

  // Ordering.
  struct Ordering {
    static constexpr Index kW = 3;
    static constexpr Index kX = 0;
    static constexpr Index kY = 1;
    static constexpr Index kZ = 2;
  };

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(QuaternionBase)

  /// Identity group element.
  /// \return Identity element.
  static auto Identity() -> Quaternion<Scalar> { return Base::Identity(); }

  /// Random group element.
  /// \return Random element.
  static auto Random() -> Quaternion<Scalar> { return Base::UnitRandom(); }

  /// Data accessor.
  /// \return Data.
  [[nodiscard]] auto data() const -> const Scalar* { return this->coeffs().data(); }

  /// Data modifier.
  /// \return Data.
  [[nodiscard]] auto data() -> ScalarWithConstIfNotLvalue* { return this->coeffs().data(); }

  /// Retrieves the manifold size.
  /// \return Manifold size.
  [[nodiscard]] auto manifoldSize() const -> Index final { return kNumParameters; }

  /// Retrieves the tangent size.
  /// \return Tangent size.
  [[nodiscard]] auto tangentSize() const -> Index final { return variables::Tangent<SU2<Scalar>>::kNumParameters; }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() const -> Eigen::Ref<const VectorX<Scalar>> final { return this->coeffs(); }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Ref<VectorXWithConstIfNotLvalue> final { return this->coeffs(); }

  /// Group inverse.
  /// \return Inverse element.
  [[nodiscard]] auto gInv() const -> Quaternion<Scalar> { return this->inverse(); }

  /// Group plus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \return Group element.
  template <typename Other_>
  auto gPlus(const Eigen::QuaternionBase<Other_>& other) const -> Quaternion<Scalar> {
    return Base::operator*(other);
  }

  /// Group logarithm (quaternion -> quaternion).
  /// \return Group element.
  [[nodiscard]] auto glog() const -> Quaternion<Scalar> {
    const auto nv2 = this->vec().squaredNorm();
    const auto w2 = this->w() * this->w();
    const auto nq2 = nv2 + w2;

    const auto nv = std::sqrt(nv2);
    const auto nq = std::sqrt(nq2);

    DLOG_IF(FATAL, nq < Eigen::NumTraits<Scalar>::epsilon()) << "Quaternion norm is zero.";
    const auto a = (nv < Eigen::NumTraits<Scalar>::epsilon()) ? (Scalar{1} / nq) : (std::atan2(nv, this->w()) / nv);
    return {std::log(nq), a * this->x(), a * this->y(), a * this->z()};
  }

  /// Group exponential (quaternion -> quaternion).
  /// \return Group element.
  [[nodiscard]] auto gexp() const -> Quaternion<Scalar> {
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

  /// Group action.
  /// \tparam Other_ Other type.
  /// \param other Other vector.
  /// \return Vector.
  template <typename Other_>
  auto act(const Eigen::MatrixBase<Other_>& other) const -> Translation {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Other_, 3);
    return *this * other;
  }
};

template <typename TDerived>
class SU2Base : public QuaternionBase<TDerived> {
 public:
  using Base = QuaternionBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;
  using Base::operator*;

  using Translation = typename Base::Translation;

  using Tangent = variables::Tangent<SU2<Scalar>>;

  static constexpr auto kAlpha = Scalar{2.0};
  static constexpr auto kiAlpha = Scalar{1} / kAlpha;

  static constexpr auto kGlobal = HYPER_DEFAULT_TO_GLOBAL_MANIFOLD_DERIVATIVES;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SU2Base)

  /// Identity group element.
  /// \return Identity element.
  static auto Identity() -> SU2<Scalar> { return Base::Identity(); }

  /// Random group element.
  /// \return Random element.
  static auto Random() -> SU2<Scalar> { return Base::UnitRandom(); }

  /// Group adjoint.
  /// \return Adjoint matrix.
  [[nodiscard]] auto gAdj() const { return this->matrix(); }

  /// Group inverse.
  /// \param J_this Jacobian w.r.t. this.
  /// \param global Global Jacobian flag.
  /// \return Inverse group element.
  [[nodiscard]] auto gInv(Scalar* J_this = nullptr, bool global = kGlobal) const -> SU2<Scalar> {
    const auto i_su2 = this->conjugate();
    if (J_this) {
      if (global) {
        Eigen::Map<JacobianNM<Tangent>>{J_this}.noalias() = Scalar{-1} * i_su2.matrix();
      } else {
        Eigen::Map<JacobianNM<Tangent>>{J_this}.noalias() = Scalar{-1} * this->matrix();
      }
    }
    return i_su2;
  }

  /// Group plus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \param J_this Jacobian w.r.t. this.
  /// \param J_other Jacobian w.r.t. other.
  /// \param global Global Jacobian flag.
  /// \return Group element.
  template <typename Other_>
  auto gPlus(const SU2Base<Other_>& other, Scalar* J_this = nullptr, Scalar* J_other = nullptr, bool global = kGlobal) const -> SU2<Scalar> {
    if (J_this) {
      if (global) {
        Eigen::Map<JacobianNM<Tangent>>{J_this}.setIdentity();
      } else {
        Eigen::Map<JacobianNM<Tangent>>{J_this}.noalias() = other.inverse().matrix();
      }
    }
    if (J_other) {
      if (global) {
        Eigen::Map<JacobianNM<Tangent>>{J_other}.noalias() = this->matrix();
      } else {
        Eigen::Map<JacobianNM<Tangent>>{J_other}.setIdentity();
      }
    }
    return (*this) * other;
  }

  /// Group logarithm (SU2 -> SU2 tangent).
  /// \param J_this Jacobian w.r.t. this.
  /// \param global Global Jacobian flag.
  /// \return Tangent element.
  [[nodiscard]] auto gLog(Scalar* J_this = nullptr, bool global = kGlobal) const -> Tangent {
    const auto nv2 = this->vec().squaredNorm();
    const auto w2 = this->w() * this->w();
    const auto nq2 = nv2 + w2;

    const auto nv = std::sqrt(nv2);
    const auto nq = std::sqrt(nq2);

    DLOG_IF(FATAL, nq < Eigen::NumTraits<Scalar>::epsilon()) << "Quaternion norm is zero.";
    const auto a = (nv < Eigen::NumTraits<Scalar>::epsilon()) ? (Scalar{1} / nq) : (std::atan2(nv, this->w()) / nv);
    Tangent tangent = kAlpha * a * this->vec();

    if (J_this) {
      const auto nw2 = kAlpha * kAlpha * a * a * nv2;
      const auto nw = kAlpha * a * nv;
      const auto nw3 = nw * nw2;
      if (nw3 < Eigen::NumTraits<Scalar>::epsilon()) {
        Eigen::Map<JacobianNM<Tangent>>{J_this}.setIdentity();
      } else {
        const auto Wx = tangent.hat();
        if (global) {
          Eigen::Map<JacobianNM<Tangent>>{J_this}.noalias() =
              JacobianNM<Tangent>::Identity() - Scalar{0.5} * Wx + (Scalar{1} / nw2 - (Scalar{1} + std::cos(nw)) / (Scalar{2} * nw * std::sin(nw))) * Wx * Wx;
        } else {
          Eigen::Map<JacobianNM<Tangent>>{J_this}.noalias() =
              JacobianNM<Tangent>::Identity() + Scalar{0.5} * Wx + (Scalar{1} / nw2 - (Scalar{1} + std::cos(nw)) / (Scalar{2} * nw * std::sin(nw))) * Wx * Wx;
        }
      }
    }

    return tangent;
  }

  /// Group exponential (SU2 -> quaternion).
  /// \return Group element.
  [[nodiscard]] auto gExp() const -> Quaternion<Scalar> { return Base::gExp(); }

  /// Tangent plus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \param global Global Jacobian flag.
  /// \return Group element.
  template <typename Other_>
  auto tPlus(const SU2TangentBase<Other_>& other, const bool global = kGlobal) const -> SU2<Scalar> {
    return (global) ? other.gExp().gPlus(*this) : (*this).gPlus(other.gExp());
  }

  /// Tangent minus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \param global Global Jacobian flag.
  /// \return Tangent element.
  template <typename Other_>
  auto tMinus(const SU2Base<Other_>& other, const bool global = kGlobal) const -> Tangent {
    return (global) ? gPlus(other.gInv()).gLog() : other.gInv().gPlus(*this).gLog();
  }

  /// Group action.
  /// \tparam Other_ Other type.
  /// \param other Other vector.
  /// \param J_this Jacobian w.r.t. this.
  /// \param J_other Jacobian w.r.t. other.
  /// \param global Global Jacobian flag.
  /// \return Vector.
  template <typename Other_>
  auto act(const Eigen::MatrixBase<Other_>& other, Scalar* J_this = nullptr, Scalar* J_other = nullptr, bool global = kGlobal) const -> Translation {
    auto w = Base::act(other);
    if (J_this) {
      if (global) {
        Eigen::Map<JacobianNM<Translation, Tangent>>{J_this}.noalias() = Scalar{-1} * w.hat();
      } else {
        Eigen::Map<JacobianNM<Translation, Tangent>>{J_this}.noalias() = Scalar{-1} * this->matrix() * other.hat();
      }
    }
    if (J_other) {
      Eigen::Map<JacobianNM<Translation>>{J_other}.noalias() = this->matrix();
    }
    return w;
  }
};

template <typename TScalar>
class Quaternion final : public QuaternionBase<Quaternion<TScalar>> {
 public:
  using Base = QuaternionBase<Quaternion<TScalar>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Quaternion)
};

template <typename TScalar>
class SU2 final : public SU2Base<SU2<TScalar>> {
 public:
  using Base = SU2Base<SU2<TScalar>>;

  /// Default constructor.
  SU2() : Base{TScalar{1}, TScalar{0}, TScalar{0}, TScalar{0}} {}

  /// Perfect forwarding constructor.
  template <typename... TArgs>
  SU2(TArgs&&... args) : Base{std::forward<TArgs>(args)...} {}  // NOLINT

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SU2)
};

template <typename TDerived>
class SU2TangentBase : public CartesianBase<TDerived> {
 public:
  // Definitions.
  using Base = CartesianBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  // Constants.
  static constexpr auto kAngularOffset = 0;
  static constexpr auto kNumAngularParameters = 3;

  static constexpr auto kGlobal = HYPER_DEFAULT_TO_GLOBAL_MANIFOLD_DERIVATIVES;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SU2TangentBase)

  /// Group exponential (SU2 tangent -> SU2).
  /// \param J_this Jacobian w.r.t. this.
  /// \param global Global Jacobian flag.
  /// \return Group element.
  auto gExp(Scalar* J_this = nullptr, bool global = kGlobal) const -> SU2<Scalar> {
    // Constants.
    constexpr auto kiAlpha = SU2<Scalar>::kiAlpha;

    // Definitions.
    using Jacobian = variables::JacobianNM<Tangent<SU2<Scalar>>>;

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

    if (J_this) {
      const auto nw = std::sqrt(nw2);
      const auto nw3 = nw * nw2;
      if (nw3 < Eigen::NumTraits<Scalar>::epsilon()) {
        Eigen::Map<Jacobian>{J_this}.setIdentity();
      } else {
        const auto Wx = this->hat();
        if (global) {
          Eigen::Map<Jacobian>{J_this}.noalias() = Jacobian::Identity() + (Scalar{1} - std::cos(nw)) / nw2 * Wx + (nw - std::sin(nw)) / nw3 * Wx * Wx;
        } else {
          Eigen::Map<Jacobian>{J_this}.noalias() = Jacobian::Identity() - (Scalar{1} - std::cos(nw)) / nw2 * Wx + (nw - std::sin(nw)) / nw3 * Wx * Wx;
        }
      }
    }

    const auto a = kiAlpha * s;
    return {c, a * this->x(), a * this->y(), a * this->z()};
  }
};

template <typename TScalar>
class Tangent<SU2<TScalar>> final : public SU2TangentBase<Tangent<SU2<TScalar>>> {
 public:
  using Base = SU2TangentBase<Tangent<SU2<TScalar>>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Tangent)
};

}  // namespace hyper::variables

HYPER_DECLARE_EIGEN_INTERFACE(hyper::variables::Quaternion)
HYPER_DECLARE_EIGEN_INTERFACE(hyper::variables::SU2)
HYPER_DECLARE_TANGENT_MAP(hyper::variables::SU2)
