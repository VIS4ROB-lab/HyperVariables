/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>
#include <Eigen/Geometry>

#include "hyper/variables/groups/forward.hpp"

#include "hyper/variables/cartesian.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper {

template <typename TDerived>
class QuaternionBase
    : public Traits<TDerived>::Base,
      public AbstractVariable<typename Traits<TDerived>::ScalarWithConstIfNotLvalue> {
 public:
  // Constants.
  static constexpr auto SizeAtCompileTime = 4;

  // Definitions.
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using VectorXWithConstIfNotLvalue = std::conditional_t<std::is_const_v<ScalarWithConstIfNotLvalue>, const TVectorX<Scalar>, TVectorX<Scalar>>;
  using Base = typename Traits<TDerived>::Base;
  using Base::Base;
  using Base::operator*;

  using Translation = Cartesian<Scalar, 3>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(QuaternionBase)

  /// Constructs an identity element.
  /// \return Identity element.
  static auto Identity() -> Quaternion<Scalar> {
    return Base::Identity();
  }

  /// Constructs a random element.
  /// \return Random element.
  static auto Random() -> Quaternion<Scalar> {
    return Base::UnitRandom();
  }

  /// Data pointer accessor.
  /// \return Data pointer.
  [[nodiscard]] auto data() const -> const Scalar*;

  /// Data pointer modifier.
  /// \return Data pointer.
  [[nodiscard]] auto data() -> ScalarWithConstIfNotLvalue*;

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() const -> Eigen::Map<const TVectorX<Scalar>> final;

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Map<VectorXWithConstIfNotLvalue> final;

  /// Group inverse.
  /// \return Inverse element.
  [[nodiscard]] auto groupInverse() const -> Quaternion<Scalar>;

  /// Group plus.
  /// \tparam TOtherDerived_ Other derived type.
  /// \param other Other input.
  /// \return Additive element.
  template <typename TOtherDerived_>
  auto groupPlus(const Eigen::QuaternionBase<TOtherDerived_>& other) const -> Quaternion<Scalar>;

  /// Vector plus.
  /// \tparam TOtherDerived_ Other derived type.
  /// \param v Input vector.
  /// \return Additive element.
  template <typename TOtherDerived_>
  auto vectorPlus(const Eigen::MatrixBase<TOtherDerived_>& v) const -> Translation;

  /// Group logarithm.
  /// \return Logarithmic element.
  [[nodiscard]] auto groupLog() const -> Quaternion<Scalar>;

  /// Group exponential.
  /// \return Exponential element.
  [[nodiscard]] auto groupExp() const -> Quaternion<Scalar>;
};

template <typename TDerived>
class SU2Base
    : public QuaternionBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = QuaternionBase<TDerived>;
  using Base::Base;
  using Base::operator*;

  using Translation = typename Base::Translation;

  static constexpr auto kDefaultDerivativesAreGlobal = HYPER_DEFAULT_TO_GLOBAL_LIE_GROUP_DERIVATIVES;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SU2Base)

  /// Constructs an identity element.
  /// \return Identity element.
  static auto Identity() -> SU2<Scalar> {
    return Base::Identity();
  }

  /// Constructs a random element.
  /// \return Random element.
  static auto Random() -> SU2<Scalar> {
    return Base::UnitRandom();
  }

  /// Group adjoint.
  /// \return Adjoint.
  [[nodiscard]] auto groupAdjoint() const {
    return this->matrix();
  }

  /// Group inverse.
  /// \param raw_J Input Jacobian (if requested).
  /// \param global Request global Jacobians flag.
  /// \return Inverse element.
  [[nodiscard]] auto groupInverse(Scalar* raw_J = nullptr, bool global = kDefaultDerivativesAreGlobal) const -> SU2<Scalar>;

  /// Group plus.
  /// \tparam TOtherDerived_ Other derived type.
  /// \param other Other input.
  /// \param raw_J_this This input Jacobian (if requested).
  /// \param raw_J_other Other input Jacobian (if requested).
  /// \param global Request global Jacobians flag.
  /// \return Additive element.
  template <typename TOtherDerived_>
  auto groupPlus(const SU2Base<TOtherDerived_>& other, Scalar* raw_J_this = nullptr, Scalar* raw_J_other = nullptr, bool global = kDefaultDerivativesAreGlobal) const -> SU2<Scalar>;

  /// Vector plus.
  /// \tparam TOtherDerived_ Other derived type.
  /// \param v Input vector.
  /// \param raw_J_this This input Jacobian (if requested).
  /// \param raw_J_vector Point input Jacobian (if requested).
  /// \param global Request global Jacobians flag.
  /// \return Additive element.
  template <typename TOtherDerived_>
  auto vectorPlus(const Eigen::MatrixBase<TOtherDerived_>& v, Scalar* raw_J_this = nullptr, Scalar* raw_J_vector = nullptr, bool global = kDefaultDerivativesAreGlobal) const -> Translation;

  /// Group logarithm.
  /// \return Logarithmic element.
  [[nodiscard]] auto groupLog() const -> Algebra<SU2<Scalar>>;

  /// Group exponential.
  /// \return Exponential element.
  [[nodiscard]] auto groupExp() const -> Quaternion<Scalar>;

  /// Conversion to tangent element.
  /// \param raw_J_this Input Jacobian (if requested).
  /// \param global Request global Jacobians flag.
  /// \return Tangent element.
  auto toTangent(Scalar* raw_J = nullptr, bool global = kDefaultDerivativesAreGlobal) const -> Tangent<SU2<Scalar>>;
};

template <typename TDerived>
class SU2AlgebraBase
    : public QuaternionBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = QuaternionBase<TDerived>;
  using Base::Base;
  using Base::operator*;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SU2AlgebraBase)

  /// Constructs an identity element.
  /// \return Identity element.
  static auto Identity() -> Algebra<SU2<Scalar>>;

  /// Constructs a random element.
  /// \return Random element.
  static auto Random() -> Algebra<SU2<Scalar>>;

  /// Conjugate.
  /// \return SU2 algebra.
  [[nodiscard]] auto conjugate() const -> Algebra<SU2<Scalar>>;

  /// Group exponential.
  /// \return Exponential element.
  [[nodiscard]] auto groupExp() const -> SU2<Scalar>;

  /// Conversion to tangent element.
  /// \return Tangent element.
  auto toTangent() const -> Tangent<SU2<Scalar>>;
};

template <typename TScalar>
class Quaternion final
    : public QuaternionBase<Quaternion<TScalar>> {
 public:
  using Base = QuaternionBase<Quaternion<TScalar>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Quaternion)
};

template <typename TScalar>
class SU2 final
    : public SU2Base<SU2<TScalar>> {
 public:
  using Base = SU2Base<SU2<TScalar>>;

  /// Default constructor.
  SU2() : Base{TScalar{1}, TScalar{0}, TScalar{0}, TScalar{0}} {}

  /// Perfect forwarding constructor.
  template <typename... TArgs>
  SU2(TArgs&&... args) // NOLINT
  : Base{std::forward<TArgs>(args)...} {
    // DCHECK(Eigen::internal::isApprox(this->norm(), TScalar{1}));
  }

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SU2)
};

template <typename TScalar>
class Algebra<SU2<TScalar>> final
    : public SU2AlgebraBase<Algebra<SU2<TScalar>>> {
 public:
  using Base = SU2AlgebraBase<Algebra<SU2<TScalar>>>;

  /// Default constructor.
  Algebra() : Base{TScalar{0}, TScalar{0}, TScalar{0}, TScalar{0}} {}

  /// Perfect forwarding constructor.
  template <typename... TArgs>
  Algebra(TArgs&&... args) // NOLINT
      : Base{std::forward<TArgs>(args)...} {
    DCHECK(Eigen::internal::isApprox(TScalar{1} + this->w(), TScalar{1}));
  }

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Algebra)
};

template <typename TDerived>
class SU2TangentBase
    : public CartesianBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = CartesianBase<TDerived>;
  using Base::Base;

  static constexpr auto kDefaultDerivativesAreGlobal = HYPER_DEFAULT_TO_GLOBAL_LIE_GROUP_DERIVATIVES;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SU2TangentBase)

  /// Conversion to algebra element.
  /// \return Algebra element.
  auto toAlgebra() const -> Algebra<SU2<Scalar>>;

  /// Conversion to manifold element.
  /// \param raw_J_this Input Jacobian (if requested).
  /// \param global Request global Jacobians flag.
  /// \return Manifold element.
  auto toManifold(Scalar* raw_J = nullptr, bool global = kDefaultDerivativesAreGlobal) const -> SU2<Scalar>;
};

template <typename TScalar>
class Tangent<SU2<TScalar>> final
    : public SU2TangentBase<Tangent<SU2<TScalar>>> {
 public:
  using Base = SU2TangentBase<Tangent<SU2<TScalar>>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Tangent)
};

template <typename TDerived>
auto QuaternionBase<TDerived>::data() const -> const Scalar* {
  return this->coeffs().data();
}

template <typename TDerived>
auto QuaternionBase<TDerived>::data() -> ScalarWithConstIfNotLvalue* {
  return this->coeffs().data();
}

template <typename TDerived>
auto QuaternionBase<TDerived>::asVector() const -> Eigen::Map<const TVectorX<Scalar>> {
  return {this->data(), Traits<TDerived>::kNumParameters, 1};
}

template <typename TDerived>
auto QuaternionBase<TDerived>::asVector() -> Eigen::Map<VectorXWithConstIfNotLvalue> {
  return {this->data(), Traits<TDerived>::kNumParameters, 1};
}

template <typename TDerived>
auto QuaternionBase<TDerived>::groupInverse() const -> Quaternion<Scalar> {
  return this->inverse();
}

template <typename TDerived>
template <typename TOtherDerived_>
auto QuaternionBase<TDerived>::groupPlus(const Eigen::QuaternionBase<TOtherDerived_>& other) const -> Quaternion<Scalar> {
  return Base::operator*(other);
}

template <typename TDerived>
template <typename TOtherDerived_>
auto QuaternionBase<TDerived>::vectorPlus(const Eigen::MatrixBase<TOtherDerived_>& v) const -> Translation {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(TOtherDerived_, 3);
  return (*this) * v;
}

template <typename TDerived>
auto QuaternionBase<TDerived>::groupLog() const -> Quaternion<Scalar> {
  const auto v = this->vec();
  const auto w = this->w();
  const auto w2 = w * w;

  const auto nv2 = v.squaredNorm();
  const auto nv = std::sqrt(nv2);

  const auto nq2 = nv2 + w2;
  const auto nq = std::sqrt(nq2);

  DLOG_IF(FATAL, nq < Eigen::NumTraits<Scalar>::epsilon()) << "Quaternion norm is zero.";
  const auto a = (nv < Eigen::NumTraits<Scalar>::epsilon()) ? (Scalar{1} / nq + Scalar{1 / (6 * nq * nq2)} * (nq2 - w2)) : (std::atan2(nv, w) / nv);
  const auto av = (a * v).eval();
  return {std::log(nq), av.x(), av.y(), av.z()};
}

template <typename TDerived>
auto QuaternionBase<TDerived>::groupExp() const -> Quaternion<Scalar> {
  const auto v = this->vec();
  const auto w = this->w();

  const auto nv2 = v.squaredNorm();
  const auto nv = std::sqrt(nv2);

  Scalar sinv, cnv;
  if (nv < Eigen::NumTraits<Scalar>::epsilon()) {
    sinv = Scalar{1};
    cnv = Scalar{1};
  } else {
    sinv = std::sin(nv) / nv;
    cnv = std::cos(nv);
  }

  const auto ew = std::exp(w);
  const auto a = ew * sinv;
  const auto av = (a * v).eval();
  return {ew * cnv, av.x(), av.y(), av.z()};
}

template <typename TDerived>
auto SU2Base<TDerived>::groupInverse(Scalar* raw_J, const bool global) const -> SU2<Scalar> {
  const auto i_q = this->conjugate();

  if (raw_J) {
    auto J = Eigen::Map<TJacobianNM<Tangent<SU2<Scalar>>>>{raw_J};
    if (global) {
      J.noalias() = Scalar{-1} * i_q.matrix();
    } else {
      J.noalias() = Scalar{-1} * this->matrix();
    }
  }

  return i_q;
}

template <typename TDerived>
template <typename TOtherDerived_>
auto SU2Base<TDerived>::groupPlus(const SU2Base<TOtherDerived_>& other, Scalar* raw_J_this, Scalar* raw_J_other, const bool global) const -> SU2<Scalar> {
  auto output = (*this) * other;

  if (raw_J_this) {
    auto J = Eigen::Map<TJacobianNM<Tangent<SU2<Scalar>>>>{raw_J_this};
    if (global) {
      J.setIdentity();
    } else {
      J.noalias() = other.inverse().matrix();
    }
  }

  if (raw_J_other) {
    auto J = Eigen::Map<TJacobianNM<Tangent<SU2<Scalar>>>>{raw_J_other};
    if (global) {
      J.noalias() = this->matrix();
    } else {
      J.setIdentity();
    }
  }

  return output;
}

template <typename TDerived>
template <typename TOtherDerived_>
auto SU2Base<TDerived>::vectorPlus(const Eigen::MatrixBase<TOtherDerived_>& v, Scalar* raw_J_this, Scalar* raw_J_vector, const bool global) const -> Translation {
  auto output = Base::vectorPlus(v);

  if (raw_J_this) {
    using Tangent = Tangent<SU2<Scalar>>;
    auto J = Eigen::Map<TJacobianNM<Translation, Tangent>>{raw_J_this};
    if (global) {
      J.noalias() = Scalar{-1} * output.hat();
    } else {
      J.noalias() = Scalar{-1} * this->matrix() * v.hat();
    }
  }

  if (raw_J_vector) {
    Eigen::Map<TJacobianNM<Translation>>{raw_J_vector}.noalias() = this->matrix();
  }

  return output;
}

template <typename TDerived>
auto SU2Base<TDerived>::groupLog() const -> Algebra<SU2<Scalar>> {
  const auto v = this->vec();
  const auto w = this->w();
  const auto w2 = w * w;

  const auto nv2 = v.squaredNorm();
  const auto nv = std::sqrt(nv2);

  const auto nq2 = nv2 + w2;
  const auto nq = std::sqrt(nq2);

  DLOG_IF(FATAL, nq < Eigen::NumTraits<Scalar>::epsilon()) << "Quaternion norm is zero.";
  const auto a = (nv < Eigen::NumTraits<Scalar>::epsilon()) ? (Scalar{1} / nq + Scalar{1 / (6 * nq * nq2)} * (nq2 - w2)) : (std::atan2(nv, w) / nv);
  const auto av = (a * v).eval();

  return {Scalar{0}, av.x(), av.y(), av.z()};
}

template <typename TDerived>
auto SU2Base<TDerived>::groupExp() const -> Quaternion<Scalar> {
  return Base::groupExp();
}

template <typename TDerived>
auto SU2Base<TDerived>::toTangent(Scalar* raw_J, const bool global) const -> Tangent<SU2<Scalar>> {
  auto output = groupLog().toTangent();

  if (raw_J) {
    using Jacobian = TJacobianNM<Tangent<SU2<Scalar>>>;
    auto J = Eigen::Map<Jacobian>{raw_J};
    const auto nw2 = output.squaredNorm();
    const auto nw = std::sqrt(nw2);
    const auto nw3 = nw * nw2;
    if (nw3 < Eigen::NumTraits<Scalar>::epsilon()) {
      J.setIdentity();
    } else {
      const auto Wx = output.hat();
      if (global) {
        J.noalias() = Jacobian::Identity() - Scalar{0.5} * Wx + (Scalar{1} / nw2 - (Scalar{1} + std::cos(nw)) / (Scalar{2} * nw * std::sin(nw))) * Wx * Wx;
      } else {
        J.noalias() = Jacobian::Identity() + Scalar{0.5} * Wx + (Scalar{1} / nw2 - (Scalar{1} + std::cos(nw)) / (Scalar{2} * nw * std::sin(nw))) * Wx * Wx;
      }
    }
  }

  return output;
}

template <typename TDerived>
auto SU2AlgebraBase<TDerived>::Identity() -> Algebra<SU2<Scalar>> {
  return {Scalar{0}, Scalar{0}, Scalar{0}, Scalar{0}};
}

template <typename TDerived>
auto SU2AlgebraBase<TDerived>::Random() -> Algebra<SU2<SU2AlgebraBase::Scalar>> {
  Algebra<SU2<Scalar>> algebra;
  algebra.vec().setRandom();
  algebra.w() = Scalar{0};
  return algebra;
}

template <typename TDerived>
auto SU2AlgebraBase<TDerived>::conjugate() const -> Algebra<SU2<Scalar>> {
  return Base::conjugate();
}

template <typename TDerived>
auto SU2AlgebraBase<TDerived>::groupExp() const -> SU2<Scalar> {
  const auto v = this->vec();

  const auto nv2 = v.squaredNorm();
  const auto nv = std::sqrt(nv2);

  Scalar sinv, cnv;
  if (nv < Eigen::NumTraits<Scalar>::epsilon()) {
    sinv = Scalar{1};
    cnv = Scalar{1};
  } else {
    sinv = std::sin(nv) / nv;
    cnv = std::cos(nv);
  }

  const auto av = (sinv * v).eval();
  return {cnv, av.x(), av.y(), av.z()};
}

template <typename TDerived>
auto SU2AlgebraBase<TDerived>::toTangent() const -> Tangent<SU2<Scalar>> {
  return Scalar{2} * this->vec();
}

template <typename TDerived>
auto SU2TangentBase<TDerived>::toAlgebra() const -> Algebra<SU2<Scalar>> {
  return {Scalar{0}, Scalar{0.5} * this->x(), Scalar{0.5} * this->y(), Scalar{0.5} * this->z()};
}

template <typename TDerived>
auto SU2TangentBase<TDerived>::toManifold(Scalar* raw_J, const bool global) const -> SU2<Scalar> {
  auto output = toAlgebra().groupExp();

  if (raw_J) {
    using Jacobian = TJacobianNM<Tangent<SU2<Scalar>>>;
    auto J = Eigen::Map<Jacobian>{raw_J};
    const auto nw2 = this->squaredNorm();
    const auto nw = std::sqrt(nw2);
    const auto nw3 = nw * nw2;
    if (nw3 < Eigen::NumTraits<Scalar>::epsilon()) {
      J.setIdentity();
    } else {
      const auto Wx = this->hat();
      if (global) {
        J.noalias() = Jacobian::Identity() + (Scalar{1} - std::cos(nw)) / nw2 * Wx + (nw - std::sin(nw)) / nw3 * Wx * Wx;
      } else {
        J.noalias() = Jacobian::Identity() - (Scalar{1} - std::cos(nw)) / nw2 * Wx + (nw - std::sin(nw)) / nw3 * Wx * Wx;
      }
    }
  }

  return output;
}

} // namespace hyper

HYPER_DECLARE_EIGEN_INTERFACE(hyper::Quaternion)
HYPER_DECLARE_EIGEN_INTERFACE(hyper::SU2)
HYPER_DECLARE_ALGEBRA_MAP(hyper::SU2)
HYPER_DECLARE_TANGENT_MAP(hyper::SU2)
