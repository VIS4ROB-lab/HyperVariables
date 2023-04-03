/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/groups/su2.hpp"

namespace hyper::variables {

template <typename TDerived>
class SE3TangentBase;

template <typename TDerived>
class SE3Base : public CartesianBase<TDerived> {
 public:
  // Definitions.
  using Base = CartesianBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, VectorX<Scalar>>;
  using Base::Base;

  using Rotation = SU2<Scalar>;
  using RotationWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Rotation>;
  using Translation = Cartesian<Scalar, 3>;
  using TranslationWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Translation>;

  using Tangent = variables::Tangent<SE3<Scalar>>;

  // Constants.
  static constexpr auto kRotationOffset = 0;
  static constexpr auto kNumRotationParameters = Rotation::kNumParameters;
  static constexpr auto kTranslationOffset = kNumRotationParameters;
  static constexpr auto kNumTranslationParameters = 3;
  static constexpr auto kNumParameters = kNumRotationParameters + kNumTranslationParameters;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SE3Base)

  /// Rotation Jacobian accessor/modifier.
  template <int NumRows, typename TDerived_>
  static auto RotationJacobian(Eigen::MatrixBase<TDerived_>& matrix, int row) {
    return matrix.template block<NumRows, kNumRotationParameters>(row, kRotationOffset);
  }

  /// Translation Jacobian accessor/modifier.
  template <int NumRows, typename TDerived_>
  static auto TranslationJacobian(Eigen::MatrixBase<TDerived_>& matrix, int row) {
    return matrix.template block<NumRows, kNumTranslationParameters>(row, kTranslationOffset);
  }

  /// Identity group element.
  /// \return Identity element.
  static auto Identity() -> SE3<Scalar> { return {}; }

  /// Random group element.
  /// \return Random element.
  static auto Random() -> SE3<Scalar> { return {Rotation::Random(), Translation::Random()}; }

  /// Sets this to identity.
  /// \return Derived type.
  auto setIdentity() -> TDerived& {
    rotation().setIdentity();
    translation().setZero();
    return this->derived();
  }

  /// Sets this to random.
  /// \return Derived type.
  auto setRandom() -> TDerived& {
    rotation().setRandom();
    translation().setRandom();
    return this->derived();
  }

  /// Rotation accessor.
  /// \return Rotation.
  auto rotation() const { return Eigen::Map<const Rotation>{this->data() + kRotationOffset}; }

  /// Rotation modifier.
  /// \return Rotation.
  auto rotation() { return Eigen::Map<RotationWithConstIfNotLvalue>{this->data() + kRotationOffset}; }

  /// Translation accessor.
  /// \return Translation.
  auto translation() const { return Eigen::Map<const Translation>{this->data() + kTranslationOffset}; }

  /// Translation modifier.
  /// \return Translation.
  auto translation() { return Eigen::Map<TranslationWithConstIfNotLvalue>{this->data() + kTranslationOffset}; }

  /// Group inverse.
  /// \param J_this Jacobian w.r.t. this.
  /// \return Group element.
  auto gInv(Scalar* J_this = nullptr) const -> SE3<Scalar> {
    const Rotation i_rotation = rotation().gInv();
    const Translation i_translation = Scalar{-1} * (i_rotation.act(translation()));
    SE3<Scalar> inv = {i_rotation, i_translation};

    if (!J_this) {
      return inv;
    }
#if HYPER_USE_GLOBAL_MANIFOLD_DERIVATIVES
    auto J = Eigen::Map<JacobianNM<Tangent>>{J_this};
    const auto i_R_this = (Scalar{-1} * i_rotation.matrix()).eval();
    Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset) = i_R_this;
    Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = i_R_this * translation().hat();
    Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
    Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset) = i_R_this;
#else
    auto J = Eigen::Map<JacobianNM<Tangent>>{J_this};
    const auto R_this = (Scalar{-1} * rotation().matrix()).eval();
    Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset) = R_this;
    Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset) = i_translation.hat();
    Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
    Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset) = R_this.transpose();
#endif
    return inv;
  }

  /// Group plus.
  /// \tparam TOther_ Other type.
  /// \param other Other element.
  /// \param J_this Jacobian w.r.t. this.
  /// \param J_other Jacobian w.r.t. other.
  /// \return Group element.
  template <typename TOther_>
  auto gPlus(const SE3Base<TOther_>& other, Scalar* J_this = nullptr, Scalar* J_other = nullptr) const -> SE3<Scalar> {
    SE3<Scalar> plus = {rotation().gPlus(other.rotation()), rotation().act(other.translation()) + this->translation()};

    if (!J_this && !J_other) {
      return plus;
    }
#if HYPER_USE_GLOBAL_MANIFOLD_DERIVATIVES
    if (J_this) {
      auto J = Eigen::Map<JacobianNM<Tangent>>{J_this};
      Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setIdentity();
      Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = Scalar{-1} * R_this_t_other.hat();
      Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
      Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
    }
    if (J_other) {
      const auto R_this = this->rotation().matrix();
      auto J = Eigen::Map<JacobianNM<Tangent>>{J_other};
      Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset) = R_this;
      Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
      Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
      Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset) = R_this;
    }
#else
    if (J_this) {
      const auto R_this = rotation().matrix();
      const auto i_R_other = other.rotation().gInv().matrix();
      auto J = Eigen::Map<JacobianNM<Tangent>>{J_this};
      Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset) = i_R_other;
      Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = Scalar{-1} * R_this * other.translation().hat();
      Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
      Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
    }
    if (J_other) {
      const auto R_this = this->rotation().matrix();
      auto J = Eigen::Map<JacobianNM<Tangent>>{J_other};
      Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setIdentity();
      Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
      Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
      Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset) = R_this;
    }
#endif
    return plus;
  }

  /// Group logarithm (SE3 -> SE3 tangent).
  /// \param J_this Jacobian w.r.t. this.
  /// \return Tangent element.
  auto gLog(Scalar* J_this = nullptr) const -> Tangent {
    Tangent log;
    if (!J_this) {
      log.angular() = rotation().gLog();
      log.linear() = translation();
      return log;
    }

    JacobianNM<typename Rotation::Tangent> J_log;
    log.angular() = rotation().gLog(J_log.data());
    log.linear() = translation();

    auto J = Eigen::Map<JacobianNM<Tangent>>{J_this};
    Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset) = J_log;
    Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
    Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
    Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
    return log;
  }

  /// Tangent plus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \return Group element.
  template <typename TOther_>
  auto tPlus(const SE3TangentBase<TOther_>& other) const -> SE3<Scalar> {
#if HYPER_USE_GLOBAL_MANIFOLD_DERIVATIVES
    return {other.angular().gExp().gPlus(rotation()), translation() + other.linear()};
#else
    return {rotation().gPlus(other.angular().gExp()), translation() + other.linear()};
#endif
  }

  /// Tangent minus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \return Tangent element.
  template <typename TOther_>
  auto tMinus(const SE3Base<TOther_>& other) const -> Tangent {
#if HYPER_USE_GLOBAL_MANIFOLD_DERIVATIVES
    Tangent minus;
    minus.angular() = rotation().gPlus(other.rotation().gInv()).gLog();
    minus.linear() = (translation() - other.translation());
    return minus;
#else
    Tangent minus;
    minus.angular() = other.rotation().gInv().gPlus(rotation()).gLog();
    minus.linear() = translation() - other.translation();
    return minus;
#endif
  }

  /// Tangent plus Jacobian.
  /// \return Jacobian.
  auto tPlusJacobian() const -> Jacobian<Scalar, Base::kNumParameters, Traits<Tangent>::kNumParameters>;

  /// Tangent minus Jacobian.
  /// \return Jacobian.
  auto tMinusJacobian() const -> Jacobian<Scalar, Traits<Tangent>::kNumParameters, Base::kNumParameters>;

  /// Group action.
  /// \tparam TOther_ Other type.
  /// \param other Other vector.
  /// \param J_this Jacobian w.r.t. this.
  /// \param J_other Jacobian w.r.t. other.
  /// \return Vector.
  template <typename TOther_>
  auto act(const Eigen::MatrixBase<TOther_>& other, Scalar* J_this = nullptr, Scalar* J_other = nullptr) const -> Translation {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(TOther_, 3)

    const auto R_this = this->rotation().matrix();
    const Translation R_this_v = R_this * other;
    Translation x = R_this_v + translation();

    if (!J_this && !J_other) {
      return x;
    }
    if (J_this) {
#if HYPER_USE_GLOBAL_MANIFOLD_DERIVATIVES
      auto J = Eigen::Map<JacobianNM<Translation, Tangent>>{J_this};
      Tangent::template AngularJacobian<Translation::kNumParameters>(J, 0).noalias() = Scalar{-1} * R_this_v.hat();
      Tangent::template LinearJacobian<Translation::kNumParameters>(J, 0).setIdentity();
#else
      auto J = Eigen::Map<JacobianNM<Translation, Tangent>>{J_this};
      Tangent::template AngularJacobian<Translation::kNumParameters>(J, 0).noalias() = Scalar{-1} * R_this * other.hat();
      Tangent::template LinearJacobian<Translation::kNumParameters>(J, 0).setIdentity();
#endif
    }
    if (J_other) {
      Eigen::Map<JacobianNM<Translation>>{J_other} = R_this;
    }
    return x;
  }

 private:
  using Base::Constant;
  using Base::LinSpaced;
  using Base::Ones;
  using Base::Unit;
  using Base::Zero;

  using Base::setConstant;
  using Base::setLinSpaced;
  using Base::setOnes;
  using Base::setUnit;
  using Base::setZero;
};

template <typename TScalar>
class SE3 final : public SE3Base<SE3<TScalar>> {
 public:
  using Base = SE3Base<SE3<TScalar>>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SE3)

  /// Default constructor.
  SE3() { this->setIdentity(); }

  /// Constructor from pointer.
  /// \param other Pointer.
  explicit SE3(const TScalar* other) : Base{other} {}

  /// Constructor from rotation and translation.
  /// \tparam TDerived_ Derived type.
  /// \tparam TOther_ Other type.
  /// \param rotation Input rotation.
  /// \param translation Input translation.
  template <typename TDerived_, typename TOther_>
  SE3(const SU2Base<TDerived_>& rotation, const Eigen::MatrixBase<TOther_>& translation) {
    this->rotation().coeffs() = rotation.coeffs();
    this->translation() = translation;
  }
};

template <typename TDerived>
class SE3TangentBase : public CartesianBase<TDerived> {
 public:
  // Definitions.
  using Base = CartesianBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  using Angular = Tangent<typename SE3<Scalar>::Rotation>;
  using AngularWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Angular>;
  using Linear = Tangent<typename SE3<Scalar>::Translation>;
  using LinearWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Linear>;

  // Constants.
  static constexpr auto kAngularOffset = 0;
  static constexpr auto kNumAngularParameters = 3;
  static constexpr auto kLinearOffset = kAngularOffset + kNumAngularParameters;
  static constexpr auto kNumLinearParameters = 3;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SE3TangentBase)

  /// Angular Jacobian accessor/modifier.
  template <int NumRows, typename TDerived_>
  static auto AngularJacobian(Eigen::MatrixBase<TDerived_>& matrix, int row) {
    return matrix.template block<NumRows, kNumAngularParameters>(row, kAngularOffset);
  }

  /// Linear Jacobian accessor/modifier.
  template <int NumRows, typename TDerived_>
  static auto LinearJacobian(Eigen::MatrixBase<TDerived_>& matrix, int row) {
    return matrix.template block<NumRows, kNumLinearParameters>(row, kLinearOffset);
  }

  /// Angular tangent accessor.
  /// \return Angular tangent.
  [[nodiscard]] auto angular() const -> Eigen::Map<const Angular> { return Eigen::Map<const Angular>{this->data() + kAngularOffset}; }

  /// Angular tangent modifier.
  /// \return Angular tangent.
  auto angular() -> Eigen::Map<AngularWithConstIfNotLvalue> { return Eigen::Map<AngularWithConstIfNotLvalue>{this->data() + kAngularOffset}; }

  /// Linear tangent accessor.
  /// \return Linear tangent.
  [[nodiscard]] auto linear() const -> Eigen::Map<const Linear> { return Eigen::Map<const Linear>{this->data() + kLinearOffset}; }

  /// Linear tangent modifier.
  /// \return Linear tangent.
  auto linear() -> Eigen::Map<LinearWithConstIfNotLvalue> { return Eigen::Map<LinearWithConstIfNotLvalue>{this->data() + kLinearOffset}; }

  /// Group exponential (SE3 tangent -> SE3).
  /// \param J_this Jacobian w.r.t. this.
  /// \return Group element.
  auto gExp(Scalar* J_this = nullptr) const -> SE3<Scalar> {
    SE3<Scalar> exp;

    if (!J_this) {
      exp.rotation() = angular().gExp();
      exp.translation() = linear();
      return exp;
    }

    JacobianNM<Tangent<SU2<Scalar>>> J_exp;
    exp.rotation() = angular().gExp(J_exp.data());
    exp.translation() = linear();

    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<JacobianNM<Tangent>>{J_this};
    Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset) = J_exp;
    Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
    Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
    Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
    return exp;
  }
};

template <typename TScalar>
class Tangent<SE3<TScalar>> final : public SE3TangentBase<Tangent<SE3<TScalar>>> {
 public:
  using Base = SE3TangentBase<Tangent<SE3<TScalar>>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Tangent)
};

template <typename TDerived>
auto SE3Base<TDerived>::tPlusJacobian() const -> Jacobian<Scalar, Base::kNumParameters, Traits<Tangent>::kNumParameters> {
  Jacobian<Scalar, Base::kNumParameters, Tangent::kNumParameters> J;
  J.template block<kNumRotationParameters, Tangent::kNumAngularParameters>(kRotationOffset, Tangent::kAngularOffset) = rotation().tPlusJacobian();
  J.template block<kNumTranslationParameters, Tangent::kNumAngularParameters>(kTranslationOffset, Tangent::kAngularOffset).setZero();
  J.template block<kNumRotationParameters, Tangent::kNumLinearParameters>(kRotationOffset, Tangent::kLinearOffset).setZero();
  J.template block<kNumTranslationParameters, Tangent::kNumLinearParameters>(kTranslationOffset, Tangent::kLinearOffset) = translation().tPlusJacobian();
  return J;
}

template <typename TDerived>
auto SE3Base<TDerived>::tMinusJacobian() const -> Jacobian<Scalar, Traits<Tangent>::kNumParameters, Base::kNumParameters> {
  Jacobian<Scalar, Traits<Tangent>::kNumParameters, Base::kNumParameters> J;
  J.template block<Tangent::kNumAngularParameters, kNumRotationParameters>(Tangent::kAngularOffset, kRotationOffset) = rotation().tMinusJacobian();
  J.template block<Tangent::kNumLinearParameters, kNumRotationParameters>(Tangent::kLinearOffset, kRotationOffset).setZero();
  J.template block<Tangent::kNumAngularParameters, kNumTranslationParameters>(Tangent::kAngularOffset, kTranslationOffset).setZero();
  J.template block<Tangent::kNumLinearParameters, kNumTranslationParameters>(Tangent::kLinearOffset, kTranslationOffset) = translation().tMinusJacobian();
  return J;
}

}  // namespace hyper::variables

HYPER_DECLARE_EIGEN_INTERFACE(hyper::variables::SE3)
HYPER_DECLARE_TANGENT_MAP(hyper::variables::SE3)
