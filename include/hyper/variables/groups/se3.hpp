/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/groups/su2.hpp"
#include "hyper/variables/jacobian.hpp"

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

  using Index = Eigen::Index;
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

  static constexpr auto kGlobal = HYPER_DEFAULT_TO_GLOBAL_MANIFOLD_DERIVATIVES;
  static constexpr auto kCoupled = HYPER_DEFAULT_TO_COUPLED_MANIFOLD_DERIVATIVES;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SE3Base)

  /// Rotation Jacobian accessor/modifier.
  template <int NumRows, typename TDerived_>
  static auto RotationJacobian(Eigen::MatrixBase<TDerived_>& matrix, const Index& row) {
    return matrix.template block<NumRows, kNumRotationParameters>(row, kRotationOffset);
  }

  /// Translation Jacobian accessor/modifier.
  template <int NumRows, typename TDerived_>
  static auto TranslationJacobian(Eigen::MatrixBase<TDerived_>& matrix, const Index& row) {
    return matrix.template block<NumRows, kNumTranslationParameters>(row, kTranslationOffset);
  }

  /// Identity group element.
  /// \return Identity element.
  static auto Identity() -> SE3<Scalar> { return {}; }

  /// Random group element.
  /// \return Random element.
  static auto Random() -> SE3<Scalar> { return {Rotation::Random(), Translation::Random()}; }

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
  /// \param global Global Jacobian flag.
  /// \param coupled Coupled Jacobian flag.
  /// \return Group element.
  auto gInv(Scalar* J_this = nullptr, bool global = kGlobal, bool coupled = kCoupled) const -> SE3<Scalar> {
    const Rotation i_rotation = rotation().gInv();
    const Translation i_translation = Scalar{-1} * (i_rotation.act(translation()));

    if (J_this) {
      auto J = Eigen::Map<JacobianNM<Tangent>>{J_this};

      if (coupled) {
        if (global) {
          const auto i_R_this = (Scalar{-1} * i_rotation.matrix()).eval();
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = i_R_this;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = Scalar{-1} * i_R_this * translation().hat();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = i_R_this;
        } else {
          const auto R_this = (Scalar{-1} * rotation().matrix()).eval();
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = R_this;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = translation().hat() * R_this;
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = R_this;
        }
      } else {
        if (global) {
          const auto i_R_this = (Scalar{-1} * i_rotation.matrix()).eval();
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = i_R_this;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = i_R_this * translation().hat();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = i_R_this;
        } else {
          const auto R_this = (Scalar{-1} * rotation().matrix()).eval();
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = R_this;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = i_translation.hat();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = R_this.transpose();
        }
      }
    }

    return {i_rotation, i_translation};
  }

  /// Group plus.
  /// \tparam TOther_ Other type.
  /// \param other Other element.
  /// \param J_this Jacobian w.r.t. this.
  /// \param J_other Jacobian w.r.t. other.
  /// \param global Global Jacobian flag.
  /// \param coupled Coupled Jacobian flag.
  /// \return Group element.
  template <typename TOther_>
  auto gPlus(const SE3Base<TOther_>& other, Scalar* J_this = nullptr, Scalar* J_other = nullptr, bool global = kGlobal, bool coupled = kCoupled) const -> SE3<Scalar> {
    const auto R_this_R_other = rotation().gPlus(other.rotation());
    const auto R_this_t_other = rotation().act(other.translation());

    if (J_this) {
      auto J = Eigen::Map<JacobianNM<Tangent>>{J_this};
      if (coupled) {
        if (global) {
          J.setIdentity();
        } else {
          const auto i_R_other = other.rotation().gInv().matrix();
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = i_R_other;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = Scalar{-1} * i_R_other * other.translation().hat();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = i_R_other;
        }
      } else {
        if (global) {
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setIdentity();
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = Scalar{-1} * R_this_t_other.hat();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
        } else {
          const auto R_this = rotation().matrix();
          const auto i_R_other = other.rotation().gInv().matrix();
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = i_R_other;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = Scalar{-1} * R_this * other.translation().hat();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
        }
      }
    }

    if (J_other) {
      auto J = Eigen::Map<JacobianNM<Tangent>>{J_other};
      if (coupled) {
        if (global) {
          const auto R_this = this->rotation().matrix();
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = R_this;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = this->translation().hat() * R_this;
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = R_this;
        } else {
          J.setIdentity();
        }
      } else {
        if (global) {
          const auto R_this = this->rotation().matrix();
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = R_this;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = R_this;
        } else {
          const auto R_this = this->rotation().matrix();
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setIdentity();
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = R_this;
        }
      }
    }

    return {R_this_R_other, R_this_t_other + this->translation()};
  }

  /// Group logarithm (SE3 -> SE3 tangent).
  /// \param J_this Jacobian w.r.t. this.
  /// \param global Global Jacobian flag.
  /// \param coupled Coupled Jacobian flag.
  /// \return Tangent element.
  auto gLog(Scalar* J_this = nullptr, bool global = kGlobal, bool coupled = kCoupled) const -> Tangent {
    Tangent tangent;

    if (J_this) {
      JacobianNM<typename Rotation::Tangent> J_r;
      tangent.angular().noalias() = rotation().gLog(J_r.data(), global);
      tangent.linear().noalias() = translation();

      auto J = Eigen::Map<JacobianNM<Tangent>>{J_this};

      if (coupled) {
        if (global) {
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = J_r;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = Scalar{-1} * translation().hat();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
        } else {
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = J_r;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = rotation().matrix();
        }
      } else {
        Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = J_r;
        Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
        Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
        Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
      }
    } else {
      tangent.angular().noalias() = rotation().gLog();
      tangent.linear().noalias() = translation();
    }

    return tangent;
  }

  /// Tangent plus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \param global Global Jacobian flag.
  /// \param coupled Coupled Jacobian flag.
  /// \return Group element.
  template <typename TOther_>
  auto tPlus(const SE3TangentBase<TOther_>& other, bool global = kGlobal, bool coupled = kCoupled) const -> SE3<Scalar> {
    if (coupled) {
      if (global) {
        return other.gExp().gPlus(*this);
      } else {
        return this->gPlus(other.gExp());
      }
    } else {
      if (global) {
        return {other.angular().gExp().gPlus(rotation()), translation() + other.linear()};
      } else {
        return {rotation().gPlus(other.angular().gExp()), translation() + other.linear()};
      }
    }
  }

  /// Tangent minus.
  /// \tparam Other_ Other type.
  /// \param other Other element.
  /// \param global Global Jacobian flag.
  /// \param coupled Coupled Jacobian flag.
  /// \return Tangent element.
  template <typename TOther_>
  auto tMinus(const SE3Base<TOther_>& other, bool global, bool coupled) const -> Tangent {
    if (coupled) {
      if (global) {
        return this->gPlus(other.gInv()).gLog();
      } else {
        return other.gInv().gPlus(*this).gLog();
      }
    } else {
      Tangent tangent;
      if (global) {
        tangent.angular() = rotation().gPlus(other.rotation().gInv()).gLog();
        tangent.linear() = (translation() - other.translation());
      } else {
        tangent.angular() = other.rotation().gInv().gPlus(rotation()).gLog();
        tangent.linear() = translation() - other.translation();
      }
      return tangent;
    }
  }

  /// Group action.
  /// \tparam TOther_ Other type.
  /// \param other Other vector.
  /// \param J_this Jacobian w.r.t. this.
  /// \param J_other Jacobian w.r.t. other.
  /// \param global Global Jacobian flag.
  /// \param coupled Coupled Jacobian flag.
  /// \return Vector.
  template <typename TOther_>
  auto act(const Eigen::MatrixBase<TOther_>& other, Scalar* J_this = nullptr, Scalar* J_other = nullptr, bool global = kGlobal, bool coupled = kCoupled) const -> Translation {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(TOther_, 3)

    const auto R_this = this->rotation().matrix();
    const Translation R_this_v = R_this * other;
    Translation output = R_this_v + translation();

    if (J_this) {
      auto J = Eigen::Map<JacobianNM<Translation, Tangent>>{J_this};

      if (coupled) {
        if (global) {
          Tangent::template AngularJacobian<Translation::kNumParameters>(J, 0).noalias() = Scalar{-1} * output.hat();
          Tangent::template LinearJacobian<Translation::kNumParameters>(J, 0).setIdentity();

        } else {
          Tangent::template AngularJacobian<Translation::kNumParameters>(J, 0).noalias() = Scalar{-1} * R_this * other.hat();
          Tangent::template LinearJacobian<Translation::kNumParameters>(J, 0).noalias() = R_this;
        }
      } else {
        if (global) {
          Tangent::template AngularJacobian<Translation::kNumParameters>(J, 0).noalias() = Scalar{-1} * R_this_v.hat();
          Tangent::template LinearJacobian<Translation::kNumParameters>(J, 0).setIdentity();
        } else {
          Tangent::template AngularJacobian<Translation::kNumParameters>(J, 0).noalias() = Scalar{-1} * R_this * other.hat();
          Tangent::template LinearJacobian<Translation::kNumParameters>(J, 0).setIdentity();
        }
      }
    }

    if (J_other) {
      Eigen::Map<JacobianNM<Translation>>{J_other}.noalias() = R_this;
    }

    return output;
  }
};

template <typename TScalar>
class SE3 final : public SE3Base<SE3<TScalar>> {
 public:
  using Base = SE3Base<SE3<TScalar>>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SE3)

  /// Default constructor.
  SE3() {
    this->rotation().setIdentity();
    this->translation().setZero();
  }

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
    this->rotation().coeffs().noalias() = rotation.coeffs();
    this->translation().noalias() = translation;
  }
};

template <typename TDerived>
class SE3TangentBase : public CartesianBase<TDerived> {
 public:
  // Definitions.
  using Base = CartesianBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  using Index = Eigen::Index;
  using Angular = Tangent<typename SE3<Scalar>::Rotation>;
  using AngularWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Angular>;
  using Linear = Tangent<typename SE3<Scalar>::Translation>;
  using LinearWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Linear>;

  // Constants.
  static constexpr auto kAngularOffset = 0;
  static constexpr auto kNumAngularParameters = 3;
  static constexpr auto kLinearOffset = kAngularOffset + kNumAngularParameters;
  static constexpr auto kNumLinearParameters = 3;

  static constexpr auto kGlobal = HYPER_DEFAULT_TO_GLOBAL_MANIFOLD_DERIVATIVES;
  static constexpr auto kCoupled = HYPER_DEFAULT_TO_COUPLED_MANIFOLD_DERIVATIVES;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SE3TangentBase)

  /// Angular Jacobian accessor/modifier.
  template <int NumRows, typename TDerived_>
  static auto AngularJacobian(Eigen::MatrixBase<TDerived_>& matrix, const Index& row) {
    return matrix.template block<NumRows, kNumAngularParameters>(row, kAngularOffset);
  }

  /// Linear Jacobian accessor/modifier.
  template <int NumRows, typename TDerived_>
  static auto LinearJacobian(Eigen::MatrixBase<TDerived_>& matrix, const Index& row) {
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
  /// \param global Global Jacobian flag.
  /// \param coupled Coupled Jacobian flag.
  /// \return Group element.
  auto gExp(Scalar* J_this = nullptr, bool global = kGlobal, bool coupled = kCoupled) const -> SE3<Scalar> {
    SE3<Scalar> se3;

    if (J_this) {
      JacobianNM<Tangent<SU2<Scalar>>> J_r;
      se3.rotation() = angular().gExp(J_r.data(), global);
      se3.translation().noalias() = linear();

      using Tangent = Tangent<SE3<Scalar>>;
      auto J = Eigen::Map<JacobianNM<Tangent>>{J_this};

      if (coupled) {
        if (global) {
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = J_r;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = linear().hat() * J_r;
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
        } else {
          Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = J_r;
          Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
          Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = se3.rotation().inverse().matrix();
        }
      } else {
        Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = J_r;
        Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
        Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
        Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
      }
    } else {
      se3.rotation() = angular().gExp();
      se3.translation().noalias() = linear();
    }

    return se3;
  }
};

template <typename TScalar>
class Tangent<SE3<TScalar>> final : public SE3TangentBase<Tangent<SE3<TScalar>>> {
 public:
  using Base = SE3TangentBase<Tangent<SE3<TScalar>>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Tangent)
};

}  // namespace hyper::variables

HYPER_DECLARE_EIGEN_INTERFACE(hyper::variables::SE3)
HYPER_DECLARE_TANGENT_MAP(hyper::variables::SE3)
