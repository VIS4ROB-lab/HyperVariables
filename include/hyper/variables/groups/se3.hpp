/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/groups/su2.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper {

template <typename TDerived>
class SE3Base
    : public Traits<TDerived>::Base,
      public AbstractVariable<ConstScalarIfVariableIsNotLValue_t<TDerived>> {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstScalarIfVariableIsNotLValue_t<TDerived>;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, TVectorX<Scalar>>;
  using Base::Base;

  using Rotation = SU2<Scalar>;
  using RotationWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Rotation>;
  using Translation = Cartesian<Scalar, 3>;
  using TranslationWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Translation>;

  // Constants.
  static constexpr auto kRotationOffset = 0;
  static constexpr auto kNumRotationParameters = SU2<Scalar>::SizeAtCompileTime;
  static constexpr auto kTranslationOffset = kNumRotationParameters;
  static constexpr auto kNumTranslationParameters = 3;

  static constexpr auto kDefaultDerivativesAreGlobal = HYPER_DEFAULT_TO_GLOBAL_LIE_GROUP_DERIVATIVES;
  static constexpr auto kDefaultDerivativesAreCoupled = HYPER_DEFAULT_TO_COUPLED_LIE_GROUP_DERIVATIVES;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SE3Base)

  /// Utility function to access correct Jacobian submatrix.
  template <int NumRows, typename TMatrix>
  static auto RotationJacobian(TMatrix& matrix, Eigen::Index row) {
    return matrix.template block<NumRows, kNumRotationParameters>(row, kRotationOffset);
  }

  /// Utility function to access correct Jacobian submatrix.
  template <int NumRows, typename TMatrix>
  static auto TranslationJacobian(TMatrix& matrix, Eigen::Index row) {
    return matrix.template block<NumRows, kNumTranslationParameters>(row, kTranslationOffset);
  }

  /// Constructs an identity element.
  /// \return Identity element.
  static auto Identity() -> SE3<Scalar>;

  /// Constructs a random element.
  /// \return Random element.
  static auto Random() -> SE3<Scalar>;

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() const -> Eigen::Map<const TVectorX<Scalar>> final;

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Map<VectorXWithConstIfNotLvalue> final;

  /// Rotation accessor.
  /// \return Rotation.
  auto rotation() const -> Eigen::Map<const Rotation>;

  /// Rotation modifier.
  /// \return Rotation.
  auto rotation() -> Eigen::Map<RotationWithConstIfNotLvalue>;

  /// Translation accessor.
  /// \return Translation.
  auto translation() const -> Eigen::Map<const Translation>;

  /// Translation modifier.
  /// \return Translation.
  auto translation() -> Eigen::Map<TranslationWithConstIfNotLvalue>;

  /// Group inverse.
  /// \param raw_J Input Jacobian (if requested).
  /// \param global Request global Jacobians flag.
  /// \param coupled Compute SE3 instead of SU2 x R3 Jacobians.
  /// \return Inverse element.
  auto groupInverse(Scalar* raw_J_this = nullptr, bool global = kDefaultDerivativesAreGlobal, bool coupled = kDefaultDerivativesAreCoupled) const -> SE3<Scalar>;

  /// Group plus.
  /// \tparam TOtherDerived_ Other derived type.
  /// \param other Other input.
  /// \param raw_J_this This input Jacobian (if requested).
  /// \param raw_J_other Other input Jacobian (if requested).
  /// \param global Request global Jacobians flag.
  /// \param coupled Compute SE3 instead of SU2 x R3 Jacobians.
  /// \return Additive element.
  template <typename TOtherDerived_>
  auto groupPlus(const SE3Base<TOtherDerived_>& other, Scalar* raw_J_this = nullptr, Scalar* raw_J_other = nullptr, bool global = kDefaultDerivativesAreGlobal, bool coupled = kDefaultDerivativesAreCoupled) const -> SE3<Scalar>;

  /// Vector plus.
  /// \tparam TOtherDerived_ Other derived type.
  /// \param v Input vector.
  /// \param raw_J_this This input Jacobian (if requested).
  /// \param raw_J_v Point input Jacobian (if requested).
  /// \param global Request global Jacobians flag.
  /// \param coupled Compute SE3 instead of SU2 x R3 Jacobians.
  /// \return Additive element.
  template <typename TOtherDerived_>
  auto vectorPlus(const Eigen::MatrixBase<TOtherDerived_>& v, Scalar* raw_J_this = nullptr, Scalar* raw_J_v = nullptr, bool global = kDefaultDerivativesAreGlobal, bool coupled = kDefaultDerivativesAreCoupled) const -> Translation;

  /// Conversion to tangent element.
  /// \param raw_J Input Jacobian (if requested).
  /// \param global Request global Jacobians flag.
  /// \param coupled Compute SE3 instead of SU2 x R3 Jacobians.
  /// \return Tangent element.
  auto toTangent(Scalar* raw_J = nullptr, bool global = kDefaultDerivativesAreGlobal, bool coupled = kDefaultDerivativesAreCoupled) const -> Tangent<SE3<Scalar>>;
};

template <typename TScalar>
class SE3 final
    : public SE3Base<SE3<TScalar>> {
 public:
  using Base = SE3Base<SE3<TScalar>>;

  /// Default constructor.
  SE3() {
    this->rotation().setIdentity();
    this->translation().setZero();
  }

  /// Constructor from address.
  /// \param other Input address.
  explicit SE3(const TScalar* other)
      : Base{other} {}

  /// Copy constructor.
  /// \tparam TOtherDerived_ Other derived type.
  /// \param other Other input instance.
  template <typename TOtherDerived_>
  SE3(const SE3Base<TOtherDerived_>& other) // NOLINT
      : Base{other} {}

  /// Assignment operator.
  /// \tparam TOtherDerived_ Other dervied type.
  /// \param other Other input instance.
  /// \return This instance.
  template <typename TOtherDerived_>
  auto operator=(const SE3Base<TOtherDerived_>& other) -> SE3& {
    Base::operator=(other);
    return *this;
  }

  /// Constructor from rotation and translation.
  /// \tparam TDerived_ Derived type.
  /// \tparam TOtherDerived_ Other derived type.
  /// \param rotation Input rotation.
  /// \param translation Input translation.
  template <typename TDerived_, typename TOtherDerived_>
  SE3(const SU2Base<TDerived_>& rotation, const Eigen::MatrixBase<TOtherDerived_>& translation) {
    this->rotation().coeffs().noalias() = rotation.coeffs();
    this->translation().noalias() = translation;
  }
};

template <typename TDerived>
class SE3TangentBase
    : public CartesianBase<TDerived> {
 public:
  // Definitions.
  using Base = CartesianBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstScalarIfVariableIsNotLValue_t<TDerived>;
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

  static constexpr auto kDefaultDerivativesAreGlobal = HYPER_DEFAULT_TO_GLOBAL_LIE_GROUP_DERIVATIVES;
  static constexpr auto kDefaultDerivativesAreCoupled = HYPER_DEFAULT_TO_COUPLED_LIE_GROUP_DERIVATIVES;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SE3TangentBase)

  /// Utility function to access correct Jacobian submatrix.
  template <int NumRows, typename TMatrix>
  static auto AngularJacobian(TMatrix& matrix, Eigen::Index row) {
    return matrix.template block<NumRows, kNumAngularParameters>(row, kAngularOffset);
  }

  /// Utility function to access correct Jacobian submatrix.
  template <int NumRows, typename TMatrix>
  static auto LinearJacobian(TMatrix& matrix, Eigen::Index row) {
    return matrix.template block<NumRows, kNumLinearParameters>(row, kLinearOffset);
  }

  /// Angular tangent accessor.
  /// \return Angular tangent.
  [[nodiscard]] auto angular() const -> Eigen::Map<const Angular> {
    return Eigen::Map<const Angular>{this->data() + kAngularOffset};
  }

  /// Angular tangent modifier.
  /// \return Angular tangent.
  auto angular() -> Eigen::Map<AngularWithConstIfNotLvalue> {
    return Eigen::Map<AngularWithConstIfNotLvalue>{this->data() + kAngularOffset};
  }

  /// Linear tangent accessor.
  /// \return Linear tangent.
  [[nodiscard]] auto linear() const -> Eigen::Map<const Linear> {
    return Eigen::Map<const Linear>{this->data() + kLinearOffset};
  }

  /// Linear tangent modifier.
  /// \return Linear tangent.
  auto linear() -> Eigen::Map<LinearWithConstIfNotLvalue> {
    return Eigen::Map<LinearWithConstIfNotLvalue>{this->data() + kLinearOffset};
  }

  /// Converts this to a manifold element.
  /// \param raw_J Input Jacobian (if requested).
  /// \param global Request global Jacobians flag.
  /// \param coupled Compute SE3 instead of SU2 x R3 Jacobians.
  /// \return Manifold element.
  auto toManifold(Scalar* raw_J = nullptr, bool global = kDefaultDerivativesAreGlobal, bool coupled = kDefaultDerivativesAreCoupled) const -> SE3<Scalar>;
};

template <typename TScalar>
class Tangent<SE3<TScalar>> final
    : public SE3TangentBase<Tangent<SE3<TScalar>>> {
 public:
  using Base = SE3TangentBase<Tangent<SE3<TScalar>>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Tangent)
};

} // namespace hyper

HYPER_DECLARE_EIGEN_INTERFACE(hyper::SE3)
HYPER_DECLARE_TANGENT_MAP(hyper::SE3)

namespace hyper {

template <typename TDerived>
auto SE3Base<TDerived>::Identity() -> SE3<Scalar> {
  return SE3<Scalar>{};
}

template <typename TDerived>
auto SE3Base<TDerived>::Random() -> SE3<Scalar> {
  return {Rotation{Rotation::UnitRandom()}, Translation::Random()};
}

template <typename TDerived>
auto SE3Base<TDerived>::asVector() const -> Eigen::Map<const TVectorX<Scalar>> {
  return {this->data(), TDerived::SizeAtCompileTime, 1};
}

template <typename TDerived>
auto SE3Base<TDerived>::asVector() -> Eigen::Map<VectorXWithConstIfNotLvalue> {
  return {this->data(), TDerived::SizeAtCompileTime, 1};
}

template <typename TDerived>
auto SE3Base<TDerived>::rotation() const -> Eigen::Map<const Rotation> {
  return Eigen::Map<const Rotation>{this->data() + kRotationOffset};
}

template <typename TDerived>
auto SE3Base<TDerived>::rotation() -> Eigen::Map<RotationWithConstIfNotLvalue> {
  return Eigen::Map<RotationWithConstIfNotLvalue>{this->data() + kRotationOffset};
}

template <typename TDerived>
auto SE3Base<TDerived>::translation() const -> Eigen::Map<const Translation> {
  return Eigen::Map<const Translation>{this->data() + kTranslationOffset};
}

template <typename TDerived>
auto SE3Base<TDerived>::translation() -> Eigen::Map<TranslationWithConstIfNotLvalue> {
  return Eigen::Map<TranslationWithConstIfNotLvalue>{this->data() + kTranslationOffset};
}

template <typename TDerived>
auto SE3Base<TDerived>::groupInverse(Scalar* raw_J, const bool global, const bool coupled) const -> SE3<Scalar> {
  const auto i_rotation = rotation().groupInverse();
  const Translation i_translation = Scalar{-1} * (i_rotation.vectorPlus(translation()));
  auto output = SE3<Scalar>{i_rotation, i_translation};

  if (raw_J) {
    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<TJacobianNM<Tangent>>{raw_J};

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

  return output;
}

template <typename TDerived>
template <typename TOtherDerived_>
auto SE3Base<TDerived>::groupPlus(const SE3Base<TOtherDerived_>& other, Scalar* raw_J_this, Scalar* raw_J_other, const bool global, const bool coupled) const -> SE3<Scalar> {
  const auto R_this_R_other = rotation().groupPlus(other.rotation());
  const auto R_this_t_other = rotation().vectorPlus(other.translation());
  auto output = SE3<Scalar>{R_this_R_other, R_this_t_other + this->translation()};

  if (raw_J_this) {
    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<TJacobianNM<Tangent>>{raw_J_this};

    if (coupled) {
      if (global) {
        J.setIdentity();
      } else {
        const auto i_R_other = other.rotation().groupInverse().matrix();
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
        const auto i_R_other = other.rotation().groupInverse().matrix();
        Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = i_R_other;
        Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = Scalar{-1} * R_this * other.translation().hat();
        Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
        Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
      }
    }
  }

  if (raw_J_other) {
    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<TJacobianNM<Tangent>>{raw_J_other};

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

  return output;
}

template <typename TDerived>
template <typename TOtherDerived_>
auto SE3Base<TDerived>::vectorPlus(const Eigen::MatrixBase<TOtherDerived_>& v, Scalar* raw_J_this, Scalar* raw_J_v, const bool global, const bool coupled) const -> Translation {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(TOtherDerived_, 3)

  const auto R_this = this->rotation().matrix();
  const Translation R_this_v = R_this * v;
  Translation output = R_this_v + translation();

  if (raw_J_this) {
    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<TJacobianNM<Translation, Tangent>>{raw_J_this};

    if (coupled) {
      if (global) {
        Tangent::template AngularJacobian<Translation::SizeAtCompileTime>(J, 0).noalias() = Scalar{-1} * output.hat();
        Tangent::template LinearJacobian<Translation::SizeAtCompileTime>(J, 0).setIdentity();

      } else {
        Tangent::template AngularJacobian<Translation::SizeAtCompileTime>(J, 0).noalias() = Scalar{-1} * R_this * v.hat();
        Tangent::template LinearJacobian<Translation::SizeAtCompileTime>(J, 0).noalias() = R_this;
      }
    } else {
      if (global) {
        Tangent::template AngularJacobian<Translation::SizeAtCompileTime>(J, 0).noalias() = Scalar{-1} * R_this_v.hat();
        Tangent::template LinearJacobian<Translation::SizeAtCompileTime>(J, 0).setIdentity();
      } else {
        Tangent::template AngularJacobian<Translation::SizeAtCompileTime>(J, 0).noalias() = Scalar{-1} * R_this * v.hat();
        Tangent::template LinearJacobian<Translation::SizeAtCompileTime>(J, 0).setIdentity();
      }
    }
  }

  if (raw_J_v) {
    Eigen::Map<TJacobianNM<Translation>>{raw_J_v}.noalias() = R_this;
  }

  return output;
}

template <typename TDerived>
auto SE3Base<TDerived>::toTangent(Scalar* raw_J, const bool global, const bool coupled) const -> Tangent<SE3<Scalar>> {
  Tangent<SE3<Scalar>> output;

  if (raw_J) {
    TJacobianNM<Tangent<SU2<Scalar>>> J_r;
    output.angular().noalias() = rotation().toTangent(J_r.data(), global);
    output.linear().noalias() = translation();

    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<TJacobianNM<Tangent>>{raw_J};

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
    output.angular().noalias() = rotation().toTangent();
    output.linear().noalias() = translation();
  }

  return output;
}

template <typename TDerived>
auto SE3TangentBase<TDerived>::toManifold(Scalar* raw_J, const bool global, const bool coupled) const -> SE3<Scalar> {
  SE3<Scalar> output;

  if (raw_J) {
    TJacobianNM<Tangent<SU2<Scalar>>> J_r;
    output.rotation() = angular().toManifold(J_r.data(), global);
    output.translation().noalias() = linear();

    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<TJacobianNM<Tangent>>{raw_J};

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
        Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).noalias() = output.rotation().inverse().matrix();
      }
    } else {
      Tangent::template AngularJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).noalias() = J_r;
      Tangent::template AngularJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setZero();
      Tangent::template LinearJacobian<Tangent::kNumAngularParameters>(J, Tangent::kAngularOffset).setZero();
      Tangent::template LinearJacobian<Tangent::kNumLinearParameters>(J, Tangent::kLinearOffset).setIdentity();
    }
  } else {
    output.rotation() = angular().toManifold();
    output.translation().noalias() = linear();
  }

  return output;
}

} // namespace hyper
