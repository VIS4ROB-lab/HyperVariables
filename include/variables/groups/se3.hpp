/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "variables/groups/su2.hpp"
#include "variables/jacobian.hpp"

namespace hyper {

template <typename TDerived>
class SE3Base
    : public Traits<TDerived>::Base,
      public AbstractVariable<typename Traits<TDerived>::ScalarWithConstIfNotLvalue> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = typename Traits<TDerived>::Base;
  using Base::Base;

  using Rotation = SU2<Scalar>;
  using RotationWithConstIfNotLvalue = std::conditional_t<std::is_const_v<ScalarWithConstIfNotLvalue>, const Rotation, Rotation>;
  using Translation = Cartesian<Scalar, 3>;
  using TranslationWithConstIfNotLvalue = std::conditional_t<std::is_const_v<ScalarWithConstIfNotLvalue>, const Translation, Translation>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SE3Base)

  /// Utility function to access correct Jacobian submatrix.
  template <int NumRows, typename TMatrix>
  static auto RotationJacobian(TMatrix& matrix, Eigen::Index row) {
    return matrix.template block<NumRows, Traits<TDerived>::kNumRotationParameters>(row, Traits<TDerived>::kRotationOffset);
  }

  /// Utility function to access correct Jacobian submatrix.
  template <int NumRows, typename TMatrix>
  static auto TranslationJacobian(TMatrix& matrix, Eigen::Index row) {
    return matrix.template block<NumRows, Traits<TDerived>::kNumTranslationParameters>(row, Traits<TDerived>::kTranslationOffset);
  }

  /// Constructs an identity element.
  /// \return Identity element.
  static auto Identity() -> SE3<Scalar>;

  /// Constructs a random element.
  /// \return Random element.
  static auto Random() -> SE3<Scalar>;

  /// Memory accessor.
  /// \return Memory block.
  [[nodiscard]] auto memory() const -> MemoryBlock<const Scalar> final;

  /// Memory modifier.
  /// \return Memory block.
  [[nodiscard]] auto memory() -> MemoryBlock<ScalarWithConstIfNotLvalue> final;

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
  /// \param frame Frame of the requested Jacobian.
  /// \return Inverse element.
  auto groupInverse(Scalar* raw_J_this = nullptr, Frame frame = Frame::DEFAULT) const -> SE3<Scalar>;

  /// Group plus.
  /// \tparam TOtherDerived_ Other derived type.
  /// \param other Other input.
  /// \param raw_J_this This input Jacobian (if requested).
  /// \param raw_J_other Other input Jacobian (if requested).
  /// \param frame Frame of the requested Jacobian(s).
  /// \return Additive element.
  template <typename TOtherDerived_>
  auto groupPlus(const SE3Base<TOtherDerived_>& other, Scalar* raw_J_this = nullptr, Scalar* raw_J_other = nullptr, Frame frame = Frame::DEFAULT) const -> SE3<Scalar>;

  /// Vector plus.
  /// \tparam TOtherDerived_ Other derived type.
  /// \param vector Input vector.
  /// \param raw_J_this This input Jacobian (if requested).
  /// \param raw_J_vector Point input Jacobian (if requested).
  /// \param frame Frame of the requested Jacobian(s).
  /// \return Additive element.
  template <typename TOtherDerived_>
  auto vectorPlus(const Eigen::MatrixBase<TOtherDerived_>& vector, Scalar* raw_J_this = nullptr, Scalar* raw_J_vector = nullptr, Frame frame = Frame::DEFAULT) const -> Translation;

  /// Conversion to tangent element.
  /// \param raw_J Input Jacobian (if requested).
  /// \param frame Frame of the requested Jacobian(s).
  /// \return Tangent element.
  auto toTangent(Scalar* raw_J = nullptr, Frame frame = Frame::DEFAULT) const -> Tangent<SE3<Scalar>>;
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
  using Base = CartesianBase<TDerived>;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  using Angular = Tangent<typename SE3<Scalar>::Rotation>;
  using AngularWithConstIfNotLvalue = std::conditional_t<std::is_const_v<ScalarWithConstIfNotLvalue>, const Angular, Angular>;
  using Linear = Tangent<typename SE3<Scalar>::Translation>;
  using LinearWithConstIfNotLvalue = std::conditional_t<std::is_const_v<ScalarWithConstIfNotLvalue>, const Linear, Linear>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SE3TangentBase)

  /// Utility function to access correct Jacobian submatrix.
  template <int NumRows, typename TMatrix>
  static auto AngularJacobian(TMatrix& matrix, Eigen::Index row) {
    return matrix.template block<NumRows, Traits<TDerived>::kNumAngularParameters>(row, Traits<TDerived>::kAngularOffset);
  }

  /// Utility function to access correct Jacobian submatrix.
  template <int NumRows, typename TMatrix>
  static auto LinearJacobian(TMatrix& matrix, Eigen::Index row) {
    return matrix.template block<NumRows, Traits<TDerived>::kNumLinearParameters>(row, Traits<TDerived>::kLinearOffset);
  }

  /// Angular tangent accessor.
  /// \return Angular tangent.
  [[nodiscard]] auto angular() const -> Eigen::Map<const Angular> {
    return Eigen::Map<const Angular>{this->data() + Traits<TDerived>::kAngularOffset};
  }

  /// Angular tangent modifier.
  /// \return Angular tangent.
  auto angular() -> Eigen::Map<AngularWithConstIfNotLvalue> {
    return Eigen::Map<AngularWithConstIfNotLvalue>{this->data() + Traits<TDerived>::kAngularOffset};
  }

  /// Linear tangent accessor.
  /// \return Linear tangent.
  [[nodiscard]] auto linear() const -> Eigen::Map<const Linear> {
    return Eigen::Map<const Linear>{this->data() + Traits<TDerived>::kLinearOffset};
  }

  /// Linear tangent modifier.
  /// \return Linear tangent.
  auto linear() -> Eigen::Map<LinearWithConstIfNotLvalue> {
    return Eigen::Map<LinearWithConstIfNotLvalue>{this->data() + Traits<TDerived>::kLinearOffset};
  }

  /// Converts this to a manifold element.
  /// \param raw_J Input Jacobian (if requested).
  /// \param frame Frame of the requested Jacobian(s).
  /// \return Manifold element.
  auto toManifold(Scalar* raw_J = nullptr, Frame frame = Frame::DEFAULT) const -> SE3<Scalar>;
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
auto SE3Base<TDerived>::memory() const -> MemoryBlock<const Scalar> {
  return {this->data(), Traits<TDerived>::kNumParameters};
}

template <typename TDerived>
auto SE3Base<TDerived>::memory() -> MemoryBlock<ScalarWithConstIfNotLvalue> {
  return {this->data(), Traits<TDerived>::kNumParameters};
}

template <typename TDerived>
auto SE3Base<TDerived>::rotation() const -> Eigen::Map<const Rotation> {
  return Eigen::Map<const Rotation>{this->data() + Traits<TDerived>::kRotationOffset};
}

template <typename TDerived>
auto SE3Base<TDerived>::rotation() -> Eigen::Map<RotationWithConstIfNotLvalue> {
  return Eigen::Map<RotationWithConstIfNotLvalue>{this->data() + Traits<TDerived>::kRotationOffset};
}

template <typename TDerived>
auto SE3Base<TDerived>::translation() const -> Eigen::Map<const Translation> {
  return Eigen::Map<const Translation>{this->data() + Traits<TDerived>::kTranslationOffset};
}

template <typename TDerived>
auto SE3Base<TDerived>::translation() -> Eigen::Map<TranslationWithConstIfNotLvalue> {
  return Eigen::Map<TranslationWithConstIfNotLvalue>{this->data() + Traits<TDerived>::kTranslationOffset};
}

template <typename TDerived>
auto SE3Base<TDerived>::groupInverse(Scalar* raw_J, const Frame frame) const -> SE3<Scalar> {
  const auto i_rotation = rotation().groupInverse();
  auto output = SE3<Scalar>{i_rotation, Scalar{-1} * (i_rotation.vectorPlus(this->translation()))};

  if (raw_J) {
    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<Jacobian<Tangent>>{raw_J};
    if (frame == Frame::GLOBAL) {
      const auto i_R = (Scalar{-1} * i_rotation.matrix()).eval();
      Tangent::template AngularJacobian<Traits<Tangent>::kNumAngularParameters>(J, Traits<Tangent>::kAngularOffset).noalias() = i_R;
      Tangent::template AngularJacobian<Traits<Tangent>::kNumLinearParameters>(J, Traits<Tangent>::kLinearOffset).noalias() = Scalar{-1} * i_R * this->translation().hat();
      Tangent::template LinearJacobian<Traits<Tangent>::kNumAngularParameters>(J, Traits<Tangent>::kAngularOffset).setZero();
      Tangent::template LinearJacobian<Traits<Tangent>::kNumLinearParameters>(J, Traits<Tangent>::kLinearOffset).noalias() = i_R;
    } else {
      const auto R = (Scalar{-1} * rotation().matrix()).eval();
      Tangent::template AngularJacobian<Traits<Tangent>::kNumAngularParameters>(J, Traits<Tangent>::kAngularOffset).noalias() = R;
      Tangent::template AngularJacobian<Traits<Tangent>::kNumLinearParameters>(J, Traits<Tangent>::kLinearOffset).noalias() = this->translation().hat() * R;
      Tangent::template LinearJacobian<Traits<Tangent>::kNumAngularParameters>(J, Traits<Tangent>::kAngularOffset).setZero();
      Tangent::template LinearJacobian<Traits<Tangent>::kNumLinearParameters>(J, Traits<Tangent>::kLinearOffset).noalias() = R;
    }
  }

  return output;
}

template <typename TDerived>
template <typename TOtherDerived_>
auto SE3Base<TDerived>::groupPlus(const SE3Base<TOtherDerived_>& other, Scalar* raw_J_this, Scalar* raw_J_other, const Frame frame) const -> SE3<Scalar> {
  auto output = SE3<Scalar>{this->rotation().groupPlus(other.rotation()), this->rotation().vectorPlus(other.translation()) + this->translation()};

  if (raw_J_this) {
    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<Jacobian<Tangent>>{raw_J_this};
    if (frame == Frame::GLOBAL) {
      J.setIdentity();
    } else {
      const auto i_R_other = other.rotation().groupInverse().matrix();
      Tangent::template AngularJacobian<Traits<Tangent>::kNumAngularParameters>(J, Traits<Tangent>::kAngularOffset).noalias() = i_R_other;
      Tangent::template AngularJacobian<Traits<Tangent>::kNumLinearParameters>(J, Traits<Tangent>::kLinearOffset).noalias() = Scalar{-1} * i_R_other * other.translation().hat();
      Tangent::template LinearJacobian<Traits<Tangent>::kNumAngularParameters>(J, Traits<Tangent>::kAngularOffset).setZero();
      Tangent::template LinearJacobian<Traits<Tangent>::kNumLinearParameters>(J, Traits<Tangent>::kLinearOffset).noalias() = i_R_other;
    }
  }

  if (raw_J_other) {
    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<Jacobian<Tangent>>{raw_J_other};
    if (frame == Frame::GLOBAL) {
      const auto R_this = this->rotation().matrix();
      Tangent::template AngularJacobian<Traits<Tangent>::kNumAngularParameters>(J, Traits<Tangent>::kAngularOffset).noalias() = R_this;
      Tangent::template AngularJacobian<Traits<Tangent>::kNumLinearParameters>(J, Traits<Tangent>::kLinearOffset).noalias() = this->translation().hat() * R_this;
      Tangent::template LinearJacobian<Traits<Tangent>::kNumAngularParameters>(J, Traits<Tangent>::kAngularOffset).setZero();
      Tangent::template LinearJacobian<Traits<Tangent>::kNumLinearParameters>(J, Traits<Tangent>::kLinearOffset).noalias() = R_this;
    } else {
      J.setIdentity();
    }
  }

  return output;
}

template <typename TDerived>
template <typename TOtherDerived_>
auto SE3Base<TDerived>::vectorPlus(const Eigen::MatrixBase<TOtherDerived_>& vector, Scalar* raw_J_this, Scalar* raw_J_vector, const Frame frame) const -> Translation {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(TOtherDerived_, 3)

  const auto R = this->rotation().matrix();
  auto output = (R * vector + translation()).eval();

  if (raw_J_this) {
    using Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<Jacobian<Translation, Tangent>>{raw_J_this};
    if (frame == Frame::GLOBAL) {
      Tangent::template AngularJacobian<Traits<Translation>::kNumParameters>(J, 0).noalias() = Scalar{-1} * output.hat();
      Tangent::template LinearJacobian<Traits<Translation>::kNumParameters>(J, 0).setIdentity();

    } else {
      Tangent::template AngularJacobian<Traits<Translation>::kNumParameters>(J, 0).noalias() = Scalar{-1} * R * vector.hat();
      Tangent::template LinearJacobian<Traits<Translation>::kNumParameters>(J, 0).noalias() = R;
    }
  }

  if (raw_J_vector) {
    Eigen::Map<Jacobian<Translation>>{raw_J_vector}.noalias() = R;
  }

  return output;
}

template <typename TDerived>
auto SE3Base<TDerived>::toTangent(Scalar* raw_J, const Frame frame) const -> Tangent<SE3<Scalar>> {
  Tangent<SE3<Scalar>> output;

  if (raw_J) {
    Jacobian<Tangent<SU2<Scalar>>> J_r;
    const auto t = rotation().toTangent(J_r.data(), frame);

    using SE3Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<Jacobian<SE3Tangent>>{raw_J};

    if (frame == Frame::GLOBAL) {
      J.setIdentity();
      SE3Tangent::template AngularJacobian<Traits<SE3Tangent>::kNumAngularParameters>(J, Traits<SE3Tangent>::kAngularOffset).noalias() = J_r;
      SE3Tangent::template AngularJacobian<Traits<SE3Tangent>::kNumLinearParameters>(J, Traits<SE3Tangent>::kLinearOffset).noalias() = Scalar{-1} * translation().hat();
      // SE3Tangent::template LinearJacobian<Traits<SE3Tangent>::kNumAngularParameters>(J, Traits<SE3Tangent>::kAngularOffset).setZero();
      // SE3Tangent::template LinearJacobian<Traits<SE3Tangent>::kNumLinearParameters>(J, Traits<SE3Tangent>::kLinearOffset).setIdentity();

    } else {
      J.setZero();
      SE3Tangent::template AngularJacobian<Traits<SE3Tangent>::kNumAngularParameters>(J, Traits<SE3Tangent>::kAngularOffset).noalias() = J_r;
      // SE3Tangent::template AngularJacobian<Traits<SE3Tangent>::kNumLinearParameters>(J, Traits<SE3Tangent>::kLinearOffset).setZero();
      // SE3Tangent::template LinearJacobian<Traits<SE3Tangent>::kNumAngularParameters>(J, Traits<SE3Tangent>::kAngularOffset).setZero();
      SE3Tangent::template LinearJacobian<Traits<SE3Tangent>::kNumLinearParameters>(J, Traits<SE3Tangent>::kLinearOffset).noalias() = rotation().matrix();
    }

    output.angular().noalias() = t;
    output.linear().noalias() = translation();

  } else {
    output.angular().noalias() = rotation().toTangent();
    output.linear().noalias() = translation();
  }

  return output;
}

template <typename TDerived>
auto SE3TangentBase<TDerived>::toManifold(Scalar* raw_J, const Frame frame) const -> SE3<Scalar> {
  SE3<Scalar> output;

  if (raw_J) {
    Jacobian<Tangent<SU2<Scalar>>> J_r;
    const auto r = angular().toManifold(J_r.data(), frame);

    using SE3Tangent = Tangent<SE3<Scalar>>;
    auto J = Eigen::Map<Jacobian<SE3Tangent>>{raw_J};

    if (frame == Frame::GLOBAL) {
      J.setIdentity();
      SE3Tangent::template AngularJacobian<Traits<SE3Tangent>::kNumAngularParameters>(J, Traits<SE3Tangent>::kAngularOffset).noalias() = J_r;
      SE3Tangent::template AngularJacobian<Traits<SE3Tangent>::kNumLinearParameters>(J, Traits<SE3Tangent>::kLinearOffset).noalias() = linear().hat() * J_r;
      // SE3Tangent::template LinearJacobian<Traits<SE3Tangent>::kNumAngularParameters>(J, Traits<SE3Tangent>::kAngularOffset).setZero();
      // SE3Tangent::template LinearJacobian<Traits<SE3Tangent>::kNumLinearParameters>(J, Traits<SE3Tangent>::kLinearOffset).setIdentity();

    } else {
      J.setZero();
      SE3Tangent::template AngularJacobian<Traits<SE3Tangent>::kNumAngularParameters>(J, Traits<SE3Tangent>::kAngularOffset).noalias() = J_r;
      // SE3Tangent::template AngularJacobian<Traits<SE3Tangent>::kNumLinearParameters>(J, Traits<SE3Tangent>::kLinearOffset).setZero();
      // SE3Tangent::template LinearJacobian<Traits<SE3Tangent>::kNumAngularParameters>(J, Traits<SE3Tangent>::kAngularOffset).setZero();
      SE3Tangent::template LinearJacobian<Traits<SE3Tangent>::kNumLinearParameters>(J, Traits<SE3Tangent>::kLinearOffset).noalias() = r.inverse().matrix();
    }

    output.rotation() = r;
    output.translation().noalias() = linear();

  } else {
    output.rotation() = angular().toManifold();
    output.translation().noalias() = linear();
  }

  return output;
}

} // namespace hyper
