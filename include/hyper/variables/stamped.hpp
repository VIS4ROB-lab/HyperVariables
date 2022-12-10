/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/cartesian.hpp"

namespace hyper {

template <typename TScalar>
class AbstractStamped
    : public AbstractVariable<TScalar> {
 public:
  using Scalar = std::remove_const_t<TScalar>;
  using ScalarWithConstIfNotLvalue = TScalar;

  /// Virtual default destructor.
  ~AbstractStamped() override = default;

  /// Stamp accessor.
  /// \return Stamp.
  [[nodiscard]] virtual auto stamp() const -> const Scalar& = 0;

  /// Stamp modifier.
  /// \return Stamp.
  [[nodiscard]] virtual auto stamp() -> ScalarWithConstIfNotLvalue& = 0;
};

template <typename TDerived>
class StampedBase
    : public Traits<TDerived>::Base,
      public AbstractStamped<ConstScalarIfVariableIsNotLValue_t<TDerived>> {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstScalarIfVariableIsNotLValue_t<TDerived>;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, TVectorX<Scalar>>;
  using Base::Base;

  using Variable = Traits<TDerived>::Variable;
  using VariableWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Variable>;

  // Constants.
  static constexpr auto kVariableOffset = 0;
  static constexpr auto kNumVariableParameters = Variable::SizeAtCompileTime;
  static constexpr auto kStampOffset = kVariableOffset + kNumVariableParameters;
  static constexpr auto kNumStampParameters = 1;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(StampedBase)

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

  /// Stamp accessor.
  /// \return Stamp.
  [[nodiscard]] auto stamp() const -> const Scalar& final {
    return this->data()[kStampOffset];
  }

  /// Stamp modifier.
  /// \return Stamp.
  auto stamp() -> ScalarWithConstIfNotLvalue& {
    return this->data()[kStampOffset];
  }

  /// Variable accessor.
  /// \return Variable.
  [[nodiscard]] auto variable() const -> Eigen::Map<const Variable> {
    return Eigen::Map<const Variable>{this->data() + kVariableOffset};
  }

  /// Variable modifier.
  /// \return Variable.
  auto variable() -> Eigen::Map<VariableWithConstIfNotLvalue> {
    return Eigen::Map<VariableWithConstIfNotLvalue>{this->data() + kVariableOffset};
  }
};

template <typename TVariable>
class Stamped final
    : public StampedBase<Stamped<TVariable>> {
 public:
  using Base = StampedBase<Stamped<TVariable>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Stamped)
};

} // namespace hyper

HYPER_DECLARE_EIGEN_INTERFACE(hyper::Stamped)
