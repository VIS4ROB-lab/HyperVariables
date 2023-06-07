/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/rn.hpp"

namespace hyper::variables {

class StampedVariable : public Variable {
 public:
  /// Time accessor.
  /// \return Time.
  [[nodiscard]] virtual auto time() const -> const Scalar& = 0;

  /// Time modifier.
  /// \return Time.
  virtual auto time() -> Scalar& = 0;

  /// Stamp accessor.
  /// \return Stamp.
  [[nodiscard]] virtual auto stamp() const -> Eigen::Map<const Stamp> = 0;

  /// Stamp modifier.
  /// \return Stamp.
  virtual auto stamp() -> Eigen::Map<Stamp> = 0;
};

class ConstStampedVariable : public ConstVariable {
 public:
  /// Time accessor.
  /// \return Time.
  [[nodiscard]] virtual auto time() const -> const Scalar& = 0;

  /// Time modifier.
  /// \return Time.
  virtual auto time() -> const Scalar& = 0;

  /// Stamp accessor.
  /// \return Stamp.
  [[nodiscard]] virtual auto stamp() const -> Eigen::Map<const Stamp> = 0;

  /// Stamp modifier.
  /// \return Stamp.
  virtual auto stamp() -> Eigen::Map<const Stamp> = 0;
};

template <typename TDerived>
class StampedBase : public Traits<TDerived>::Base, public ConditionalConstBase_t<TDerived, StampedVariable, ConstStampedVariable> {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using ScalarWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Scalar>;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, VectorX>;
  using Base::Base;

  using Variable = typename Traits<TDerived>::Variable;
  using VariableWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Variable>;
  using StampWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Stamp>;

  // Constants.
  static constexpr auto kVariableOffset = 0;
  static constexpr auto kNumVariableParameters = Variable::kNumParameters;
  static constexpr auto kStampOffset = kVariableOffset + kNumVariableParameters;
  static constexpr auto kNumStampParameters = 1;
  static constexpr auto kNumParameters = kNumVariableParameters + kNumStampParameters;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(StampedBase)

  /// Random group element.
  /// \return Random element.
  static auto Random() -> Stamped<Variable> {
    Stamped<Variable> stamped_variable;
    stamped_variable.stamp() = Stamp::Random();
    stamped_variable.variable() = Variable::Random();
    return stamped_variable;
  }

  /// Map as Eigen vector.
  /// \return Vector.
  [[nodiscard]] auto asVector() const -> Eigen::Ref<const VectorX> final { return *this; }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Ref<VectorXWithConstIfNotLvalue> final { return *this; }

  /// Time accessor.
  /// \return Time.
  [[nodiscard]] auto time() const -> const Scalar& final { return this->data()[kStampOffset]; }

  /// Time modifier.
  /// \return Time.
  auto time() -> ScalarWithConstIfNotLvalue& final { return this->data()[kStampOffset]; }

  /// Stamp accessor.
  /// \return Stamp.
  [[nodiscard]] auto stamp() const -> Eigen::Map<const Stamp> final { return Eigen::Map<const Stamp>{this->data() + kStampOffset}; }

  /// Stamp modifier.
  /// \return Stamp.
  auto stamp() -> Eigen::Map<StampWithConstIfNotLvalue> final { return Eigen::Map<StampWithConstIfNotLvalue>{this->data() + kStampOffset}; }

  /// Variable accessor.
  /// \return Variable.
  [[nodiscard]] auto variable() const -> Eigen::Map<const Variable> { return Eigen::Map<const Variable>{this->data() + kVariableOffset}; }

  /// Variable modifier.
  /// \return Variable.
  auto variable() -> Eigen::Map<VariableWithConstIfNotLvalue> { return Eigen::Map<VariableWithConstIfNotLvalue>{this->data() + kVariableOffset}; }

  /// Tangent plus.
  /// \param other Other element.
  /// \return Element.
  auto tPlus(const Stamped<Tangent<Variable>>& other) const -> Stamped<Variable> {
    Stamped<Variable> stamped_variable;
    stamped_variable.stamp() = stamp() + other.stamp();
    stamped_variable.variable() = variable().tPlus(other.variable());
    return stamped_variable;
  }

  /// Tangent minus.
  /// \param other Other element.
  /// \return Tangent.
  auto tMinus(const Stamped<Variable>& other) const -> Stamped<Tangent<Variable>> {
    Stamped<Tangent<Variable>> stamped_tangent;
    stamped_tangent.stamp() = stamp() - other.stamp();
    stamped_tangent.variable() = variable().tMinus(other.variable());
    return stamped_tangent;
  }

  /// Tangent plus Jacobian.
  /// \return Jacobian.
  auto tPlusJacobian() const -> Jacobian<kNumParameters, Tangent<Variable>::kNumParameters + kNumStampParameters> {
    Jacobian<kNumParameters, Tangent<Variable>::kNumParameters + 1> J;
    J.template block<kNumVariableParameters, Tangent<Variable>::kNumParameters>(kVariableOffset, 0) = variable().tPlusJacobian();
    J.template block<kNumStampParameters, Tangent<Variable>::kNumParameters>(kStampOffset, 0).setZero();
    J.template block<kNumVariableParameters, kNumStampParameters>(kVariableOffset, Tangent<Variable>::kNumParameters).setZero();
    J.template block<kNumStampParameters, kNumStampParameters>(kStampOffset, Tangent<Variable>::kNumParameters).setIdentity();
    return J;
  }

  /// Tangent minus Jacobian.
  /// \return Jacobian.
  auto tMinusJacobian() const -> Jacobian<Tangent<Variable>::kNumParameters + 1, kNumParameters> {
    Jacobian<Tangent<Variable>::kNumParameters + 1, kNumParameters> J;
    J.template block<Tangent<Variable>::kNumParameters, kNumVariableParameters>(0, kVariableOffset) = variable().tMinusJacobian();
    J.template block<kNumStampParameters, kNumVariableParameters>(Tangent<Variable>::kNumParameters, kVariableOffset).setZero();
    J.template block<Tangent<Variable>::kNumParameters, kNumStampParameters>(0, kStampOffset).setZero();
    J.template block<kNumStampParameters, kNumStampParameters>(Tangent<Variable>::kNumParameters, kStampOffset).setIdentity();
    return J;
  }
};

template <typename TVariable>
class Stamped final : public StampedBase<Stamped<TVariable>> {
 public:
  using Base = StampedBase<Stamped<TVariable>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Stamped)
};

}  // namespace hyper::variables

HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_INTERFACE(hyper::variables, Stamped, typename)
