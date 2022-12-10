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
      public AbstractStamped<std::conditional_t<VariableIsLValue<TDerived>::value, typename Traits<TDerived>::Base::Scalar, const typename Traits<TDerived>::Base::Scalar>> {
 public:
  // Constants.
  static constexpr auto kVariableOffset = 0;
  static constexpr auto kNumVariableParameters = Traits<TDerived>::Variable::SizeAtCompileTime;
  static constexpr auto kStampOffset = kVariableOffset + kNumVariableParameters;
  static constexpr auto kNumStampParameters = 1;

  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = Base::Scalar;
  using ScalarWithConstIfNotLvalue = std::conditional_t<VariableIsLValue<TDerived>::value, Scalar, const Scalar>;
  using VectorXWithConstIfNotLvalue = std::conditional_t<VariableIsLValue<TDerived>::value, TVectorX<Scalar>, const TVectorX<Scalar>>;
  using Base::Base;

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
  [[nodiscard]] auto variable() const -> Eigen::Map<const typename Traits<TDerived>::Variable> {
    return Eigen::Map<const typename Traits<TDerived>::Variable>{this->data() + kVariableOffset};
  }

  /// Variable modifier.
  /// \return Variable.
  auto variable() -> Eigen::Map<typename Traits<TDerived>::VariableWithConstIfNotLvalue> {
    return Eigen::Map<typename Traits<TDerived>::VariableWithConstIfNotLvalue>{this->data() + kVariableOffset};
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
