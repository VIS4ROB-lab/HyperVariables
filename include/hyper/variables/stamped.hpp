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
  [[nodiscard]] virtual auto Stamp() -> ScalarWithConstIfNotLvalue& = 0;
};

template <typename TDerived>
class StampedBase
    : public Traits<TDerived>::Base,
      public AbstractStamped<typename Traits<TDerived>::ScalarWithConstIfNotLvalue> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = typename Traits<TDerived>::Base;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(StampedBase)

  /// Memory accessor.
  /// \return Memory block.
  [[nodiscard]] auto memory() const -> MemoryBlock<const Scalar> final {
    return {this->data(), this->size()};
  }

  /// Memory modifier.
  /// \return Memory block.
  [[nodiscard]] auto memory() -> MemoryBlock<ScalarWithConstIfNotLvalue> final {
    return {this->data(), this->size()};
  }

  /// Stamp accessor.
  /// \return Stamp.
  [[nodiscard]] auto stamp() const -> const Scalar& final {
    return this->data()[Traits<TDerived>::kStampOffset];
  }

  /// Stamp modifier.
  /// \return Stamp.
  auto stamp() -> ScalarWithConstIfNotLvalue& {
    return this->data()[Traits<TDerived>::kStampOffset];
  }

  /// Variable accessor.
  /// \return Variable.
  [[nodiscard]] auto variable() const -> Eigen::Map<const typename Traits<TDerived>::Variable> {
    return Eigen::Map<const typename Traits<TDerived>::Variable>{this->data() + Traits<TDerived>::kVariableOffset};
  }

  /// Variable modifier.
  /// \return Variable.
  auto variable() -> Eigen::Map<typename Traits<TDerived>::VariableWithConstIfNotLvalue> {
    return Eigen::Map<typename Traits<TDerived>::VariableWithConstIfNotLvalue>{this->data() + Traits<TDerived>::kVariableOffset};
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
