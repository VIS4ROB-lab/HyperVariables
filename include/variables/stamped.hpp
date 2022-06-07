/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "variables/cartesian.hpp"

namespace hyper {

template <typename TScalar>
class AbstractStampedVariable
    : public AbstractVariable<TScalar> {
 public:
  using Scalar = std::remove_const_t<TScalar>;
  using ScalarWithConstIfNotLvalue = TScalar;

  /// Virtual default destructor.
  ~AbstractStampedVariable() override = default;

  /// Time accessor.
  /// \return Time.
  [[nodiscard]] virtual auto time() const -> const Scalar& = 0;

  /// Time modifier.
  /// \return Time.
  [[nodiscard]] virtual auto time() -> ScalarWithConstIfNotLvalue& = 0;
};

template <typename TDerived>
class StampedVariableBase
    : public AbstractStampedVariable<typename Traits<TDerived>::ScalarWithConstIfNotLvalue> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;

  /// Forwarding constructor.
  /// \tparam TArgs_ Input argument types.
  /// \param args Inputs arguments.
  template <typename... TArgs_>
  StampedVariableBase(TArgs_&&... args) // NOLINT
      : cartesian_{std::forward<TArgs_>(args)...} {}

  /// Memory accessor.
  /// \return Memory block.
  [[nodiscard]] auto memory() const -> MemoryBlock<const Scalar> final {
    return {cartesian_.data(), Traits<TDerived>::kNumParameters};
  }

  /// Memory accessor.
  /// \return Memory block.
  [[nodiscard]] auto memory() -> MemoryBlock<ScalarWithConstIfNotLvalue> final {
    return {cartesian_.data(), Traits<TDerived>::kNumParameters};
  }

  /// Time accessor.
  /// \return Time.
  [[nodiscard]] auto time() const -> const Scalar& final {
    return cartesian_.data()[Traits<TDerived>::kStampOffset];
  }

  /// Time modifier.
  /// \return Time.
  auto time() -> ScalarWithConstIfNotLvalue& {
    return cartesian_.data()[Traits<TDerived>::kStampOffset];
  }

  /// Variable accessor.
  /// \return Variable.
  [[nodiscard]] auto variable() const -> Eigen::Map<const typename Traits<TDerived>::Variable> {
    return Eigen::Map<const typename Traits<TDerived>::Variable>{cartesian_.data() + Traits<TDerived>::kVariableOffset};
  }

  /// Variable modifier.
  /// \return Variable.
  auto variable() -> Eigen::Map<typename Traits<TDerived>::VariableWithConstIfNotLvalue> {
    return Eigen::Map<typename Traits<TDerived>::VariableWithConstIfNotLvalue>{cartesian_.data() + Traits<TDerived>::kVariableOffset};
  }

  /// Cartesian accessor.
  /// \return Cartesian.
  [[nodiscard]] auto cartesian() const -> const CartesianBase<TDerived>& {
    return cartesian_;
  }

  /// Cartesian modifier.
  /// \return Cartesian.
  auto cartesian() -> CartesianBase<TDerived>& {
    return const_cast<CartesianBase<TDerived>&>(std::as_const(*this).cartesian());
  }

 private:
  CartesianBase<TDerived> cartesian_;
};

template <typename TVariable>
class StampedVariable final
    : public StampedVariableBase<StampedVariable<TVariable>> {
 public:
  using Base = StampedVariableBase<StampedVariable<TVariable>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(StampedVariable)
};

} // namespace hyper

HYPER_DECLARE_EIGEN_MAP(StampedVariable)
