/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "variables/distortions/abstract.hpp"

namespace hyper {

template <typename TDerived>
class DistortionBase<TDerived, true>
    : public Traits<TDerived>::Base,
      public AbstractDistortion<typename Traits<TDerived>::ScalarWithConstIfNotLvalue> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = typename Traits<TDerived>::Base;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(DistortionBase)

  /// Memory accessor.
  /// \return Memory block.
  [[nodiscard]] auto memory() const -> MemoryBlock<const Scalar> final;

  /// Memory modifier.
  /// \return Memory block.
  [[nodiscard]] auto memory() -> MemoryBlock<ScalarWithConstIfNotLvalue> final;

  /// Maps a distortion.
  /// \param raw_distortion Raw distortion.
  /// \return Mapped distortion.
  auto map(const Scalar* raw_distortion) const -> std::unique_ptr<AbstractDistortion<const Scalar>> final;

  /// Maps a distortion.
  /// \param raw_distortion Raw distortion.
  /// \return Mapped distortion.
  auto map(Scalar* raw_distortion) const -> std::unique_ptr<AbstractDistortion<Scalar>> final;
};

template <typename TDerived>
class DistortionBase<TDerived, false>
    : public DistortionBase<TDerived, true> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = DistortionBase<TDerived, true>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(DistortionBase)

  /// Sets the default parameters.
  auto setDefault() -> DistortionBase& final;

  /// Perturbs this.
  /// \param scale Perturbation scale.
  auto perturb(const Scalar& scale) -> DistortionBase& final;
};

template <typename TDerived>
auto DistortionBase<TDerived, true>::memory() const -> MemoryBlock<const Scalar> {
  return {this->data(), this->size()};
}

template <typename TDerived>
auto DistortionBase<TDerived, true>::memory() -> MemoryBlock<ScalarWithConstIfNotLvalue> {
  return {this->data(), this->size()};
}

template <typename TDerived>
auto DistortionBase<TDerived, true>::map(const Scalar* raw_distortion) const -> std::unique_ptr<AbstractDistortion<const Scalar>> {
  return std::make_unique<Eigen::Map<const typename Traits<TDerived>::PlainDerivedType>>(raw_distortion);
}

template <typename TDerived>
auto DistortionBase<TDerived, true>::map(Scalar* raw_distortion) const -> std::unique_ptr<AbstractDistortion<Scalar>> {
  return std::make_unique<Eigen::Map<typename Traits<TDerived>::PlainDerivedType>>(raw_distortion);
}

template <typename TDerived>
auto DistortionBase<TDerived, false>::setDefault() -> DistortionBase& {
  return static_cast<TDerived&>(*this).setDefault();
}

template <typename TDerived>
auto DistortionBase<TDerived, false>::perturb(const Scalar& scale) -> DistortionBase& {
  return static_cast<TDerived&>(*this).perturb(scale);
}

} // namespace hyper
