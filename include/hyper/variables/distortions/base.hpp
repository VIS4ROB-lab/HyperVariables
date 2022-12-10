/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/distortions/abstract.hpp"

namespace hyper {

template <typename TDerived>
class DistortionBase<TDerived, true>
    : public Traits<TDerived>::Base,
      public AbstractDistortion<ConstScalarIfVariableIsNotLValue_t<TDerived>> {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstScalarIfVariableIsNotLValue_t<TDerived>;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, TVectorX<Scalar>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(DistortionBase)

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() const -> Eigen::Map<const TVectorX<Scalar>> final;

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Map<VectorXWithConstIfNotLvalue> final;

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
  using Base = DistortionBase<TDerived, true>;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstScalarIfVariableIsNotLValue_t<TDerived>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(DistortionBase)

  /// Sets the default parameters.
  auto setDefault() -> DistortionBase& final;

  /// Perturbs this.
  /// \param scale Perturbation scale.
  auto perturb(const Scalar& scale) -> DistortionBase& final;
};

template <typename TDerived>
auto DistortionBase<TDerived, true>::asVector() const -> Eigen::Map<const TVectorX<Scalar>> {
  return {this->data(), this->size(), 1};
}

template <typename TDerived>
auto DistortionBase<TDerived, true>::asVector() -> Eigen::Map<VectorXWithConstIfNotLvalue> {
  return {this->data(), this->size(), 1};
}

template <typename TDerived>
auto DistortionBase<TDerived, true>::map(const Scalar* raw_distortion) const -> std::unique_ptr<AbstractDistortion<const Scalar>> {
  return std::make_unique<Eigen::Map<const typename Traits<TDerived>::Distortion>>(raw_distortion);
}

template <typename TDerived>
auto DistortionBase<TDerived, true>::map(Scalar* raw_distortion) const -> std::unique_ptr<AbstractDistortion<Scalar>> {
  return std::make_unique<Eigen::Map<typename Traits<TDerived>::Distortion>>(raw_distortion);
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
