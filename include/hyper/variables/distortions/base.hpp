/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/distortions/abstract.hpp"

namespace hyper {

template <typename TDerived, typename TBase>
class DistortionBase : public Traits<TDerived>::Base, public TBase {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(DistortionBase)

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() const -> Eigen::Ref<const VectorX<Scalar>> final;

  /// Maps a distortion.
  /// \param raw_distortion Raw distortion.
  /// \return Mapped distortion.
  auto map(const Scalar* raw_distortion) const -> std::unique_ptr<ConstAbstractDistortion<Scalar>> final;

  /// Maps a distortion.
  /// \param raw_distortion Raw distortion.
  /// \return Mapped distortion.
  auto map(Scalar* raw_distortion) const -> std::unique_ptr<AbstractDistortion<Scalar>> final;
};

template <typename TDerived>
class ConstDistortion : public DistortionBase<TDerived, ConstAbstractDistortion<typename Traits<TDerived>::Base::Scalar>> {
 public:
  // Definitions.
  using Scalar = typename Traits<TDerived>::Base::Scalar;
  using Base = DistortionBase<TDerived, ConstAbstractDistortion<Scalar>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(ConstDistortion)

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Ref<const VectorX<Scalar>> final;
};

template <typename TDerived>
class Distortion : public DistortionBase<TDerived, AbstractDistortion<typename Traits<TDerived>::Base::Scalar>> {
 public:
  // Definitions.
  using Scalar = typename Traits<TDerived>::Base::Scalar;
  using Base = DistortionBase<TDerived, AbstractDistortion<Scalar>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Distortion)

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Ref<VectorX<Scalar>> final;

  /// Sets the default parameters.
  auto setDefault() -> Distortion& final;

  /// Perturbs this.
  /// \param scale Perturbation scale.
  auto perturb(const Scalar& scale) -> Distortion& final;
};

template <typename TDerived, typename TBase>
auto DistortionBase<TDerived, TBase>::asVector() const -> Eigen::Ref<const VectorX<Scalar>> {
  return *this;
}

template <typename TDerived, typename TBase>
auto DistortionBase<TDerived, TBase>::map(const Scalar* raw_distortion) const -> std::unique_ptr<ConstAbstractDistortion<Scalar>> {
  return std::make_unique<Eigen::Map<const typename Traits<TDerived>::Distortion>>(raw_distortion);
}

template <typename TDerived, typename TBase>
auto DistortionBase<TDerived, TBase>::map(Scalar* raw_distortion) const -> std::unique_ptr<AbstractDistortion<Scalar>> {
  return std::make_unique<Eigen::Map<typename Traits<TDerived>::Distortion>>(raw_distortion);
}

template <typename TDerived>
auto ConstDistortion<TDerived>::asVector() -> Eigen::Ref<const VectorX<Scalar>> {
  return *this;
}

template <typename TDerived>
auto Distortion<TDerived>::asVector() -> Eigen::Ref<VectorX<Scalar>> {
  return *this;
}

template <typename TDerived>
auto Distortion<TDerived>::setDefault() -> Distortion& {
  return static_cast<TDerived&>(*this).setDefault();
}

template <typename TDerived>
auto Distortion<TDerived>::perturb(const Scalar& scale) -> Distortion& {
  return static_cast<TDerived&>(*this).perturb(scale);
}

} // namespace hyper
