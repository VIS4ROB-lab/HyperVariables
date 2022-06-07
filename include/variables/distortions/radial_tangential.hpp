/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "variables/distortions/base.hpp"

namespace hyper {

template <typename TDerived>
class RadialTangentialDistortionBase
    : public DistortionBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = DistortionBase<TDerived>;
  using Base::Base;

  using PixelRef = typename Base::PixelRef;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(RadialTangentialDistortionBase)

  /// Radial order accessor.
  /// \return Order.
  [[nodiscard]] auto radialOrder() const -> Eigen::Index;

  /// Sets the radial order.
  /// \param order Input order.
  auto setRadialOrder(Eigen::Index order) -> void;

  /// Tangential order accessor.
  /// \return Order.
  [[nodiscard]] auto tangentialOrder() const -> Eigen::Index;

  /// Radial parameters accessor.
  /// \return Radial parameters.
  auto radial() const {
    return this->segment(Traits<TDerived>::kRadialOffset, radialOrder());
  }

  /// Radial parameters modifier.
  /// \return Radial parameters.
  auto radial() {
    return this->segment(Traits<TDerived>::kRadialOffset, radialOrder());
  }

  /// Tangential parameters accessor.
  /// \return Tangential parameters.
  auto tangential() const {
    return this->segment(Traits<TDerived>::kRadialOffset + radialOrder(), tangentialOrder());
  }

  /// Tangential parameters modifier.
  /// \return Tangential parameters.
  auto tangential() {
    return this->segment(Traits<TDerived>::kRadialOffset + radialOrder(), tangentialOrder());
  }

  /// Sets the default parameters.
  template <typename TScalar_ = ScalarWithConstIfNotLvalue, std::enable_if_t<!std::is_const_v<TScalar_>, bool> = true>
  auto setDefault() -> RadialTangentialDistortionBase&;

  /// Perturbs this.
  /// \param scale Perturbation scale.
  template <typename TScalar_ = ScalarWithConstIfNotLvalue, std::enable_if_t<!std::is_const_v<TScalar_>, bool> = true>
  auto perturb(const Scalar& scale) -> RadialTangentialDistortionBase&;

  /// Distorts a pixel.
  /// \param pixel Pixel to distort.
  /// \param raw_J_p_p Pixel Jacobian.
  /// \param raw_J_p_d  Distortion Jacobian.
  /// \return Distorted pixel.
  auto distort(const PixelRef& pixel, Scalar* raw_J_p_p, Scalar* raw_J_p_d) const -> Pixel<Scalar> final;
};

template <typename TScalar, int TOrder>
class RadialTangentialDistortion final
    : public RadialTangentialDistortionBase<RadialTangentialDistortion<TScalar, TOrder>> {
 public:
  using Base = RadialTangentialDistortionBase<RadialTangentialDistortion<TScalar, TOrder>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(RadialTangentialDistortion)
};

template <typename TDerived>
auto RadialTangentialDistortionBase<TDerived>::radialOrder() const -> Eigen::Index {
  return this->size() - tangentialOrder();
}

template <typename TDerived>
auto RadialTangentialDistortionBase<TDerived>::setRadialOrder(const Eigen::Index /* order */) -> void {
  this->resize(radialOrder() + tangentialOrder());
}

template <typename TDerived>
auto RadialTangentialDistortionBase<TDerived>::tangentialOrder() const -> Eigen::Index {
  return 2;
}

template <typename TDerived>
template <typename TScalar_, std::enable_if_t<!std::is_const_v<TScalar_>, bool>>
auto RadialTangentialDistortionBase<TDerived>::setDefault() -> RadialTangentialDistortionBase& {
  this->setZero();
  return *this;
}

template <typename TDerived>
template <typename TScalar_, std::enable_if_t<!std::is_const_v<TScalar_>, bool>>
auto RadialTangentialDistortionBase<TDerived>::perturb(const Scalar& scale) -> RadialTangentialDistortionBase& {
  using Parameters = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  this->radial() += Scalar{0.1} * scale * Parameters::Random(this->radialOrder(), 1);
  this->tangential() += scale * Parameters::Random(this->tangentialOrder(), 1);
  return *this;
}

template <typename TDerived>
auto RadialTangentialDistortionBase<TDerived>::distort(const PixelRef& pixel, Scalar* raw_J_p_p, Scalar* raw_J_p_d) const -> Pixel<Scalar> { // NOLINT
  // Map inputs.
  const auto radial_order = this->radialOrder();
  const auto x2 = pixel.x() * pixel.x();
  const auto y2 = pixel.y() * pixel.y();
  const auto rho2 = x2 + y2;
  const auto K = radial();
  const auto P = tangential();

  using Rhos = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  auto rhos = Rhos{radial_order, 1};
  for (auto i = 0; i < radial_order; ++i) {
    rhos[i] = (i == 0) ? rho2 : rhos[i - 1] * rho2;
  }

  const auto a = Scalar{1} + K.dot(rhos) + Scalar{2} * P.dot(pixel);

  if (raw_J_p_p) {
    auto d_radial_sum = K[0];
    for (auto i = 1; i < radial_order; ++i) {
      d_radial_sum += (i + 1) * K[i] * rhos[i - 1];
    }

    const auto d_radial = (Scalar{2} * d_radial_sum * pixel.transpose()).eval();
    const auto d_tangential = (Scalar{2} * P.transpose()).eval();

    using Jacobian = Jacobian<Pixel<Scalar>>;
    auto J = Eigen::Map<Jacobian>{raw_J_p_p};
    J = a * Jacobian::Identity() + pixel * (d_radial + d_tangential) + Scalar{2} * P * pixel.transpose();
  }

  if (raw_J_p_d) {
    const auto size = this->size();
    const auto tangential_order = this->tangentialOrder();
    auto J = Eigen::Map<DynamicInputJacobian<Pixel<Scalar>>>{raw_J_p_d, Traits<Pixel<Scalar>>::kNumParameters, size};
    for (auto i = 0; i < radial_order; ++i) {
      J.col(i) = rhos[i] * pixel;
    }
    J.middleCols(radial_order, tangential_order) = Scalar{2} * pixel * pixel.transpose() + rho2 * Jacobian<Pixel<Scalar>>::Identity();
  }

  return a * pixel + rho2 * P;
}

} // namespace hyper

HYPER_DECLARE_TEMPLATED_EIGEN_MAP(RadialTangentialDistortion, int)
