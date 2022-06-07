/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "variables/distortions/base.hpp"

namespace hyper {

template <typename TDerived>
class EquidistantDistortionBase
    : public DistortionBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = DistortionBase<TDerived>;
  using Base::Base;

  using PixelRef = typename Base::PixelRef;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(EquidistantDistortionBase)

  /// Order accessor.
  /// \return Order.
  [[nodiscard]] auto order() const -> Eigen::Index;

  /// Sets the order.
  /// \param order Input order.
  auto setOrder(Eigen::Index order) -> void;

  /// Sets the default parameters.
  template <typename TScalar_ = ScalarWithConstIfNotLvalue, std::enable_if_t<!std::is_const_v<TScalar_>, bool> = true>
  auto setDefault() -> EquidistantDistortionBase&;

  /// Perturbs this.
  /// \param scale Perturbation scale.
  template <typename TScalar_ = ScalarWithConstIfNotLvalue, std::enable_if_t<!std::is_const_v<TScalar_>, bool> = true>
  auto perturb(const Scalar& scale) -> EquidistantDistortionBase&;

  /// Distorts a pixel.
  /// \param pixel Pixel to distort.
  /// \param raw_J_p_p Pixel Jacobian.
  /// \param raw_J_p_d  Distortion Jacobian.
  /// \return Distorted pixel.
  auto distort(const PixelRef& pixel, Scalar* raw_J_p_p, Scalar* raw_J_p_d) const -> Pixel<Scalar> final;

 private:
  using ThetaJacobian = SizedJacobian<Scalar, 1, Eigen::Dynamic>;

  /// Computes the theta distortion.
  /// \param theta Input theta.
  /// \param raw_J_t Theta Jacobian.
  /// \param raw_J_p Parameter Jacobian.
  /// \return Distorted theta.
  auto distortTheta(const Scalar& theta, Scalar* raw_J_t, Scalar* raw_J_p) const -> Scalar;
};

template <typename TScalar, int TOrder>
class EquidistantDistortion final
    : public EquidistantDistortionBase<EquidistantDistortion<TScalar, TOrder>> {
 public:
  using Base = EquidistantDistortionBase<EquidistantDistortion>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(EquidistantDistortion)
};

template <typename TDerived>
auto EquidistantDistortionBase<TDerived>::order() const -> Eigen::Index {
  return this->size();
}

template <typename TDerived>
auto EquidistantDistortionBase<TDerived>::setOrder(const Eigen::Index order) -> void {
  this->resize(order);
}

template <typename TDerived>
template <typename TScalar_, std::enable_if_t<!std::is_const_v<TScalar_>, bool>>
auto EquidistantDistortionBase<TDerived>::setDefault() -> EquidistantDistortionBase& {
  this->setIdentity();
  return *this;
}

template <typename TDerived>
template <typename TScalar_, std::enable_if_t<!std::is_const_v<TScalar_>, bool>>
auto EquidistantDistortionBase<TDerived>::perturb(const Scalar& scale) -> EquidistantDistortionBase& {
  using Parameters = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  const auto size = this->size();
  auto perturbations = (scale * Parameters::Random(size, 1)).eval();
  const auto data = perturbations.data();
  std::sort(data, data + size, [](const Scalar& a, const Scalar& b) { return std::abs(a) > std::abs(b); });
  (*this) += perturbations;
  (*this)[0] = Scalar{1};
  return *this;
}

template <typename TDerived>
auto EquidistantDistortionBase<TDerived>::distort(const PixelRef& pixel, Scalar* raw_J_p_p, Scalar* raw_J_p_d) const -> Pixel<Scalar> {
  // Map inputs.
  const auto size = this->size();
  const auto x2 = pixel.x() * pixel.x();
  const auto y2 = pixel.y() * pixel.y();
  const auto rho2 = x2 + y2;
  const auto rho = std::sqrt(rho2);
  const auto theta = std::atan2(rho, Scalar{1});

  Scalar J_theta;
  auto J_p_d = ThetaJacobian{1, size};
  const auto d_theta = distortTheta(theta, raw_J_p_p ? &J_theta : nullptr, raw_J_p_d ? J_p_d.data() : nullptr);

  const auto is_small_angle = (rho < NumericVariableTraits<Scalar>::kSmallAngleTolerance);
  const auto a = is_small_angle ? Scalar{1} : (d_theta / rho);

  if (raw_J_p_p) {
    using Jacobian = Jacobian<Pixel<Scalar>>;
    auto J = Eigen::Map<Jacobian>{raw_J_p_p};
    if (is_small_angle) {
      J.setZero();
    } else {
      J = ((J_theta / (Scalar{1} + rho2) - a) / rho2) * pixel * pixel.transpose() + a * Jacobian::Identity();
    }
  }

  if (raw_J_p_d) {
    auto J = Eigen::Map<DynamicInputJacobian<Pixel<Scalar>>>{raw_J_p_d, Traits<Pixel<Scalar>>::kNumParameters, size};
    if (is_small_angle) {
      J.setZero();
    } else {
      J = pixel * (J_p_d / rho);
    }
  }

  return a * pixel;
}

template <typename TDerived>
auto EquidistantDistortionBase<TDerived>::distortTheta(const Scalar& theta, Scalar* raw_J_t, Scalar* raw_J_p) const -> Scalar {
  const auto size = this->size();
  const auto theta2 = theta * theta;

  using Thetas = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  auto thetas = Thetas{size, 1};
  auto theta_i = theta;

  for (auto i = 0; i < size; ++i) {
    thetas[i] = theta_i;
    theta_i *= theta2;
  }

  if (raw_J_t) {
    auto d_thetas = Thetas{size, 1};
    auto theta_j = Scalar{1};
    for (auto j = 0; j < d_thetas.rows(); ++j) {
      d_thetas[j] = (Scalar{1} + Scalar{2} * j) * theta_j;
      theta_j *= theta2;
    }
    *raw_J_t = this->dot(d_thetas);
  }

  if (raw_J_p) {
    auto J = Eigen::Map<ThetaJacobian>{raw_J_p, 1, size};
    J = thetas.transpose();
  }

  return this->dot(thetas);
}

} // namespace hyper

HYPER_DECLARE_TEMPLATED_EIGEN_MAP(EquidistantDistortion, int)
