/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "variables/distortions/base.hpp"

namespace hyper {

template <typename TDerived>
class IterativeRadialDistortionBase
    : public DistortionBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = DistortionBase<TDerived>;
  using Base::Base;

  using PixelRef = typename Base::PixelRef;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(IterativeRadialDistortionBase)

  /// Order accessor.
  /// \return Order.
  [[nodiscard]] auto order() const -> Eigen::Index;

  /// Sets the order.
  /// \param order Input order.
  auto setOrder(Eigen::Index order) -> void;

  /// Sets the default parameters.
  template <typename TScalar_ = ScalarWithConstIfNotLvalue, std::enable_if_t<!std::is_const_v<TScalar_>, bool> = true>
  auto setDefault() -> IterativeRadialDistortionBase&;

  /// Perturbs this.
  /// \param scale Perturbation scale.
  template <typename TScalar_ = ScalarWithConstIfNotLvalue, std::enable_if_t<!std::is_const_v<TScalar_>, bool> = true>
  auto perturb(const Scalar& scale) -> IterativeRadialDistortionBase&;

  /// Distorts a pixel.
  /// \param pixel Pixel to distort.
  /// \param raw_J_p Pixel Jacobian.
  /// \param raw_J_d  Distortion Jacobian.
  /// \return Distorted pixel.
  auto distort(const PixelRef& pixel, Scalar* raw_J_p_p, Scalar* raw_J_p_d) const -> Pixel<Scalar> final;

  /// Undistorts a pixel.
  /// \param pixel Pixel to undistort.
  /// \param raw_J_p_p Pixel Jacobian.
  /// \param raw_J_p_d  Distortion Jacobian.
  /// \return Undistorted pixel.
  auto undistort(const PixelRef& pixel, Scalar* raw_J_p_p, Scalar* raw_J_p_d) const -> Pixel<Scalar> final;

 private:
  auto IterativePixelPixelJacobian(const Scalar& x2, const Scalar& xy, const Scalar& y2, const Scalar& rho2, const Scalar& alpha) const -> Jacobian<Pixel<Scalar>>;
  auto InverseIterativePixelPixelJacobian(const Scalar& x2, const Scalar& xy, const Scalar& y2, const Scalar& rho2, const Scalar& alpha) const -> Jacobian<Pixel<Scalar>>;
  auto IterativePixelDistortionJacobian(const Pixel<Scalar>& pixel, const Scalar& rho2, const Scalar& alpha) const -> Pixel<Scalar>;
  auto InverseIterativePixelDistortionJacobian(const Pixel<Scalar>& pixel, const Scalar& rho2, const Scalar& alpha) const -> Pixel<Scalar>;
};

template <typename TScalar, int TOrder>
class IterativeRadialDistortion final
    : public IterativeRadialDistortionBase<IterativeRadialDistortion<TScalar, TOrder>> {
 public:
  using Base = IterativeRadialDistortionBase<IterativeRadialDistortion>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(IterativeRadialDistortion)
};

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::order() const -> Eigen::Index {
  return this->size();
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::setOrder(const Eigen::Index order) -> void {
  this->resize(order);
}

template <typename TDerived>
template <typename TScalar_, std::enable_if_t<!std::is_const_v<TScalar_>, bool>>
auto IterativeRadialDistortionBase<TDerived>::setDefault() -> IterativeRadialDistortionBase& {
  this->setZero();
  return *this;
}

template <typename TDerived>
template <typename TScalar_, std::enable_if_t<!std::is_const_v<TScalar_>, bool>>
auto IterativeRadialDistortionBase<TDerived>::perturb(const Scalar& scale) -> IterativeRadialDistortionBase& {
  using Parameters = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  const auto size = this->size();
  auto perturbations = (scale * Parameters::Random(size, 1)).eval();
  const auto data = perturbations.data();
  std::sort(data, data + size, [](const Scalar& a, const Scalar& b) { return std::abs(a) > std::abs(b); });
  (*this) += perturbations;
  return *this;
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::distort(const PixelRef& pixel, Scalar* raw_J_p_p, Scalar* raw_J_p_d) const -> Pixel<Scalar> {
  Pixel<Scalar> output = pixel;

  const auto size = this->size();
  const auto last = size - 1;
  for (auto i = last; i >= 0; --i) {
    const auto x2 = output.x() * output.x();
    const auto xy = output.x() * output.y();
    const auto y2 = output.y() * output.y();
    const auto rho2 = x2 + y2;

    if (raw_J_p_p) {
      auto J = Eigen::Map<Jacobian<Pixel<Scalar>>>{raw_J_p_p};
      if (i == last) {
        J = IterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]);
      } else {
        J = IterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]) * J;
      }
    }

    if (raw_J_p_d) {
      auto J = Eigen::Map<DynamicInputJacobian<Pixel<Scalar>>>{raw_J_p_d, Traits<Pixel<Scalar>>::kNumParameters, size};
      J.col(i) = IterativePixelDistortionJacobian(output, rho2, (*this)[i]);
      if (i < last) {
        J.rightCols(last - i) = IterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]) * J.rightCols(last - i);
      }
    }

    const auto a = Scalar{1} / (Scalar{0.5} + std::sqrt(Scalar{0.25} - (*this)[i] * rho2));
    output = a * output;
  }

  return output;
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::undistort(const PixelRef& pixel, Scalar* raw_J_p_p, Scalar* raw_J_p_d) const -> Pixel<Scalar> {
  Pixel<Scalar> output = pixel;

  const auto size = this->size();
  for (auto i = 0; i < size; ++i) {
    const auto x2 = output.x() * output.x();
    const auto xy = output.x() * output.y();
    const auto y2 = output.y() * output.y();
    const auto rho2 = x2 + y2;

    if (raw_J_p_p) {
      auto J = Eigen::Map<Jacobian<Pixel<Scalar>>>{raw_J_p_p};
      if (i == 0) {
        J = InverseIterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]);
      } else {
        J = InverseIterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]) * J;
      }
    }

    if (raw_J_p_d) {
      auto J = Eigen::Map<DynamicInputJacobian<Pixel<Scalar>>>{raw_J_p_d, Traits<Pixel<Scalar>>::kNumParameters, size};
      J.col(i) = InverseIterativePixelDistortionJacobian(output, rho2, (*this)[i]);
      if (i > 0) {
        J.leftCols(i) = InverseIterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]) * J.leftCols(i);
      }
    }

    const auto a = Scalar{1} / (Scalar{1} + (*this)[i] * rho2);
    output = a * output;
  }

  return output;
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::IterativePixelPixelJacobian(const Scalar& x2, const Scalar& xy, const Scalar& y2, const Scalar& rho2, const Scalar& alpha) const -> Jacobian<Pixel<Scalar>> {
  const auto a = std::sqrt(Scalar{0.25} - alpha * rho2);
  const auto b = Scalar{0.5} + a;
  const auto c = Scalar{1} / (b * b);
  const auto d = c * alpha * xy / a;

  Jacobian<Pixel<Scalar>> J;
  J(0, 0) = c * (Scalar{0.5} + (Scalar{0.25} - alpha * y2) / a);
  J(0, 1) = d;
  J(1, 0) = d;
  J(1, 1) = c * (Scalar{0.5} + (Scalar{0.25} - alpha * x2) / a);
  return J;
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::InverseIterativePixelPixelJacobian(const Scalar& x2, const Scalar& xy, const Scalar& y2, const Scalar& rho2, const Scalar& alpha) const -> Jacobian<Pixel<Scalar>> {
  const auto a = Scalar{1} + alpha * rho2;
  const auto b = Scalar{1} / (a * a);
  const auto c = alpha * (y2 - x2);
  const auto d = b * Scalar{-2} * alpha * xy;

  Jacobian<Pixel<Scalar>> J;
  J(0, 0) = b * (Scalar{1} + c);
  J(0, 1) = d;
  J(1, 0) = d;
  J(1, 1) = b * (Scalar{1} - c);
  return J;
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::IterativePixelDistortionJacobian(const Pixel<Scalar>& pixel, const Scalar& rho2, const Scalar& alpha) const -> Pixel<Scalar> {
  const auto a = std::sqrt(Scalar{0.25} - alpha * rho2);
  const auto b = Scalar{0.5} + a;
  const auto c = (rho2 / (Scalar{2} * a * b * b));
  return c * pixel;
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::InverseIterativePixelDistortionJacobian(const Pixel<Scalar>& pixel, const Scalar& rho2, const Scalar& alpha) const -> Pixel<Scalar> {
  const auto a = Scalar{1} + alpha * rho2;
  const auto b = Scalar{-1} * rho2 / (a * a);
  return b * pixel;
}

} // namespace hyper

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(hyper::IterativeRadialDistortion, int)
