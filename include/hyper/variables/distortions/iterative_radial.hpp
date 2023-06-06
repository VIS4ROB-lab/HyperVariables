/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/distortions/distortion.hpp"

namespace hyper::variables {

template <typename TDerived>
class IterativeRadialDistortionBase : public DistortionBase<TDerived> {
 public:
  // Definitions.
  using Base = DistortionBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Scalar>;
  using Base::Base;

  using Pixel = variables::Pixel<Scalar>;
  using PixelJacobian = hyper::JacobianNM<Pixel>;
  using PlainDistortion = typename Traits<TDerived>::PlainDistortion;

  // Constants.
  static constexpr auto kOrder = Traits<TDerived>::kOrder;
  static constexpr auto kRadialOffset = 0;
  static constexpr auto kNumRadialParameters = kOrder;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(IterativeRadialDistortionBase)

  /// Default distortion (i.e. trivial).
  static auto Default() -> PlainDistortion;

  /// Perturbation.
  /// \param scale Perturbation scale.
  /// \return Perturbation.
  static auto Perturbation(const Scalar& scale) -> PlainDistortion;

  /// Perturbed distortion.
  /// \param scale Perturbation scale.
  /// \return Perturbed distortion.
  static auto Perturbed(const Scalar& scale) -> PlainDistortion;

  /// Order accessor.
  /// \return Order.
  [[nodiscard]] inline auto order() const -> int { return this->size(); }

  /// Sets the order.
  /// \param order Input order.
  inline auto setOrder(int order) -> void { this->resize(order); }

  /// Perturbed distortion.
  /// \param scale Perturbation scale.
  /// \return Perturbed distortion.
  auto perturbed(const Scalar& scale) const -> VectorX final { return Perturbed(scale); }

  /// Distorts a pixel.
  /// \param p Pixel to distort.
  /// \param J_p Pixel Jacobian.
  /// \param J_d  Distortion Jacobian.
  /// \param parameters Distort with external parameters.
  /// \return Distorted p.
  auto distort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel final;

  /// Undistorts a pixel.
  /// \param p Pixel to undistort.
  /// \param J_p Pixel Jacobian.
  /// \param J_d  Distortion Jacobian.
  /// \param parameters Undistort with external parameters.
  /// \return Undistorted p.
  auto undistort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel final;

 private:
  auto IterativePixelPixelJacobian(const Scalar& x2, const Scalar& xy, const Scalar& y2, const Scalar& rho2, const Scalar& alpha) const -> PixelJacobian;
  auto InverseIterativePixelPixelJacobian(const Scalar& x2, const Scalar& xy, const Scalar& y2, const Scalar& rho2, const Scalar& alpha) const -> PixelJacobian;
  auto IterativePixelDistortionJacobian(const Pixel& p, const Scalar& rho2, const Scalar& alpha) const -> Pixel;
  auto InverseIterativePixelDistortionJacobian(const Pixel& p, const Scalar& rho2, const Scalar& alpha) const -> Pixel;
};

template <typename TScalar, int TOrder>
class IterativeRadialDistortion final : public IterativeRadialDistortionBase<IterativeRadialDistortion<TScalar, TOrder>> {
 public:
  using Base = IterativeRadialDistortionBase<IterativeRadialDistortion>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(IterativeRadialDistortion)
};

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::Default() -> PlainDistortion {
  return PlainDistortion::Zero();
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::Perturbation(const Scalar& scale) -> PlainDistortion {
  PlainDistortion plain_distortion = scale * PlainDistortion::Random();
  const auto data = plain_distortion.data();
  const auto size = plain_distortion.size();
  std::sort(data, data + size, [](const Scalar& a, const Scalar& b) { return std::abs(a) > std::abs(b); });
  return plain_distortion;
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::Perturbed(const Scalar& scale) -> PlainDistortion {
  return Default() + Perturbation(scale);
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::distort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel {
  if (!parameters) {
    Pixel output = p;

    const auto size = this->size();
    const auto last = size - 1;
    for (auto i = last; i >= 0; --i) {
      const auto x2 = output.x() * output.x();
      const auto xy = output.x() * output.y();
      const auto y2 = output.y() * output.y();
      const auto rho2 = x2 + y2;

      if (J_p) {
        auto J = Eigen::Map<PixelJacobian>{J_p};
        if (i == last) {
          J = IterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]);
        } else {
          J = IterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]) * J;
        }
      }

      if (J_d) {
        auto J = Eigen::Map<JacobianNX<Pixel>>{J_d, Pixel::kNumParameters, size};
        J.col(i) = IterativePixelDistortionJacobian(output, rho2, (*this)[i]);
        if (i < last) {
          J.rightCols(last - i) = IterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]) * J.rightCols(last - i);
        }
      }

      const auto a = Scalar{1} / (Scalar{0.5} + std::sqrt(Scalar{0.25} - (*this)[i] * rho2));
      output = a * output;
    }

    return output;
  } else {
    return Eigen::Map<const PlainDistortion>{parameters}.distort(p, J_p, J_d, nullptr);
  }
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::undistort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel {
  if (!parameters) {
    Pixel output = p;

    const auto size = this->size();
    for (auto i = 0; i < size; ++i) {
      const auto x2 = output.x() * output.x();
      const auto xy = output.x() * output.y();
      const auto y2 = output.y() * output.y();
      const auto rho2 = x2 + y2;

      if (J_p) {
        auto J = Eigen::Map<PixelJacobian>{J_p};
        if (i == 0) {
          J = InverseIterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]);
        } else {
          J = InverseIterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]) * J;
        }
      }

      if (J_d) {
        auto J = Eigen::Map<JacobianNX<Pixel>>{J_d, Pixel::kNumParameters, size};
        J.col(i) = InverseIterativePixelDistortionJacobian(output, rho2, (*this)[i]);
        if (i > 0) {
          J.leftCols(i) = InverseIterativePixelPixelJacobian(x2, xy, y2, rho2, (*this)[i]) * J.leftCols(i);
        }
      }

      const auto a = Scalar{1} / (Scalar{1} + (*this)[i] * rho2);
      output = a * output;
    }

    return output;
  } else {
    return Eigen::Map<const PlainDistortion>{parameters}.undistort(p, J_p, J_d, nullptr);
  }
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::IterativePixelPixelJacobian(const Scalar& x2, const Scalar& xy, const Scalar& y2, const Scalar& rho2, const Scalar& alpha) const
    -> PixelJacobian {
  const auto a = std::sqrt(Scalar{0.25} - alpha * rho2);
  const auto b = Scalar{0.5} + a;
  const auto c = Scalar{1} / (b * b);
  const auto d = c * alpha * xy / a;

  PixelJacobian J;
  J(0, 0) = c * (Scalar{0.5} + (Scalar{0.25} - alpha * y2) / a);
  J(0, 1) = d;
  J(1, 0) = d;
  J(1, 1) = c * (Scalar{0.5} + (Scalar{0.25} - alpha * x2) / a);
  return J;
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::InverseIterativePixelPixelJacobian(const Scalar& x2, const Scalar& xy, const Scalar& y2, const Scalar& rho2,
                                                                                 const Scalar& alpha) const -> PixelJacobian {
  const auto a = Scalar{1} + alpha * rho2;
  const auto b = Scalar{1} / (a * a);
  const auto c = alpha * (y2 - x2);
  const auto d = b * Scalar{-2} * alpha * xy;

  PixelJacobian J;
  J(0, 0) = b * (Scalar{1} + c);
  J(0, 1) = d;
  J(1, 0) = d;
  J(1, 1) = b * (Scalar{1} - c);
  return J;
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::IterativePixelDistortionJacobian(const Pixel& p, const Scalar& rho2, const Scalar& alpha) const -> Pixel {
  const auto a = std::sqrt(Scalar{0.25} - alpha * rho2);
  const auto b = Scalar{0.5} + a;
  const auto c = (rho2 / (Scalar{2} * a * b * b));
  return c * p;
}

template <typename TDerived>
auto IterativeRadialDistortionBase<TDerived>::InverseIterativePixelDistortionJacobian(const Pixel& p, const Scalar& rho2, const Scalar& alpha) const -> Pixel {
  const auto a = Scalar{1} + alpha * rho2;
  const auto b = Scalar{-1} * rho2 / (a * a);
  return b * p;
}

}  // namespace hyper::variables

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(hyper::variables::IterativeRadialDistortion, int)
