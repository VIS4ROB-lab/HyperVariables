/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/distortions/distortion.hpp"

namespace hyper::variables {

template <typename TDerived>
class EquidistantDistortionBase : public DistortionBase<TDerived> {
 public:
  // Definitions.
  using Base = DistortionBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Scalar>;
  using Base::Base;

  using Pixel = variables::Pixel<Scalar>;
  using PixelJacobian = variables::JacobianNM<Pixel>;
  using PlainDistortion = typename Traits<TDerived>::PlainDistortion;

  // Constants.
  static constexpr auto kOrder = Traits<TDerived>::kOrder;
  static constexpr auto kRadialOffset = 0;
  static constexpr auto kNumRadialParameters = kOrder;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(EquidistantDistortionBase)

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
  auto perturbed(const Scalar& scale) const -> VectorX<Scalar> final { return Perturbed(scale); }

  /// Distorts a pixel.
  /// \param p Pixel to distort.
  /// \param J_p Pixel Jacobian.
  /// \param J_d  Distortion Jacobian.
  /// \param parameters Distort with external parameters.
  /// \return Distorted p.
  auto distort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel final;

 private:
  using ParameterJacobian = Jacobian<Scalar, 1, Eigen::Dynamic>;

  /// Computes the theta distortion.
  /// \param theta Input theta.
  /// \param J_t Theta Jacobian.
  /// \param J_p Parameter Jacobian.
  /// \return Distorted theta.
  auto distortTheta(const Scalar& theta, Scalar* J_t, Scalar* J_p) const -> Scalar;
};

template <typename TScalar, int TOrder>
class EquidistantDistortion final : public EquidistantDistortionBase<EquidistantDistortion<TScalar, TOrder>> {
 public:
  using Base = EquidistantDistortionBase<EquidistantDistortion>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(EquidistantDistortion)
};

template <typename TDerived>
auto EquidistantDistortionBase<TDerived>::Default() -> PlainDistortion {
  return PlainDistortion::Identity();
}

template <typename TDerived>
auto EquidistantDistortionBase<TDerived>::Perturbation(const Scalar& scale) -> PlainDistortion {
  PlainDistortion plain_distortion = scale * PlainDistortion::Random();
  const auto data = plain_distortion.data();
  const auto size = plain_distortion.size();
  std::sort(data, data + size, [](const Scalar& a, const Scalar& b) { return std::abs(a) > std::abs(b); });
  plain_distortion[0] = Scalar{0};
  return plain_distortion;
}

template <typename TDerived>
auto EquidistantDistortionBase<TDerived>::Perturbed(const Scalar& scale) -> PlainDistortion {
  return Default() + Perturbation(scale);
}

template <typename TDerived>
auto EquidistantDistortionBase<TDerived>::distort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel {
  if (!parameters) {
    // Map inputs.
    const auto size = this->size();
    const auto x2 = p.x() * p.x();
    const auto y2 = p.y() * p.y();
    const auto rho2 = x2 + y2;
    const auto rho = std::sqrt(rho2);
    const auto theta = std::atan2(rho, Scalar{1});

    Scalar J_theta;
    auto J_p_i = ParameterJacobian{1, size};
    const auto d_theta = distortTheta(theta, J_p ? &J_theta : nullptr, J_d ? J_p_i.data() : nullptr);

    const auto is_small_angle = (rho < NumericVariableTraits<Scalar>::kSmallAngleTolerance);
    const auto a = is_small_angle ? Scalar{1} : (d_theta / rho);

    if (J_p) {
      auto J = Eigen::Map<PixelJacobian>{J_p};
      if (is_small_angle) {
        J.setZero();
      } else {
        J = ((J_theta / (Scalar{1} + rho2) - a) / rho2) * p * p.transpose() + a * PixelJacobian::Identity();
      }
    }

    if (J_d) {
      auto J = Eigen::Map<JacobianNX<Pixel>>{J_d, Pixel::kNumParameters, size};
      if (is_small_angle) {
        J.setZero();
      } else {
        J = p * (J_p_i / rho);
      }
    }

    return a * p;
  } else {
    return Eigen::Map<const PlainDistortion>{parameters}.distort(p, J_p, J_d, nullptr);
  }
}

template <typename TDerived>
auto EquidistantDistortionBase<TDerived>::distortTheta(const Scalar& theta, Scalar* J_t, Scalar* J_p) const -> Scalar {
  const auto size = this->size();
  const auto theta2 = theta * theta;

  using Thetas = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  auto thetas = Thetas{size, 1};
  auto theta_i = theta;

  for (auto i = 0; i < size; ++i) {
    thetas[i] = theta_i;
    theta_i *= theta2;
  }

  if (J_t) {
    auto d_thetas = Thetas{size, 1};
    auto theta_j = Scalar{1};
    for (auto j = 0; j < d_thetas.rows(); ++j) {
      d_thetas[j] = (Scalar{1} + Scalar{2} * j) * theta_j;
      theta_j *= theta2;
    }
    *J_t = this->dot(d_thetas);
  }

  if (J_p) {
    auto J = Eigen::Map<ParameterJacobian>{J_p, 1, size};
    J = thetas.transpose();
  }

  return this->dot(thetas);
}

}  // namespace hyper::variables

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(hyper::variables::EquidistantDistortion, int)
