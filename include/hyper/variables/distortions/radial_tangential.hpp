/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/distortions/distortion.hpp"

namespace hyper::variables {

template <typename TDerived>
class RadialTangentialDistortionBase : public DistortionBase<TDerived> {
 public:
  // Definitions.
  using Base = DistortionBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Scalar>;
  using Base::Base;

  using Pixel = R2;
  using PixelJacobian = hyper::JacobianNM<Pixel>;
  using PlainDistortion = typename Traits<TDerived>::PlainDistortion;

  // Constants.
  static constexpr auto kOrder = Traits<TDerived>::kOrder;
  static constexpr auto kRadialOffset = 0;
  static constexpr auto kNumRadialParameters = kOrder;
  static constexpr auto kTangentialOffset = kRadialOffset + kNumRadialParameters;
  static constexpr auto kNumTangentialParameters = 2;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(RadialTangentialDistortionBase)

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

  /// Radial order accessor.
  /// \return Order.
  [[nodiscard]] inline auto radialOrder() const -> int { return this->size() - tangentialOrder(); }

  /// Sets the radial order.
  /// \param order Input order.
  inline auto setRadialOrder(int order) -> void { this->resize(order + tangentialOrder()); }

  /// Tangential order accessor.
  /// \return Order.
  [[nodiscard]] inline auto tangentialOrder() const -> int { return 2; }

  /// Radial parameters accessor.
  /// \return Radial parameters.
  inline auto radial() const { return this->segment(kRadialOffset, radialOrder()); }

  /// Radial parameters modifier.
  /// \return Radial parameters.
  inline auto radial() { return this->segment(kRadialOffset, radialOrder()); }

  /// Tangential parameters accessor.
  /// \return Tangential parameters.
  inline auto tangential() const { return this->segment(kRadialOffset + radialOrder(), tangentialOrder()); }

  /// Tangential parameters modifier.
  /// \return Tangential parameters.
  inline auto tangential() { return this->segment(kRadialOffset + radialOrder(), tangentialOrder()); }

  /// Perturbed distortion.
  /// \param scale Perturbation scale.
  /// \return Perturbed distortion.
  auto perturbed(const Scalar& scale) const -> VectorX final { return Perturbed(scale); }

  /// Distorts a pixel.
  /// \param p Pixel to distort.
  /// \param J_p Pixel Jacobian.
  /// \param J_d  Distortion Jacobian.
  /// \param parameters Distort with external parameters.
  /// \return Distorted pixel.
  auto distort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel final;
};

template <int TOrder>
class RadialTangentialDistortion final : public RadialTangentialDistortionBase<RadialTangentialDistortion<TOrder>> {
 public:
  using Base = RadialTangentialDistortionBase<RadialTangentialDistortion>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(RadialTangentialDistortion)
};

template <typename TDerived>
auto RadialTangentialDistortionBase<TDerived>::Default() -> PlainDistortion {
  return PlainDistortion::Zero();
}

template <typename TDerived>
auto RadialTangentialDistortionBase<TDerived>::Perturbation(const Scalar& scale) -> PlainDistortion {
  PlainDistortion plain_distortion = scale * PlainDistortion::Random();
  plain_distortion.radial() *= Scalar{0.1} * scale;
  plain_distortion.tangential() * scale;
  return plain_distortion;
}

template <typename TDerived>
auto RadialTangentialDistortionBase<TDerived>::Perturbed(const Scalar& scale) -> PlainDistortion {
  return Default() + Perturbation(scale);
}

template <typename TDerived>
auto RadialTangentialDistortionBase<TDerived>::distort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel {  // NOLINT
  if (!parameters) {
    // Map inputs.
    const auto radial_order = this->radialOrder();
    const auto x2 = p.x() * p.x();
    const auto y2 = p.y() * p.y();
    const auto rho2 = x2 + y2;
    const auto K = radial();
    const auto P = tangential();

    using Rhos = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    auto rhos = Rhos{radial_order, 1};
    for (auto i = 0; i < radial_order; ++i) {
      rhos[i] = (i == 0) ? rho2 : rhos[i - 1] * rho2;
    }

    const auto a = Scalar{1} + K.dot(rhos) + Scalar{2} * P.dot(p);

    if (J_p) {
      auto d_radial_sum = K[0];
      for (auto i = 1; i < radial_order; ++i) {
        d_radial_sum += (i + 1) * K[i] * rhos[i - 1];
      }

      const auto d_radial = (Scalar{2} * d_radial_sum * p.transpose()).eval();
      const auto d_tangential = (Scalar{2} * P.transpose()).eval();

      auto J = Eigen::Map<PixelJacobian>{J_p};
      J = a * PixelJacobian::Identity() + p * (d_radial + d_tangential) + Scalar{2} * P * p.transpose();
    }

    if (J_d) {
      const auto size = this->size();
      const auto tangential_order = this->tangentialOrder();
      auto J = Eigen::Map<JacobianNX<Pixel>>{J_d, Pixel::kNumParameters, size};
      for (auto i = 0; i < radial_order; ++i) {
        J.col(i) = rhos[i] * p;
      }
      J.middleCols(radial_order, tangential_order) = Scalar{2} * p * p.transpose() + rho2 * PixelJacobian::Identity();
    }

    return a * p + rho2 * P;
  } else {
    return Eigen::Map<const PlainDistortion>{parameters}.distort(p, J_p, J_d, nullptr);
  }
}

}  // namespace hyper::variables

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(hyper::variables::RadialTangentialDistortion, int)
