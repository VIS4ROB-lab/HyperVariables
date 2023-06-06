/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/distortions/forward.hpp"

#include "hyper/variables/rn.hpp"

namespace hyper::variables {

template <typename>
struct NumDistortionTraits;

template <>
struct NumDistortionTraits<float> {
  static constexpr float kDistortionTolerance = 1e-6;
  static constexpr float kDistortionTolerance2 = kDistortionTolerance * kDistortionTolerance;
  static constexpr auto kMaxNumDistortionSteps = 20;
};

template <>
struct NumDistortionTraits<double> {
  static constexpr double kDistortionTolerance = 1e-12;
  static constexpr double kDistortionTolerance2 = kDistortionTolerance * kDistortionTolerance;
  static constexpr auto kMaxNumDistortionSteps = 20;
};

class ConstDistortion : public ConstVariable {
 public:
  // Definitions.
  using Pixel = R2;
  using PixelJacobian = JacobianNM<R2>;
  using ParameterJacobian = JacobianNX<R2>;

  /// Perturbed distortion.
  /// \param scale Perturbation scale.
  /// \return Perturbed distortion.
  [[nodiscard]] virtual auto perturbed(const Scalar& scale) const -> VectorX = 0;

  /// Distorts a pixel.
  /// \param p Pixel to distort.
  /// \param J_p Pixel Jacobian.
  /// \param J_d  Distortion Jacobian.
  /// \param parameters Distort with external parameters.
  /// \return Distorted p.
  virtual auto distort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel = 0;

  /// Undistorts a pixel.
  /// \param p Pixel to undistort.
  /// \param J_p Pixel Jacobian.
  /// \param J_d  Distortion Jacobian.
  /// \param parameters Undistort with external parameters.
  /// \return Undistorted p.
  virtual auto undistort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel;
};

class Distortion : public Variable {
 public:
  // Definitions.
  using Pixel = R2;
  using PixelJacobian = JacobianNM<R2>;
  using ParameterJacobian = JacobianNX<R2>;

  /// Perturbed distortion.
  /// \param scale Perturbation scale.
  /// \return Perturbed distortion.
  [[nodiscard]] virtual auto perturbed(const Scalar& scale) const -> VectorX = 0;

  /// Distorts a pixel.
  /// \param p Pixel to distort.
  /// \param J_p Pixel Jacobian.
  /// \param J_d  Distortion Jacobian.
  /// \param parameters Distort with external parameters.
  /// \return Distorted p.
  virtual auto distort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel = 0;

  /// Undistorts a pixel.
  /// \param p Pixel to undistort.
  /// \param J_p Pixel Jacobian.
  /// \param J_d  Distortion Jacobian.
  /// \param parameters Undistort with external parameters.
  /// \return Undistorted p.
  virtual auto undistort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel;
};

template <typename TDerived>
class DistortionBase : public Traits<TDerived>::Base, public ConditionalConstBase_t<TDerived, Distortion, ConstDistortion> {
 public:
  // Constants.
  static constexpr auto kNumParameters = Traits<TDerived>::kNumParameters;

  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = typename Base::Scalar;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, VectorX>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(DistortionBase)

  /// Map as Eigen vector.
  /// \return Vector.
  [[nodiscard]] auto asVector() const -> Eigen::Ref<const VectorX> final { return *this; }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Ref<VectorXWithConstIfNotLvalue> final { return *this; }
};

}  // namespace hyper::variables
