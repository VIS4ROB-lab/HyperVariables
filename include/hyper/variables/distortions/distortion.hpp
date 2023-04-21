/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>

#include <glog/logging.h>
#include <Eigen/LU>

#include "hyper/variables/distortions/forward.hpp"

#include "hyper/jacobian.hpp"
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

namespace distortion {

template <typename TScalar, typename TDistortion>
auto Undistort(const TDistortion* distortion, const Eigen::Ref<const Pixel<TScalar>>& p, TScalar* J_p, TScalar* J_d, const TScalar* parameters) -> Pixel<TScalar> {
  // Definitions.
  using Pixel = variables::Pixel<TScalar>;

  Pixel output = p;
  JacobianNM<Pixel> J_p_p, J_p_p_i;

  for (auto i = 0; i <= NumDistortionTraits<TScalar>::kMaxNumDistortionSteps; ++i) {
    const auto b = (distortion->distort(output, J_p_p_i.data(), nullptr, parameters) - p).eval();
    J_p_p.noalias() = J_p_p_i.inverse();

    if (NumDistortionTraits<TScalar>::kDistortionTolerance2 < b.dot(b)) {
      DLOG_IF(WARNING, i == NumDistortionTraits<TScalar>::kMaxNumDistortionSteps) << "Maximum number of iterations reached.";
      DLOG_IF_EVERY_N(WARNING, J_p_p_i.determinant() < NumTraits<TScalar>::kSmallAngleTolerance, NumDistortionTraits<TScalar>::kMaxNumDistortionSteps)
          << "Numerical issues detected.";
      output.noalias() -= J_p_p * b;
    } else {
      break;
    }
  }

  if (J_p) {
    Eigen::Map<JacobianNM<Pixel>>{J_p}.noalias() = J_p_p;
  }

  if (J_d) {
    const auto size = distortion->asVector().size();
    auto J_p_d_i = JacobianNX<Pixel>{Pixel::kNumParameters, size};
    distortion->distort(output, nullptr, J_p_d_i.data(), parameters);
    Eigen::Map<JacobianNX<Pixel>>{J_d, Pixel::kNumParameters, size}.noalias() = TScalar{-1} * J_p_p * J_p_d_i;
  }

  return output;
}

}  // namespace distortion

template <typename TScalar>
class ConstDistortion : public ConstVariable<TScalar> {
 public:
  // Definitions.
  using Scalar = TScalar;
  using Pixel = variables::Pixel<TScalar>;

  /// Perturbed distortion.
  /// \param scale Perturbation scale.
  /// \return Perturbed distortion.
  virtual auto perturbed(const Scalar& scale) const -> VectorX<Scalar> = 0;

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
  virtual auto undistort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel {
    return distortion::Undistort(this, p, J_p, J_d, parameters);
  }
};

template <typename TScalar>
class Distortion : public Variable<TScalar> {
 public:
  // Definitions.
  using Scalar = TScalar;
  using Pixel = variables::Pixel<TScalar>;

  /// Perturbed distortion.
  /// \param scale Perturbation scale.
  /// \return Perturbed distortion.
  virtual auto perturbed(const Scalar& scale) const -> VectorX<Scalar> = 0;

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
  virtual auto undistort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel {
    return distortion::Undistort(this, p, J_p, J_d, parameters);
  }
};

template <typename TDerived>
class DistortionBase : public Traits<TDerived>::Base, public ConditionalConstBase_t<TDerived, Distortion<DerivedScalar_t<TDerived>>, ConstDistortion<DerivedScalar_t<TDerived>>> {
 public:
  // Definitions.
  using Base = typename Traits<TDerived>::Base;
  using Scalar = typename Base::Scalar;
  using VectorXWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, VectorX<Scalar>>;
  using Base::Base;

  // Constants.
  static constexpr auto kNumParameters = (int)Base::SizeAtCompileTime;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(DistortionBase)

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() const -> Eigen::Ref<const VectorX<Scalar>> final { return *this; }

  /// Map as Eigen vector.
  /// \return Vector.
  auto asVector() -> Eigen::Ref<VectorXWithConstIfNotLvalue> final { return *this; }
};

}  // namespace hyper::variables
