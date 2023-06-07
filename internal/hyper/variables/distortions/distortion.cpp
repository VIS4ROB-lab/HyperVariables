/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>
#include <Eigen/LU>

#include "hyper/variables/distortions/distortion.hpp"

namespace hyper::variables {

auto ConstDistortion::undistort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Pixel {
  Pixel up = p;
  PixelJacobian J_p_p, J_p_p_i;

  for (auto i = 0; i <= NumDistortionTraits<Scalar>::kMaxNumDistortionSteps; ++i) {
    const auto b = (distort(up, J_p_p_i.data(), nullptr, parameters) - p).eval();
    J_p_p.noalias() = J_p_p_i.inverse();

    if (NumDistortionTraits<Scalar>::kDistortionTolerance2 < b.dot(b)) {
      DLOG_IF(WARNING, i == NumDistortionTraits<Scalar>::kMaxNumDistortionSteps) << "Maximum number of iterations reached.";
      DLOG_IF_EVERY_N(WARNING, J_p_p_i.determinant() < NumTraits<Scalar>::kSmallAngleTolerance, NumDistortionTraits<Scalar>::kMaxNumDistortionSteps)
          << "Numerical issues detected.";
      up.noalias() -= J_p_p * b;
    } else {
      break;
    }
  }

  if (J_p) {
    Eigen::Map<PixelJacobian>{J_p}.noalias() = J_p_p;
  }

  if (J_d) {
    const auto size = asVector().size();
    auto J_p_d_i = ParameterJacobian{Pixel::kNumParameters, size};
    distort(up, nullptr, J_p_d_i.data(), parameters);
    Eigen::Map<ParameterJacobian>{J_d, Pixel::kNumParameters, size}.noalias() = Scalar{-1} * J_p_p * J_p_d_i;
  }

  return up;
}

auto Distortion::undistort(const Eigen::Ref<const Pixel>& p, Scalar* J_p, Scalar* J_d, const Scalar* parameters) const -> Distortion::Pixel {
  Pixel up = p;
  PixelJacobian J_p_p, J_p_p_i;

  for (auto i = 0; i <= NumDistortionTraits<Scalar>::kMaxNumDistortionSteps; ++i) {
    const auto b = (distort(up, J_p_p_i.data(), nullptr, parameters) - p).eval();
    J_p_p.noalias() = J_p_p_i.inverse();

    if (NumDistortionTraits<Scalar>::kDistortionTolerance2 < b.dot(b)) {
      DLOG_IF(WARNING, i == NumDistortionTraits<Scalar>::kMaxNumDistortionSteps) << "Maximum number of iterations reached.";
      DLOG_IF_EVERY_N(WARNING, J_p_p_i.determinant() < NumTraits<Scalar>::kSmallAngleTolerance, NumDistortionTraits<Scalar>::kMaxNumDistortionSteps)
          << "Numerical issues detected.";
      up.noalias() -= J_p_p * b;
    } else {
      break;
    }
  }

  if (J_p) {
    Eigen::Map<PixelJacobian>{J_p}.noalias() = J_p_p;
  }

  if (J_d) {
    const auto size = asVector().size();
    auto J_p_d_i = ParameterJacobian{Pixel::kNumParameters, size};
    distort(up, nullptr, J_p_d_i.data(), parameters);
    Eigen::Map<ParameterJacobian>{J_d, Pixel::kNumParameters, size}.noalias() = Scalar{-1} * J_p_p * J_p_d_i;
  }

  return up;
}

}  // namespace hyper::variables
