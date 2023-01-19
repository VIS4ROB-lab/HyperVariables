/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include <glog/logging.h>

#include "hyper/variables/distortions/equidistant.hpp"
#include "hyper/variables/distortions/iterative_radial.hpp"
#include "hyper/variables/distortions/radial_tangential.hpp"

namespace hyper::variables::tests {

using Scalar = double;

class DistortionTests : public testing::TestWithParam<Distortion<Scalar>*> {
 public:
  // Constants.
  static constexpr auto kMaxPerturbation = 0.03;
  static constexpr auto kNumOuterIterations = 5;
  static constexpr auto kNumInnerIterations = 25;
  static constexpr auto kNumericIncrement = 1e-7;
  static constexpr auto kNumericTolerance = 1e-5;

  // Definitions.
  using Pixel = variables::Pixel<Scalar>;
  using PixelJacobian = variables::JacobianNM<Pixel>;
  using Distortion = variables::Distortion<Scalar>;
  using DistortionJacobian = variables::JacobianNX<Pixel>;

  DistortionTests() : distortion_{nullptr} {}

  ~DistortionTests() override { delete distortion_; }

  void SetUp() override { distortion_ = GetParam(); }

  void TearDown() override {
    delete distortion_;
    distortion_ = nullptr;
  }

  auto checkDuality() const -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      const Pixel px = Pixel::Random();
      const Pixel py = distortion_->distort(px, nullptr, nullptr, nullptr);
      const Pixel pz = distortion_->undistort(py, nullptr, nullptr, nullptr);
      EXPECT_TRUE(pz.isApprox(px, kNumericTolerance));
    }
  }

  auto checkInputJacobian() const -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      PixelJacobian J_a;
      const Pixel px = Pixel::Random();
      const Pixel d_px = distortion_->distort(px, J_a.data(), nullptr, nullptr);

      PixelJacobian J_n;
      for (auto j = 0; j < Pixel::kNumParameters; ++j) {
        const Pixel py_0 = px - kNumericIncrement * Pixel::Unit(j);
        const Pixel d_py_0 = distortion_->distort(py_0, nullptr, nullptr, nullptr);
        const Pixel py_1 = px + kNumericIncrement * Pixel::Unit(j);
        const Pixel d_py_1 = distortion_->distort(py_1, nullptr, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kNumericTolerance));
    }
  }

  auto checkParameterJacobian() const -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      const auto num_distortion_parameters = distortion_->asVector().size();
      auto J_a = DistortionJacobian{Pixel::kNumParameters, num_distortion_parameters};
      const Pixel px = Pixel::Random();
      const Pixel d_px = distortion_->distort(px, nullptr, J_a.data(), nullptr);

      auto J_n = DistortionJacobian{Pixel::kNumParameters, num_distortion_parameters};
      auto vector = distortion_->asVector();
      for (auto j = 0; j < vector.size(); ++j) {
        const auto tmp = vector[j];
        vector[j] = tmp - kNumericIncrement;
        const Pixel d_py_0 = distortion_->distort(px, nullptr, nullptr, nullptr);
        vector[j] = tmp + kNumericIncrement;
        const Pixel d_py_1 = distortion_->distort(px, nullptr, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
        vector[j] = tmp;
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kNumericTolerance));
    }
  }

  auto checkInverseTheorem() const -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      const Pixel px = Pixel::Random();

      PixelJacobian J_a, J_b;
      const Pixel py = distortion_->distort(px, J_a.data(), nullptr, nullptr);
      const Pixel pz = distortion_->undistort(py, J_b.data(), nullptr, nullptr);

      EXPECT_TRUE((J_a * J_b).isIdentity(kNumericTolerance));
    }
  }

  auto checkInverseInputJacobian() -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      const Pixel px = Pixel::Random();

      PixelJacobian J_a;
      const Pixel d_px = distortion_->undistort(px, J_a.data(), nullptr, nullptr);

      PixelJacobian J_n;
      for (auto j = 0; j < Pixel::kNumParameters; ++j) {
        const Pixel py_0 = px - kNumericIncrement * Pixel::Unit(j);
        const Pixel d_py_0 = distortion_->undistort(py_0, nullptr, nullptr, nullptr);
        const Pixel py_1 = px + kNumericIncrement * Pixel::Unit(j);
        const Pixel d_py_1 = distortion_->undistort(py_1, nullptr, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kNumericTolerance));
    }
  }

  auto checkInverseParameterJacobian() const -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      const Pixel px = Pixel::Random();

      const auto num_distortion_parameters = distortion_->asVector().size();
      auto J_a = DistortionJacobian{Pixel::kNumParameters, num_distortion_parameters};
      const Pixel d_px = distortion_->undistort(px, nullptr, J_a.data(), nullptr);

      auto J_n = DistortionJacobian{Pixel::kNumParameters, num_distortion_parameters};
      auto vector = distortion_->asVector();
      for (auto j = 0; j < vector.size(); ++j) {
        const auto tmp = vector[j];
        vector[j] = tmp - kNumericIncrement;
        const Pixel d_py_0 = distortion_->undistort(px, nullptr, nullptr, nullptr);
        vector[j] = tmp + kNumericIncrement;
        const Pixel d_py_1 = distortion_->undistort(px, nullptr, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
        vector[j] = tmp;
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kNumericTolerance));
    }
  }

 protected:
  Distortion* distortion_;
};

TEST_P(DistortionTests, Duality) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkDuality();
  }
}

TEST_P(DistortionTests, InputJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkInputJacobian();
  }
}

TEST_P(DistortionTests, ParameterJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkParameterJacobian();
  }
}

TEST_P(DistortionTests, InverseTheorem) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkInverseTheorem();
  }
}

TEST_P(DistortionTests, InverseInputJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkInverseInputJacobian();
  }
}

TEST_P(DistortionTests, InverseParameterJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkInverseParameterJacobian();
  }
}

INSTANTIATE_TEST_SUITE_P(, DistortionTests,
                         testing::Values(new variables::EquidistantDistortion<Scalar, 5>(), new variables::RadialTangentialDistortion<Scalar, 2>(),
                                         new variables::RadialTangentialDistortion<Scalar, 4>(), new variables::IterativeRadialDistortion<Scalar, 2>(),
                                         new variables::IterativeRadialDistortion<Scalar, 4>()));

}  // namespace hyper::variables::tests
