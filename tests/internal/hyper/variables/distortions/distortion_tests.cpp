/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include <glog/logging.h>

#include "hyper/variables/distortions/distortions.hpp"

namespace hyper::variables::tests {

class DistortionTests : public testing::TestWithParam<Distortion*> {
 public:
  // Constants.
  static constexpr auto kMaxPerturbation = 0.005;
  static constexpr auto kOuterItr = 5;
  static constexpr auto kInnerItr = 25;
  static constexpr auto kInc = 1e-6;
  static constexpr auto kTol = 1e-5;

  // Definitions.
  using PixelJacobian = hyper::JacobianNM<Pixel>;
  using DistortionJacobian = hyper::JacobianNX<Pixel>;

  DistortionTests() : distortion_{nullptr} {}

  ~DistortionTests() override { delete distortion_; }

  void SetUp() override { distortion_ = GetParam(); }

  void TearDown() override {
    delete distortion_;
    distortion_ = nullptr;
  }

  auto checkDuality() const -> void {
    for (auto i = 0; i < kInnerItr; ++i) {
      const Pixel px = Pixel::Random();
      const Pixel py = distortion_->distort(px, nullptr, nullptr, nullptr);
      const Pixel pz = distortion_->undistort(py, nullptr, nullptr, nullptr);
      EXPECT_TRUE(pz.isApprox(px, kTol));
    }
  }

  auto checkInputJacobian() const -> void {
    for (auto i = 0; i < kInnerItr; ++i) {
      PixelJacobian J_a;
      const Pixel px = Pixel::Random();
      const Pixel d_px = distortion_->distort(px, J_a.data(), nullptr, nullptr);

      PixelJacobian J_n;
      for (auto j = 0; j < Pixel::kNumParameters; ++j) {
        const Pixel py_0 = px - kInc * Pixel::Unit(j);
        const Pixel d_py_0 = distortion_->distort(py_0, nullptr, nullptr, nullptr);
        const Pixel py_1 = px + kInc * Pixel::Unit(j);
        const Pixel d_py_1 = distortion_->distort(py_1, nullptr, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kInc);
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kTol));
    }
  }

  auto checkParameterJacobian() const -> void {
    for (auto i = 0; i < kInnerItr; ++i) {
      const auto num_distortion_parameters = distortion_->asVector().size();
      auto J_a = DistortionJacobian{Pixel::kNumParameters, num_distortion_parameters};
      const Pixel px = Pixel::Random();
      const Pixel d_px = distortion_->distort(px, nullptr, J_a.data(), nullptr);

      auto J_n = DistortionJacobian{Pixel::kNumParameters, num_distortion_parameters};
      auto vector = distortion_->asVector();
      for (auto j = 0; j < vector.size(); ++j) {
        const auto tmp = vector[j];
        vector[j] = tmp - kInc;
        const Pixel d_py_0 = distortion_->distort(px, nullptr, nullptr, nullptr);
        vector[j] = tmp + kInc;
        const Pixel d_py_1 = distortion_->distort(px, nullptr, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kInc);
        vector[j] = tmp;
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kTol));
    }
  }

  auto checkInverseTheorem() const -> void {
    for (auto i = 0; i < kInnerItr; ++i) {
      const Pixel px = Pixel::Random();

      PixelJacobian J_a, J_b;
      const Pixel py = distortion_->distort(px, J_a.data(), nullptr, nullptr);
      const Pixel pz = distortion_->undistort(py, J_b.data(), nullptr, nullptr);

      EXPECT_TRUE((J_a * J_b).isIdentity(kTol));
    }
  }

  auto checkInverseInputJacobian() -> void {
    for (auto i = 0; i < kInnerItr; ++i) {
      const Pixel px = Pixel::Random();

      PixelJacobian J_a;
      const Pixel d_px = distortion_->undistort(px, J_a.data(), nullptr, nullptr);

      PixelJacobian J_n;
      for (auto j = 0; j < Pixel::kNumParameters; ++j) {
        const Pixel py_0 = px - kInc * Pixel::Unit(j);
        const Pixel d_py_0 = distortion_->undistort(py_0, nullptr, nullptr, nullptr);
        const Pixel py_1 = px + kInc * Pixel::Unit(j);
        const Pixel d_py_1 = distortion_->undistort(py_1, nullptr, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kInc);
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kTol));
    }
  }

  auto checkInverseParameterJacobian() const -> void {
    for (auto i = 0; i < kInnerItr; ++i) {
      const Pixel px = Pixel::Random();

      const auto num_distortion_parameters = distortion_->asVector().size();
      auto J_a = DistortionJacobian{Pixel::kNumParameters, num_distortion_parameters};
      const Pixel d_px = distortion_->undistort(px, nullptr, J_a.data(), nullptr);

      auto J_n = DistortionJacobian{Pixel::kNumParameters, num_distortion_parameters};
      auto vector = distortion_->asVector();
      for (auto j = 0; j < vector.size(); ++j) {
        const auto tmp = vector[j];
        vector[j] = tmp - kInc;
        const Pixel d_py_0 = distortion_->undistort(px, nullptr, nullptr, nullptr);
        vector[j] = tmp + kInc;
        const Pixel d_py_1 = distortion_->undistort(px, nullptr, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kInc);
        vector[j] = tmp;
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kTol));
    }
  }

 protected:
  Distortion* distortion_;
};

TEST_P(DistortionTests, Duality) {
  for (auto i = 0; i < kOuterItr; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkDuality();
  }
}

TEST_P(DistortionTests, InputJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkInputJacobian();
  }
}

TEST_P(DistortionTests, ParameterJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkParameterJacobian();
  }
}

TEST_P(DistortionTests, InverseTheorem) {
  for (auto i = 0; i < kOuterItr; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkInverseTheorem();
  }
}

TEST_P(DistortionTests, InverseInputJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkInverseInputJacobian();
  }
}

TEST_P(DistortionTests, InverseParameterJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    distortion_->asVector() = distortion_->perturbed(kMaxPerturbation);
    checkInverseParameterJacobian();
  }
}

INSTANTIATE_TEST_SUITE_P(, DistortionTests,
                         testing::Values(new variables::EquidistantDistortion<5>(), new variables::RadialTangentialDistortion<2>(), new variables::RadialTangentialDistortion<4>(),
                                         new variables::IterativeRadialDistortion<2>(), new variables::IterativeRadialDistortion<4>()));

}  // namespace hyper::variables::tests
