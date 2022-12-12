/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include <glog/logging.h>

#include "hyper/variables/distortions/equidistant.hpp"
#include "hyper/variables/distortions/iterative_radial.hpp"
#include "hyper/variables/distortions/radial_tangential.hpp"

namespace hyper::variables::tests {

using Scalar = double;

template <typename TDistortion>
auto createDefaultDistortion() -> TDistortion* {
  auto distortion = new TDistortion();
  distortion->setDefault();
  return distortion;
}

class DistortionTests
    : public testing::TestWithParam<AbstractDistortion<Scalar>*> {
 public:
  static constexpr auto kNumOuterIterations = 5;
  static constexpr auto kNumInnerIterations = 25;
  static constexpr auto kNumericIncrement = 1e-7;
  static constexpr auto kNumericTolerance = 1e-5;

  DistortionTests()
      : distortion_{nullptr} {}

  ~DistortionTests() override {
    delete distortion_;
  }

  void SetUp() override {
    distortion_ = GetParam();
  }

  void TearDown() override {
    delete distortion_;
    distortion_ = nullptr;
  }

  auto checkDuality() const -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      const Pixel<Scalar> px = Pixel<Scalar>::Random();
      const Pixel<Scalar> py = distortion_->distort(px, nullptr, nullptr);
      const Pixel<Scalar> pz = distortion_->undistort(py, nullptr, nullptr);
      EXPECT_TRUE(pz.isApprox(px, kNumericTolerance));
    }
  }

  auto checkInputJacobian() const -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      auto J_a = TJacobianNM<Pixel<Scalar>>{};
      const Pixel<Scalar> px = Pixel<Scalar>::Random();
      const Pixel<Scalar> d_px = distortion_->distort(px, J_a.data(), nullptr);

      auto J_n = TJacobianNM<Pixel<Scalar>>{};
      for (auto j = 0; j < Pixel<Scalar>::kNumParameters; ++j) {
        const Pixel<Scalar> py_0 = px - kNumericIncrement * Pixel<Scalar>::Unit(j);
        const Pixel<Scalar> d_py_0 = distortion_->distort(py_0, nullptr, nullptr);
        const Pixel<Scalar> py_1 = px + kNumericIncrement * Pixel<Scalar>::Unit(j);
        const Pixel<Scalar> d_py_1 = distortion_->distort(py_1, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kNumericTolerance));
    }
  }

  auto checkParameterJacobian() const -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      auto J_a = distortion_->allocatePixelDistortionJacobian();
      const Pixel<Scalar> px = Pixel<Scalar>::Random();
      const Pixel<Scalar> d_px = distortion_->distort(px, nullptr, J_a.data());

      auto J_n = distortion_->allocatePixelDistortionJacobian();
      auto vector = distortion_->asVector();
      for (auto j = 0; j < vector.size(); ++j) {
        const auto tmp = vector[j];
        vector[j] = tmp - kNumericIncrement;
        const Pixel<Scalar> d_py_0 = distortion_->distort(px, nullptr, nullptr);
        vector[j] = tmp + kNumericIncrement;
        const Pixel<Scalar> d_py_1 = distortion_->distort(px, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
        vector[j] = tmp;
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kNumericTolerance));
    }
  }

  auto checkInverseTheorem() const -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      const Pixel<Scalar> px = Pixel<Scalar>::Random();

      auto J_a = TJacobianNM<Pixel<Scalar>>{};
      auto J_b = TJacobianNM<Pixel<Scalar>>{};
      const Pixel<Scalar> py = distortion_->distort(px, J_a.data(), nullptr);
      const Pixel<Scalar> pz = distortion_->undistort(py, J_b.data(), nullptr);

      EXPECT_TRUE((J_a * J_b).isIdentity(kNumericTolerance));
    }
  }

  auto checkInverseInputJacobian() -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      const Pixel<Scalar> px = Pixel<Scalar>::Random();

      auto J_a = TJacobianNM<Pixel<Scalar>>{};
      const Pixel<Scalar> d_px = distortion_->undistort(px, J_a.data(), nullptr);

      auto J_n = TJacobianNM<Pixel<Scalar>>{};
      for (auto j = 0; j < Pixel<Scalar>::kNumParameters; ++j) {
        const Pixel<Scalar> py_0 = px - kNumericIncrement * Pixel<Scalar>::Unit(j);
        const Pixel<Scalar> d_py_0 = distortion_->undistort(py_0, nullptr, nullptr);
        const Pixel<Scalar> py_1 = px + kNumericIncrement * Pixel<Scalar>::Unit(j);
        const Pixel<Scalar> d_py_1 = distortion_->undistort(py_1, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kNumericTolerance));
    }
  }

  auto checkInverseParameterJacobian() const -> void {
    for (auto i = 0; i < kNumInnerIterations; ++i) {
      const Pixel<Scalar> px = Pixel<Scalar>::Random();

      auto J_a = distortion_->allocatePixelDistortionJacobian();
      const Pixel<Scalar> d_px = distortion_->undistort(px, nullptr, J_a.data());

      auto J_n = distortion_->allocatePixelDistortionJacobian();
      auto vector = distortion_->asVector();
      for (auto j = 0; j < vector.size(); ++j) {
        const auto tmp = vector[j];
        vector[j] = tmp - kNumericIncrement;
        const Pixel<Scalar> d_py_0 = distortion_->undistort(px, nullptr, nullptr);
        vector[j] = tmp + kNumericIncrement;
        const Pixel<Scalar> d_py_1 = distortion_->undistort(px, nullptr, nullptr);
        J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
        vector[j] = tmp;
      }

      EXPECT_TRUE(J_n.isApprox(J_a, kNumericTolerance));
    }
  }

  auto setPerturbed() -> void {
    constexpr auto kMaxPerturbation = 0.002;
    distortion_->setDefault();
    distortion_->perturb(kMaxPerturbation);
  }

 protected:
  AbstractDistortion<Scalar>* distortion_;
};

TEST_P(DistortionTests, Duality) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    checkDuality();
    setPerturbed();
  }
}

TEST_P(DistortionTests, InputJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    checkInputJacobian();
    setPerturbed();
  }
}

TEST_P(DistortionTests, ParameterJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    checkParameterJacobian();
    setPerturbed();
  }
}

TEST_P(DistortionTests, InverseTheorem) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    checkInverseTheorem();
    setPerturbed();
  }
}

TEST_P(DistortionTests, InverseInputJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    checkInverseInputJacobian();
    setPerturbed();
  }
}

TEST_P(DistortionTests, InverseParameterJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    checkInverseParameterJacobian();
    setPerturbed();
  }
}

INSTANTIATE_TEST_SUITE_P(, DistortionTests,
    testing::Values(
        createDefaultDistortion<EquidistantDistortion<Scalar, 5>>(),
        createDefaultDistortion<RadialTangentialDistortion<Scalar, 2>>(),
        createDefaultDistortion<RadialTangentialDistortion<Scalar, 4>>(),
        createDefaultDistortion<IterativeRadialDistortion<Scalar, 2>>(),
        createDefaultDistortion<IterativeRadialDistortion<Scalar, 4>>()));

} // namespace hyper::variables::tests
