/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/intrinsics.hpp"

namespace hyper::tests {

using Scalar = double;

class IntrinsicsTests
    : public testing::Test {
 protected:
  static constexpr auto kNumOuterIterations = 5;
  static constexpr auto kNumInnerIterations = 25;
  static constexpr auto kNumericIncrement = 1e-6;
  static constexpr auto kNumericTolerance = 1e-6;
  static constexpr auto kSensorWidth_2 = 640 / 2;
  static constexpr auto kSensorHeight_2 = 480 / 2;

  static auto randomPixel() -> Pixel<Scalar> {
    return Pixel<Scalar>::Random();
  }

  auto setRandom() -> void {
    static constexpr auto kMinDeltaWidth = Scalar{0.25} * kSensorWidth_2;
    static constexpr auto kMaxDeltaWidth = Scalar{16} * kSensorWidth_2;
    static constexpr auto kMinDeltaHeight = Scalar{0.25} * kSensorHeight_2;
    static constexpr auto kMaxDeltaHeight = Scalar{16} * kSensorHeight_2;
    intrinsics_.cx() = kSensorWidth_2 + kMinDeltaWidth * Eigen::internal::random<Scalar>();
    intrinsics_.cy() = kSensorHeight_2 + kMinDeltaHeight * Eigen::internal::random<Scalar>();
    intrinsics_.fx() = Eigen::internal::random<Scalar>(kMinDeltaWidth, kMaxDeltaWidth);
    intrinsics_.fy() = Eigen::internal::random<Scalar>(kMinDeltaHeight, kMaxDeltaHeight);
  }

  [[nodiscard]] auto checkDuality() const -> bool {
    const auto px = randomPixel();
    const auto py = intrinsics_.denormalize(px, nullptr, nullptr);
    const auto pz = intrinsics_.normalize(py, nullptr, nullptr);
    return (pz - px).norm() < kNumericTolerance;
  }

  [[nodiscard]] auto checkInputJacobian() const -> bool {
    auto J_a = JacobianNM<Pixel<Scalar>>{};
    const auto px = randomPixel();
    const auto py = intrinsics_.denormalize(px, nullptr, nullptr);
    intrinsics_.normalize(py, J_a.data(), nullptr);

    auto J_n = JacobianNM<Pixel<Scalar>>{};
    for (auto j = 0; j < Pixel<Scalar>::kNumParameters; ++j) {
      const Pixel<Scalar> py_0 = py - kNumericIncrement * Pixel<Scalar>::Unit(j);
      const auto d_py_0 = intrinsics_.normalize(py_0, nullptr, nullptr);
      const Pixel<Scalar> py_1 = py + kNumericIncrement * Pixel<Scalar>::Unit(j);
      const auto d_py_1 = intrinsics_.normalize(py_1, nullptr, nullptr);
      J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

  auto checkParameterJacobian() -> bool {
    auto J_a = JacobianNM<Pixel<Scalar>, Intrinsics<Scalar>>{};
    const Pixel<Scalar> px = Pixel<Scalar>::Random();
    const Pixel<Scalar> d_px = intrinsics_.normalize(px, nullptr, J_a.data());

    auto J_n = JacobianNM<Pixel<Scalar>, Intrinsics<Scalar>>{};
    for (auto j = 0; j < Intrinsics<Scalar>::kNumParameters; ++j) {
      const auto tmp = intrinsics_[j];
      intrinsics_[j] = tmp - kNumericIncrement;
      const Pixel<Scalar> d_py_0 = intrinsics_.normalize(px, nullptr, nullptr);
      intrinsics_[j] = tmp + kNumericIncrement;
      const Pixel<Scalar> d_py_1 = intrinsics_.normalize(px, nullptr, nullptr);
      J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
      intrinsics_[j] = tmp;
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkInverseTheorem() const -> bool {
    const auto px = randomPixel();

    auto J_a = JacobianNM<Pixel<Scalar>>{};
    auto J_b = JacobianNM<Pixel<Scalar>>{};
    const auto py = intrinsics_.denormalize(px, J_a.data(), nullptr);
    const auto pz = intrinsics_.normalize(py, J_b.data(), nullptr);

    return pz.isApprox(px, kNumericTolerance) && (J_a * J_b).isIdentity(kNumericTolerance);
  }

  [[nodiscard]] auto checkInverseInputJacobian() const -> bool {
    auto J_a = JacobianNM<Pixel<Scalar>>{};
    const auto px = randomPixel();
    intrinsics_.denormalize(px, J_a.data(), nullptr);

    auto J_n = JacobianNM<Pixel<Scalar>>{};
    for (auto j = 0; j < Pixel<Scalar>::kNumParameters; ++j) {
      const Pixel<Scalar> py_0 = px - kNumericIncrement * Pixel<Scalar>::Unit(j);
      const auto d_py_0 = intrinsics_.denormalize(py_0, nullptr, nullptr);
      const Pixel<Scalar> py_1 = px + kNumericIncrement * Pixel<Scalar>::Unit(j);
      const auto d_py_1 = intrinsics_.denormalize(py_1, nullptr, nullptr);
      J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

  auto checkInverseParameterJacobian() -> bool {
    const auto px = randomPixel();

    auto J_a = JacobianNM<Pixel<Scalar>, Intrinsics<Scalar>>{};
    intrinsics_.denormalize(px, nullptr, J_a.data());

    auto J_n = JacobianNM<Pixel<Scalar>, Intrinsics<Scalar>>{};
    for (auto j = 0; j < Intrinsics<Scalar>::kNumParameters; ++j) {
      const auto tmp = intrinsics_[j];
      intrinsics_[j] = tmp - kNumericIncrement;
      const Pixel<Scalar> d_py_0 = intrinsics_.denormalize(px, nullptr, nullptr);
      intrinsics_[j] = tmp + kNumericIncrement;
      const Pixel<Scalar> d_py_1 = intrinsics_.denormalize(px, nullptr, nullptr);
      J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kNumericIncrement);
      intrinsics_[j] = tmp;
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

 private:
  Intrinsics<Scalar> intrinsics_;
};

TEST_F(IntrinsicsTests, Duality) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    setRandom();
    for (auto j = 0; j < kNumInnerIterations; ++j) {
      EXPECT_TRUE(checkDuality());
    }
  }
}

TEST_F(IntrinsicsTests, InputJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    setRandom();
    for (auto j = 0; j < kNumInnerIterations; ++j) {
      EXPECT_TRUE(checkInputJacobian());
    }
  }
}

TEST_F(IntrinsicsTests, ParameterJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    setRandom();
    for (auto j = 0; j < kNumInnerIterations; ++j) {
      EXPECT_TRUE(checkParameterJacobian());
    }
  }
}

TEST_F(IntrinsicsTests, InverseTheorem) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    setRandom();
    for (auto j = 0; j < kNumInnerIterations; ++j) {
      EXPECT_TRUE(checkInverseTheorem());
    }
  }
}

TEST_F(IntrinsicsTests, InverseInputJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    setRandom();
    for (auto j = 0; j < kNumInnerIterations; ++j) {
      EXPECT_TRUE(checkInverseInputJacobian());
    }
  }
}

TEST_F(IntrinsicsTests, InverseParameterJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    setRandom();
    for (auto j = 0; j < kNumInnerIterations; ++j) {
      EXPECT_TRUE(checkInverseParameterJacobian());
    }
  }
}

} // namespace hyper::tests
