/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/intrinsics.hpp"

namespace hyper::variables::tests {

class IntrinsicsTests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kOuterItr = 5;
  static constexpr auto kInnerItr = 25;
  static constexpr auto kInc = 1e-6;
  static constexpr auto kTol = 1e-6;
  static constexpr auto kSensorWidth_2 = 640 / 2;
  static constexpr auto kSensorHeight_2 = 480 / 2;

  // Definitions.
  using Scalar = double;
  using Pixel = variables::Pixel<Scalar>;
  using Intrinsics = variables::Intrinsics<Scalar>;
  using PixelJacobian = hyper::JacobianNM<Pixel>;
  using IntrinsicsJacobian = hyper::JacobianNM<Pixel, Intrinsics>;

  static auto randomPixel() -> Pixel { return Pixel::Random(); }

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
    return (pz - px).norm() < kTol;
  }

  [[nodiscard]] auto checkInputJacobian() const -> bool {
    PixelJacobian J_a;
    const auto px = randomPixel();
    const auto py = intrinsics_.denormalize(px, nullptr, nullptr);
    intrinsics_.normalize(py, J_a.data(), nullptr);

    PixelJacobian J_n;
    for (auto j = 0; j < Pixel::kNumParameters; ++j) {
      const Pixel py_0 = py - kInc * Pixel::Unit(j);
      const auto d_py_0 = intrinsics_.normalize(py_0, nullptr, nullptr);
      const Pixel py_1 = py + kInc * Pixel::Unit(j);
      const auto d_py_1 = intrinsics_.normalize(py_1, nullptr, nullptr);
      J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kInc);
    }

    return J_n.isApprox(J_a, kTol);
  }

  auto checkParameterJacobian() -> bool {
    IntrinsicsJacobian J_a;
    const Pixel px = Pixel::Random();
    const Pixel d_px = intrinsics_.normalize(px, nullptr, J_a.data());

    IntrinsicsJacobian J_n;
    for (auto j = 0; j < Intrinsics::kNumParameters; ++j) {
      const auto tmp = intrinsics_[j];
      intrinsics_[j] = tmp - kInc;
      const Pixel d_py_0 = intrinsics_.normalize(px, nullptr, nullptr);
      intrinsics_[j] = tmp + kInc;
      const Pixel d_py_1 = intrinsics_.normalize(px, nullptr, nullptr);
      J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kInc);
      intrinsics_[j] = tmp;
    }

    return J_n.isApprox(J_a, kTol);
  }

  [[nodiscard]] auto checkInverseTheorem() const -> bool {
    const auto px = randomPixel();

    PixelJacobian J_a;
    PixelJacobian J_b;
    const auto py = intrinsics_.denormalize(px, J_a.data(), nullptr);
    const auto pz = intrinsics_.normalize(py, J_b.data(), nullptr);

    return pz.isApprox(px, kTol) && (J_a * J_b).isIdentity(kTol);
  }

  [[nodiscard]] auto checkInverseInputJacobian() const -> bool {
    PixelJacobian J_a;
    const auto px = randomPixel();
    intrinsics_.denormalize(px, J_a.data(), nullptr);

    PixelJacobian J_n;
    for (auto j = 0; j < Pixel::kNumParameters; ++j) {
      const Pixel py_0 = px - kInc * Pixel::Unit(j);
      const auto d_py_0 = intrinsics_.denormalize(py_0, nullptr, nullptr);
      const Pixel py_1 = px + kInc * Pixel::Unit(j);
      const auto d_py_1 = intrinsics_.denormalize(py_1, nullptr, nullptr);
      J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kInc);
    }

    return J_n.isApprox(J_a, kTol);
  }

  auto checkInverseParameterJacobian() -> bool {
    const auto px = randomPixel();

    IntrinsicsJacobian J_a;
    intrinsics_.denormalize(px, nullptr, J_a.data());

    IntrinsicsJacobian J_n;
    for (auto j = 0; j < Intrinsics::kNumParameters; ++j) {
      const auto tmp = intrinsics_[j];
      intrinsics_[j] = tmp - kInc;
      const Pixel d_py_0 = intrinsics_.denormalize(px, nullptr, nullptr);
      intrinsics_[j] = tmp + kInc;
      const Pixel d_py_1 = intrinsics_.denormalize(px, nullptr, nullptr);
      J_n.col(j) = (d_py_1 - d_py_0) / (Scalar{2} * kInc);
      intrinsics_[j] = tmp;
    }

    return J_n.isApprox(J_a, kTol);
  }

 private:
  Intrinsics intrinsics_;
};

TEST_F(IntrinsicsTests, Duality) {
  for (auto i = 0; i < kOuterItr; ++i) {
    setRandom();
    for (auto j = 0; j < kInnerItr; ++j) {
      EXPECT_TRUE(checkDuality());
    }
  }
}

TEST_F(IntrinsicsTests, InputJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    setRandom();
    for (auto j = 0; j < kInnerItr; ++j) {
      EXPECT_TRUE(checkInputJacobian());
    }
  }
}

TEST_F(IntrinsicsTests, ParameterJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    setRandom();
    for (auto j = 0; j < kInnerItr; ++j) {
      EXPECT_TRUE(checkParameterJacobian());
    }
  }
}

TEST_F(IntrinsicsTests, InverseTheorem) {
  for (auto i = 0; i < kOuterItr; ++i) {
    setRandom();
    for (auto j = 0; j < kInnerItr; ++j) {
      EXPECT_TRUE(checkInverseTheorem());
    }
  }
}

TEST_F(IntrinsicsTests, InverseInputJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    setRandom();
    for (auto j = 0; j < kInnerItr; ++j) {
      EXPECT_TRUE(checkInverseInputJacobian());
    }
  }
}

TEST_F(IntrinsicsTests, InverseParameterJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    setRandom();
    for (auto j = 0; j < kInnerItr; ++j) {
      EXPECT_TRUE(checkInverseParameterJacobian());
    }
  }
}

}  // namespace hyper::variables::tests
