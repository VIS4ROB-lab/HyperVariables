/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/orthonormality_alignment.hpp"

namespace hyper::variables::tests {

using Scalar = double;

class OrthonormalityAlignmentTests
    : public testing::Test {
 protected:
  static constexpr auto kNumOuterIterations = 5;
  static constexpr auto kNumInnerIterations = 25;
  static constexpr auto kNumericIncrement = 1e-6;
  static constexpr auto kNumericTolerance = 1e-8;

  static constexpr auto Order = 3;
  using Alignment = OrthonormalityAlignment<Scalar, Order>;
  using Input = Alignment::Input;

  using InputJacobian = TJacobianNM<Alignment::Output, Alignment::Input>;
  using ParameterJacobian = TJacobianNM<Alignment::Output, Alignment>;

  auto setRandom() -> void {
    alignment_.setRandom();
  }

  [[nodiscard]] auto checkDuality() const -> bool {
    const auto S = alignment_.scalingMatrix();
    const auto A = alignment_.alignmentMatrix();
    const auto SA = alignment_.asMatrix();
    return (S * A).isApprox(SA, kNumericTolerance);
  }

  [[nodiscard]] auto checkInputJacobian() const -> bool {
    Input input = Input::Random();

    InputJacobian J_a;
    const auto output = alignment_.align(input, J_a.data(), nullptr);

    InputJacobian J_n;
    for (auto j = 0; j < Traits<Alignment>::kOrder; ++j) {
      const Input d_input = input + kNumericIncrement * Input::Unit(j);
      const auto d_output = alignment_.align(d_input, nullptr, nullptr);
      J_n.col(j) = (d_output - output) / kNumericIncrement;
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkParameterJacobian() const -> bool {
    Input input = Input::Random();

    ParameterJacobian J_a;
    const auto output = alignment_.align(input, nullptr, J_a.data());

    ParameterJacobian J_n;
    for (auto j = 0; j < alignment_.size(); ++j) {
      const Alignment d_alignment = alignment_ + kNumericIncrement * Alignment::Unit(j);
      const auto d_output = d_alignment.align(input, nullptr, nullptr);
      J_n.col(j) = (d_output - output) / kNumericIncrement;
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

 private:
  Alignment alignment_;
};

TEST_F(OrthonormalityAlignmentTests, Duality) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    setRandom();
    EXPECT_TRUE(checkDuality());
  }
}

TEST_F(OrthonormalityAlignmentTests, InputJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    setRandom();
    for (auto j = 0; j < kNumInnerIterations; ++j) {
      EXPECT_TRUE(checkInputJacobian());
    }
  }
}

TEST_F(OrthonormalityAlignmentTests, ParameterJacobian) {
  for (auto i = 0; i < kNumOuterIterations; ++i) {
    setRandom();
    for (auto j = 0; j < kNumInnerIterations; ++j) {
      EXPECT_TRUE(checkParameterJacobian());
    }
  }
}

} // namespace hyper::variables::tests
