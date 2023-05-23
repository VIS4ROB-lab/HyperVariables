/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/variables/lie_group_tests.hpp"
#include "hyper/variables/se3.hpp"

namespace hyper::variables::tests {

class QuaternionTests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kItr = 10;
  static constexpr auto kTol = 1e-7;

  // Definitions.
  using Scalar = double;
  using Group = Quaternion<Scalar>;

  static auto Random() -> Group { return Group{Eigen::internal::random<Scalar>(0.5, 1.5) * Group::UnitRandom().coeffs()}; }

  [[nodiscard]] auto checkGroupExponentials() const -> bool {
    const auto qle = group_.glog().gexp();
    const auto qel = group_.gexp().glog();
    return qle.isApprox(group_, kTol) && qel.isApprox(group_, kTol);
  }

  Group group_;
};

TEST_F(QuaternionTests, GroupExponentials) {
  for (auto i = 0; i < kItr; ++i) {
    group_ = Random();
    EXPECT_TRUE(checkGroupExponentials());
  }
}

using TestTypes = ::testing::Types<SU2<double>, SE3<double>>;
TYPED_TEST_SUITE(LieGroupTests, TestTypes);

TYPED_TEST(LieGroupTests, GroupOperators) {
  this->checkGroupOperators(10);
}

TYPED_TEST(LieGroupTests, TangentOperators) {
  this->checkTangentOperators(10);
}

}  // namespace hyper::variables::tests
