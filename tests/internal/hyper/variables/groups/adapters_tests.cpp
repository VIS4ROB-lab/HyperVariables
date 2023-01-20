/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/groups/adapters.hpp"

namespace hyper::variables::tests {

template <typename TVariable>
class JacobianAdapterTests : public ::testing::Test {
 public:
  // Constants.
  static constexpr auto kNumInnerIterations = 10;

  // Definitions.
  using Manifold = TVariable;
  using Tangent = variables::Tangent<Manifold>;
  // using ManifoldJacobian = variables::JacobianNM<Manifold>;
  using TangentJacobian = variables::JacobianNM<Tangent>;

  /// Checks the adapter duality.
  /// \return True on success.
  auto checkDuality() -> bool {
    manifold = Manifold::Random();
    const auto J_l = JacobianAdapter<Manifold>(manifold.data());
    const auto J_p = InverseJacobianAdapter<Manifold>(manifold.data());
    return (J_l * J_p).isApprox(TangentJacobian::Identity());
  }

  Manifold manifold;
};

using TestTypes = ::testing::Types<SU2<double>, SE3<double>>;
TYPED_TEST_SUITE(JacobianAdapterTests, TestTypes);

TYPED_TEST(JacobianAdapterTests, Duality) {
  for (auto i = 0; i < TestFixture::kNumInnerIterations; ++i) {
    EXPECT_TRUE(this->checkDuality());
  }
}

}  // namespace hyper::variables::tests
