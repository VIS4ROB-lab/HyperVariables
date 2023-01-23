/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/cartesian.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper::variables {

template <typename TDerived>
class IntrinsicsBase : public CartesianBase<TDerived> {
 public:
  // Definitions.
  using Base = CartesianBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using ScalarWithConstIfNotLvalue = ConstValueIfVariableIsNotLValue_t<TDerived, Scalar>;
  using Base::Base;

  // Constants.
  static constexpr auto kPrincipalOffset = 0;
  static constexpr auto kPrincipalOffsetX = kPrincipalOffset;
  static constexpr auto kPrincipalOffsetY = kPrincipalOffset + 1;
  static constexpr auto kNumPrincipalParameters = 2;
  static constexpr auto kFocalOffset = kPrincipalOffset + kNumPrincipalParameters;
  static constexpr auto kFocalOffsetX = kFocalOffset;
  static constexpr auto kFocalOffsetY = kFocalOffset + 1;
  static constexpr auto kNumFocalParameters = 2;

  // using Index = Eigen::Index;
  using Pixel = variables::Pixel<Scalar>;

  using Input = Pixel;
  using InputJacobian = variables::JacobianNM<Pixel>;
  using ParameterJacobian = variables::JacobianNM<Pixel, Base>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(IntrinsicsBase)

  /// Retrieves the x-component of the principal point.
  /// \return Principal point in x-direction.
  [[nodiscard]] auto cx() const -> const Scalar& { return this->data()[kPrincipalOffsetX]; }

  /// Retrieves the x-component of the principal point.
  /// \return Principal point in x-direction.
  auto cx() -> ScalarWithConstIfNotLvalue& { return this->data()[kPrincipalOffsetX]; }

  /// Retrieves the y-component of the principal point.
  /// \return Principal point in y-direction.
  [[nodiscard]] auto cy() const -> const Scalar& { return this->data()[kPrincipalOffsetY]; }

  /// Retrieves the y-component of the principal point.
  /// \return Principal point in y-direction.
  auto cy() -> ScalarWithConstIfNotLvalue& { return this->data()[kPrincipalOffsetY]; }

  /// Retrieves the x-component of the focal length.
  /// \return Focal length in x-direction.
  [[nodiscard]] auto fx() const -> const Scalar& { return this->data()[kFocalOffsetX]; }

  /// Retrieves the x-component of the focal length.
  /// \return Focal length in x-direction.
  auto fx() -> ScalarWithConstIfNotLvalue& { return this->data()[kFocalOffsetX]; }

  /// Retrieves the y-component of the focal length.
  /// \return Focal length in y-direction.
  [[nodiscard]] auto fy() const -> const Scalar& { return this->data()[kFocalOffsetY]; }

  /// Retrieves the y-component of the focal length.
  /// \return Focal length in y-direction.
  auto fy() -> ScalarWithConstIfNotLvalue& { return this->data()[kFocalOffsetY]; }

  /// Principal parameters accessor.
  /// \return Principal parameters.
  [[nodiscard]] auto principalParameters() const { return this->template head<kNumPrincipalParameters>(); }

  /// Principal parameters modifier.
  /// \return Principal parameters.
  auto principalParameters() { return this->template head<kNumPrincipalParameters>(); }

  /// Focal parameters accessor.
  /// \return Focal parameters.
  [[nodiscard]] auto focalParameters() const { return this->template tail<kNumFocalParameters>(); }

  /// Focal parameters modifier.
  /// \return Focal parameters.
  auto focalParameters() { return this->template tail<kNumFocalParameters>(); }

  /// Normalizes pixel coordinates on the image plane.
  /// \param input Non-normalized input pixel.
  /// \param J_i Input Jacobian.
  /// \param J_p Parameter Jacobian.
  /// \return Normalized pixel coordinates.
  auto normalize(const Eigen::Ref<const Input>& input, Scalar* J_i = nullptr, Scalar* J_p = nullptr) const -> Pixel {  // NOLINT
    const auto ifx = Scalar{1} / fx();
    const auto ify = Scalar{1} / fy();
    const auto dx = input.x() - cx();
    const auto dy = input.y() - cy();
    const auto dx_ifx = dx * ifx;
    const auto dy_ify = dy * ify;

    if (J_i) {
      auto J = Eigen::Map<InputJacobian>{J_i};
      J(0, 0) = ifx;
      J(1, 0) = Scalar{0};
      J(0, 1) = Scalar{0};
      J(1, 1) = ify;
    }

    if (J_p) {
      auto J = Eigen::Map<ParameterJacobian>{J_p};
      J(0, kPrincipalOffsetX) = Scalar{-1} * ifx;
      J(1, kPrincipalOffsetX) = Scalar{0};
      J(0, kPrincipalOffsetY) = Scalar{0};
      J(1, kPrincipalOffsetY) = Scalar{-1} * ify;
      J(0, kFocalOffsetX) = Scalar{-1} * dx_ifx * ifx;
      J(1, kFocalOffsetX) = Scalar{0};
      J(0, kFocalOffsetY) = Scalar{0};
      J(1, kFocalOffsetY) = Scalar{-1} * dy_ify * ify;
    }

    return {dx_ifx, dy_ify};
  }

  /// Denormalizes normalized pixel coordinates.
  /// \param input Normalized input pixel.
  /// \param J_i Input Jacobian.
  /// \param J_p Parameter Jacobian.
  /// \return Denormalized pixel coordinates.
  auto denormalize(const Eigen::Ref<const Input>& input, Scalar* J_i = nullptr, Scalar* J_p = nullptr) const -> Pixel {  // NOLINT
    if (J_i) {
      auto J = Eigen::Map<InputJacobian>{J_i};
      J(0, 0) = fx();
      J(1, 0) = Scalar{0};
      J(0, 1) = Scalar{0};
      J(1, 1) = fy();
    }

    if (J_p) {
      auto J = Eigen::Map<ParameterJacobian>{J_p};
      J(0, kPrincipalOffsetX) = Scalar{1};
      J(1, kPrincipalOffsetX) = Scalar{0};
      J(0, kPrincipalOffsetY) = Scalar{0};
      J(1, kPrincipalOffsetY) = Scalar{1};
      J(0, kFocalOffsetX) = input.x();
      J(1, kFocalOffsetX) = Scalar{0};
      J(0, kFocalOffsetY) = Scalar{0};
      J(1, kFocalOffsetY) = input.y();
    }

    return {cx() + fx() * input.x(), cy() + fy() * input.y()};
  }
};

template <typename TScalar>
class Intrinsics final : public IntrinsicsBase<Intrinsics<TScalar>> {
 public:
  using Base = IntrinsicsBase<Intrinsics>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Intrinsics)
};

}  // namespace hyper::variables

HYPER_DECLARE_EIGEN_INTERFACE(hyper::variables::Intrinsics)
