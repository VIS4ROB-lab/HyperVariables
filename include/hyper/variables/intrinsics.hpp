/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/cartesian.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper {

template <typename TDerived>
class IntrinsicsBase
    : public CartesianBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = CartesianBase<TDerived>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(IntrinsicsBase)

  /// Retrieves the x-component of the principal point.
  /// \return Principal point in x-direction.
  [[nodiscard]] auto cx() const -> const Scalar& {
    return this->data()[Traits<Intrinsics<Scalar>>::kPrincipalOffsetX];
  }

  /// Retrieves the x-component of the principal point.
  /// \return Principal point in x-direction.
  auto cx() -> ScalarWithConstIfNotLvalue& {
    return this->data()[Traits<Intrinsics<Scalar>>::kPrincipalOffsetX];
  }

  /// Retrieves the y-component of the principal point.
  /// \return Principal point in y-direction.
  [[nodiscard]] auto cy() const -> const Scalar& {
    return this->data()[Traits<Intrinsics<Scalar>>::kPrincipalOffsetY];
  }

  /// Retrieves the y-component of the principal point.
  /// \return Principal point in y-direction.
  auto cy() -> ScalarWithConstIfNotLvalue& {
    return this->data()[Traits<Intrinsics<Scalar>>::kPrincipalOffsetY];
  }

  /// Retrieves the x-component of the focal length.
  /// \return Focal length in x-direction.
  [[nodiscard]] auto fx() const -> const Scalar& {
    return this->data()[Traits<Intrinsics<Scalar>>::kFocalOffsetX];
  }

  /// Retrieves the x-component of the focal length.
  /// \return Focal length in x-direction.
  auto fx() -> ScalarWithConstIfNotLvalue& {
    return this->data()[Traits<Intrinsics<Scalar>>::kFocalOffsetX];
  }

  /// Retrieves the y-component of the focal length.
  /// \return Focal length in y-direction.
  [[nodiscard]] auto fy() const -> const Scalar& {
    return this->data()[Traits<Intrinsics<Scalar>>::kFocalOffsetY];
  }

  /// Retrieves the y-component of the focal length.
  /// \return Focal length in y-direction.
  auto fy() -> ScalarWithConstIfNotLvalue& {
    return this->data()[Traits<Intrinsics<Scalar>>::kFocalOffsetY];
  }

  /// Principal parameters accessor.
  /// \return Principal parameters.
  [[nodiscard]] auto principalParameters() const {
    return this->template head<Traits<Intrinsics<Scalar>>::kNumPrincipalParameters>();
  }

  /// Principal parameters modifier.
  /// \return Principal parameters.
  auto principalParameters() {
    return this->template head<Traits<Intrinsics<Scalar>>::kNumPrincipalParameters>();
  }

  /// Focal parameters accessor.
  /// \return Focal parameters.
  [[nodiscard]] auto focalParameters() const {
    return this->template tail<Traits<Intrinsics<Scalar>>::kNumFocalParameters>();
  }

  /// Focal parameters modifier.
  /// \return Focal parameters.
  auto focalParameters() {
    return this->template tail<Traits<Intrinsics<Scalar>>::kNumFocalParameters>();
  }

  /// Normalizes pixel coordinates on the image plane.
  /// \param input Non-normalized input pixel.
  /// \param raw_J_p_p Pointer to pixel to pixel Jacobian.
  /// \param raw_J_p_i Pointer to pixel to intrinsics Jacobian.
  /// \return Normalized pixel coordinates.
  auto normalize(const Eigen::Ref<const typename Traits<Pixel<Scalar>>::Base>& input, Scalar* raw_J_p_p = nullptr, Scalar* raw_J_p_i = nullptr) const -> Pixel<Scalar> { // NOLINT
    const auto ifx = Scalar{1} / fx();
    const auto ify = Scalar{1} / fy();
    const auto dx = input.x() - cx();
    const auto dy = input.y() - cy();
    const auto dx_ifx = dx * ifx;
    const auto dy_ify = dy * ify;

    if (raw_J_p_p) {
      auto J = Eigen::Map<TJacobianNM<Pixel<Scalar>>>{raw_J_p_p};
      J(0, 0) = ifx;
      J(1, 0) = Scalar{0};
      J(0, 1) = Scalar{0};
      J(1, 1) = ify;
    }

    if (raw_J_p_i) {
      using Traits = Traits<Intrinsics<Scalar>>;
      auto J = Eigen::Map<TJacobianNM<Pixel<Scalar>, Intrinsics<Scalar>>>{raw_J_p_i};
      J(0, Traits::kPrincipalOffsetX) = Scalar{-1} * ifx;
      J(1, Traits::kPrincipalOffsetX) = Scalar{0};
      J(0, Traits::kPrincipalOffsetY) = Scalar{0};
      J(1, Traits::kPrincipalOffsetY) = Scalar{-1} * ify;
      J(0, Traits::kFocalOffsetX) = Scalar{-1} * dx_ifx * ifx;
      J(1, Traits::kFocalOffsetX) = Scalar{0};
      J(0, Traits::kFocalOffsetY) = Scalar{0};
      J(1, Traits::kFocalOffsetY) = Scalar{-1} * dy_ify * ify;
    }

    return {dx_ifx, dy_ify};
  }

  /// Denormalizes normalized pixel coordinates.
  /// \param input Normalized input pixel.
  /// \param raw_J_p_p Pointer to pixel to pixel Jacobian.
  /// \param raw_J_p_i Pointer to pixel to intrinsics Jacobian.
  /// \return Denormalized pixel coordinates.
  auto denormalize(const Eigen::Ref<const typename Traits<Pixel<Scalar>>::Base>& input, Scalar* raw_J_p_p = nullptr, Scalar* raw_J_p_i = nullptr) const -> Pixel<Scalar> { // NOLINT
    if (raw_J_p_p) {
      auto J = Eigen::Map<TJacobianNM<Pixel<Scalar>>>{raw_J_p_p};
      J(0, 0) = fx();
      J(1, 0) = Scalar{0};
      J(0, 1) = Scalar{0};
      J(1, 1) = fy();
    }

    if (raw_J_p_i) {
      using Traits = Traits<Intrinsics<Scalar>>;
      auto J = Eigen::Map<TJacobianNM<Pixel<Scalar>, Intrinsics<Scalar>>>{raw_J_p_i};
      J(0, Traits::kPrincipalOffsetX) = Scalar{1};
      J(1, Traits::kPrincipalOffsetX) = Scalar{0};
      J(0, Traits::kPrincipalOffsetY) = Scalar{0};
      J(1, Traits::kPrincipalOffsetY) = Scalar{1};
      J(0, Traits::kFocalOffsetX) = input.x();
      J(1, Traits::kFocalOffsetX) = Scalar{0};
      J(0, Traits::kFocalOffsetY) = Scalar{0};
      J(1, Traits::kFocalOffsetY) = input.y();
    }

    return {cx() + fx() * input.x(), cy() + fy() * input.y()};
  }
};

template <typename TScalar>
class Intrinsics final
    : public IntrinsicsBase<Intrinsics<TScalar>> {
 public:
  using Base = IntrinsicsBase<Intrinsics>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Intrinsics)
};

} // namespace hyper

HYPER_DECLARE_EIGEN_INTERFACE(hyper::Intrinsics)
