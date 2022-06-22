/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>
#include <Eigen/LU>

#include "variables/cartesian.hpp"

namespace hyper {

template <typename TDerived>
class BearingBase
    : public CartesianBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = CartesianBase<TDerived>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(BearingBase)

  /// Checks the norm.
  /// \return True if correct.
  [[nodiscard]] auto checkNorm() const -> bool {
    return Eigen::internal::isApprox(this->norm(), Traits<TDerived>::kNorm);
  }

  /// Finds an orthonormal basis where the first unit
  /// vector points along the direction of the bearing.
  /// \return Local frame.
  auto localFrame() -> typename Base::SquareMatrixType {
    // Definitions.
    using Matrix = typename Base::SquareMatrixType;

    // Find orthonormal basis.
    DCHECK(checkNorm());
    const auto& vx = this->x();
    const auto& vy = this->y();
    const auto& vz = this->z();

    const auto c0 = Scalar{1} / (Scalar{1} + std::abs(vx));
    const auto hy = -c0 * vy; // y-component of Householder vector.
    const auto hz = -c0 * vz; // z-component of Householder vector.

    Matrix matrix;
    matrix(0, 0) = vx;
    matrix(1, 0) = vy;
    matrix(2, 0) = vz;
    matrix(1, 1) = vy * hy + Scalar{1};
    matrix(2, 1) = vz * hy;

    if (vx > Scalar{0}) {
      const auto hx = vx + Scalar{1}; // x-component of Householder vector.
      matrix(0, 1) = hx * hy;
      matrix(0, 2) = hx * hz;
      matrix(1, 2) = vy * hz;
      matrix(2, 2) = vz * hz + Scalar{1};
    } else {
      const auto hx = vx - Scalar{1}; // x-component of Householder vector.
      matrix(0, 1) = hx * hy;
      matrix(0, 2) = -hx * hz;
      matrix(1, 2) = -vy * hz;
      matrix(2, 2) = -vz * hz - Scalar{1};
    }

    DCHECK(Eigen::internal::isApprox(matrix.determinant(), Scalar{1}));
    DCHECK((matrix * matrix.transpose()).isIdentity());
    return matrix;
  }
};

template <typename TScalar>
class Bearing final
    : public BearingBase<Bearing<TScalar>> {
 public:
  using Base = BearingBase<Bearing<TScalar>>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Bearing)

  /// Deleted default constructor.
  Bearing() = delete;

  /// Forwarding constructor with norm check.
  /// \tparam TArgs_ Input argument types.
  /// \param args Inputs arguments.
  template <typename... TArgs_>
  Bearing(TArgs_&&... args) // NOLINT
      : Base{std::forward<TArgs_>(args)...} {
    DCHECK(this->checkNorm());
  }
};

} // namespace hyper

HYPER_DECLARE_EIGEN_INTERFACE(Bearing)
