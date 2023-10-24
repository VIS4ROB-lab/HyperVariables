/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "hyper/stochastics/forward.hpp"

#include "hyper/variables/rn.hpp"

namespace hyper::stochastics {

namespace internal {

template <int TSize, typename TDerived>
auto invertPSDMatrix(const Eigen::MatrixBase<TDerived>& matrix, const bool full_rank = true) -> Eigen::Matrix<Scalar, TSize, TSize> {
  using Matrix = Eigen::Matrix<Scalar, TSize, TSize>;
  using SVDMatrix = Eigen::Matrix<Scalar, TSize, Eigen::Dynamic>;

  const auto size = matrix.rows();

  if (full_rank) {
    if (size > 0 && size < 5) {
      return matrix.inverse();
    }
    return matrix.template selfadjointView<Eigen::Lower>().llt().solve(Matrix::Identity(size, size));
  }

  Eigen::JacobiSVD<SVDMatrix> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.solve(SVDMatrix::Identity(size, size));
}

}  // namespace internal

template <int TOrder>
class Gaussian final {
 public:
  // Definitions.
  using Mu = hyper::Vector<TOrder>;
  using Sigma = hyper::Matrix<TOrder>;

  /// Constructor from order.
  /// \param order Order.
  explicit Gaussian(int order = TOrder) : matrix_{order, order + 1} {}

  /// Constructor from mu and sigma.
  /// \param mu Mu.
  /// \param sigma Sigma.
  template <typename TDerived_, typename TOtherDerived_>
  Gaussian(const Eigen::MatrixBase<TDerived_>& mu, const Eigen::MatrixBase<TOtherDerived_>& sigma) : Gaussian{mu.size()} {
    this->mu() = mu;
    this->sigma() = sigma;
  }

  /// Order accessor.
  /// \return Order.
  [[nodiscard]] inline auto order() const { return matrix_.rows(); }

  /// Mu accessor.
  /// \return Mu.
  inline auto mu() const { return matrix_.template leftCols<1>(); }

  /// Mu setter.
  /// \tparam TDerived_ Derived type.
  /// \param mu Mu.
  inline auto mu() { return matrix_.template leftCols<1>(); }

  /// Sigma accessor.
  /// \return Sigma.
  inline auto sigma() const {
    if constexpr (TOrder != Eigen::Dynamic) {
      return matrix_.template rightCols<TOrder>();
    } else {
      return matrix_.rightCols(order());
    }
  }

  /// Sigma setter.
  /// \tparam TDerived_ Derived type.
  /// \param sigma Sigma.
  inline auto sigma() {
    if constexpr (TOrder != Eigen::Dynamic) {
      return matrix_.template rightCols<TOrder>();
    } else {
      return matrix_.rightCols(order());
    }
  }

  /// Inverse sigma.
  /// \return Inverse sigma.
  inline auto sigmaInverse() const { return internal::invertPSDMatrix<TOrder>(sigma()); }

  /// Converts this.
  /// \param inverse_gaussian Inverse Gaussian.
  auto toInverseGaussian(InverseGaussian<TOrder>& inverse_gaussian) const -> void;

  /// Converts this.
  /// \return Inverse Gaussian.
  auto toInverseGaussian() const -> InverseGaussian<TOrder>;

 private:
  // Definitions.
  using Matrix = hyper::Traits<Gaussian<TOrder>>::Base;

  Matrix matrix_;  ///< Matrix.
};

template <int TOrder>
class InverseGaussian final {
 public:
  // Definitions.
  using Eta = hyper::Vector<TOrder>;
  using Lambda = hyper::Matrix<TOrder>;

  /// Zero inverse Gaussian.
  /// \return Inverse Gaussian.
  static auto Zero(int size) -> InverseGaussian {
    InverseGaussian inverse_gaussian{size};
    inverse_gaussian.setZero();
    return inverse_gaussian;
  }

  /// Constructor from order.
  /// \param order Order.
  explicit InverseGaussian(int order = TOrder) : matrix_{order, order + 1} {}

  /// Constructor from eta and lambda.
  /// \param eta Eta.
  /// \param sigma Lambda.
  template <typename TDerived_, typename TOtherDerived_>
  InverseGaussian(const Eigen::MatrixBase<TDerived_>& eta, const Eigen::MatrixBase<TOtherDerived_>& lambda) : InverseGaussian{eta.size()} {
    this->eta() = eta;
    this->lambda() = lambda;
  }

  /// Sets this to zero.
  /// \return Inverse Gaussian.
  inline auto setZero() -> InverseGaussian& {
    const auto order = this->order();
    matrix_.setZero(order, order + 1);
    return *this;
  }

  /// Order accessor.
  /// \return Order.
  [[nodiscard]] inline auto order() const { return matrix_.rows(); }

  /// Eta accessor.
  /// \return Eta.
  inline auto eta() const { return matrix_.template leftCols<1>(); }

  /// Eta setter.
  /// \tparam TDerived_ Derived type.
  /// \param eta Eta.
  inline auto eta() { return matrix_.template leftCols<1>(); }

  /// Lambda accessor.
  /// \return Lambda.
  inline auto lambda() const {
    if constexpr (TOrder != Eigen::Dynamic) {
      return matrix_.template rightCols<TOrder>();
    } else {
      return matrix_.rightCols(order());
    }
  }

  /// Lambda setter.
  /// \tparam TDerived_ Derived type.
  /// \param lambda Lambda.
  inline auto lambda() {
    if constexpr (TOrder != Eigen::Dynamic) {
      return matrix_.template rightCols<TOrder>();
    } else {
      return matrix_.rightCols(order());
    }
  }

  /// Inverse lambda.
  /// \return Inverse lambda.
  inline auto lambdaInverse() const { return internal::invertPSDMatrix<TOrder>(lambda()); }

  /// Plus operator.
  /// \param other Other inverse Gaussian.
  /// \return Inverse Gaussian.
  inline auto operator+(const InverseGaussian& other) const -> InverseGaussian { return {matrix_ + other.matrix_}; }

  /// Plus operator (in-place).
  /// \param other Other Gaussian.
  /// \return Gaussian.
  inline auto operator+=(const InverseGaussian& other) -> InverseGaussian& {
    matrix_ += other.matrix_;
    return *this;
  }

  /// Product operator.
  /// \param scalar Scalar.
  /// \return Inverse Gaussian.
  inline auto operator*(const Scalar scalar) const -> InverseGaussian { return {scalar * matrix_}; }

  /// Product operator (in-place).
  /// \param scalar Scalar.
  /// \return Gaussian.
  inline auto operator*=(const Scalar scalar) -> InverseGaussian& {
    matrix_ *= scalar;
    return *this;
  }

  /// Converts this.
  /// \param gaussian Gaussian.
  auto toGaussian(Gaussian<TOrder>& gaussian) const -> void;

  /// Converts this.
  /// \return Gaussian.
  auto toGaussian() const -> Gaussian<TOrder>;

 private:
  // Definitions.
  using Matrix = hyper::Traits<InverseGaussian<TOrder>>::Base;

  /// Constructor from matrix.
  /// \tparam TDerived_ Derived type.
  /// \param matrix Matrix.
  template <typename TDerived_>
  InverseGaussian(const Eigen::MatrixBase<TDerived_>& matrix) : InverseGaussian{matrix.rows()} {
    matrix_ = matrix;
  }

  Matrix matrix_;  ///< Matrix.
};

template <int TOrder>
auto operator*(const Scalar scalar, const InverseGaussian<TOrder>& inverseGaussian) -> InverseGaussian<TOrder> {
  return inverseGaussian * scalar;
}

template <int TOrder>
auto operator*(const Scalar scalar, InverseGaussian<TOrder>& inverseGaussian) -> InverseGaussian<TOrder>& {
  return inverseGaussian *= scalar;
}

template <int TOrder>
auto Gaussian<TOrder>::toInverseGaussian(InverseGaussian<TOrder>& inverse_gaussian) const -> void {
  inverse_gaussian.lambda().noalias() = sigmaInverse();
  inverse_gaussian.eta().noalias() = inverse_gaussian.lambda() * mu();
}

template <int TOrder>
auto Gaussian<TOrder>::toInverseGaussian() const -> InverseGaussian<TOrder> {
  InverseGaussian<TOrder> inverse_gaussian{order()};
  toInverseGaussian(inverse_gaussian);
  return inverse_gaussian;
}

template <int TOrder>
auto InverseGaussian<TOrder>::toGaussian(Gaussian<TOrder>& gaussian) const -> void {
  gaussian.sigma().noalias() = lambdaInverse();
  gaussian.mu().noalias() = gaussian.sigma() * eta();
}

template <int TOrder>
auto InverseGaussian<TOrder>::toGaussian() const -> Gaussian<TOrder> {
  Gaussian<TOrder> gaussian{order()};
  toGaussian(gaussian);
  return gaussian;
}

}  // namespace hyper::stochastics
