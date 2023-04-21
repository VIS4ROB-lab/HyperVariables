/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Cholesky>

#include "hyper/variables/stochastic/forward.hpp"

#include "hyper/variables/rn.hpp"

namespace hyper::variables {

template <typename TScalar, int TOrder>
class Gaussian<TScalar, TOrder, GaussianType::STANDARD> final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using Mu = Eigen::Matrix<TScalar, TOrder, 1>;
  using Sigma = Eigen::Matrix<TScalar, TOrder, TOrder>;
  using Matrix = Eigen::Matrix<TScalar, TOrder, (TOrder != Eigen::Dynamic) ? (TOrder + 1) : Eigen::Dynamic>;

  /// Zero Gaussian.
  /// \return Gaussian.
  static auto Zero(Index size) -> Gaussian {
    auto gaussian = Gaussian{size};
    gaussian.setZero(size);
    return gaussian;
  }

  /// Constructor from order.
  /// \param order Input order.
  explicit Gaussian(Index order = TOrder) : matrix_{order, order + 1} {}

  /// Constructor from mu and sigma.
  /// \param mu Mu.
  /// \param sigma Sigma.
  template <typename TDerived_, typename TOtherDerived_>
  Gaussian(const Eigen::MatrixBase<TDerived_>& mu, const Eigen::MatrixBase<TOtherDerived_>& sigma) : Gaussian{mu.size()} {
    setMu(mu);
    setSigma(sigma);
  }

  /// Constructor from order.
  /// \param order Input order.
  /// \return Gaussian.
  inline auto setZero(Index order) -> Gaussian& {
    matrix_.setZero(order, order + 1);
    return *this;
  }

  /// Retrieves the order.
  /// \return Order.
  [[nodiscard]] inline auto order() const { return matrix_.rows(); }

  /// Mu accessor.
  /// \return Mu.
  inline auto mu() const { return matrix_.template leftCols<1>(); }

  /// Mu setter.
  /// \tparam TDerived_ Derived type.
  /// \param mu Mu.
  template <typename TDerived_>
  inline auto setMu(const Eigen::MatrixBase<TDerived_>& mu) -> void {
    matrix_.template leftCols<1>() = mu;
  }

  /// Sigma accessor.
  /// \return Sigma.
  inline auto sigma() const {
    if constexpr (TOrder != Eigen::Dynamic) {
      return matrix_.template rightCols<TOrder>().template selfadjointView<Eigen::Lower>();
    } else {
      return matrix_.rightCols(order()).template selfadjointView<Eigen::Lower>();
    }
  }

  /// Sigma setter.
  /// \tparam TDerived_ Derived type.
  /// \param sigma Sigma.
  template <typename TDerived_>
  inline auto setSigma(const Eigen::MatrixBase<TDerived_>& sigma) -> void {
    if constexpr (TOrder != Eigen::Dynamic) {
      matrix_.template rightCols<TOrder>().template triangularView<Eigen::Lower>() = sigma;
    } else {
      matrix_.rightCols(order()).template triangularView<Eigen::Lower>() = sigma;
    }
  }

  /// Matrix accessor.
  /// \return Matrix
  auto matrix() const -> const Matrix& { return matrix_; }

  /// Matrix setter.
  /// \tparam TDerived_ Derived type.
  /// \param matrix Matrix.
  template <typename TDerived_>
  auto setMatrix(const Eigen::MatrixBase<TDerived_>& matrix) -> void {
    matrix_ = matrix;
  }

  /// Inverse sigma.
  /// \return Inverse sigma.
  inline auto sigmaInverse() const { return sigma().llt().solve(Sigma::Identity(order(), order())); }

  /// Converts this.
  /// \param canonical_gaussian Canonical Gaussian to write to.
  auto toCanonicalForm(Gaussian<TScalar, TOrder, GaussianType::CANONICAL>& canonical_gaussian) const -> void;

  /// Converts this.
  /// \return Canonical form.
  auto toCanonicalForm() const -> Gaussian<TScalar, TOrder, GaussianType::CANONICAL>;

 private:
  Matrix matrix_;  ///< Matrix.
};

template <typename TScalar, int TOrder>
class Gaussian<TScalar, TOrder, GaussianType::CANONICAL> final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using Eta = Eigen::Matrix<TScalar, TOrder, 1>;
  using Lambda = Eigen::Matrix<TScalar, TOrder, TOrder>;
  using Matrix = Eigen::Matrix<TScalar, TOrder, (TOrder != Eigen::Dynamic) ? (TOrder + 1) : Eigen::Dynamic>;

  /// Zero Gaussian.
  /// \return Gaussian.
  static auto Zero(Index size) -> Gaussian {
    auto gaussian = Gaussian{size};
    gaussian.setZero(size);
    return gaussian;
  }

  /// Constructor from order.
  /// \param order Input order.
  explicit Gaussian(Index order = TOrder) : matrix_{order, order + 1} {}

  /// Constructor from eta and lambda.
  /// \param eta Eta.
  /// \param sigma Lambda.
  template <typename TDerived_, typename TOtherDerived_>
  Gaussian(const Eigen::MatrixBase<TDerived_>& eta, const Eigen::MatrixBase<TOtherDerived_>& lambda) : Gaussian{eta.size()} {
    setEta(eta);
    setLambda(lambda);
  }

  /// Constructor from order.
  /// \param order Input order.
  /// \return Gaussian.
  inline auto setZero(Index order) -> Gaussian& {
    matrix_.setZero(order, order + 1);
    return *this;
  }

  /// Retrieves the order.
  /// \return Order.
  [[nodiscard]] inline auto order() const { return matrix_.rows(); }

  /// Eta accessor.
  /// \return Eta.
  inline auto eta() const { return matrix_.template leftCols<1>(); }

  /// Eta setter.
  /// \tparam TDerived_ Derived type.
  /// \param eta Eta.
  template <typename TDerived_>
  inline auto setEta(const Eigen::MatrixBase<TDerived_>& eta) -> void {
    matrix_.template leftCols<1>() = eta;
  }

  /// Lambda accessor.
  /// \return Lambda.
  inline auto lambda() const {
    if constexpr (TOrder != Eigen::Dynamic) {
      return matrix_.template rightCols<TOrder>().template selfadjointView<Eigen::Lower>();
    } else {
      return matrix_.rightCols(order()).template selfadjointView<Eigen::Lower>();
    }
  }

  /// Lambda setter.
  /// \tparam TDerived_ Derived type.
  /// \param lambda Lambda.
  template <typename TDerived_>
  inline auto setLambda(const Eigen::MatrixBase<TDerived_>& lambda) -> void {
    if constexpr (TOrder != Eigen::Dynamic) {
      matrix_.template rightCols<TOrder>().template triangularView<Eigen::Lower>() = lambda;
    } else {
      matrix_.rightCols(order()).template triangularView<Eigen::Lower>() = lambda;
    }
  }

  /// Inverse lambda.
  /// \return Inverse lambda.
  inline auto lambdaInverse() const { return lambda().llt().solve(Lambda::Identity(order(), order())); }

  /// Matrix accessor.
  /// \return Matrix
  auto matrix() const -> const Matrix& { return matrix_; }

  /// Matrix setter.
  /// \tparam TDerived_ Derived type.
  /// \param matrix Matrix.
  template <typename TDerived_>
  auto setMatrix(const Eigen::MatrixBase<TDerived_>& matrix) -> void {
    matrix_ = matrix;
  }

  /// Converts this.
  /// \param standard_gaussian Standard Gaussian to write to.
  auto toStandardForm(Gaussian<TScalar, TOrder, GaussianType::STANDARD>& standard_gaussian) const -> void;

  /// Converts this.
  /// \return Standard form.
  auto toStandardForm() const -> Gaussian<TScalar, TOrder, GaussianType::STANDARD>;

 private:
  Matrix matrix_;  ///< Matrix.
};

template <typename TScalar, int TOrder>
class DualGaussian final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using StandardGaussian = Gaussian<TScalar, TOrder, GaussianType::STANDARD>;
  using CanonicalGaussian = Gaussian<TScalar, TOrder, GaussianType::CANONICAL>;

  /// Constructor from order.
  /// \param order Input order.
  explicit DualGaussian(Index order = TOrder) : standard_gaussian_{order}, canonical_gaussian_{order} {}

  /// Standard form accessor.
  /// \return Gaussian in standard form.
  inline auto standardForm() const -> const StandardGaussian& { return standard_gaussian_; }

  /// Canonical form accessor.
  /// \return Gaussian in canonical form.
  inline auto canonicalForm() const -> const CanonicalGaussian& { return canonical_gaussian_; }

  /// Sets the standard Gaussian form.
  /// \param standard_gaussian Standard Gaussian.
  auto setStandardForm(const StandardGaussian& standard_gaussian) -> void {
    standard_gaussian.toCanonicalForm(canonical_gaussian_);
    standard_gaussian_ = standard_gaussian;
  }

  /// Sets the canonical Gaussian form.
  /// \param canonical_gaussian Canonical Gaussian.
  auto setCanonicalForm(const CanonicalGaussian& canonical_gaussian) -> void {
    canonical_gaussian.toStandardForm(standard_gaussian_);
    canonical_gaussian_ = canonical_gaussian;
  }

 private:
  StandardGaussian standard_gaussian_;
  CanonicalGaussian canonical_gaussian_;
};

template <typename TScalar, int TOrder>
auto Gaussian<TScalar, TOrder, GaussianType::STANDARD>::toCanonicalForm(Gaussian<TScalar, TOrder, GaussianType::CANONICAL>& canonical_gaussian) const -> void {
  canonical_gaussian.setLambda(sigmaInverse());
  canonical_gaussian.setEta(canonical_gaussian.lambda() * mu());
}

template <typename TScalar, int TOrder>
auto Gaussian<TScalar, TOrder, GaussianType::STANDARD>::toCanonicalForm() const -> Gaussian<TScalar, TOrder, GaussianType::CANONICAL> {
  using CanonicalGaussian = Gaussian<TScalar, TOrder, GaussianType::CANONICAL>;
  CanonicalGaussian canonical_gaussian{order()};
  toCanonicalForm(canonical_gaussian);
  return canonical_gaussian;
}

template <typename TScalar, int TOrder>
auto Gaussian<TScalar, TOrder, GaussianType::CANONICAL>::toStandardForm(Gaussian<TScalar, TOrder, GaussianType::STANDARD>& standard_gaussian) const -> void {
  standard_gaussian.setSigma(lambdaInverse());
  standard_gaussian.setMu(standard_gaussian.sigma() * eta());
}

template <typename TScalar, int TOrder>
auto Gaussian<TScalar, TOrder, GaussianType::CANONICAL>::toStandardForm() const -> Gaussian<TScalar, TOrder, GaussianType::STANDARD> {
  using StandardGaussian = Gaussian<TScalar, TOrder, GaussianType::STANDARD>;
  StandardGaussian standard_gaussian{order()};
  toStandardForm(standard_gaussian);
  return standard_gaussian;
}

}  // namespace hyper::variables
