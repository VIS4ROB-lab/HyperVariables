/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "hyper/stochastics/forward.hpp"

#include "hyper/variables/rn.hpp"

namespace hyper::stochastics {

namespace gaussian {

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

}  // namespace gaussian

template <int TOrder>
class Gaussian<TOrder, GaussianType::STANDARD> final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using Mu = Vector<TOrder>;
  using Sigma = Matrix<TOrder, TOrder>;
  using Matrix = Matrix<TOrder, (TOrder != Eigen::Dynamic) ? (TOrder + 1) : Eigen::Dynamic>;

  /// Zero Gaussian.
  /// \return Gaussian.
  static auto Zero(Index size) -> Gaussian {
    Gaussian gaussian{size};
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
    this->mu() = mu;
    this->sigma() = sigma;
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

  /// Plus operator.
  /// \param other Other Gaussian.
  /// \return Gaussian.
  auto operator+(const Gaussian& other) -> Gaussian { return {mu() + other.mu(), sigma() + other.sigma()}; }

  /// Minus operator.
  /// \param other Other Gaussian.
  /// \return Gaussian.
  auto operator-(const Gaussian& other) -> Gaussian { return {mu() - other.mu(), sigma() - other.sigma()}; }

  /// Plus operator (in-place).
  /// \param other Other Gaussian.
  /// \return Gaussian.
  auto operator+=(const Gaussian& other) -> Gaussian& {
    mu() += other.mu();
    sigma() += other.sigma();
    return *this;
  }

  /// Minus operator (in-place).
  /// \param other Other Gaussian.
  /// \return Gaussian.
  auto operator-=(const Gaussian& other) -> Gaussian& {
    mu() -= other.mu();
    sigma() -= other.sigma();
    return *this;
  }

  /// Inverse sigma.
  /// \return Inverse sigma.
  inline auto sigmaInverse() const { return gaussian::invertPSDMatrix<TOrder>(sigma()); }

  /// Converts this.
  /// \param information_gaussian Information Gaussian to write to.
  auto toInformationGaussian(Gaussian<TOrder, GaussianType::INFORMATION>& information_gaussian) const -> void;

  /// Converts this.
  /// \return Information form.
  auto toInformationGaussian() const -> Gaussian<TOrder, GaussianType::INFORMATION>;

 private:
  Matrix matrix_;  ///< Matrix.
};

template <int TOrder>
class Gaussian<TOrder, GaussianType::INFORMATION> final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using Eta = Vector<TOrder>;
  using Lambda = Matrix<TOrder, TOrder>;
  using Matrix = Matrix<TOrder, (TOrder != Eigen::Dynamic) ? (TOrder + 1) : Eigen::Dynamic>;

  /// Zero Gaussian.
  /// \return Gaussian.
  static auto Zero(Index size) -> Gaussian {
    Gaussian gaussian{size};
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
    this->eta() = eta;
    this->lambda() = lambda;
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
  inline auto lambdaInverse() const { return gaussian::invertPSDMatrix<TOrder>(lambda()); }

  /// Plus operator.
  /// \param other Other Gaussian.
  /// \return Gaussian.
  auto operator+(const Gaussian& other) -> Gaussian { return {eta() + other.eta(), lambda() + other.lambda()}; }

  /// Minus operator.
  /// \param other Other Gaussian.
  /// \return Gaussian.
  auto operator-(const Gaussian& other) -> Gaussian { return {eta() - other.eta(), lambda() - other.lambda()}; }

  /// Plus operator (in-place).
  /// \param other Other Gaussian.
  /// \return Gaussian.
  auto operator+=(const Gaussian& other) -> Gaussian& {
    eta() += other.eta();
    lambda() += other.lambda();
    return *this;
  }

  /// Minus operator (in-place).
  /// \param other Other Gaussian.
  /// \return Gaussian.
  auto operator-=(const Gaussian& other) -> Gaussian& {
    eta() -= other.eta();
    lambda() -= other.lambda();
    return *this;
  }

  /// Converts this.
  /// \param standard_gaussian Standard Gaussian to write to.
  auto toStandardGaussian(Gaussian<TOrder, GaussianType::STANDARD>& standard_gaussian) const -> void;

  /// Converts this.
  /// \return Standard form.
  auto toStandardGaussian() const -> Gaussian<TOrder, GaussianType::STANDARD>;

 private:
  Matrix matrix_;  ///< Matrix.
};

template <int TOrder>
class DualGaussian final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using StandardGaussian = Gaussian<TOrder, GaussianType::STANDARD>;
  using InformationGaussian = Gaussian<TOrder, GaussianType::INFORMATION>;

  using Mu = typename StandardGaussian::Mu;
  using Sigma = typename StandardGaussian::Sigma;
  using Eta = typename InformationGaussian::Eta;
  using Lambda = typename InformationGaussian::Lambda;

  /// Initialization as identity.
  /// \param order Order.
  /// \return Uncertainty.
  static auto Identity(Index order = TOrder) -> DualGaussian {
    DualGaussian dual_gaussian{order};
    dual_gaussian.standard_gaussian_ = {Mu::Zero(order), Sigma::Identity(order, order)};
    dual_gaussian.information_gaussian_ = {Eta::Zero(order), Lambda::Identity(order, order)};
    return dual_gaussian;
  }

  /// Constructor from order.
  /// \param order Input order.
  explicit DualGaussian(Index order = TOrder) : standard_gaussian_{order}, information_gaussian_{order} {}

  /// Constructor from order.
  /// \param order Input order.
  /// \return Gaussian.
  inline auto setIdentity(Index order = TOrder) -> DualGaussian& {
    *this = Identity(order);
    return *this;
  }

  /// Retrieves the order.
  /// \return Order.
  [[nodiscard]] inline auto order() const { return standard_gaussian_.order(); }

  /// Standard form accessor.
  /// \return Gaussian in standard form.
  inline auto standardGaussian() const -> const StandardGaussian& { return standard_gaussian_; }

  /// Information form accessor.
  /// \return Gaussian in information form.
  inline auto informationGaussian() const -> const InformationGaussian& { return information_gaussian_; }

  /// Mu setter.
  /// \tparam TDerived_ Derived type.
  /// \param mu Mu.
  template <typename TDerived_>
  inline auto setMu(const Eigen::MatrixBase<TDerived_>& mu) -> void {
    standard_gaussian_.mu() = mu;
    information_gaussian_.eta() = information_gaussian_.lambda() * mu;
  }

  /// Sigma setter.
  /// \tparam TDerived_ Derived type.
  /// \param sigma Sigma.
  template <typename TDerived_>
  inline auto setSigma(const Eigen::MatrixBase<TDerived_>& sigma) -> void {
    standard_gaussian_.sigma() = sigma;
    standard_gaussian_.toInformationGaussian(information_gaussian_);
  }

  /// Eta setter.
  /// \tparam TDerived_ Derived type.
  /// \param eta Eta.
  template <typename TDerived_>
  inline auto setEta(const Eigen::MatrixBase<TDerived_>& eta) -> void {
    information_gaussian_.eta() = eta;
    standard_gaussian_.mu() = standard_gaussian_.sigma() * eta;
  }

  /// Lambda setter.
  /// \tparam TDerived_ Derived type.
  /// \param lambda Lambda.
  template <typename TDerived_>
  inline auto setLambda(const Eigen::MatrixBase<TDerived_>& lambda) -> void {
    information_gaussian_.lambda() = lambda;
    information_gaussian_.toStandardGaussian(standard_gaussian_);
  }

  /// Sets the standard Gaussian form.
  /// \param standard_gaussian Standard Gaussian.
  auto setStandardGaussian(const StandardGaussian& standard_gaussian) -> void {
    standard_gaussian.toInformationGaussian(information_gaussian_);
    standard_gaussian_ = standard_gaussian;
  }

  /// Sets the information Gaussian form.
  /// \param information_gaussian Information Gaussian.
  auto setInformationGaussian(const InformationGaussian& information_gaussian) -> void {
    information_gaussian.toStandardGaussian(standard_gaussian_);
    information_gaussian_ = information_gaussian;
  }

 private:
  StandardGaussian standard_gaussian_;
  InformationGaussian information_gaussian_;
};

template <int TOrder>
auto Gaussian<TOrder, GaussianType::STANDARD>::toInformationGaussian(Gaussian<TOrder, GaussianType::INFORMATION>& information_gaussian) const -> void {
  information_gaussian.lambda().noalias() = sigmaInverse();
  information_gaussian.eta().noalias() = information_gaussian.lambda() * mu();
}

template <int TOrder>
auto Gaussian<TOrder, GaussianType::STANDARD>::toInformationGaussian() const -> Gaussian<TOrder, GaussianType::INFORMATION> {
  using InformationGaussian = Gaussian<TOrder, GaussianType::INFORMATION>;
  InformationGaussian information_gaussian{order()};
  toInformationGaussian(information_gaussian);
  return information_gaussian;
}

template <int TOrder>
auto Gaussian<TOrder, GaussianType::INFORMATION>::toStandardGaussian(Gaussian<TOrder, GaussianType::STANDARD>& standard_gaussian) const -> void {
  standard_gaussian.sigma().noalias() = lambdaInverse();
  standard_gaussian.mu().noalias() = standard_gaussian.sigma() * eta();
}

template <int TOrder>
auto Gaussian<TOrder, GaussianType::INFORMATION>::toStandardGaussian() const -> Gaussian<TOrder, GaussianType::STANDARD> {
  using StandardGaussian = Gaussian<TOrder, GaussianType::STANDARD>;
  StandardGaussian standard_gaussian{order()};
  toStandardGaussian(standard_gaussian);
  return standard_gaussian;
}

}  // namespace hyper::stochastics
