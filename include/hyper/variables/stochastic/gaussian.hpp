/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <Eigen/Cholesky>

#include "hyper/variables/stochastic/forward.hpp"

#include "hyper/variables/cartesian.hpp"

namespace hyper {

template <typename TDerived>
class GaussianBase : public Traits<TDerived>::Base {
 public:
  // Definitions.
  using Base = Traits<TDerived>::Base;
  using Index = typename Base::Index;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  // Constants.
  static constexpr auto kOrder = Base::RowsAtCompileTime;
  static constexpr auto kMuColOffset = 0;
  static constexpr auto kNumMuCols = 1;
  static constexpr auto kSigmaColOffset = 1;
  static constexpr auto kNumSigmaCols = kOrder;

  using Mu = Eigen::Matrix<Scalar, kOrder, 1>;
  using Sigma = Eigen::Matrix<Scalar, kOrder, kOrder>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(GaussianBase)

  /// Retrieves the order.
  /// \return Order.
  [[nodiscard]] auto order() const -> Index {
    return this->rows();
  }

  /// Mean accessor.
  /// \return Mean.
  inline auto mu() const { return this->template leftCols<1>(); }

  /// Mean modifier.
  /// \return Mean.
  inline auto mu() { return this->template leftCols<1>(); }

  /// Covariance accessor.
  /// \return Covariance.
  inline auto sigma() const { return this->rightCols(order()); }

  /// Covariance modifier.
  /// \return Covariance.
  inline auto sigma() { return this->rightCols(order()); }

  /// Inverse covariance.
  /// \return Inverse covariance.
  inline auto inverseSigma() const {
    const auto order = this->order();
    return sigma().llt().solve(Eigen::Matrix<Scalar, kOrder, kOrder>::Identity(order, order));
  }

  /// Converts this.
  /// \return Canonical Gaussian.
  auto toCanonicalGaussian() const -> CanonicalGaussian<Scalar, kOrder>;
};

template <typename TScalar, int TOrder>
class Gaussian final
    : public GaussianBase<Gaussian<TScalar, TOrder>> {
 public:
  using Base = GaussianBase<Gaussian<TScalar, TOrder>>;
  using Index = typename Base::Index;
  using Mu = typename Base::Mu;
  using Sigma = typename Base::Sigma;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Gaussian)

  /// Zero Gaussian.
  /// \return Gaussian.
  static auto Zero(const Index& size) -> Gaussian {
    auto gaussian = Gaussian{size};
    gaussian.setZero(size);
    return gaussian;
  }

  /// Default constructor.
  Gaussian() = default;

  /// Constructor from order.
  /// \param order Input order.
  explicit Gaussian(const Index& order) : Base{order, order + 1} {}

  /// Constructor from mu and sigma.
  /// \param mu Input Mu.
  /// \param sigma Input Sigma.
  Gaussian(const Mu& mu, const Sigma& sigma) : Gaussian{mu.size()} {
    this->mu() = mu;
    this->sigma() = sigma;
  }

  /// Constructor from order.
  /// \param order Input order.
  /// \return Gaussian.
  auto setZero(const Index& order) -> Gaussian& {
    Base::setZero(order, order + 1);
    return *this;
  }

 private:
  using Base::conservativeResize;
  using Base::conservativeResizeLike;
  using Base::resize;
  using Base::resizeLike;
  using Base::setConstant;
  using Base::setIdentity;
  using Base::setLinSpaced;
  using Base::setOnes;
  using Base::setRandom;
  using Base::setUnit;
};

template <typename TDerived>
class CanonicalGaussianBase : public Traits<TDerived>::Base {
 public:
  // Definitions.
  using Base = Traits<TDerived>::Base;
  using Index = typename Base::Index;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  // Constants.
  static constexpr auto kOrder = Base::RowsAtCompileTime;
  static constexpr auto kEtaColOffset = 0;
  static constexpr auto kNumEtaCols = 1;
  static constexpr auto kLambdaColOffset = 1;
  static constexpr auto kNumLambdaCols = kOrder;

  using Eta = Eigen::Matrix<Scalar, kOrder, 1>;
  using Lambda = Eigen::Matrix<Scalar, kOrder, kOrder>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(CanonicalGaussianBase)

  /// Retrieves the order.
  /// \return Order.
  [[nodiscard]] auto order() const -> Index {
    return this->rows();
  }

  /// Information accessor.
  /// \return Information.
  inline auto eta() const { return this->template leftCols<1>(); }

  /// Information modifier.
  /// \return Information.
  inline auto eta() { return this->template leftCols<1>(); }

  /// Precision accessor.
  /// \return Precision.
  inline auto lambda() const { return this->rightCols(order()); }

  /// Precision modifier.
  /// \return Precision.
  inline auto lambda() { return this->rightCols(order()); }

  /// Inverse precision.
  /// \return Inverse precision.
  inline auto inverseLambda() const {
    const auto order = this->order();
    return lambda().llt().solve(Eigen::Matrix<Scalar, kOrder, kOrder>::Identity(order, order));
  }

  /// Converts this.
  /// \return Gaussian.
  auto toGaussian() const -> Gaussian<Scalar, kOrder>;
};

template <typename TScalar, int TOrder>
class CanonicalGaussian final
    : public CanonicalGaussianBase<CanonicalGaussian<TScalar, TOrder>> {
 public:
  using Base = CanonicalGaussianBase<CanonicalGaussian<TScalar, TOrder>>;
  using Index = typename Base::Index;
  using Eta = typename Base::Eta;
  using Lambda = typename Base::Lambda;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(CanonicalGaussian)

  /// Zero Gaussian.
  /// \return Gaussian.
  static auto Zero(const Index& size) -> CanonicalGaussian {
    auto canonical_gaussian = CanonicalGaussian{size};
    canonical_gaussian.setZero(size);
    return canonical_gaussian;
  }

  /// Default constructor.
  CanonicalGaussian() = default;

  /// Constructor from order.
  /// \param order Input order.
  explicit CanonicalGaussian(const Index& order) : Base{order, order + 1} {}

  /// Constructor from eta and lambda.
  /// \param eta Input eta.
  /// \param lambda Input lambda.
  CanonicalGaussian(const Eta& eta, const Lambda& lambda) : CanonicalGaussian{eta.size()} {
    this->eta() = eta;
    this->lambda() = lambda;
  }

  /// Constructor from order.
  /// \param order Input order.
  /// \return Gaussian.
  auto setZero(const Index& order) -> CanonicalGaussian& {
    Base::setZero(order, order + 1);
    return *this;
  }

 private:
  using Base::conservativeResize;
  using Base::conservativeResizeLike;
  using Base::resize;
  using Base::resizeLike;
  using Base::setConstant;
  using Base::setIdentity;
  using Base::setLinSpaced;
  using Base::setOnes;
  using Base::setRandom;
  using Base::setUnit;
};

template <typename TDerived>
auto GaussianBase<TDerived>::toCanonicalGaussian() const -> CanonicalGaussian<Scalar, kOrder> {
  const auto order = this->order();
  CanonicalGaussian<Scalar, kOrder> canonical_gaussian{order};
  canonical_gaussian.lambda().noalias() = inverseSigma();
  canonical_gaussian.eta().noalias() = canonical_gaussian.lambda().lazyProduct(mu());
  return canonical_gaussian;
}

template <typename TDerived>
auto CanonicalGaussianBase<TDerived>::toGaussian() const -> Gaussian<Scalar, kOrder> {
  const auto order = this->order();
  Gaussian<Scalar, kOrder> gaussian{order};
  gaussian.sigma().noalias() = inverseLambda();
  gaussian.mu().noalias() = gaussian.sigma().lazyProduct(eta());
  return gaussian;
}

} // namespace hyper

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(hyper::Gaussian, int)
