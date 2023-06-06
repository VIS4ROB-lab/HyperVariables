/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>

#include <Eigen/Cholesky>

#include "hyper/stochastics/forward.hpp"

namespace hyper::stochastics {

template <int TOrder>
class Uncertainty {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using Covariance = Matrix<Scalar, TOrder, TOrder>;  ///< Covariance matrix type.
  using Precision = Matrix<Scalar, TOrder, TOrder>;   ///< Precision matrix type.

  /// Initialization as identity.
  /// \param order Order.
  /// \return Uncertainty.
  static auto Identity(Index order = TOrder) -> Uncertainty {
    Uncertainty uncertainty{order};
    uncertainty.setCovariance(Covariance::Identity(order, order));
    return uncertainty;
  }

  /// Initialization from covariance.
  /// \param covariance Covariance.
  /// \return Uncertainty.
  template <typename TDerived_>
  static auto FromCovariance(const Eigen::EigenBase<TDerived_>& covariance) -> Uncertainty {
    Uncertainty uncertainty{covariance.rows()};
    uncertainty.setCovariance(covariance);
    return uncertainty;
  }

  /// Initialization from precision.
  /// \param covariance Precision.
  /// \return Uncertainty.
  template <typename TDerived_>
  static auto FromPrecision(const Eigen::EigenBase<TDerived_>& precision) -> Uncertainty {
    Uncertainty uncertainty{precision.rows()};
    uncertainty.setPrecision(precision);
    return uncertainty;
  }

  /// Constructor from order.
  /// \param order Order.
  explicit Uncertainty(Index order = TOrder) : covariance_{order, order}, precision_{order, order} {}

  /// Order accessor.
  /// \return Order.
  [[nodiscard]] inline auto order() const -> Index { return covariance_.rows(); }

  /// Covariance accessor.
  /// \return Covariance.
  inline auto covariance() const { return covariance_.template selfadjointView<Eigen::Lower>(); }

  /// Precision accessor.
  /// \return Precision.
  inline auto precision() const { return precision_.template selfadjointView<Eigen::Lower>(); }

  /// Square root covariance.
  /// \return Square root covariance.
  inline auto sqrtCovariance() const -> const Eigen::LLT<Covariance, Eigen::Lower>& { return covariance_llt_; }

  /// Square root precision.
  /// \return Square root precision.
  inline auto sqrtPrecision() const -> const Eigen::LLT<Precision, Eigen::Lower>& { return precision_llt_; }

  /// Covariance setter.
  /// \param covariance Covariance.
  template <typename TDerived_>
  auto setCovariance(const Eigen::EigenBase<TDerived_>& covariance) -> void {
    const auto& derived = covariance.const_derived();
    if constexpr (std::is_base_of_v<Eigen::MatrixBase<TDerived_>, TDerived_>) {
      DCHECK(derived.isApprox(derived.transpose()) || derived.isLowerTriangular());
    }
    covariance_.template triangularView<Eigen::Lower>() = derived;
    covariance_llt_ = this->covariance().llt();
    DCHECK(covariance_llt_.info() != Eigen::NumericalIssue);
    precision_.template triangularView<Eigen::Lower>() = covariance_llt_.solve(Covariance::Identity(order(), order()));
    precision_llt_ = this->precision().llt();
    DCHECK(precision_llt_.info() != Eigen::NumericalIssue);
  }

  /// Precision setter.
  /// \param precision Precision.
  template <typename TDerived_>
  auto setPrecision(const Eigen::EigenBase<TDerived_>& precision) -> void {
    const auto& derived = precision.const_derived();
    if constexpr (std::is_base_of_v<Eigen::MatrixBase<TDerived_>, TDerived_>) {
      DCHECK(derived.isApprox(derived.transpose()) || derived.isLowerTriangular());
    }
    precision_.template triangularView<Eigen::Lower>() = derived;
    precision_llt_ = this->precision().llt();
    DCHECK(precision_llt_.info() != Eigen::NumericalIssue);
    covariance_.template triangularView<Eigen::Lower>() = precision_llt_.solve(Precision::Identity(order(), order()));
    covariance_llt_ = this->covariance().llt();
    DCHECK(covariance_llt_.info() != Eigen::NumericalIssue);
  }

 private:
  Covariance covariance_;                                ///< Covariance.
  Eigen::LLT<Covariance, Eigen::Lower> covariance_llt_;  ///< Cholesky decomposition of covariance.
  Precision precision_;                                  ///< Precision (i.e. inverse covariance).
  Eigen::LLT<Precision, Eigen::Lower> precision_llt_;    ///< Cholesky decomposition of precision.
};

}  // namespace hyper::stochastics
