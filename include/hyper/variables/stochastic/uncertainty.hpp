/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <glog/logging.h>

#include <Eigen/Cholesky>

#include "hyper/variables/stochastic/forward.hpp"

#include "hyper/matrix.hpp"

namespace hyper::variables {

template <typename TScalar, int TOrder>
class Uncertainty {
 public:
  // Definitions.
  using Scalar = TScalar;
  using Index = Eigen::Index;
  using Covariance = Matrix<TScalar, TOrder, TOrder>;   ///< Covariance matrix type.
  using Information = Matrix<TScalar, TOrder, TOrder>;  ///< Information matrix type.

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
  static auto FromCovariance(const Eigen::MatrixBase<TDerived_>& covariance) -> Uncertainty {
    Uncertainty uncertainty{covariance.rows()};
    uncertainty.setCovariance(covariance);
    return uncertainty;
  }

  /// Initialization from information.
  /// \param covariance Information.
  /// \return Uncertainty.
  template <typename TDerived_>
  static auto FromInformation(const Eigen::MatrixBase<TDerived_>& information) -> Uncertainty {
    Uncertainty uncertainty{information.rows()};
    uncertainty.setInformation(information);
    return uncertainty;
  }

  /// Constructor from order.
  /// \param order Order.
  explicit Uncertainty(Index order = TOrder) : covariance_{order, order}, information_{order, order} {}

  /// Order accessor.
  /// \return Order.
  [[nodiscard]] inline auto order() const -> Index { return covariance_.rows(); }

  /// Covariance accessor.
  /// \return Covariance.
  inline auto covariance() const { return covariance_.template selfadjointView<Eigen::Lower>(); }

  /// Information accessor.
  /// \return Information.
  inline auto information() const { return information_.template selfadjointView<Eigen::Lower>(); }

  /// Square root covariance.
  /// \return Square root covariance.
  inline auto sqrtCovariance() const -> const Eigen::LLT<Covariance, Eigen::Lower>& { return covariance_llt_; }

  /// Square root information.
  /// \return Square root information.
  inline auto sqrtInformation() const -> const Eigen::LLT<Information, Eigen::Lower>& { return information_llt_; }

  /// Covariance setter.
  /// \param covariance Covariance.
  template <typename TDerived_>
  auto setCovariance(const Eigen::MatrixBase<TDerived_>& covariance) -> void {
    DCHECK_EQ(covariance_.rows(), covariance.rows());
    DCHECK_EQ(covariance_.cols(), covariance.cols());
    covariance_.template triangularView<Eigen::Lower>() = covariance;
    covariance_llt_ = this->covariance().llt();
    information_.template triangularView<Eigen::Lower>() = covariance_llt_.solve(Covariance::Identity(order(), order()));
    information_llt_ = this->information().llt();
  }

  /// Information setter.
  /// \param information Information.
  template <typename TDerived_>
  auto setInformation(const Eigen::MatrixBase<TDerived_>& information) -> void {
    DCHECK_EQ(information_.rows(), information.rows());
    DCHECK_EQ(information_.cols(), information.cols());
    information_.template triangularView<Eigen::Lower>() = information;
    information_llt_ = this->information().llt();
    covariance_.template triangularView<Eigen::Lower>() = information_llt_.solve(Information::Identity(order(), order()));
    covariance_llt_ = this->covariance().llt();
  }

 private:
  Covariance covariance_;                                  ///< Covariance.
  Eigen::LLT<Covariance, Eigen::Lower> covariance_llt_;    ///< Cholesky decomposition of covariance.
  Information information_;                                ///< Information (i.e. inverse covariance).
  Eigen::LLT<Information, Eigen::Lower> information_llt_;  ///< Cholesky decomposition of information.
};

}  // namespace hyper::variables
