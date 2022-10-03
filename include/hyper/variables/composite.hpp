/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <vector>

#include <glog/logging.h>

#include "hyper/variables/abstract.hpp"

namespace hyper {

template <typename TScalar>
class CompositeVariable {
 public:
  using Size = std::size_t;
  using Variable = AbstractVariable<TScalar>;
  using Variables = std::vector<std::unique_ptr<Variable>>;

  /// Default constructor.
  CompositeVariable() = default;

  /// Constructor from number of variables.
  /// \param num_variables Number of variables.
  explicit CompositeVariable(const Size& num_variables)
      : variables_(num_variables) {}

  /// Variables accessor.
  /// \return Variables.
  [[nodiscard]] auto variables() const -> const Variables& {
    return variables_;
  }

  /// Variables modifier.
  /// \return Variables.
  [[nodiscard]] auto variables() -> Variables& {
    return const_cast<Variables&>(std::as_const(*this).variables());
  }

  /// Variable accessor.
  /// \param index Variable index.
  /// \return Variable.
  [[nodiscard]] auto variable(const Size& index) const -> Variable& {
    DCHECK_LT(index, variables_.size());
    DCHECK(variables_[index] != nullptr);
    return *variables_[index];
  }

  /// Sets a variable at index.
  /// \param index Variable index.
  /// \param variable Input variable.
  auto setVariable(const Size& index, std::unique_ptr<Variable>&& variable) -> void {
    DCHECK_LT(index, variables_.size());
    variables_[index] = std::move(variable);
  }

 private:
  Variables variables_; ///< Variables.
};

} // namespace hyper
