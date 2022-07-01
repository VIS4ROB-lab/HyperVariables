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
  using Variable = AbstractVariable<TScalar>;
  using Variables = std::vector<std::unique_ptr<Variable>>;
  using Size = typename Variables::size_type;

  /// Deleted default constructor.
  CompositeVariable() = delete;

  /// Default destructor.
  virtual ~CompositeVariable() = default;

  /// Retrieves the number of variables.
  /// \return Number of variables.
  [[nodiscard]] auto numVariables() const -> Size {
    return variables_.size();
  }

  /// Variables accessor.
  /// \return variables.
  [[nodiscard]] auto variables() const -> const Variables& {
    return variables_;
  }

  /// Variable modifier.
  /// \param index Variable index.
  /// \return Variable.
  [[nodiscard]] auto variable(const Size index) const -> Variable& {
    DCHECK_LT(index, variables_.size());
    DCHECK(variables_[index] != nullptr);
    return *variables_[index];
  }

  /// Memory block collector.
  /// \return Memory blocks.
  [[nodiscard]] virtual auto memoryBlocks() const -> MemoryBlocks<TScalar> {
    MemoryBlocks<TScalar> memory_blocks;
    memory_blocks.reserve(variables_.size());
    for (const auto& variable : variables_) {
      memory_blocks.template emplace_back(variable->memory());
    }
    return memory_blocks;
  }

 protected:
  /// Constructor from number of variables.
  /// \param num_variables Number of variables.
  explicit CompositeVariable(int num_variables)
      : variables_(num_variables) {}

  /// Sets a variable at index.
  /// \param index Variable index.
  /// \param variable Input variable.
  auto setVariable(const Size index, std::unique_ptr<Variable>&& variable) -> void {
    DCHECK_LT(index, variables_.size());
    variables_[index] = std::move(variable);
  }

 private:
  Variables variables_; ///< Variables.
};

} // namespace hyper
