/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#include "hyper/variables/memory.hpp"

namespace hyper {

template <typename TScalar>
class AbstractVariable {
 public:
  /// Virtual default destructor.
  virtual ~AbstractVariable() = default;

  /// Memory accessor.
  /// \return Memory block.
  [[nodiscard]] virtual auto memory() const -> MemoryBlock<std::add_const_t<TScalar>> = 0;

  /// Memory modifier.
  /// \return Memory block.
  [[nodiscard]] virtual auto memory() -> MemoryBlock<TScalar> = 0;
};

} // namespace hyper
