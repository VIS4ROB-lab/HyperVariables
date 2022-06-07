/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "variables/forward.hpp"

#include <algorithm>

namespace hyper {

/// Address and size.
/// \tparam TValue Value type.
/// \tparam TSize Size type.
template <typename TValue, typename TSize>
struct MemoryBlock {
  using Value = TValue;
  using Address = TValue*;
  using Size = TSize;
  Address address;
  Size size;
};

/// Collection of memory blocks.
/// \tparam TValue Value type.
/// \tparam TSize Size type.
template <typename TValue, typename TSize>
class MemoryBlocks : public std::vector<MemoryBlock<TValue, TSize>> {
 public:
  /// Inherited constructors.
  using Base = std::vector<MemoryBlock<TValue, TSize>>;
  using Base::Base;

  /// Collects the memory block addresses.
  /// \tparam TValue_ Target value type.
  /// \return Memory block addresses.
  template <typename TValue_ = TValue>
  [[nodiscard]] auto addresses() const -> std::vector<TValue_*> {
    std::vector<TValue_*> v(this->size());
    std::transform(this->begin(), this->end(), v.begin(), [](const auto& element) { return element.address; });
    return v;
  }

  /// Collects the memory block sizes.
  /// \tparam TSize_ Target size type.
  /// \return Memory block sizes.
  template <typename TSize_ = TSize>
  [[nodiscard]] auto sizes() const -> std::vector<TSize_> {
    std::vector<TSize_> v(this->size());
    std::transform(this->begin(), this->end(), v.begin(), [](const auto& element) { return static_cast<TSize_>(element.size); });
    return v;
  }
};

} // namespace hyper
