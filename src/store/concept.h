#pragma once

#include <concepts>
#include <cstddef>

#include "core/types.h"

namespace vextor {

template <typename T>
concept VectorStore = requires(const T store, Offset offset) {
    { store.get_vector(offset) } -> std::same_as<const float*>;
    { store.size() } -> std::same_as<std::size_t>;
    { store.dimensions() } -> std::same_as<Dim>;
};

}  // namespace vextor