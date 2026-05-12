#pragma once

#include <compare>

#include "core/types.h"

namespace vexdb {

struct SearchResult {
    Offset offset;
    float distance;

    // Natural ordering: smaller distance = "less than".
    // For std::priority_queue (max-heap by default), use std::greater<SearchResult>
    // to keep the closest result on top.
    std::partial_ordering operator<=>(const SearchResult& other) const {
        return distance <=> other.distance;
    }

    bool operator==(const SearchResult& other) const = default;
};

}  // namespace vexdb