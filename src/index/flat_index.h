#pragma once

#include <algorithm>
#include <cstddef>
#include <queue>
#include <vector>

#include "core/distance.h"
#include "core/search_result.h"
#include "core/types.h"
#include "store/concept.h"

namespace vexdb {

template <VectorStore Store>
class FlatIndex {
   public:
    explicit FlatIndex(const Store& store) : store_(store) {}

    // Brute-force search: compute distance to every vector, return top-k closest.
    [[nodiscard]] std::vector<SearchResult> search(const float* query, std::size_t k) const {
        if (k == 0 || store_.size() == 0) return {};

        // Max-heap: farthest on top, so we can evict the worst candidate.
        std::priority_queue<SearchResult> pq;

        for (Offset i = 0; i < static_cast<Offset>(store_.size()); i++) {
            float dist = l2_distance(query, store_.get_vector(i), store_.dimensions());

            if (pq.size() < k) {
                pq.push({.offset = i, .distance = dist});
            } else if (dist < pq.top().distance) {
                pq.pop();
                pq.push({.offset = i, .distance = dist});
            }
        }

        // Drain heap into vector, closest first.
        std::vector<SearchResult> results;
        results.reserve(pq.size());
        while (!pq.empty()) {
            results.push_back(pq.top());
            pq.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

   private:
    const Store& store_;
};

}  // namespace vexdb