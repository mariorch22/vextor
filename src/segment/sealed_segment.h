#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "core/search_result.h"
#include "core/types.h"
#include "index/hnsw_index.h"
#include "segment/active_segment.h"
#include "segment/id_mapping.h"
#include "store/in_memory_store.h"
#include "store/mmap_store.h"

namespace vexdb {

class SealedSegment {
   public:
    static SealedSegment from_memory(InMemoryStore store, HnswGraph graph, IdMapping id_mapping);
    static SealedSegment from_mmap(MmapStore store, HnswGraph graph, IdMapping id_mapping);

    std::vector<QueryResult> search(const float* query, std::size_t k, int ef_search = 128) const;

    std::size_t size() const;
    Dim dimensions() const;

   private:
    SealedSegment() = default;

    struct Backend {
        virtual ~Backend() = default;
        virtual std::vector<SearchResult> search(const float* query, std::size_t k,
                                                 int ef_search) const = 0;
        virtual std::size_t size() const = 0;
        virtual Dim dimensions() const = 0;
    };

    template <typename Store>
    struct TypedBackend : Backend {
        Store store;
        HnswIndex<Store> index;

        TypedBackend(Store s, int m) : store(std::move(s)), index(store, m) {}

        std::vector<SearchResult> search(const float* query, std::size_t k,
                                         int ef_search) const override {
            return index.search(query, k, ef_search);
        }
        std::size_t size() const override { return store.size(); }
        Dim dimensions() const override { return store.dimensions(); }
    };

    std::unique_ptr<Backend> backend_;
    IdMapping id_mapping_;
};

}  // namespace vexdb